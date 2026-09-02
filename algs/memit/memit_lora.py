"""All-Transformer-layer LoRA editing with E2E CE and covariance preservation.

PEFT owns LoRA injection, trainable adapter parameters, and safe merging.  The
repository keeps ownership of the global edit dataset, E2E cross-entropy,
covariance-trace preservation, optimizer loop, and evaluation checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

PEFT_IMPORT_ERROR = None
try:
    import peft
    from peft import LoraConfig, TaskType, get_peft_model
    from peft.tuners.lora import LoraLayer
except ImportError as error:  # Keep non-LoRA algorithms importable without PEFT.
    PEFT_IMPORT_ERROR = error
    peft = None
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    LoraLayer = None

from util import nethook

from .memit_joint import (
    build_joint_sample_loss_weights,
    build_joint_training_samples,
    chunks,
    get_joint_ce_losses,
    get_stable_joint_context_templates,
    prepare_joint_ce_batch,
)


LORA_CHECKPOINT_FORMAT = "memit_lora_v1"
SUPPORTED_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "gate_up_proj",
    "down_proj",
)


@dataclass
class LoraAdapter:
    module_name: str
    module: object
    adapter_name: str
    covariance_path: Path

    @property
    def a(self) -> torch.nn.Parameter:
        return self.module.lora_A[self.adapter_name].weight

    @property
    def b(self) -> torch.nn.Parameter:
        return self.module.lora_B[self.adapter_name].weight

    @property
    def scale(self) -> float:
        return float(self.module.scaling[self.adapter_name])

    @property
    def transpose_for_weight(self) -> bool:
        return bool(getattr(self.module, "fan_in_fan_out", False))

    def delta_linear(self) -> torch.Tensor:
        """Return Delta W in conventional [out_features, in_features] form."""
        return (self.b @ self.a) * self.scale


def apply_memit_lora_to_model(model, tok, requests, cfg):
    """Train all-layer LoRA adapters and merge them into ``model``."""
    if peft is None:
        raise ImportError(
            "MEMIT-LoRA requires PEFT. Install the project requirements or "
            "run `pip install peft==0.15.2`."
        ) from PEFT_IMPORT_ERROR
    if not requests:
        raise ValueError("MEMIT-LoRA requires at least one edit")
    if not hasattr(cfg.llms, "memit_lora"):
        raise ValueError(
            f"Model config {cfg.llms.alias!r} has no llms.memit_lora block"
        )
    hparams = cfg.llms.memit_lora
    _validate_hparams(hparams)
    # Keep dropout disabled: only the explicit LoRA factors are optimized.
    model.eval()
    device = torch.device(
        f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"
    )
    requests = [dict(request) for request in requests]
    for request in requests:
        request["target_new"] = " " + request["target_new"]

    layers = _parse_layers(hparams.layers, _num_transformer_layers(model, cfg))
    targets = [str(target) for target in hparams.target_modules]
    if len(targets) != len(set(targets)):
        raise ValueError("MEMIT-LoRA target_modules must not contain duplicates")
    unknown = sorted(set(targets) - set(SUPPORTED_TARGETS))
    if unknown:
        raise ValueError(
            f"Unsupported LoRA target modules {unknown}; supported targets are "
            f"{list(SUPPORTED_TARGETS)}"
        )
    module_names = [
        _target_module_name(cfg, layer, target)
        for layer in layers
        for target in targets
    ]
    missing = [
        name for name in module_names if not _module_exists(model, name)
    ]
    if missing:
        preview = "\n  ".join(missing[:20])
        raise ValueError(
            "Some all-layer LoRA targets do not exist in this model:\n  "
            f"{preview}"
        )

    covariance_paths = _resolve_covariance_paths(
        cfg=cfg,
        required_modules=module_names,
        explicit_manifests=[str(path) for path in hparams.covariance_manifests],
        cache_namespace=str(hparams.covariance_cache_namespace),
    )
    covariance_device = _resolve_covariance_device(
        str(hparams.covariance_device), device
    )
    covariances = _load_covariances(
        covariance_paths, covariance_device=covariance_device
    )

    peft_model, training_model, adapters = _build_peft_adapters(
        model=model,
        module_names=module_names,
        covariance_paths=covariance_paths,
        rank=int(hparams.rank),
        lora_alpha=float(hparams.lora_alpha),
    )
    parameters = _get_only_peft_trainable_parameters(peft_model, adapters)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(hparams.lr),
        weight_decay=float(hparams.weight_decay),
    )

    context_templates = (
        get_stable_joint_context_templates(
            cfg, training_model, tok, cache_entries=[]
        )
        if bool(cfg.algs.joint_context_training)
        else [["{}"]]
    )
    training_samples = build_joint_training_samples(
        requests=requests,
        context_templates=context_templates,
        use_context_templates=bool(cfg.algs.joint_context_training),
    )
    num_edits = len(requests)
    num_contexts = len(training_samples) // num_edits
    sample_weights = torch.tensor(
        build_joint_sample_loss_weights(
            training_samples,
            num_edits,
            hparams.original_prompt_loss_share,
        ),
        dtype=torch.float64,
    )
    original_share = sum(
        weight.item()
        for weight, sample in zip(sample_weights, training_samples)
        if sample["is_original_prompt"]
    ) / num_edits
    print(
        f"MEMIT-LoRA (PEFT {peft.__version__}): {len(adapters)} adapters "
        f"across {len(layers)} layers; "
        f"rank={int(hparams.rank)}, scale="
        f"{float(hparams.lora_alpha) / int(hparams.rank):.6g}"
    )
    print(
        f"MEMIT-LoRA training set: {num_edits} edits * {num_contexts} prompts; "
        f"original/context loss share={original_share:.2%}/"
        f"{1.0 - original_share:.2%}"
    )
    print(
        f"MEMIT-LoRA covariance preservation: alpha="
        f"{float(hparams.preservation_alpha):.3e}, device={covariance_device}, "
        f"{len(covariances)} unique matrices"
    )

    checkpoint_path, checkpoint_metadata = _checkpoint_entry(
        cfg, requests, adapters, training_samples, covariance_paths
    )
    start_epoch = _load_training_checkpoint(
        checkpoint_path,
        checkpoint_metadata,
        adapters,
        optimizer,
        resume=bool(cfg.algs.checkpoint_resume),
    )
    previous_epoch_loss = None
    try:
        for epoch in range(start_epoch, int(hparams.epochs)):
            permutation = torch.randperm(len(training_samples)).tolist()
            optimizer.zero_grad(set_to_none=True)
            epoch_target_sum = 0.0
            per_edit_loss_sum = torch.zeros(num_edits, dtype=torch.float64)
            per_edit_weight_sum = torch.zeros(num_edits, dtype=torch.float64)
            micro_batches = (
                len(training_samples) + int(hparams.micro_batch_size) - 1
            ) // int(hparams.micro_batch_size)

            for micro_step, batch_indices in enumerate(
                chunks(permutation, int(hparams.micro_batch_size)), start=1
            ):
                samples = [training_samples[index] for index in batch_indices]
                model_inputs, target_positions, target_ids = prepare_joint_ce_batch(
                    tok=tok, samples=samples, device=device
                )
                losses = get_joint_ce_losses(
                    model=training_model,
                    model_inputs=model_inputs,
                    target_positions=target_positions,
                    target_ids=target_ids,
                    loss_layer=max(int(cfg.llms.v_loss_layer), layers[-1]),
                    layer_module_tmp=cfg.llms.layer_module_tmp,
                    ln_f_module=cfg.llms.ln_f_module,
                    lm_head_module=cfg.llms.lm_head_module,
                )
                if not torch.isfinite(losses).all():
                    raise FloatingPointError(
                        f"Non-finite LoRA CE at epoch {epoch + 1}, "
                        f"micro-batch {micro_step}"
                    )
                gpu_weights = sample_weights[batch_indices].to(
                    device=losses.device, dtype=losses.dtype
                )
                weighted_sum = (losses * gpu_weights).sum()
                (weighted_sum / num_edits).backward()
                epoch_target_sum += weighted_sum.detach().item()

                request_indices = torch.tensor(
                    [sample["request_index"] for sample in samples],
                    dtype=torch.int64,
                )
                cpu_weights = sample_weights[batch_indices]
                per_edit_loss_sum.index_add_(
                    0,
                    request_indices,
                    losses.detach().cpu().to(torch.float64) * cpu_weights,
                )
                per_edit_weight_sum.index_add_(0, request_indices, cpu_weights)
                if (
                    micro_step == 1
                    or micro_step % int(cfg.algs.log_interval) == 0
                    or micro_step == micro_batches
                ):
                    local_mean = weighted_sum / gpu_weights.sum()
                    print(
                        f"lora epoch {epoch + 1}/{int(hparams.epochs)}, "
                        f"micro-batch {micro_step}/{micro_batches}: CE "
                        f"{local_mean.detach().item():.6e} "
                        "(gradient accumulated)"
                    )

            preserve_value = _backward_preservation(
                adapters=adapters,
                covariances=covariances,
                coefficient=float(hparams.preservation_alpha),
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(hparams.max_grad_norm)
            ).item()
            if not np.isfinite(grad_norm):
                raise FloatingPointError("Non-finite MEMIT-LoRA gradient norm")
            optimizer.step()

            target_value = epoch_target_sum / num_edits
            epoch_loss = target_value + float(hparams.preservation_alpha) * preserve_value
            adapter_norm = _global_adapter_delta_norm(adapters)
            per_edit_losses = per_edit_loss_sum / per_edit_weight_sum
            median = torch.quantile(per_edit_losses.float(), 0.5).item()
            p90 = torch.quantile(per_edit_losses.float(), 0.9).item()
            maximum = per_edit_losses.max().item()
            print(
                f"lora epoch {epoch + 1} complete: loss {epoch_loss:.6e} = "
                f"target {target_value:.6e} + preservation_alpha "
                f"{float(hparams.preservation_alpha):.3e} * "
                f"{preserve_value:.6e}; gradient norm {grad_norm:.6e}; "
                f"adapter delta norm {adapter_norm:.6e}; per-edit CE "
                f"median/p90/max {median:.6e}/{p90:.6e}/{maximum:.6e}"
            )

            should_save = (
                (epoch + 1) % int(hparams.checkpoint_interval) == 0
                or epoch + 1 == int(hparams.epochs)
            )
            if bool(cfg.algs.checkpoint_enabled) and should_save:
                _save_training_checkpoint(
                    checkpoint_path,
                    checkpoint_metadata,
                    epoch + 1,
                    epoch_loss,
                    adapters,
                    optimizer,
                )
                _save_evaluation_adapter_checkpoint(cfg, adapters, epoch + 1)

            if previous_epoch_loss is not None:
                relative_change = abs(previous_epoch_loss - epoch_loss) / max(
                    abs(previous_epoch_loss), torch.finfo(torch.float32).eps
                )
                if relative_change <= float(cfg.algs.tolerance):
                    print(
                        f"MEMIT-LoRA converged at epoch {epoch + 1}: "
                        f"relative loss change {relative_change:.3e}"
                    )
                    break
            previous_epoch_loss = epoch_loss
    except BaseException:
        # PEFT mutates the supplied model in place when it injects adapters.
        # Remove those wrappers without merging if training is interrupted or
        # fails, so callers never receive a silently partially edited model.
        peft_model.unload()
        raise

    payload = _adapter_payload(adapters)
    merged_model = peft_model.merge_and_unload(safe_merge=True)
    # save_model() consumes this compact payload instead of writing nearly a
    # second full model. load_model() knows how to merge it for test_only.
    merged_model._memit_lora_payload = payload
    return merged_model


def _num_transformer_layers(model, cfg) -> int:
    config = model.config
    value = getattr(config, "num_hidden_layers", None)
    if value is None and hasattr(config, "text_config"):
        value = getattr(config.text_config, "num_hidden_layers", None)
    if value is None:
        raise ValueError("Cannot determine the number of Transformer layers")
    count = int(value)
    nethook.get_module(model, cfg.llms.layer_module_tmp.format(count - 1))
    return count


def _validate_hparams(hparams):
    positive = {
        "rank": int(hparams.rank),
        "lora_alpha": float(hparams.lora_alpha),
        "lr": float(hparams.lr),
        "epochs": int(hparams.epochs),
        "micro_batch_size": int(hparams.micro_batch_size),
        "max_grad_norm": float(hparams.max_grad_norm),
        "checkpoint_interval": int(hparams.checkpoint_interval),
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"MEMIT-LoRA parameters must be positive: {invalid}")
    if float(hparams.weight_decay) < 0:
        raise ValueError("MEMIT-LoRA weight_decay must be non-negative")
    if float(hparams.preservation_alpha) < 0:
        raise ValueError("MEMIT-LoRA preservation_alpha must be non-negative")
    if not hparams.target_modules:
        raise ValueError("MEMIT-LoRA target_modules must not be empty")


def _parse_layers(value, count: int) -> List[int]:
    if value is None or str(value).lower() == "all":
        return list(range(count))
    layers = sorted({int(layer) for layer in value})
    if not layers or layers[0] < 0 or layers[-1] >= count:
        raise ValueError(f"LoRA layers must be within 0..{count - 1}")
    return layers


def _target_module_name(cfg, layer: int, target: str) -> str:
    if target in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return f"{cfg.llms.attn_module_tmp.format(layer)}.{target}"
    return f"{cfg.llms.mlp_module_tmp.format(layer)}.{target}"


def _module_exists(model, name: str) -> bool:
    try:
        nethook.get_module(model, name)
        return True
    except LookupError:
        return False


def _resolve_covariance_device(value: str, training_device) -> torch.device:
    if value == "cuda":
        if training_device.type != "cuda":
            raise ValueError("covariance_device=cuda requires CUDA")
        return training_device
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError("covariance_device must be either 'cuda' or 'cpu'")


def _resolve_covariance_paths(
    cfg, required_modules: Sequence[str], explicit_manifests, cache_namespace
) -> Dict[str, Path]:
    stats_dir = (
        Path(cfg.cache_dir)
        / "stats"
        / str(cfg.llms.alias).replace("/", "-")
    )
    manifest_paths = [Path(path) for path in explicit_manifests]
    if not manifest_paths:
        manifest_paths = sorted(
            stats_dir.glob(f"{cache_namespace}-*-manifest.json")
        )
    if not manifest_paths:
        raise FileNotFoundError(
            f"No LoRA covariance manifests found in {stats_dir}; run "
            "experiments/precompute_lora_covariances.py first"
        )

    required = set(required_modules)
    candidates: Dict[str, set] = {name: set() for name in required}
    compatible_manifests = 0
    for manifest_path in manifest_paths:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if (
            manifest.get("model_alias") != str(cfg.llms.alias)
            or manifest.get("dataset") != str(cfg.llms.mom2_dataset)
            or int(manifest.get("sample_size", -1))
            != int(cfg.llms.mom2_n_samples)
            or manifest.get("precision") != str(cfg.llms.mom2_dtype)
            or bool(manifest.get("probability_weighted", False))
            != bool(cfg.cov_probability_weighted)
        ):
            continue
        compatible_manifests += 1
        for module_name, mapping in manifest.get(
            "module_to_covariance", {}
        ).items():
            if module_name not in required:
                continue
            cache_path = Path(mapping["cache_file"])
            if not cache_path.is_file():
                fallback = manifest_path.parent / cache_path.name
                cache_path = fallback if fallback.is_file() else cache_path
            if cache_path.is_file():
                candidates[module_name].add(cache_path.resolve())

    missing = sorted(name for name, paths in candidates.items() if not paths)
    ambiguous = {
        name: sorted(str(path) for path in paths)
        for name, paths in candidates.items()
        if len(paths) > 1
    }
    if missing:
        preview = "\n  ".join(missing[:20])
        raise FileNotFoundError(
            f"Compatible manifests: {compatible_manifests}. Missing covariance "
            f"for {len(missing)} LoRA modules:\n  {preview}"
        )
    if ambiguous:
        name, paths = next(iter(ambiguous.items()))
        raise ValueError(
            f"Multiple covariance caches map to {name}: {paths}. Set "
            "llms.memit_lora.covariance_manifests explicitly."
        )
    return {name: next(iter(paths)) for name, paths in candidates.items()}


def _load_covariances(paths_by_module, covariance_device):
    tensors: Dict[Path, torch.Tensor] = {}
    for index, path in enumerate(sorted(set(paths_by_module.values())), start=1):
        with np.load(path, allow_pickle=False) as state:
            matrix = np.asarray(state["mom2.mom2"])
            if "mom2.weight_sum" in state:
                denominator = float(state["mom2.weight_sum"])
            else:
                denominator = float(state["mom2.count"])
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Invalid covariance shape {matrix.shape} in {path}")
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError(f"Invalid covariance denominator in {path}")
        covariance = torch.from_numpy(matrix).to(
            device=covariance_device, dtype=torch.float32
        )
        covariance.div_(denominator)
        tensors[path] = covariance
        print(
            f"Loaded LoRA covariance {index}/"
            f"{len(set(paths_by_module.values()))}: {path} "
            f"shape={tuple(covariance.shape)} device={covariance.device}"
        )
    return tensors


def _build_peft_adapters(
    model, module_names, covariance_paths, rank: int, lora_alpha: float
):
    if rank <= 0 or lora_alpha <= 0:
        raise ValueError("LoRA rank and lora_alpha must be positive")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        # Full paths make injection exact: PEFT will not adapt a same-named
        # auxiliary layer outside the configured Transformer blocks.
        target_modules=list(module_names),
        init_lora_weights=True,
    )
    peft_model = get_peft_model(
        model,
        lora_config,
        # Keep PEFT's trainable factors in fp32 even when the base model is
        # bf16/fp16.  Besides being PEFT's stable-training default, this keeps
        # the A C A^T covariance calculation in fp32 without hidden casts.
        autocast_adapter_dtype=True,
    )
    # PEFT parameters remain trainable in eval mode; this only keeps base-model
    # dropout and other stochastic inference layers disabled.
    peft_model.eval()
    training_model = peft_model.get_base_model()
    adapter_name = peft_model.active_adapter
    if not isinstance(adapter_name, str):
        raise RuntimeError(
            f"Expected one active PEFT adapter, got {adapter_name!r}"
        )

    adapters = []
    for module_name in module_names:
        module = nethook.get_module(training_model, module_name)
        if not isinstance(module, LoraLayer):
            raise TypeError(
                f"PEFT did not replace target {module_name} with a LoraLayer; "
                f"got {type(module).__name__}"
            )
        if (
            adapter_name not in module.lora_A
            or adapter_name not in module.lora_B
            or adapter_name not in module.scaling
        ):
            raise RuntimeError(
                f"PEFT adapter {adapter_name!r} is missing from {module_name}"
            )
        adapters.append(
            LoraAdapter(
                module_name=module_name,
                module=module,
                adapter_name=adapter_name,
                covariance_path=covariance_paths[module_name],
            )
        )
    if len(adapters) != len(module_names):
        raise RuntimeError(
            f"PEFT created {len(adapters)} adapters for "
            f"{len(module_names)} requested modules"
        )
    peft_model.print_trainable_parameters()
    return peft_model, training_model, adapters


def _get_only_peft_trainable_parameters(peft_model, adapters):
    expected = {
        id(parameter)
        for adapter in adapters
        for parameter in (adapter.a, adapter.b)
    }
    trainable = [
        parameter for parameter in peft_model.parameters() if parameter.requires_grad
    ]
    actual = {id(parameter) for parameter in trainable}
    if actual != expected:
        unexpected = [
            name
            for name, parameter in peft_model.named_parameters()
            if parameter.requires_grad and id(parameter) not in expected
        ]
        missing = len(expected - actual)
        raise RuntimeError(
            "PEFT trainable parameters do not exactly match configured LoRA "
            f"A/B factors: {missing} missing, unexpected={unexpected[:20]}"
        )
    if not trainable:
        raise RuntimeError("PEFT created no trainable LoRA parameters")
    return trainable


def _backward_preservation(adapters, covariances, coefficient: float) -> float:
    if coefficient < 0:
        raise ValueError("preservation_alpha must be non-negative")
    if coefficient == 0:
        return 0.0
    grouped: Dict[Path, List[LoraAdapter]] = {}
    for adapter in adapters:
        grouped.setdefault(adapter.covariance_path, []).append(adapter)
    total = 0.0
    for path, path_adapters in grouped.items():
        cached_covariance = covariances[path]
        adapter_device = path_adapters[0].a.device
        covariance = cached_covariance.to(adapter_device)
        for adapter in path_adapters:
            if covariance.shape != (adapter.a.shape[1], adapter.a.shape[1]):
                raise ValueError(
                    f"Covariance {path} shape {tuple(covariance.shape)} does "
                    f"not match {adapter.module_name} input {adapter.a.shape[1]}"
                )
            a_cov_a_t = (adapter.a @ covariance) @ adapter.a.T
            b_t_b = adapter.b.T @ adapter.b
            loss = (adapter.scale ** 2) * torch.trace(b_t_b @ a_cov_a_t)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite preservation loss for {adapter.module_name}"
                )
            (coefficient * loss).backward()
            total += loss.detach().item()
        if covariance.data_ptr() != cached_covariance.data_ptr():
            del covariance
    return total


def _global_adapter_delta_norm(adapters) -> float:
    squared = 0.0
    with torch.no_grad():
        for adapter in adapters:
            value = torch.linalg.vector_norm(adapter.delta_linear()).item()
            squared += value * value
    return float(squared ** 0.5)


def _adapter_payload(adapters):
    return {
        "__format__": LORA_CHECKPOINT_FORMAT,
        "lora_backend": "peft",
        "peft_version": str(peft.__version__),
        "adapters": {
            adapter.module_name: {
                "a": adapter.a.detach().cpu().to(torch.float32),
                "b": adapter.b.detach().cpu().to(torch.float32),
                "scale": float(adapter.scale),
                "transpose_for_weight": bool(adapter.transpose_for_weight),
            }
            for adapter in adapters
        },
    }


def _save_evaluation_adapter_checkpoint(cfg, adapters, epoch):
    load_name = f"{cfg.save_name}-epoch-{epoch:04d}"
    path = (
        Path(cfg.cache_dir)
        / "saved_weights"
        / cfg.algs.name
        / f"{cfg.data}-{load_name}-{str(cfg.llms.alias).replace('/', '-')}.pt"
    )
    _atomic_torch_save(_adapter_payload(adapters), path)
    print(
        f"Saved evaluation-compatible LoRA checkpoint: {path}; "
        f"use load_name={load_name}"
    )


def _checkpoint_entry(cfg, requests, adapters, training_samples, cov_paths):
    metadata = {
        "format_version": 2,
        "lora_backend": "peft",
        "peft_version": str(peft.__version__),
        "model": str(cfg.llms.alias),
        "modules": [adapter.module_name for adapter in adapters],
        "rank": int(cfg.llms.memit_lora.rank),
        "lora_alpha": float(cfg.llms.memit_lora.lora_alpha),
        "lr": float(cfg.llms.memit_lora.lr),
        "weight_decay": float(cfg.llms.memit_lora.weight_decay),
        "preservation_alpha": float(cfg.llms.memit_lora.preservation_alpha),
        "original_prompt_loss_share": float(
            cfg.llms.memit_lora.original_prompt_loss_share
        ),
        "request_digest": hashlib.sha256(
            json.dumps(requests, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "training_digest": hashlib.sha256(
            json.dumps(training_samples, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "covariances": {
            name: str(path) for name, path in sorted(cov_paths.items())
        },
    }
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    path = (
        Path(cfg.cache_dir)
        / "memit_lora_checkpoints"
        / str(cfg.llms.alias).replace("/", "-")
        / f"{key}.pt"
    )
    return path, metadata


def _save_training_checkpoint(
    path, metadata, completed_epochs, previous_loss, adapters, optimizer
):
    payload = {
        "metadata": metadata,
        "completed_epochs": completed_epochs,
        "previous_epoch_loss": previous_loss,
        "adapter_state": _adapter_payload(adapters),
        "optimizer": optimizer.state_dict(),
    }
    _atomic_torch_save(payload, path)
    print(f"Saved MEMIT-LoRA training checkpoint: {path}")


def _load_training_checkpoint(
    path, metadata, adapters, optimizer, resume: bool
) -> int:
    if not resume or not path.is_file():
        return 0
    try:
        checkpoint = torch.load(path, map_location="cpu")
        if checkpoint["metadata"] != metadata:
            return 0
        state = checkpoint["adapter_state"]["adapters"]
        for adapter in adapters:
            saved = state[adapter.module_name]
            adapter.a.data.copy_(saved["a"].to(adapter.a.device))
            adapter.b.data.copy_(saved["b"].to(adapter.b.device))
        optimizer.load_state_dict(checkpoint["optimizer"])
        completed = int(checkpoint["completed_epochs"])
        print(f"Resumed MEMIT-LoRA from epoch {completed}: {path}")
        return completed
    except (EOFError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Ignoring invalid MEMIT-LoRA checkpoint {path}: {error}")
        return 0


def _atomic_torch_save(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

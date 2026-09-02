#!/usr/bin/env python3
"""Compare original MEMIT and E2E checkpoints with preservation diagnostics.

For every configured MEMIT rewrite layer, this script reports the same raw
covariance penalty used by joint training::

    tr(Delta W C Delta W.T),  Delta W = W_edited - W_base

Subject KL is intentionally *not* decomposed by layer.  It is evaluated once
for each complete edited model at both the configured ``fact_token`` and the
last token of the complete ``"{} is a"`` prompt.  Both use
``KL(P_base || P_edited)``.

Neighborhood diagnostics use the last non-padding prompt token and report:

* ``||Delta W_l h_l^base||^2`` at every configured rewrite module; and
* ``||h_l^edited - h_l^base||^2`` after every Transformer block.

The neighborhood phase keeps one immutable base model and one sequentially
restored working model resident together.  Activations are compared and
discarded one micro-batch at a time, so no multi-gigabyte hidden-state cache is
written to CPU memory or disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


# Running ``python experiments/compare_memit_e2e_losses.py`` otherwise places
# only experiments/ on sys.path and cannot import the repository packages.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from algs.memit.memit_joint import (
    build_subject_kl_samples,
    chunks,
    get_joint_logits_at_positions,
    get_subject_kl_losses,
    prepare_subject_kl_batch,
    trace_preservation_loss,
)
from locate_edit_utils.layer_stats import get_cov
from util import nethook


DEFAULT_DATA = PROJECT_ROOT / "data" / "multi_counterfact_20877.json"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "llms" / "llama3-8b.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-layer covariance loss and whole-model subject KL "
            "for original MEMIT and E2E checkpoints."
        )
    )
    parser.add_argument(
        "--memit-weights",
        type=Path,
        required=True,
        help="Original MEMIT full-weight checkpoint written by save_model().",
    )
    parser.add_argument(
        "--e2e-weights",
        type=Path,
        required=True,
        help="E2E MEMIT full-weight checkpoint written by save_model().",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="configs/llms YAML defining layers and module templates.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Optional override for llms.name in --model-config.",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--record-start", type=int, default=0)
    parser.add_argument("--num-edits", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--neighborhood-batch-size",
        type=int,
        default=32,
        help="Batch size for paired base/edited activation forwards.",
    )
    parser.add_argument(
        "--neighborhood-prompts-per-edit",
        type=int,
        default=0,
        help="0 uses every neighborhood prompt; a positive value uses the first N.",
    )
    parser.add_argument(
        "--max-neighborhood-prompts",
        type=int,
        default=None,
        help="Optional total cap for quick diagnostic runs.",
    )
    parser.add_argument(
        "--skip-neighborhood-diagnostics",
        action="store_true",
        help="Skip Delta-W activation and full-model hidden-drift measurements.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--covariance-device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Device used for one layer's Delta-W/covariance matrix product.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Root containing stats/<model-alias>/layer-N.npz.",
    )
    parser.add_argument(
        "--cov-probability-weighted",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Must match the covariance variant used by editing.",
    )
    parser.add_argument("--cov-prob-weight-epsilon", type=float, default=0.1)
    parser.add_argument("--cov-prob-weight-alpha", type=float, default=0.5)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/memit_e2e_loss_comparison.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional per-layer CSV; defaults beside --output-json.",
    )
    parser.add_argument(
        "--output-neighborhood-csv",
        type=Path,
        default=None,
        help="Optional hidden-drift/update-action CSV; defaults beside JSON.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for label, path in (
        ("MEMIT checkpoint", args.memit_weights),
        ("E2E checkpoint", args.e2e_weights),
        ("model config", args.model_config),
        ("dataset", args.data),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if args.record_start < 0:
        raise ValueError("--record-start must be non-negative")
    if args.num_edits <= 0:
        raise ValueError("--num-edits must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.neighborhood_batch_size <= 0:
        raise ValueError("--neighborhood-batch-size must be positive")
    if args.neighborhood_prompts_per_edit < 0:
        raise ValueError("--neighborhood-prompts-per-edit must be non-negative")
    if (
        args.max_neighborhood_prompts is not None
        and args.max_neighborhood_prompts <= 0
    ):
        raise ValueError("--max-neighborhood-prompts must be positive")
    if args.gpu < 0:
        raise ValueError("--gpu must be non-negative")
    if args.covariance_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--covariance-device cuda was requested but CUDA is unavailable"
        )
    if not 0 <= args.cov_prob_weight_epsilon <= 1:
        raise ValueError("--cov-prob-weight-epsilon must be in [0, 1]")
    if args.cov_prob_weight_alpha <= 0:
        raise ValueError("--cov-prob-weight-alpha must be positive")


def model_dtype(name: str):
    if name == "auto":
        return "auto"
    return getattr(torch, name)


def load_config(args: argparse.Namespace):
    llms = OmegaConf.load(args.model_config)
    if args.base_model is not None:
        llms.name = args.base_model
    required = (
        "name",
        "alias",
        "layers",
        "rewrite_module_tmp",
        "layer_module_tmp",
        "ln_f_module",
        "lm_head_module",
        "fact_token",
        "v_loss_layer",
        "mom2_dataset",
        "mom2_n_samples",
        "mom2_dtype",
        "mom2_maxseqlen",
    )
    missing = [key for key in required if llms.get(key) is None and key != "mom2_maxseqlen"]
    if missing:
        raise ValueError(
            f"Model config {args.model_config} is missing: {', '.join(missing)}"
        )
    cfg = OmegaConf.create(
        {
            "gpu": args.gpu,
            "cache_dir": str(args.cache_dir),
            "cov_probability_weighted": args.cov_probability_weighted,
            "cov_prob_weight_epsilon": args.cov_prob_weight_epsilon,
            "cov_prob_weight_alpha": args.cov_prob_weight_alpha,
            "llms": OmegaConf.to_container(llms, resolve=True),
        }
    )
    return cfg


def load_requests(path: Path, start: int, count: int) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Dataset must contain a JSON list: {path}")
    selected = records[start : start + count]
    if len(selected) != count:
        raise ValueError(
            f"Requested records [{start}, {start + count}), but dataset has "
            f"only {len(records)} records"
        )
    for offset, record in enumerate(selected):
        if not isinstance(record, dict) or not record.get("subject"):
            raise ValueError(
                f"Dataset record {start + offset} has no non-empty subject"
            )
    return selected


def build_neighborhood_prompts(
    requests: Sequence[Mapping[str, Any]],
    prompts_per_edit: int,
    max_prompts: int | None,
) -> List[Dict[str, Any]]:
    """Flatten dataset locality prompts without using their target answers."""
    samples: List[Dict[str, Any]] = []
    for request_index, request in enumerate(requests):
        prompts = request.get("neighborhood_prompts")
        if not isinstance(prompts, list) or not prompts:
            raise ValueError(
                f"Edit {request_index} has no non-empty neighborhood_prompts list"
            )
        selected = prompts[:prompts_per_edit] if prompts_per_edit else prompts
        for prompt_index, prompt in enumerate(selected):
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"Invalid neighborhood prompt at edit {request_index}, "
                    f"position {prompt_index}"
                )
            samples.append(
                {
                    "request_index": request_index,
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                }
            )
            if max_prompts is not None and len(samples) >= max_prompts:
                return samples
    if not samples:
        raise ValueError("No neighborhood prompts were selected")
    return samples


def discover_num_transformer_layers(model, layer_template: str) -> int:
    config = getattr(model, "config", None)
    candidates = (
        getattr(config, "num_hidden_layers", None),
        getattr(getattr(config, "text_config", None), "num_hidden_layers", None),
        getattr(config, "n_layer", None),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        count = int(candidate)
        nethook.get_module(model, layer_template.format(0))
        nethook.get_module(model, layer_template.format(count - 1))
        return count

    count = 0
    while True:
        try:
            nethook.get_module(model, layer_template.format(count))
        except LookupError:
            break
        count += 1
    if count == 0:
        raise ValueError(
            f"Could not discover Transformer blocks from {layer_template!r}"
        )
    return count


def torch_load_mapping(path: Path) -> Dict[str, torch.Tensor]:
    print(f"Loading checkpoint: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Checkpoint is not a non-empty mapping: {path}")
    bad = [name for name, value in payload.items() if not isinstance(value, torch.Tensor)]
    if bad:
        raise ValueError(
            f"Checkpoint contains non-tensor entries (first: {bad[0]!r}); "
            "expected the evaluation-compatible full-weight format"
        )
    return payload


def configured_weight_names(cfg) -> List[str]:
    return [
        f"{cfg.llms.rewrite_module_tmp.format(int(layer))}.weight"
        for layer in cfg.llms.layers
    ]


def validate_checkpoint(
    checkpoint: Mapping[str, torch.Tensor],
    checkpoint_path: Path,
    base_weights: Mapping[str, torch.Tensor],
) -> None:
    missing = [name for name in base_weights if name not in checkpoint]
    if missing:
        raise KeyError(
            f"Checkpoint {checkpoint_path} is missing configured weight(s): "
            + ", ".join(missing)
        )
    for name, base in base_weights.items():
        edited = checkpoint[name]
        if edited.shape != base.shape:
            raise ValueError(
                f"Shape mismatch for {name} in {checkpoint_path}: "
                f"checkpoint={tuple(edited.shape)}, base={tuple(base.shape)}"
            )
        if not torch.isfinite(edited).all():
            raise FloatingPointError(
                f"Checkpoint {checkpoint_path} contains non-finite {name}"
            )


def capture_base_weights(model, weight_names: Sequence[str]) -> Dict[str, torch.Tensor]:
    return {
        name: nethook.get_parameter(model, name).detach().to("cpu").clone()
        for name in weight_names
    }


def compute_covariance_rows(
    label: str,
    checkpoint: Mapping[str, torch.Tensor],
    base_weights: Mapping[str, torch.Tensor],
    cfg,
    model,
    tokenizer,
    covariance_device: torch.device,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    layers = [int(layer) for layer in cfg.llms.layers]
    for position, layer in enumerate(layers, start=1):
        weight_name = f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
        base = base_weights[weight_name]
        edited = checkpoint[weight_name]
        delta_cpu = edited.to(torch.float32) - base.to(torch.float32)
        delta_norm = torch.linalg.vector_norm(delta_cpu).item()
        base_norm = torch.linalg.vector_norm(base.float()).item()

        # Load/move only one covariance at a time. This bounds peak device
        # memory even when the MLP input dimension is large.
        covariance = get_cov(
            cfg,
            model,
            tokenizer,
            layer,
            cfg.llms.mom2_dataset,
            cfg.llms.mom2_n_samples,
            cfg.llms.mom2_dtype,
            force_recompute=False,
        ).to(device=covariance_device, dtype=torch.float32)
        delta = delta_cpu.to(device=covariance_device, dtype=torch.float32)
        with torch.no_grad():
            covariance_loss = trace_preservation_loss(delta, covariance).item()
        if not math.isfinite(covariance_loss):
            raise FloatingPointError(
                f"Non-finite covariance loss for {label}, layer {layer}"
            )

        # The raw loss is the training objective. Per-output normalization is
        # included only as a scale-friendly diagnostic across architectures.
        if delta.shape[1] == covariance.shape[0]:
            output_dim = delta.shape[0]
        elif delta.shape[0] == covariance.shape[0]:
            output_dim = delta.shape[1]
        else:  # trace_preservation_loss already gives the detailed error.
            raise AssertionError("unreachable covariance/update shape branch")
        row = {
            "method": label,
            "layer": layer,
            "module": cfg.llms.rewrite_module_tmp.format(layer),
            "covariance_loss": covariance_loss,
            "covariance_loss_per_output": covariance_loss / output_dim,
            "update_frobenius_norm": delta_norm,
            "base_frobenius_norm": base_norm,
            "relative_update_norm": delta_norm / base_norm,
        }
        rows.append(row)
        print(
            f"[{label}] covariance {position}/{len(layers)}: layer={layer}, "
            f"loss={covariance_loss:.6e}, relative_update="
            f"{row['relative_update_norm']:.6e}"
        )
        del covariance, delta, delta_cpu
        if covariance_device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def apply_checkpoint(
    model,
    checkpoint: Mapping[str, torch.Tensor],
    base_weights: Mapping[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name in base_weights:
            destination = nethook.get_parameter(model, name)
            destination.copy_(
                checkpoint[name].to(
                    device=destination.device, dtype=destination.dtype
                )
            )


def restore_base_weights(model, base_weights: Mapping[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, base in base_weights.items():
            destination = nethook.get_parameter(model, name)
            destination.copy_(
                base.to(device=destination.device, dtype=destination.dtype)
            )


def cache_reference_subject_logits(
    model,
    tokenizer,
    samples: Sequence[Mapping[str, Any]],
    cfg,
    device: torch.device,
    batch_size: int,
    fact_token_strategy: str,
) -> torch.Tensor:
    loss_layer = max(int(cfg.llms.v_loss_layer), max(map(int, cfg.llms.layers)))
    result = None
    with torch.no_grad():
        for batch_indices in chunks(list(range(len(samples))), batch_size):
            batch = [samples[index] for index in batch_indices]
            model_inputs, lookup_idxs = prepare_subject_kl_batch(
                tok=tokenizer,
                samples=batch,
                fact_token_strategy=fact_token_strategy,
                device=device,
            )
            logits = get_joint_logits_at_positions(
                model=model,
                model_inputs=model_inputs,
                lookup_idxs=lookup_idxs,
                loss_layer=loss_layer,
                layer_module_tmp=cfg.llms.layer_module_tmp,
                ln_f_module=cfg.llms.ln_f_module,
                lm_head_module=cfg.llms.lm_head_module,
            )
            if result is None:
                result = torch.empty(
                    (len(samples), logits.shape[-1]),
                    dtype=logits.dtype,
                    device="cpu",
                )
            result[batch_indices] = logits.detach().cpu()
            del logits, model_inputs
    if result is None:
        raise ValueError("Cannot cache subject logits for an empty sample set")
    return result


def evaluate_subject_kl(
    model,
    tokenizer,
    samples: Sequence[Mapping[str, Any]],
    reference_logits: torch.Tensor,
    cfg,
    device: torch.device,
    batch_size: int,
    fact_token_strategy: str,
) -> np.ndarray:
    loss_layer = max(int(cfg.llms.v_loss_layer), max(map(int, cfg.llms.layers)))
    losses: List[torch.Tensor] = []
    with torch.no_grad():
        for batch_indices in chunks(list(range(len(samples))), batch_size):
            batch = [samples[index] for index in batch_indices]
            model_inputs, lookup_idxs = prepare_subject_kl_batch(
                tok=tokenizer,
                samples=batch,
                fact_token_strategy=fact_token_strategy,
                device=device,
            )
            edited_logits = get_joint_logits_at_positions(
                model=model,
                model_inputs=model_inputs,
                lookup_idxs=lookup_idxs,
                loss_layer=loss_layer,
                layer_module_tmp=cfg.llms.layer_module_tmp,
                ln_f_module=cfg.llms.ln_f_module,
                lm_head_module=cfg.llms.lm_head_module,
            )
            batch_losses = get_subject_kl_losses(
                edited_logits, reference_logits[batch_indices]
            )
            if not torch.isfinite(batch_losses).all():
                raise FloatingPointError("Non-finite subject KL detected")
            losses.append(batch_losses.detach().cpu())
            del edited_logits, batch_losses, model_inputs
    return torch.cat(losses).float().numpy()


def _last_nonpadding_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must be rank 2, got {tuple(attention_mask.shape)}"
        )
    positions = torch.arange(
        attention_mask.shape[1], device=attention_mask.device
    ).unsqueeze(0)
    positions = positions.expand_as(attention_mask)
    result = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
    if (result < 0).any():
        raise ValueError("Neighborhood batch contains an empty token sequence")
    return result


def _batch_first_activation(
    value: Any,
    expected_batch: int,
    expected_sequence: int,
    module_name: str,
) -> torch.Tensor:
    tensor = value[0] if isinstance(value, (tuple, list)) else value
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
        shape = getattr(tensor, "shape", None)
        raise ValueError(
            f"Activation for {module_name} must be rank 3, got {shape}"
        )
    if tensor.shape[:2] == (expected_batch, expected_sequence):
        return tensor
    if tensor.shape[:2] == (expected_sequence, expected_batch):
        return tensor.transpose(0, 1)
    raise ValueError(
        f"Activation for {module_name} has shape {tuple(tensor.shape)}, "
        f"incompatible with token shape "
        f"{(expected_batch, expected_sequence)}"
    )


def collect_last_token_activations(
    model,
    model_inputs,
    last_indices: torch.Tensor,
    block_names: Mapping[int, str],
    rewrite_names: Mapping[int, str],
) -> tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Collect small [batch, hidden] slices without retaining full sequences."""
    expected_batch, expected_sequence = model_inputs["input_ids"].shape
    batch_rows = torch.arange(expected_batch, device=last_indices.device)
    block_outputs: Dict[int, torch.Tensor] = {}
    rewrite_inputs: Dict[int, torch.Tensor] = {}
    handles = []

    def make_block_hook(layer: int, module_name: str):
        def hook(_module, _inputs, output):
            activation = _batch_first_activation(
                output, expected_batch, expected_sequence, module_name
            )
            block_outputs[layer] = (
                activation[batch_rows, last_indices, :].detach().float()
            )

        return hook

    def make_rewrite_pre_hook(layer: int, module_name: str):
        def hook(_module, inputs):
            if not inputs:
                raise ValueError(f"Rewrite module {module_name} received no input")
            activation = _batch_first_activation(
                inputs[0], expected_batch, expected_sequence, module_name
            )
            rewrite_inputs[layer] = (
                activation[batch_rows, last_indices, :].detach().float()
            )

        return hook

    try:
        for layer, module_name in block_names.items():
            module = nethook.get_module(model, module_name)
            handles.append(
                module.register_forward_hook(make_block_hook(layer, module_name))
            )
        for layer, module_name in rewrite_names.items():
            module = nethook.get_module(model, module_name)
            handles.append(
                module.register_forward_pre_hook(
                    make_rewrite_pre_hook(layer, module_name)
                )
            )
        with torch.no_grad():
            outputs = model(**model_inputs, use_cache=False)
        del outputs
    finally:
        for handle in handles:
            handle.remove()

    missing_blocks = set(block_names) - set(block_outputs)
    missing_rewrites = set(rewrite_names) - set(rewrite_inputs)
    if missing_blocks or missing_rewrites:
        raise RuntimeError(
            "Activation hooks did not run: "
            f"blocks={sorted(missing_blocks)}, rewrites={sorted(missing_rewrites)}"
        )
    return block_outputs, rewrite_inputs


def checkpoint_deltas_on_device(
    checkpoint: Mapping[str, torch.Tensor],
    base_weights: Mapping[str, torch.Tensor],
    cfg,
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    deltas = {}
    for layer_value in cfg.llms.layers:
        layer = int(layer_value)
        weight_name = f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
        deltas[layer] = (
            checkpoint[weight_name].detach().float()
            - base_weights[weight_name].detach().float()
        ).to(device=device, dtype=torch.float32)
    return deltas


def update_action_l2_squared(
    activations: torch.Tensor,
    delta: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """Return one ||Delta-W h||^2 value per neighborhood prompt."""
    if delta.shape[1] == activations.shape[1]:
        changed_output = activations @ delta.T
    elif delta.shape[0] == activations.shape[1]:
        # GPT-2 Conv1D stores weights as [in, out].
        changed_output = activations @ delta
    else:
        raise ValueError(
            f"Layer {layer} update shape {tuple(delta.shape)} is incompatible "
            f"with activation shape {tuple(activations.shape)}"
        )
    return changed_output.square().sum(dim=1)


def _metric_record(
    method: str,
    metric: str,
    layer: int,
    module: str,
    values: np.ndarray,
) -> Dict[str, Any]:
    return {
        "method": method,
        "metric": metric,
        "layer": layer,
        "module": module,
        **summarize(values),
    }


def evaluate_neighborhood_diagnostics(
    method: str,
    base_model,
    edited_model,
    tokenizer,
    samples: Sequence[Mapping[str, Any]],
    deltas: Mapping[int, torch.Tensor],
    cfg,
    device: torch.device,
    batch_size: int,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Compare base/edited activations at locality-prompt prediction points."""
    num_blocks = discover_num_transformer_layers(
        base_model, str(cfg.llms.layer_module_tmp)
    )
    block_names = {
        layer: str(cfg.llms.layer_module_tmp).format(layer)
        for layer in range(num_blocks)
    }
    rewrite_names = {
        int(layer): str(cfg.llms.rewrite_module_tmp).format(int(layer))
        for layer in cfg.llms.layers
    }
    hidden_drift_parts: Dict[int, List[torch.Tensor]] = {
        layer: [] for layer in block_names
    }
    relative_drift_parts: Dict[int, List[torch.Tensor]] = {
        layer: [] for layer in block_names
    }
    update_action_parts: Dict[int, List[torch.Tensor]] = {
        layer: [] for layer in rewrite_names
    }
    num_batches = (len(samples) + batch_size - 1) // batch_size

    for batch_number, batch_samples in enumerate(
        chunks(list(samples), batch_size), start=1
    ):
        prompts = [sample["prompt"] for sample in batch_samples]
        model_inputs = tokenizer(
            prompts, return_tensors="pt", padding=True
        ).to(device)
        last_indices = _last_nonpadding_indices(model_inputs["attention_mask"])

        base_blocks, base_rewrite_inputs = collect_last_token_activations(
            model=base_model,
            model_inputs=model_inputs,
            last_indices=last_indices,
            block_names=block_names,
            rewrite_names=rewrite_names,
        )
        edited_blocks, _ = collect_last_token_activations(
            model=edited_model,
            model_inputs=model_inputs,
            last_indices=last_indices,
            block_names=block_names,
            rewrite_names={},
        )

        for layer in block_names:
            base_hidden = base_blocks[layer]
            edited_hidden = edited_blocks[layer]
            drift_squared = (edited_hidden - base_hidden).square().sum(dim=1)
            base_squared = base_hidden.square().sum(dim=1)
            relative_squared = drift_squared / base_squared.clamp_min(
                torch.finfo(torch.float32).eps
            )
            hidden_drift_parts[layer].append(drift_squared.detach().cpu())
            relative_drift_parts[layer].append(relative_squared.detach().cpu())

        for layer in rewrite_names:
            action_squared = update_action_l2_squared(
                base_rewrite_inputs[layer], deltas[layer], layer
            )
            update_action_parts[layer].append(action_squared.detach().cpu())

        del (
            model_inputs,
            last_indices,
            base_blocks,
            edited_blocks,
            base_rewrite_inputs,
        )
        if batch_number == 1 or batch_number == num_batches or batch_number % 50 == 0:
            print(
                f"[{method}] neighborhood activations: "
                f"batch {batch_number}/{num_batches}"
            )

    csv_rows: List[Dict[str, Any]] = []
    hidden_rows = []
    for layer, module_name in block_names.items():
        drift = torch.cat(hidden_drift_parts[layer]).detach().numpy()
        relative = torch.cat(relative_drift_parts[layer]).detach().numpy()
        drift_summary = summarize(drift)
        relative_summary = summarize(relative)
        hidden_rows.append(
            {
                "layer": layer,
                "module": module_name,
                "hidden_drift_l2_squared": drift_summary,
                "hidden_relative_drift_squared": relative_summary,
            }
        )
        csv_rows.append(
            _metric_record(
                method,
                "hidden_drift_l2_squared",
                layer,
                module_name,
                drift,
            )
        )
        csv_rows.append(
            _metric_record(
                method,
                "hidden_relative_drift_squared",
                layer,
                module_name,
                relative,
            )
        )

    update_rows = []
    for layer, module_name in rewrite_names.items():
        values = torch.cat(update_action_parts[layer]).detach().numpy()
        value_summary = summarize(values)
        update_rows.append(
            {
                "layer": layer,
                "module": module_name,
                "update_action_l2_squared": value_summary,
            }
        )
        csv_rows.append(
            _metric_record(
                method,
                "update_action_l2_squared",
                layer,
                module_name,
                values,
            )
        )

    result = {
        "num_prompts": len(samples),
        "position": "last_nonpadding_prompt_token",
        "update_action_definition": "||Delta W_l h_l^base||_2^2",
        "hidden_drift_definition": "||h_l^edited-h_l^base||_2^2",
        "update_action": update_rows,
        "hidden_state_drift": hidden_rows,
    }
    return result, csv_rows


def summarize(values: np.ndarray) -> Dict[str, float | int]:
    if values.size == 0:
        raise ValueError("Cannot summarize an empty array")
    return {
        "count": int(values.size),
        "mean": float(values.mean(dtype=np.float64)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def print_summary(method_results: Sequence[Mapping[str, Any]]) -> None:
    print("\nComparison summary")
    print(
        f"{'Method':<12} {'Covariance sum':>18} {'Subject fact KL':>18} "
        f"{'Subject last KL':>18}"
    )
    for result in method_results:
        print(
            f"{result['method']:<12} "
            f"{result['covariance_loss_sum']:>18.6e} "
            f"{result['subject_fact_kl']['mean']:>18.6e} "
            f"{result['subject_prompt_last_kl']['mean']:>18.6e}"
        )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)
    cfg = load_config(args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    covariance_device = (
        torch.device(f"cuda:{args.gpu}")
        if args.covariance_device == "cuda"
        else torch.device("cpu")
    )

    print(f"Loading base model {cfg.llms.name} on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.llms.name,
        torch_dtype=model_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()
    nethook.set_requires_grad(False, model)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.llms.name, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    requests = load_requests(args.data, args.record_start, args.num_edits)
    subject_samples = build_subject_kl_samples(requests)
    print(
        f"Caching base subject distributions for {len(subject_samples)} edit(s) "
        f"at fact_token={cfg.llms.fact_token}"
    )
    reference_fact_logits = cache_reference_subject_logits(
        model,
        tokenizer,
        subject_samples,
        cfg,
        device,
        args.batch_size,
        fact_token_strategy=str(cfg.llms.fact_token),
    )
    if str(cfg.llms.fact_token) == "last":
        reference_prompt_last_logits = reference_fact_logits
    else:
        print(
            "Caching base subject distributions at the last token of the "
            'complete "{} is a" prompt'
        )
        reference_prompt_last_logits = cache_reference_subject_logits(
            model,
            tokenizer,
            subject_samples,
            cfg,
            device,
            args.batch_size,
            fact_token_strategy="last",
        )

    weight_names = configured_weight_names(cfg)
    base_weights = capture_base_weights(model, weight_names)
    method_specs = (
        ("memit_ori", args.memit_weights),
        ("e2e", args.e2e_weights),
    )
    method_results: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []

    for label, checkpoint_path in method_specs:
        checkpoint = torch_load_mapping(checkpoint_path)
        validate_checkpoint(checkpoint, checkpoint_path, base_weights)
        rows = compute_covariance_rows(
            label=label,
            checkpoint=checkpoint,
            base_weights=base_weights,
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            covariance_device=covariance_device,
        )
        all_rows.extend(rows)

        apply_checkpoint(model, checkpoint, base_weights)
        try:
            subject_fact_losses = evaluate_subject_kl(
                model=model,
                tokenizer=tokenizer,
                samples=subject_samples,
                reference_logits=reference_fact_logits,
                cfg=cfg,
                device=device,
                batch_size=args.batch_size,
                fact_token_strategy=str(cfg.llms.fact_token),
            )
            if str(cfg.llms.fact_token) == "last":
                subject_prompt_last_losses = subject_fact_losses.copy()
            else:
                subject_prompt_last_losses = evaluate_subject_kl(
                    model=model,
                    tokenizer=tokenizer,
                    samples=subject_samples,
                    reference_logits=reference_prompt_last_logits,
                    cfg=cfg,
                    device=device,
                    batch_size=args.batch_size,
                    fact_token_strategy="last",
                )
        finally:
            restore_base_weights(model, base_weights)

        subject_fact_summary = summarize(subject_fact_losses)
        subject_prompt_last_summary = summarize(subject_prompt_last_losses)
        method_results.append(
            {
                "method": label,
                "checkpoint": str(checkpoint_path.resolve()),
                "covariance_loss_sum": float(
                    sum(row["covariance_loss"] for row in rows)
                ),
                # Retain the old key for consumers of the first script
                # version; it is the fact-token KL used by E2E training.
                "subject_kl": subject_fact_summary,
                "subject_fact_kl": subject_fact_summary,
                "subject_prompt_last_kl": subject_prompt_last_summary,
                "layers": rows,
            }
        )
        print(
            f"[{label}] subject fact KL mean="
            f"{subject_fact_summary['mean']:.6e}; subject prompt-last KL "
            f"mean={subject_prompt_last_summary['mean']:.6e}"
        )
        del checkpoint, subject_fact_losses, subject_prompt_last_losses
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    neighborhood_samples: List[Dict[str, Any]] = []
    neighborhood_csv_rows: List[Dict[str, Any]] = []
    if not args.skip_neighborhood_diagnostics:
        neighborhood_samples = build_neighborhood_prompts(
            requests=requests,
            prompts_per_edit=args.neighborhood_prompts_per_edit,
            max_prompts=args.max_neighborhood_prompts,
        )
        print(
            f"Selected {len(neighborhood_samples)} neighborhood prompt(s). "
            "Loading a second immutable base model for paired activation "
            "diagnostics."
        )
        reference_model = AutoModelForCausalLM.from_pretrained(
            cfg.llms.name,
            torch_dtype=model_dtype(args.dtype),
            trust_remote_code=args.trust_remote_code,
        ).to(device)
        reference_model.eval()
        nethook.set_requires_grad(False, reference_model)

        for result, (label, checkpoint_path) in zip(
            method_results, method_specs
        ):
            checkpoint = torch_load_mapping(checkpoint_path)
            validate_checkpoint(checkpoint, checkpoint_path, base_weights)
            deltas = checkpoint_deltas_on_device(
                checkpoint=checkpoint,
                base_weights=base_weights,
                cfg=cfg,
                device=device,
            )
            apply_checkpoint(model, checkpoint, base_weights)
            try:
                diagnostics, csv_rows = evaluate_neighborhood_diagnostics(
                    method=label,
                    base_model=reference_model,
                    edited_model=model,
                    tokenizer=tokenizer,
                    samples=neighborhood_samples,
                    deltas=deltas,
                    cfg=cfg,
                    device=device,
                    batch_size=args.neighborhood_batch_size,
                )
            finally:
                restore_base_weights(model, base_weights)
            result["neighborhood_diagnostics"] = diagnostics
            neighborhood_csv_rows.extend(csv_rows)
            final_block = diagnostics["hidden_state_drift"][-1]
            print(
                f"[{label}] final-block neighborhood hidden drift mean="
                f"{final_block['hidden_drift_l2_squared']['mean']:.6e}"
            )
            del checkpoint, deltas, diagnostics, csv_rows
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del reference_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = {
        "definition": {
            "covariance_loss": "tr((W_edited-W_base) C (W_edited-W_base)^T)",
            "covariance_loss_weighting": "raw; no alpha or mom2_update_weight",
            "subject_probe": "{} is a",
            "subject_fact_position": str(cfg.llms.fact_token),
            "subject_prompt_last_position": "last_nonpadding_prompt_token",
            "subject_kl_direction": "KL(P_base || P_edited)",
            "subject_kl_scope": "one value per complete edited model, not per layer",
            "neighborhood_position": "last_nonpadding_prompt_token",
            "neighborhood_update_action": "||Delta W_l h_l^base||_2^2",
            "neighborhood_hidden_drift": "||h_l^edited-h_l^base||_2^2",
        },
        "model": str(cfg.llms.name),
        "model_alias": str(cfg.llms.alias),
        "model_config": str(args.model_config.resolve()),
        "dataset": str(args.data.resolve()),
        "record_start": args.record_start,
        "num_edits": args.num_edits,
        "num_neighborhood_prompts": len(neighborhood_samples),
        "neighborhood_prompts_per_edit": args.neighborhood_prompts_per_edit,
        "neighborhood_diagnostics_skipped": args.skip_neighborhood_diagnostics,
        "covariance": {
            "dataset": str(cfg.llms.mom2_dataset),
            "num_samples": int(cfg.llms.mom2_n_samples),
            "dtype": str(cfg.llms.mom2_dtype),
            "probability_weighted": bool(args.cov_probability_weighted),
        },
        "results": method_results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    output_csv = args.output_csv or args.output_json.with_suffix(".csv")
    write_csv(output_csv, all_rows)
    output_neighborhood_csv = args.output_neighborhood_csv or (
        args.output_json.parent
        / f"{args.output_json.stem}_neighborhood.csv"
    )
    if neighborhood_csv_rows:
        write_csv(output_neighborhood_csv, neighborhood_csv_rows)
    print_summary(method_results)
    print(f"\nJSON written to {args.output_json}")
    print(f"Per-layer CSV written to {output_csv}")
    if neighborhood_csv_rows:
        print(f"Neighborhood diagnostics CSV written to {output_neighborhood_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Precompute input covariances for LoRA targets in every Transformer layer.

For a linear module y = W x, the preservation penalty uses the non-centered
second moment C = E[x x^T].  Modules that receive the exact same tensor share C:

* q_proj, k_proj, v_proj -> attn_qkv
* gate_proj, up_proj     -> mlp_gate_up
* o_proj                 -> attn_o
* down_proj              -> mlp_down

The cached ``.npz`` files use the repository's existing ``SecondMoment``
format (sum(x x^T) plus count).  ``get_cov``/``SecondMoment.moment()`` performs
the division by count when the covariance is consumed.  A JSON manifest maps
every requested LoRA module to its shared cache file.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

# Executing this file by path sets sys.path[0] to ``experiments/`` rather than
# the repository root. Add the root explicitly so local packages such as
# ``locate_edit_utils`` and ``util`` can always be imported, regardless of the
# caller's current working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from locate_edit_utils.layer_stats import (
    _next_token_probabilities,
    _probability_weighted_cache_suffix,
    _smooth_probability_weights,
)
from locate_edit_utils.tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)
from util import nethook
from util.runningstats import (
    CombinedStat,
    SecondMoment,
    WeightedSecondMoment,
    load_cached_state,
    make_loader,
    save_cached_state,
)


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "gate_up_proj",
    "down_proj",
)

# The order also determines which module is used as the representative hook.
SHARED_INPUT_GROUPS: Mapping[str, Tuple[str, ...]] = {
    "attn_qkv": ("q_proj", "k_proj", "v_proj"),
    "attn_o": ("o_proj",),
    "mlp_gate_up": ("gate_proj", "up_proj", "gate_up_proj"),
    "mlp_down": ("down_proj",),
}


@dataclass(frozen=True)
class CovarianceJob:
    layer: int
    group: str
    representative: str
    modules: Tuple[str, ...]
    input_dim: int
    cache_suffix: str
    cache_file: Path


def _safe_component(value: object, max_length: int = 48) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")
    return (text or "value")[:max_length]


def _short_hash(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def _get_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    try:
        return nethook.get_module(model, name)
    except LookupError as exc:
        raise ValueError(f"Model does not contain module {name!r}") from exc


def _linear_input_dim(module: torch.nn.Module, module_name: str) -> int:
    in_features = getattr(module, "in_features", None)
    if in_features is not None:
        return int(in_features)
    weight = getattr(module, "weight", None)
    if weight is None or weight.ndim != 2:
        raise TypeError(
            f"LoRA target {module_name!r} is not a supported linear module: "
            f"{type(module).__name__}"
        )
    # This fallback is correct for torch.nn.Linear-like modules. Architectures
    # with transposed Conv1D weights should expose an explicit target adapter.
    return int(weight.shape[1])


def _module_name(cfg: DictConfig, layer: int, target: str) -> str:
    if target in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return f"{cfg.llms.attn_module_tmp.format(layer)}.{target}"
    if target in {"gate_proj", "up_proj", "gate_up_proj", "down_proj"}:
        return f"{cfg.llms.mlp_module_tmp.format(layer)}.{target}"
    raise ValueError(f"Unsupported LoRA target: {target}")


def _discover_num_layers(model: torch.nn.Module, layer_template: str) -> int:
    config = getattr(model, "config", None)
    candidates = [
        getattr(config, "num_hidden_layers", None),
        getattr(getattr(config, "text_config", None), "num_hidden_layers", None),
        getattr(config, "n_layer", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            count = int(candidate)
            # Detect a mismatched model config before starting an expensive run.
            _get_module(model, layer_template.format(0))
            _get_module(model, layer_template.format(count - 1))
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


def _parse_layers(value: object, num_layers: int) -> List[int]:
    if value is None or str(value).lower() == "all":
        return list(range(num_layers))
    if isinstance(value, str):
        result: List[int] = []
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = (int(item) for item in part.split("-", 1))
                if end < start:
                    raise ValueError(f"Invalid descending layer range: {part}")
                result.extend(range(start, end + 1))
            else:
                result.append(int(part))
    else:
        result = [int(layer) for layer in value]

    result = sorted(set(result))
    invalid = [layer for layer in result if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(
            f"Layers {invalid} are outside the valid range 0..{num_layers - 1}"
        )
    if not result:
        raise ValueError("No layers selected")
    return result


def _group_targets(targets: Sequence[str]) -> List[Tuple[str, Tuple[str, ...]]]:
    unknown = set(targets) - set(DEFAULT_TARGET_MODULES)
    if unknown:
        raise ValueError(
            "Unsupported target_modules: "
            f"{sorted(unknown)}. Supported values are {list(DEFAULT_TARGET_MODULES)}"
        )
    target_set = set(targets)
    return [
        (group, tuple(target for target in members if target in target_set))
        for group, members in SHARED_INPUT_GROUPS.items()
        if target_set.intersection(members)
    ]


def _effective_cache_file(
    cfg: DictConfig, layer: int, suffix: str, probability_weighted: bool
) -> Path:
    if probability_weighted:
        suffix = _probability_weighted_cache_suffix(
            suffix,
            float(cfg.cov_prob_weight_epsilon),
            float(cfg.cov_prob_weight_alpha),
        )
    alias = cfg.llms.alias.replace("/", "-")
    return Path(cfg.cache_dir) / "stats" / alias / f"layer-{layer}-{suffix}.npz"


def _cache_suffix(cfg: DictConfig, group: str, representative: str) -> str:
    options = cfg.lora_cov
    identity = {
        "dataset": str(cfg.llms.mom2_dataset),
        "sample_size": int(cfg.llms.mom2_n_samples),
        "precision": str(cfg.llms.mom2_dtype),
        "max_sequence_length": cfg.llms.mom2_maxseqlen,
        "random_sample": int(options.random_sample),
        "probability_weighted": bool(cfg.cov_probability_weighted),
        "prob_weight_epsilon": float(cfg.cov_prob_weight_epsilon),
        "prob_weight_alpha": float(cfg.cov_prob_weight_alpha),
        "representative": representative,
    }
    dataset_name = Path(str(cfg.llms.mom2_dataset)).name
    return "-".join(
        [
            _safe_component(options.cache_namespace),
            _safe_component(group),
            _safe_component(dataset_name, max_length=24),
            f"n{int(cfg.llms.mom2_n_samples)}",
            _safe_component(cfg.llms.mom2_dtype, max_length=12),
            _short_hash(json.dumps(identity, sort_keys=True)),
        ]
    )


def _build_jobs(
    cfg: DictConfig,
    model: torch.nn.Module,
    layers: Sequence[int],
    targets: Sequence[str],
) -> Tuple[List[CovarianceJob], Dict[str, dict], List[str]]:
    jobs: List[CovarianceJob] = []
    module_map: Dict[str, dict] = {}
    missing: List[str] = []
    for layer in layers:
        for group, group_targets in _group_targets(targets):
            existing: List[Tuple[str, str, torch.nn.Module]] = []
            for target in group_targets:
                name = _module_name(cfg, layer, target)
                try:
                    module = nethook.get_module(model, name)
                except LookupError:
                    missing.append(name)
                    continue
                existing.append((target, name, module))
            if not existing:
                continue

            representative = existing[0][1]
            input_dims = {
                name: _linear_input_dim(module, name)
                for _, name, module in existing
            }
            if len(set(input_dims.values())) != 1:
                raise ValueError(
                    f"Modules assigned to shared group {group!r} have different "
                    f"input dimensions: {input_dims}"
                )
            input_dim = next(iter(input_dims.values()))
            suffix = _cache_suffix(cfg, group, representative)
            cache_file = _effective_cache_file(
                cfg, layer, suffix, bool(cfg.cov_probability_weighted)
            )
            module_names = tuple(name for _, name, _ in existing)
            job = CovarianceJob(
                layer=layer,
                group=group,
                representative=representative,
                modules=module_names,
                input_dim=input_dim,
                cache_suffix=suffix,
                cache_file=cache_file,
            )
            jobs.append(job)
            for target, name, _ in existing:
                module_map[name] = {
                    "layer": layer,
                    "target": target,
                    "covariance_group": group,
                    "representative_module": representative,
                    "cache_file": str(cache_file.resolve()),
                    "input_dim": input_dim,
                }
    return jobs, module_map, missing


def _write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _model_dtype(value: object):
    text = str(value)
    if text == "auto":
        return "auto"
    if not hasattr(torch, text):
        raise ValueError(f"Unknown model_dtype: {text}")
    return getattr(torch, text)


def _model_max_length(model: torch.nn.Module, batch_tokens) -> Tuple[int, int]:
    config = model.config
    if hasattr(config, "max_position_embeddings"):
        max_length = int(config.max_position_embeddings)
    elif hasattr(config, "seq_length"):
        max_length = int(config.seq_length)
    else:
        raise ValueError(
            "The model config has neither max_position_embeddings nor "
            "seq_length; set an architecture-specific maximum sequence length."
        )
    if batch_tokens is not None:
        max_length = min(max_length, int(batch_tokens))
        effective_batch_tokens = int(batch_tokens)
    else:
        effective_batch_tokens = max_length * 3
    if max_length <= 0 or effective_batch_tokens <= 0:
        raise ValueError("mom2_maxseqlen and model maximum length must be positive")
    return max_length, effective_batch_tokens


def _load_tokenized_dataset(
    dataset_name: str, tokenizer, max_length: int
) -> TokenizedDataset:
    if dataset_name in {"wikipedia", "wikitext"}:
        subset = {
            "wikitext": "wikitext-103-raw-v1",
            "wikipedia": "20220301.en",
        }[dataset_name]
        raw_dataset = load_dataset(dataset_name, subset)
    else:
        raw_dataset = load_dataset(
            "json", data_files={"train": dataset_name}
        )
    return TokenizedDataset(
        raw_dataset["train"], tokenizer, maxlen=max_length
    )


def _new_stat(probability_weighted: bool) -> CombinedStat:
    moment = WeightedSecondMoment() if probability_weighted else SecondMoment()
    return CombinedStat(mom2=moment)


def _validate_stat(job: CovarianceJob, stat: CombinedStat) -> None:
    matrix = stat.mom2.mom2
    if matrix is None:
        raise ValueError(f"Covariance cache is empty for {job.representative}")
    actual_shape = tuple(matrix.shape)
    expected_shape = (job.input_dim, job.input_dim)
    if actual_shape != expected_shape:
        raise ValueError(
            f"Invalid covariance shape for {job.representative}: expected "
            f"{expected_shape}, got {actual_shape}"
        )
    if int(stat.mom2.count) <= 0:
        raise ValueError(f"Covariance count is zero for {job.representative}")


def _job_record(job: CovarianceJob, stat: CombinedStat, source: str) -> dict:
    _validate_stat(job, stat)
    return {
        "layer": job.layer,
        "group": job.group,
        "representative_module": job.representative,
        "shared_by_modules": list(job.modules),
        "cache_file": str(job.cache_file.resolve()),
        "input_dim": job.input_dim,
        "matrix_shape": list(stat.mom2.mom2.shape),
        "count": int(stat.mom2.count),
        "status": "complete",
        "source": source,
    }


def _replace_manifest_record(manifest: dict, record: dict) -> None:
    manifest["jobs"] = [
        previous
        for previous in manifest["jobs"]
        if not (
            previous["layer"] == record["layer"]
            and previous["group"] == record["group"]
        )
    ]
    manifest["jobs"].append(record)
    manifest["jobs"].sort(key=lambda item: (item["layer"], item["group"]))


def _load_cached_job(
    job: CovarianceJob,
    sample_size: int,
    probability_weighted: bool,
) -> CombinedStat | None:
    state = load_cached_state(
        str(job.cache_file), {"sample_size": sample_size}, quiet=True
    )
    if state is None:
        return None
    stat = _new_stat(probability_weighted)
    try:
        stat.load_state_dict(state)
        _validate_stat(job, stat)
    except (KeyError, TypeError, ValueError) as error:
        print(f"Ignoring invalid cache {job.cache_file}: {error}")
        del stat
        return None
    return stat


def _save_stat_atomic(
    job: CovarianceJob, stat: CombinedStat, sample_size: int
) -> None:
    job.cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = job.cache_file.with_name(
        f".{job.cache_file.stem}.{os.getpid()}.tmp.npz"
    )
    try:
        save_cached_state(
            str(temporary), stat, {"sample_size": sample_size}
        )
        os.replace(temporary, job.cache_file)
    finally:
        if temporary.exists():
            temporary.unlink()


def _make_group_loader(
    dataset: TokenizedDataset,
    sample_size: int,
    document_batch_size: int,
    batch_tokens: int,
    random_sample: int,
    num_workers: int,
):
    return make_loader(
        dataset,
        sample_size=sample_size,
        batch_size=document_batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=random_sample,
        num_workers=num_workers,
    )


def _collect_group_stats(
    cfg: DictConfig,
    model: torch.nn.Module,
    dataset: TokenizedDataset,
    jobs: Sequence[CovarianceJob],
    device: torch.device,
    batch_tokens: int,
) -> Dict[str, CombinedStat]:
    """Collect one covariance group across all selected layers per forward."""
    if not jobs:
        return {}
    options = cfg.lora_cov
    probability_weighted = bool(cfg.cov_probability_weighted)
    dtype = getattr(torch, str(cfg.llms.mom2_dtype))
    stats = {
        job.representative: _new_stat(probability_weighted) for job in jobs
    }
    module_names = [job.representative for job in jobs]
    loader = _make_group_loader(
        dataset,
        sample_size=int(cfg.llms.mom2_n_samples),
        document_batch_size=int(options.document_batch_size),
        batch_tokens=batch_tokens,
        random_sample=int(options.random_sample),
        num_workers=int(options.num_workers),
    )
    effective_sample_size = min(int(cfg.llms.mom2_n_samples), len(dataset))
    outer_batches = math.ceil(
        effective_sample_size / int(options.document_batch_size)
    )
    description = f"covariance {jobs[0].group} ({len(jobs)} layers)"

    with torch.inference_mode():
        for batch_group in tqdm(loader, total=outer_batches, desc=description):
            for batch in batch_group:
                batch = dict_to_(batch, device)
                # In the ordinary unweighted case, TraceDict stops at the last
                # representative module. This avoids constructing lm_head
                # logits while retaining all group inputs from this forward.
                if probability_weighted:
                    with nethook.TraceDict(
                        model,
                        module_names,
                        retain_input=True,
                        retain_output=False,
                        detach=True,
                        stop=False,
                    ) as traces:
                        outputs = model(**batch, use_cache=False)
                else:
                    with nethook.TraceDict(
                        model,
                        module_names,
                        retain_input=True,
                        retain_output=False,
                        detach=True,
                        stop=True,
                    ) as traces:
                        model(**batch, use_cache=False)
                    outputs = None

                weights = None
                feature_mask = batch["attention_mask"].bool()
                if probability_weighted:
                    logits = (
                        outputs.logits
                        if hasattr(outputs, "logits")
                        else outputs[0]
                    )
                    probabilities = _next_token_probabilities(
                        logits, batch["input_ids"]
                    )
                    feature_mask = (
                        batch["attention_mask"][:, :-1].bool()
                        & batch["attention_mask"][:, 1:].bool()
                    )
                    weights = _smooth_probability_weights(
                        probabilities[feature_mask],
                        float(cfg.cov_prob_weight_epsilon),
                        float(cfg.cov_prob_weight_alpha),
                    ).to(dtype=dtype)
                    del outputs, logits, probabilities

                for job in jobs:
                    inputs = traces[job.representative].input
                    if not torch.is_tensor(inputs):
                        raise TypeError(
                            f"Expected tensor input for {job.representative}, "
                            f"got {type(inputs).__name__}"
                        )
                    if probability_weighted:
                        # Match layer_stats exactly: p(token_t | token_<t)
                        # weights the module input at token_t.
                        inputs = inputs[:, 1:]
                    features = flatten_masked_batch(inputs, feature_mask).to(
                        dtype=dtype
                    )
                    stat = stats[job.representative]
                    if probability_weighted:
                        stat.add(features, weights)
                    else:
                        stat.add(features)
                    del features
                    # Drop each retained activation once its moment has been
                    # accumulated instead of holding the whole group until the
                    # next model forward.
                    del traces[job.representative].input

                del traces, batch
                if weights is not None:
                    del weights
    return stats


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    options = cfg.lora_cov
    targets = [str(target) for target in options.target_modules]
    if not targets:
        raise ValueError("lora_cov.target_modules must not be empty")
    if int(cfg.llms.mom2_n_samples) <= 0:
        raise ValueError("llms.mom2_n_samples must be positive")
    if int(options.document_batch_size) <= 0:
        raise ValueError("lora_cov.document_batch_size must be positive")
    if int(options.num_workers) < 0:
        raise ValueError("lora_cov.num_workers must be non-negative")

    device = torch.device(
        f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Loading {cfg.llms.name} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.llms.name,
        torch_dtype=_model_dtype(cfg.model_dtype),
        trust_remote_code=True,
    ).to(device)
    model.eval()
    nethook.set_requires_grad(False, model)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.llms.name, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = _discover_num_layers(model, str(cfg.llms.layer_module_tmp))
    layers = _parse_layers(options.layers, num_layers)
    jobs, module_map, missing = _build_jobs(cfg, model, layers, targets)
    if missing and not bool(options.allow_missing):
        preview = "\n  ".join(missing[:20])
        more = "" if len(missing) <= 20 else f"\n  ... and {len(missing) - 20} more"
        raise ValueError(
            "Some requested LoRA targets do not exist. Set "
            "lora_cov.allow_missing=true only if this is intentional:\n  "
            f"{preview}{more}"
        )
    if not jobs:
        raise ValueError("No covariance jobs were created")

    manifest_identity = {
        "model": str(cfg.llms.name),
        "dataset": str(cfg.llms.mom2_dataset),
        "sample_size": int(cfg.llms.mom2_n_samples),
        "precision": str(cfg.llms.mom2_dtype),
        "max_sequence_length": cfg.llms.mom2_maxseqlen,
        "random_sample": int(options.random_sample),
        "probability_weighted": bool(cfg.cov_probability_weighted),
        "layers": layers,
        "targets": targets,
    }
    manifest_run_id = _short_hash(json.dumps(manifest_identity, sort_keys=True))
    manifest_path = (
        Path(cfg.cache_dir)
        / "stats"
        / cfg.llms.alias.replace("/", "-")
        / (
            f"{_safe_component(options.cache_namespace)}-"
            f"{manifest_run_id}-manifest.json"
        )
    )
    manifest = {
        "format_version": 2,
        "description": "Non-centered input covariance caches for LoRA targets",
        "collection_mode": "one_forward_per_covariance_group",
        "model": str(cfg.llms.name),
        "model_alias": str(cfg.llms.alias),
        "dataset": str(cfg.llms.mom2_dataset),
        "sample_size": int(cfg.llms.mom2_n_samples),
        "precision": str(cfg.llms.mom2_dtype),
        "max_sequence_length": cfg.llms.mom2_maxseqlen,
        "random_sample": int(options.random_sample),
        "probability_weighted": bool(cfg.cov_probability_weighted),
        "selected_layers": layers,
        "target_modules": targets,
        "missing_modules": missing,
        "module_to_covariance": module_map,
        "jobs": [],
    }

    print(
        f"Prepared {len(jobs)} unique covariance jobs for {len(module_map)} "
        f"LoRA modules across {len(layers)}/{num_layers} layers."
    )
    num_groups = len({job.group for job in jobs})
    print(
        f"Collection plan: at most {num_groups} dataset passes; every pass "
        "collects all uncached layers in one covariance group."
    )
    estimated_bytes = sum(job.input_dim * job.input_dim * 4 for job in jobs)
    print(
        "Estimated covariance payload (float32, before filesystem overhead): "
        f"{estimated_bytes / 1024 ** 3:.2f} GiB"
    )
    print(f"Manifest: {manifest_path}")
    if bool(options.dry_run):
        for index, job in enumerate(jobs, start=1):
            print(
                f"[{index}/{len(jobs)}] layer={job.layer} group={job.group} "
                f"representative={job.representative} dim={job.input_dim} "
                f"cache={job.cache_file}"
            )
        print("Dry run complete; no covariance was computed or written.")
        return

    sample_size = int(cfg.llms.mom2_n_samples)
    probability_weighted = bool(cfg.cov_probability_weighted)
    force_recompute = bool(options.force_recompute)
    pending_by_group: Dict[str, List[CovarianceJob]] = {}

    # Validate and publish existing caches before loading the source dataset.
    # A partially completed run therefore resumes only the missing layers.
    for job in jobs:
        cached_stat = None
        if not force_recompute:
            cached_stat = _load_cached_job(
                job, sample_size, probability_weighted
            )
        if cached_stat is None:
            pending_by_group.setdefault(job.group, []).append(job)
            continue
        _replace_manifest_record(
            manifest, _job_record(job, cached_stat, source="cache")
        )
        del cached_stat

    _write_manifest(manifest_path, manifest)
    pending_jobs = sum(len(group_jobs) for group_jobs in pending_by_group.values())
    if pending_jobs == 0:
        print("All covariance caches are complete; no model forward is required.")
        return
    print(
        f"Reusing {len(jobs) - pending_jobs} cached jobs; computing "
        f"{pending_jobs} jobs in {len(pending_by_group)} dataset passes."
    )

    max_length, batch_tokens = _model_max_length(
        model, cfg.llms.mom2_maxseqlen
    )
    print(
        f"Loading covariance dataset {cfg.llms.mom2_dataset!r}; "
        f"max_length={max_length}, batch_tokens={batch_tokens}."
    )
    dataset = _load_tokenized_dataset(
        str(cfg.llms.mom2_dataset), tokenizer, max_length
    )

    # Preserve the declared group order so the run and its logs are stable.
    for group in SHARED_INPUT_GROUPS:
        group_jobs = pending_by_group.get(group, [])
        if not group_jobs:
            continue
        group_jobs.sort(key=lambda job: job.layer)
        group_payload = sum(
            job.input_dim * job.input_dim * 4 for job in group_jobs
        )
        print(
            f"Starting group {group}: {len(group_jobs)} layers, "
            f"approximately {group_payload / 1024 ** 3:.2f} GiB of "
            "float32 accumulators."
        )
        start_time = time.time()
        group_stats = _collect_group_stats(
            cfg,
            model,
            dataset,
            group_jobs,
            device,
            batch_tokens,
        )
        print(
            f"Group {group} accumulation completed in "
            f"{(time.time() - start_time) / 3600:.2f} hours; saving caches."
        )

        # Move and save one matrix at a time so host memory does not need a
        # second full copy of the whole group.
        for job in group_jobs:
            stat = group_stats.pop(job.representative)
            _validate_stat(job, stat)
            stat.to_("cpu")
            _save_stat_atomic(job, stat, sample_size)
            _replace_manifest_record(
                manifest, _job_record(job, stat, source="computed")
            )
            _write_manifest(manifest_path, manifest)
            print(
                f"Saved layer={job.layer} group={job.group}: "
                f"{job.cache_file}"
            )
            del stat
        del group_stats
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        f"Completed {len(jobs)} covariance jobs. Manifest saved to "
        f"{manifest_path}"
    )


if __name__ == "__main__":
    main()

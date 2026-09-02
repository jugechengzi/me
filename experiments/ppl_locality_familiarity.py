#!/usr/bin/env python3
"""Evaluate familiarity-conditioned locality with target-only perplexity.

The script has two reusable phases:

1. ``prepare`` uses the original model to
   - score each dataset's neighborhood prompt/reference-answer pair;
   - split prompts into low/high reference-NLL groups at the median;
   - generate preservation targets from the original model; and
   - score those generated targets with the original model.
2. ``score`` scores the saved preservation targets with an edited model and
   reports signed/absolute target-NLL drift for the two familiarity groups.

``all`` runs both phases.  Splitting the phases makes the expensive original
model preparation reusable across multiple edited checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F


# 直接执行 experiments/ 下的脚本时，sys.path 默认只包含 experiments，无法
# 导入项目根目录中的 util、load 等模块。这里与仓库内其他实验脚本保持一致，
# 显式加入项目根目录，使脚本不依赖当前工作目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA = PROJECT_ROOT / "data" / "multi_counterfact_20877.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Familiarity-stratified target-PPL locality evaluation."
    )
    parser.add_argument("--phase", choices=("prepare", "score", "all"), default="all")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--dataset-format",
        choices=("auto", "counterfact", "zsre"),
        default="auto",
        help=(
            "How locality answers are read. auto uses neighborhood_prompts_answers "
            "when present (ZSRE), otherwise broadcasts target_true (CounterFact)."
        ),
    )
    parser.add_argument("--base-model", required=True, help="Original HF model name/path.")

    parser.add_argument(
        "--edited-weights",
        type=Path,
        default=None,
        help=(
            "Optional explicit .pt edited-weight path. If omitted, the path is "
            "built exactly like load.py: cache_dir/saved_weights/algorithm/"
            "data_name-load_name-model_alias.pt."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--algorithm", default="memit", help="Equivalent to cfg.algs.name.")
    parser.add_argument("--load-name", default="default", help="Equivalent to cfg.load_name.")
    parser.add_argument("--model-alias", default=None, help="Equivalent to cfg.llms.alias.")
    parser.add_argument(
        "--data-name",
        default=None,
        help="Equivalent to cfg.data; defaults to the JSON filename stem.",
    )

    parser.add_argument("--output-dir", type=Path, default=Path("results/ppl_locality"))
    parser.add_argument("--prepared-file", type=Path, default=None)
    parser.add_argument("--results-file", type=Path, default=None)
    parser.add_argument("--summary-file", type=Path, default=None)

    parser.add_argument("--record-start", type=int, default=0)
    parser.add_argument(
        "--num-edits",
        "--max-records",
        dest="num_edits",
        type=int,
        default=None,
        help=(
            "Number of edited records to evaluate, starting at --record-start. "
            "CounterFact has 10 locality prompts per edit; ZSRE has 1. "
            "--max-records is retained as a compatibility alias."
        ),
    )
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument(
        "--exact-new-tokens",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Force generation to produce exactly max-new-tokens. "
            "By default, generation may stop at EOS."
        ),
    )

    parser.add_argument("--generation-mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of preservation targets per prompt (sampling mode only).",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)

    parser.add_argument("--device", default=None, help="For example cuda:0, mps, or cpu.")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1000)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.phase in {"score", "all"} and args.edited_weights is None and (
        args.cache_dir is None or args.model_alias is None
    ):
        raise ValueError(
            f"--phase {args.phase} requires --edited-weights, or both "
            "--cache-dir and --model-alias to locate the main.py-style checkpoint."
        )
    if args.record_start < 0:
        raise ValueError("--record-start must be non-negative.")
    if args.phase in {"prepare", "all"} and not args.data.exists():
        raise FileNotFoundError(f"Dataset does not exist: {args.data}")
    if args.num_edits is not None and args.num_edits <= 0:
        raise ValueError("--num-edits must be positive.")
    if args.score_batch_size <= 0 or args.generation_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if args.max_input_tokens is not None and args.max_input_tokens <= args.max_new_tokens:
        raise ValueError("--max-input-tokens must be larger than --max-new-tokens.")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive.")
    if args.generation_mode == "greedy" and args.num_samples != 1:
        raise ValueError("Greedy decoding supports only --num-samples 1.")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.generation_mode == "sample" and args.temperature <= 0:
        raise ValueError("Sampling temperature must be positive.")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1].")
    if args.top_k < 0:
        raise ValueError("--top-k must be non-negative.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype: str):
    return {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


def load_original_model(model_name_or_path: str, args: argparse.Namespace):
    from transformers import AutoModelForCausalLM

    # 与 main.py 保持一致：先加载一个原始模型，完成 pre 评测后再在同一个
    # model 对象上覆盖编辑层权重。这样不需要在显存中同时放两份大模型。
    print(f"Loading original model: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    model.to(resolve_device(args.device))
    model.eval()
    return model


def load_tokenizer(args: argparse.Namespace):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError("Tokenizer has neither pad, EOS, nor UNK token.")
    return tokenizer


def model_context_length(model, tokenizer, override: int | None) -> int:
    if override is not None:
        return override

    candidates = []
    for attr in ("max_position_embeddings", "n_positions", "seq_length"):
        value = getattr(model.config, attr, None)
        if isinstance(value, int) and 1 < value < 10_000_000:
            candidates.append(value)
    tok_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_max, int) and 1 < tok_max < 10_000_000:
        candidates.append(tok_max)
    return min(candidates) if candidates else 2048


def safe_exp(value: float) -> float | None:
    """Exponentiate a log metric without writing non-standard JSON infinities."""
    if not math.isfinite(value) or value > math.log(np.finfo(np.float64).max):
        return None
    return math.exp(value)


def batched(values: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def read_jsonl(path: Path, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                batch.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def target_with_separator(prompt: str, target: str) -> str:
    """Return target text with a separator only when one is needed."""
    if not target:
        raise ValueError("Reference target is empty.")
    if prompt and not prompt[-1].isspace() and not target[0].isspace():
        return " " + target
    return target


def locality_prompt_answer_pairs(
    record: dict[str, Any],
    source_index: int,
    dataset_format: str,
) -> tuple[list[tuple[str, str]], str]:
    """把不同数据集统一成 (无关问题, 保留答案) 对。"""
    prompts = record.get("neighborhood_prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(
            f"Record {source_index} has no non-empty neighborhood_prompts list."
        )
    if any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise ValueError(f"Record {source_index} contains an invalid neighborhood prompt.")

    resolved_format = dataset_format
    if dataset_format == "auto":
        # ZSRE 为每个无关问题显式提供答案；CounterFact 没有这个字段，
        # 它的一组 neighborhood prompts 共用当前编辑记录的 target_true。
        answers = record.get("neighborhood_prompts_answers")
        if isinstance(answers, list):
            resolved_format = "zsre"
        else:
            resolved_format = "counterfact"

    if resolved_format == "zsre":
        answers = record.get("neighborhood_prompts_answers")
        if not isinstance(answers, list):
            raise ValueError(
                f"ZSRE record {source_index} lacks neighborhood_prompts_answers."
            )
        if len(answers) != len(prompts):
            raise ValueError(
                f"ZSRE record {source_index} has {len(prompts)} locality prompts but "
                f"{len(answers)} answers."
            )
    elif resolved_format == "counterfact":
        target_true = record.get("target_true")
        if not isinstance(target_true, str) or not target_true:
            raise ValueError(
                f"CounterFact record {source_index} has no non-empty target_true."
            )
        answers = [target_true] * len(prompts)
    else:
        raise ValueError(f"Unsupported dataset format: {resolved_format}")

    if any(not isinstance(answer, str) or not answer for answer in answers):
        raise ValueError(f"Record {source_index} contains an empty/non-string locality answer.")
    return list(zip(prompts, answers)), resolved_format


def load_locality_prompts(args: argparse.Namespace, tokenizer) -> list[dict[str, Any]]:
    """读取前 num_edits 条编辑记录，并展开其中所有 locality 问题。"""
    with args.data.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {args.data}.")

    end = None if args.num_edits is None else args.record_start + args.num_edits
    selected = records[args.record_start:end]
    if args.num_edits is not None and len(selected) != args.num_edits:
        raise ValueError(
            f"Requested {args.num_edits} edits from record {args.record_start}, "
            f"but only {len(selected)} records are available."
        )
    rows = []
    resolved_formats = set()
    for source_index, record in enumerate(selected, start=args.record_start):
        record_id = record.get("id", source_index)
        pairs, resolved_format = locality_prompt_answer_pairs(
            record,
            source_index,
            args.dataset_format,
        )
        resolved_formats.add(resolved_format)
        for neighborhood_index, (prompt, reference_target) in enumerate(pairs):
            model_target = target_with_separator(prompt, reference_target)
            target_ids = tokenizer(
                model_target,
                add_special_tokens=False,
            )["input_ids"]
            if not target_ids:
                raise ValueError(
                    f"Empty target tokenization at record {source_index}, prompt {neighborhood_index}."
                )
            rows.append(
                {
                    "record_id": record_id,
                    "source_record_index": source_index,
                    "neighborhood_index": neighborhood_index,
                    "dataset_format": resolved_format,
                    "subject": record.get("subject"),
                    "prompt": prompt,
                    "reference_target": reference_target,
                    "reference_target_model_text": model_target,
                    "_reference_target_token_ids": target_ids,
                }
            )
    if not rows:
        raise ValueError("No neighborhood prompts were loaded.")
    print(
        f"Loaded {len(selected)} records and {len(rows)} locality prompts "
        f"using format(s): {sorted(resolved_formats)}."
    )
    return rows


def encode_prompt(tokenizer, prompt: str, prompt_token_limit: int) -> list[int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    if len(prompt_ids) > prompt_token_limit:
        prompt_ids = prompt_ids[-prompt_token_limit:]
    if not prompt_ids:
        raise ValueError("A prompt became empty after tokenization/truncation.")
    return prompt_ids


@torch.inference_mode()
def score_target_token_ids(
    model,
    tokenizer,
    prompts: Sequence[str],
    target_token_ids: Sequence[Sequence[int]],
    max_input_tokens: int,
    prompt_token_limit: int | None = None,
) -> list[dict[str, float | int]]:
    """只计算 target token 的逐样本平均 NLL，prompt token 不计入 loss。"""
    if len(prompts) != len(target_token_ids):
        raise ValueError("Prompt and target counts do not match.")

    pad_id = tokenizer.pad_token_id
    sequences = []
    labels = []
    for prompt, raw_target_ids in zip(prompts, target_token_ids):
        target_ids = list(raw_target_ids)
        if not target_ids:
            raise ValueError("Cannot score an empty target token sequence.")
        if len(target_ids) >= max_input_tokens:
            raise ValueError(
                f"Target has {len(target_ids)} tokens but context limit is {max_input_tokens}."
            )

        max_prompt = max_input_tokens - len(target_ids)
        if prompt_token_limit is not None:
            max_prompt = min(max_prompt, prompt_token_limit)
        prompt_ids = encode_prompt(tokenizer, prompt, max_prompt)
        # labels 中 prompt 的位置设为 -100。Causal LM shift 后，第一个
        # target token 正好由 prompt 的最后一个 token 预测。
        sequence = prompt_ids + target_ids
        label = [-100] * len(prompt_ids) + target_ids
        sequences.append(sequence)
        labels.append(label)

    width = max(map(len, sequences))
    input_tensor = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
    label_tensor = torch.full((len(sequences), width), -100, dtype=torch.long)
    for row_index, (sequence, label) in enumerate(zip(sequences, labels)):
        size = len(sequence)
        input_tensor[row_index, :size] = torch.tensor(sequence)
        attention_mask[row_index, :size] = 1
        label_tensor[row_index, :size] = torch.tensor(label)

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    attention_mask = attention_mask.to(device)
    label_tensor = label_tensor.to(device)

    logits = model(
        input_ids=input_tensor,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = label_tensor[:, 1:].contiguous()
    # 不能直接使用 outputs.loss：它会把整个 batch 的有效 token 一起平均，
    # 无法得到逐问题结果。这里保留 reduction="none" 后按样本分别求均值。
    token_losses = F.cross_entropy(
        shift_logits.transpose(1, 2),
        shift_labels,
        reduction="none",
        ignore_index=-100,
    )
    valid = shift_labels.ne(-100)
    loss_sums = (token_losses * valid).sum(dim=1)
    counts = valid.sum(dim=1)
    if torch.any(counts == 0):
        raise RuntimeError("A target had no scoreable token after the causal shift.")

    mean_nlls = loss_sums / counts
    results = []
    for mean_nll, loss_sum, count in zip(mean_nlls, loss_sums, counts):
        nll = float(mean_nll.item())
        results.append(
            {
                "nll": nll,
                "ppl": safe_exp(nll),
                "loss_sum": float(loss_sum.item()),
                "num_tokens": int(count.item()),
            }
        )
    return results


def score_rows(
    model,
    tokenizer,
    rows: Sequence[dict[str, Any]],
    target_key: str,
    batch_size: int,
    max_input_tokens: int,
    prompt_token_limit: int | None = None,
) -> list[dict[str, float | int]]:
    scores = []
    for batch in batched(rows, batch_size):
        scores.extend(
            score_target_token_ids(
                model,
                tokenizer,
                [row["prompt"] for row in batch],
                [row[target_key] for row in batch],
                max_input_tokens=max_input_tokens,
                prompt_token_limit=prompt_token_limit,
            )
        )
    return scores


@torch.inference_mode()
def generate_preservation_targets(
    model,
    tokenizer,
    prompts: Sequence[str],
    args: argparse.Namespace,
    max_input_tokens: int,
) -> list[list[dict[str, Any]]]:
    """由原模型生成 preservation target，并保留未经二次分词的 token IDs。"""
    prompt_limit = max_input_tokens - args.max_new_tokens
    if prompt_limit < 1:
        raise ValueError("Context limit must exceed --max-new-tokens.")

    old_padding_side = tokenizer.padding_side
    old_truncation_side = tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    encoded = tokenizer(
        list(prompts),
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=prompt_limit,
        return_tensors="pt",
    )
    tokenizer.padding_side = old_padding_side
    tokenizer.truncation_side = old_truncation_side

    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_width = encoded["input_ids"].shape[1]
    generated_by_prompt: list[list[dict[str, Any]]] = [[] for _ in prompts]

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if args.exact_new_tokens:
        generation_kwargs["min_new_tokens"] = args.max_new_tokens
    if args.generation_mode == "sample":
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    for sample_index in range(args.num_samples):
        output_ids = model.generate(**encoded, **generation_kwargs)
        for prompt_index in range(len(prompts)):
            target_ids = output_ids[prompt_index, input_width:].tolist()
            if not args.exact_new_tokens and tokenizer.eos_token_id in target_ids:
                eos_index = target_ids.index(tokenizer.eos_token_id)
                target_ids = target_ids[: eos_index + 1]
            if not target_ids:
                raise RuntimeError("Generation returned an empty preservation target.")
            generated_by_prompt[prompt_index].append(
                {
                    "sample_index": sample_index,
                    "token_ids": target_ids,
                    "text": tokenizer.decode(target_ids, skip_special_tokens=True),
                    "num_tokens": len(target_ids),
                }
            )
    return generated_by_prompt


def assign_median_groups(rows: list[dict[str, Any]]) -> float:
    values = np.asarray([row["reference_nll_pre"] for row in rows], dtype=np.float64)
    median = float(np.median(values))
    # 按 reference NLL 排序后平分，而不简单使用 <= median。这样即使大量
    # 样本恰好等于中位数，两组仍严格保持 50/50，且 stable sort 可复现。
    order = np.argsort(values, kind="stable")
    low_count = (len(rows) + 1) // 2
    low_indices = set(order[:low_count].tolist())
    for index, row in enumerate(rows):
        row["familiarity_group"] = (
            "low_reference_ppl" if index in low_indices else "high_reference_ppl"
        )
    return median


def prepare(args: argparse.Namespace, model, tokenizer, prepared_file: Path) -> dict[str, Any]:
    """完成所有只依赖原模型的工作，结果可供多个编辑 checkpoint 复用。"""
    rows = load_locality_prompts(args, tokenizer)
    max_tokens = model_context_length(model, tokenizer, args.max_input_tokens)
    print(f"Using model context limit: {max_tokens}")

    # 第一步：以数据集保留答案的原模型 NLL 作为 familiarity 分层依据。
    print("Scoring reference-answer familiarity with the original model...")
    reference_scores = score_rows(
        model,
        tokenizer,
        rows,
        target_key="_reference_target_token_ids",
        batch_size=args.score_batch_size,
        max_input_tokens=max_tokens,
    )
    for row, score in zip(rows, reference_scores):
        row["reference_nll_pre"] = score["nll"]
        row["reference_ppl_pre"] = score["ppl"]
        row["reference_num_tokens"] = score["num_tokens"]
    median = assign_median_groups(rows)
    counts = defaultdict(int)
    for row in rows:
        counts[row["familiarity_group"]] += 1
    print(f"Reference-NLL median: {median:.6f}; groups: {dict(counts)}")

    prepared_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_limit = max_tokens - args.max_new_tokens
    num_targets = 0
    with prepared_file.open("w", encoding="utf-8") as handle:
        for batch_number, batch in enumerate(
            batched(rows, args.generation_batch_size), start=1
        ):
            prompts = [row["prompt"] for row in batch]
            # 第二步：原模型生成固定 token 数的答案，作为之后必须保持的行为。
            generations = generate_preservation_targets(
                model,
                tokenizer,
                prompts,
                args,
                max_input_tokens=max_tokens,
            )

            expanded = []
            for row, prompt_generations in zip(batch, generations):
                for generation in prompt_generations:
                    expanded.append((row, generation))
            # 第三步：在原模型上 teacher-force 同一批生成 token，得到 pre NLL。
            # 后续 post 评测会直接复用这些 token IDs，避免 decode/re-tokenize 偏差。
            generated_scores = []
            for score_batch in batched(expanded, args.score_batch_size):
                generated_scores.extend(
                    score_target_token_ids(
                        model,
                        tokenizer,
                        [row["prompt"] for row, _ in score_batch],
                        [generation["token_ids"] for _, generation in score_batch],
                        max_input_tokens=max_tokens,
                        prompt_token_limit=prompt_limit,
                    )
                )

            for (row, generation), score in zip(expanded, generated_scores):
                output_row = {key: value for key, value in row.items() if not key.startswith("_")}
                output_row.update(
                    {
                        "preservation_sample_index": generation["sample_index"],
                        "preservation_target": generation["text"],
                        "preservation_target_token_ids": generation["token_ids"],
                        "preservation_num_tokens": generation["num_tokens"],
                        "preservation_nll_pre": score["nll"],
                        "preservation_ppl_pre": score["ppl"],
                    }
                )
                handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                num_targets += 1

            processed = min(
                batch_number * args.generation_batch_size,
                len(rows),
            )
            if processed % args.log_every < args.generation_batch_size or processed == len(rows):
                print(f"Prepared {processed}/{len(rows)} prompts ({num_targets} targets).")

    metadata = {
        "data": str(args.data.resolve()),
        "requested_dataset_format": args.dataset_format,
        "resolved_dataset_formats": sorted({row["dataset_format"] for row in rows}),
        "base_model": args.base_model,
        "record_start": args.record_start,
        "requested_num_edits": args.num_edits,
        "num_records": len({row["source_record_index"] for row in rows}),
        "num_prompts": len(rows),
        "num_preservation_targets": num_targets,
        "reference_nll_median": median,
        "familiarity_split_method": (
            "stable rank split at the reference-NLL median; exact median ties are split reproducibly"
        ),
        "num_prompts_equal_to_median": sum(
            math.isclose(row["reference_nll_pre"], median, rel_tol=0.0, abs_tol=1e-12)
            for row in rows
        ),
        "familiarity_group_counts": dict(counts),
        "generation_mode": args.generation_mode,
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "exact_new_tokens": args.exact_new_tokens,
        "temperature": args.temperature if args.generation_mode == "sample" else None,
        "top_p": args.top_p if args.generation_mode == "sample" else None,
        "top_k": args.top_k if args.generation_mode == "sample" else None,
        "max_input_tokens": max_tokens,
        "seed": args.seed,
        "prepared_file": str(prepared_file.resolve()),
    }
    meta_file = prepared_file.with_suffix(prepared_file.suffix + ".meta.json")
    with meta_file.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"Prepared targets written to {prepared_file}")
    print(f"Preparation metadata written to {meta_file}")
    return metadata


def resolve_edited_weights_path(args: argparse.Namespace, prepared_file: Path) -> Path:
    """按 load.py 的命名规则找到编辑权重；显式路径优先。"""
    if args.edited_weights is not None:
        return args.edited_weights

    data_name = args.data_name
    if data_name is None:
        meta_file = prepared_file.with_suffix(prepared_file.suffix + ".meta.json")
        if meta_file.exists():
            with meta_file.open("r", encoding="utf-8") as handle:
                prepared_meta = json.load(handle)
            data_name = Path(prepared_meta["data"]).stem
        else:
            data_name = args.data.stem

    model_alias = args.model_alias.replace("/", "-")
    filename = f"{data_name}-{args.load_name}-{model_alias}.pt"
    return args.cache_dir / "saved_weights" / args.algorithm / filename


def load_edited_weights_into_model(model, weights_file: Path) -> None:
    """复用 main.py/load.py 的策略：在原模型对象上原地覆盖编辑层权重。"""
    from util import nethook

    if not weights_file.exists():
        raise FileNotFoundError(f"Edited checkpoint does not exist: {weights_file}")
    print(f"Loading edited weights: {weights_file}")
    weights = torch.load(weights_file, map_location="cpu")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Edited checkpoint must be a non-empty parameter dictionary.")

    # save_model() 只保存被修改的层。这里逐项找到原模型参数并覆盖，未编辑层
    # 保持原值；这与 load.load_model(model, cfg) 的行为一致。
    with torch.no_grad():
        for name, value in weights.items():
            destination = nethook.get_parameter(model, name)
            if destination.shape != value.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: model {tuple(destination.shape)}, "
                    f"checkpoint {tuple(value.shape)}."
                )
            destination.copy_(value.to(device=destination.device, dtype=destination.dtype))
    model.eval()
    print(f"Applied {len(weights)} edited tensors to the original model.")


def percentile(values: Sequence[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if len(values) else None


def describe(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "p95": percentile(array, 95),
    }


def summarize_prompt_level(
    aggregates: dict[tuple[Any, int], dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"signed_delta_nll": [], "absolute_delta_nll": []}
    )
    for aggregate in aggregates.values():
        count = aggregate["count"]
        group = aggregate["group"]
        groups[group]["signed_delta_nll"].append(aggregate["signed_sum"] / count)
        groups[group]["absolute_delta_nll"].append(aggregate["absolute_sum"] / count)

    group_summary = {}
    for group, metrics in groups.items():
        signed = metrics["signed_delta_nll"]
        absolute = metrics["absolute_delta_nll"]
        group_summary[group] = {
            "num_prompts": len(signed),
            "signed_delta_nll": describe(signed),
            "absolute_delta_nll": describe(absolute),
            "positive_damage_rate": float(np.mean(np.asarray(signed) > 0)),
        }

    low = group_summary.get("low_reference_ppl")
    high = group_summary.get("high_reference_ppl")
    contrasts = {}
    if low and high:
        contrasts = {
            "high_minus_low_mean_signed_delta_nll": (
                high["signed_delta_nll"]["mean"] - low["signed_delta_nll"]["mean"]
            ),
            "high_minus_low_mean_absolute_delta_nll": (
                high["absolute_delta_nll"]["mean"] - low["absolute_delta_nll"]["mean"]
            ),
            "high_minus_low_positive_damage_rate": (
                high["positive_damage_rate"] - low["positive_damage_rate"]
            ),
        }
    return {
        "analysis_unit": "one neighborhood prompt; multiple samples are averaged per prompt",
        "groups": group_summary,
        "contrasts": contrasts,
    }


def score_edited(
    args: argparse.Namespace,
    model,
    tokenizer,
    prepared_file: Path,
    results_file: Path,
    summary_file: Path,
) -> dict[str, Any]:
    """在编辑模型上重算 preservation-target NLL，并与保存的 pre NLL 配对。"""
    if not prepared_file.exists():
        raise FileNotFoundError(f"Prepared file does not exist: {prepared_file}")
    current_context = model_context_length(model, tokenizer, args.max_input_tokens)
    prepared_meta_file = prepared_file.with_suffix(prepared_file.suffix + ".meta.json")
    prepared_meta: dict[str, Any] = {}
    if prepared_meta_file.exists():
        with prepared_meta_file.open("r", encoding="utf-8") as handle:
            prepared_meta = json.load(handle)
        prepared_context = int(prepared_meta["max_input_tokens"])
        prepared_new_tokens = int(prepared_meta["max_new_tokens"])
        prepared_num_edits = int(prepared_meta["num_records"])
        if args.num_edits is not None and args.num_edits != prepared_num_edits:
            raise ValueError(
                f"--num-edits is {args.num_edits}, but {prepared_file} was prepared "
                f"for {prepared_num_edits} edits. Run prepare again for that edit count."
            )
        if current_context < prepared_context:
            raise ValueError(
                f"Edited model context limit ({current_context}) is smaller than the "
                f"context used in preparation ({prepared_context})."
            )
        max_tokens = prepared_context
        prompt_limit = prepared_context - prepared_new_tokens
    else:
        print(
            f"Warning: preparation metadata not found at {prepared_meta_file}; "
            "falling back to current CLI context settings."
        )
        max_tokens = current_context
        prompt_limit = max_tokens - args.max_new_tokens
    results_file.parent.mkdir(parents=True, exist_ok=True)

    aggregates: dict[tuple[Any, int], dict[str, Any]] = {}
    row_count = 0
    with results_file.open("w", encoding="utf-8") as output:
        for batch in read_jsonl(prepared_file, args.score_batch_size):
            scores = score_target_token_ids(
                model,
                tokenizer,
                [row["prompt"] for row in batch],
                [row["preservation_target_token_ids"] for row in batch],
                max_input_tokens=max_tokens,
                prompt_token_limit=prompt_limit,
            )
            for row, score in zip(batch, scores):
                pre_nll = float(row["preservation_nll_pre"])
                post_nll = float(score["nll"])
                signed_delta = post_nll - pre_nll
                absolute_delta = abs(signed_delta)
                # signed_delta > 0：编辑模型降低了对原模型答案的支持；
                # absolute_delta 衡量严格的行为漂移，不关心变化方向。
                row.update(
                    {
                        "preservation_nll_post": post_nll,
                        "preservation_ppl_post": score["ppl"],
                        "signed_delta_nll": signed_delta,
                        "absolute_delta_nll": absolute_delta,
                        "ppl_ratio_post_over_pre": safe_exp(signed_delta),
                    }
                )
                output.write(json.dumps(row, ensure_ascii=False) + "\n")

                # sampling 模式下同一问题会有 K 个生成答案。统计时先对 K 个
                # 样本求均值，再比较 high/low familiarity，避免伪造样本量。
                key = (row["source_record_index"], row["neighborhood_index"])
                if key not in aggregates:
                    aggregates[key] = {
                        "group": row["familiarity_group"],
                        "signed_sum": 0.0,
                        "absolute_sum": 0.0,
                        "count": 0,
                    }
                aggregate = aggregates[key]
                aggregate["signed_sum"] += signed_delta
                aggregate["absolute_sum"] += absolute_delta
                aggregate["count"] += 1
                row_count += 1

            if row_count % args.log_every < len(batch):
                print(f"Scored {row_count} preservation targets with the edited model.")

    summary = summarize_prompt_level(aggregates)
    summary.update(
        {
            "base_model": args.base_model,
            "edited_weights": str(args.edited_weights.resolve()),
            "record_start": prepared_meta.get("record_start"),
            "num_edits": prepared_meta.get("num_records"),
            "data": prepared_meta.get("data"),
            "dataset_formats": prepared_meta.get("resolved_dataset_formats"),
            "prepared_file": str(prepared_file.resolve()),
            "results_file": str(results_file.resolve()),
            "num_result_rows": row_count,
            "num_unique_prompts": len(aggregates),
        }
    )
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Detailed results written to {results_file}")
    print(f"Group summary written to {summary_file}")
    return summary


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared_file = args.prepared_file or args.output_dir / "prepared.jsonl"/ "prepared.jsonl"
    results_file = args.results_file or args.output_dir / f"{args.load_name}_scored.jsonl"
    summary_file = args.summary_file or args.output_dir / f"{args.load_name}_summary.json"
    tokenizer = load_tokenizer(args)

    model = None
    if args.phase in {"prepare", "all"}:
        model = load_original_model(args.base_model, args)
        prepare(args, model, tokenizer, prepared_file)

    if args.phase in {"score", "all"}:
        if model is None:
            model = load_original_model(args.base_model, args)

        # prepare 阶段已经用这个 model 得到 pre 指标。现在像 main.py 一样把
        # 保存的编辑层权重加载到同一个对象，再计算 post 指标。
        args.edited_weights = resolve_edited_weights_path(args, prepared_file)
        load_edited_weights_into_model(model, args.edited_weights)
        summary = score_edited(
            args,
            model,
            tokenizer,
            prepared_file,
            results_file,
            summary_file,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

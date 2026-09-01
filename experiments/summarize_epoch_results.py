#!/usr/bin/env python3
"""Summarize every evaluated epoch belonging to one model-edit save name."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


RESULT_MARKER = "The Evaluation Results after Editing:"
OPTIONAL_TEXT_SUFFIXES = (".json", ".txt", ".log")
LOCALITY_KEYS = (
    "neighborhood_kl_",
    "neighborhood_top",
    "neighborhood_locality_",
)


@dataclass
class ResultFiles:
    """Regular and locality result files for one evaluated checkpoint."""

    epoch: Optional[int]
    result_file: Optional[Path] = None
    locality_file: Optional[Path] = None


def strip_optional_text_suffix(filename: str) -> str:
    for suffix in OPTIONAL_TEXT_SUFFIXES:
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def classify_result_filename(
    filename: str, prefix: str
) -> Optional[Tuple[Optional[int], bool]]:
    """Return ``(epoch, is_locality)`` when a filename belongs to prefix."""
    stem = strip_optional_text_suffix(filename)
    match = re.fullmatch(
        rf"{re.escape(prefix)}(?:-epoch-(\d+))?(-locality)?", stem
    )
    if match is None:
        return None
    epoch = int(match.group(1)) if match.group(1) is not None else None
    return epoch, match.group(2) is not None


def discover_result_files(result_dir: Path, prefix: str) -> Dict[Optional[int], ResultFiles]:
    """Find final and epoch result pairs directly under ``result_dir``."""
    discovered: Dict[Optional[int], ResultFiles] = {}
    for path in result_dir.iterdir():
        if not path.is_file():
            continue
        classification = classify_result_filename(path.name, prefix)
        if classification is None:
            continue
        epoch, is_locality = classification
        files = discovered.setdefault(epoch, ResultFiles(epoch=epoch))
        attribute = "locality_file" if is_locality else "result_file"
        existing = getattr(files, attribute)
        if existing is not None:
            raise RuntimeError(
                f"Multiple files match {prefix!r} for "
                f"epoch {epoch if epoch is not None else 'final'}: "
                f"{existing} and {path}"
            )
        setattr(files, attribute, path)
    return discovered


def extract_metrics_text(text: str, source: str) -> Dict[str, object]:
    """Extract the JSON object following the evaluation marker."""
    marker_index = text.rfind(RESULT_MARKER)
    candidate = (
        text[marker_index + len(RESULT_MARKER) :]
        if marker_index >= 0
        else text
    )
    decoder = json.JSONDecoder()
    for index, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"Cannot find evaluation JSON object in {source}")


def load_metrics(path: Optional[Path]) -> Dict[str, object]:
    if path is None:
        return {}
    return extract_metrics_text(path.read_text(encoding="utf-8"), str(path))


def merge_metrics(files: ResultFiles) -> Dict[str, object]:
    """Merge regular S/A metrics with KL/Top-k from the locality run."""
    regular = load_metrics(files.result_file)
    locality = load_metrics(files.locality_file)
    merged = {**locality, **regular}
    for key, value in locality.items():
        if key.startswith(LOCALITY_KEYS):
            merged[key] = value
    return merged


def metric_as_float(metrics: Dict[str, object], key: str) -> Optional[float]:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def success_accuracy(
    metrics: Dict[str, object], prefix: str
) -> Tuple[Optional[float], Optional[float]]:
    """Return dataset-compatible success and accuracy values.

    CounterFact-style evaluators report success as ``*_prompts_probs``.
    ZSRE instead reports success as ``*_prompts_correct`` and accuracy as
    strict whole-answer correctness in ``*_strict_correct``.
    """
    success = metric_as_float(metrics, f"{prefix}_probs")
    prompt_correct = metric_as_float(metrics, f"{prefix}_correct")
    if success is not None:
        return success, prompt_correct

    strict_prefix = (
        prefix[: -len("_prompts")]
        if prefix.endswith("_prompts")
        else prefix
    )
    strict_correct = metric_as_float(
        metrics, f"{strict_prefix}_strict_correct"
    )
    return prompt_correct, strict_correct


def format_pair(metrics: Dict[str, object], prefix: str) -> str:
    success, accuracy = success_accuracy(metrics, prefix)
    if success is None and accuracy is None:
        return "-"
    success_text = "-" if success is None else f"{100 * success:.1f}"
    accuracy_text = "-" if accuracy is None else f"{100 * accuracy:.1f}"
    return f"{success_text} / {accuracy_text}"


def format_metric(
    metrics: Dict[str, object], key: str, scale: float, digits: int
) -> str:
    value = metric_as_float(metrics, key)
    return "-" if value is None else f"{scale * value:.{digits}f}"


def sorted_result_items(
    discovered: Dict[Optional[int], ResultFiles]
) -> Iterable[Tuple[Optional[int], ResultFiles]]:
    return sorted(
        discovered.items(),
        key=lambda item: (item[0] is None, item[0] if item[0] is not None else 0),
    )


def print_markdown_table(
    rows: Iterable[Tuple[Optional[int], ResultFiles, Dict[str, object]]]
) -> None:
    print(
        "| Epoch | Rewrite S/A | Paraphrase S/A | Neighborhood S/A | "
        "KL ↓ | Top-1 ↑ | Top-5 ↑ | Top-10 ↑ |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for epoch, _, metrics in rows:
        epoch_label = "final" if epoch is None else str(epoch)
        print(
            f"| {epoch_label} "
            f"| {format_pair(metrics, 'rewrite_prompts')} "
            f"| {format_pair(metrics, 'paraphrase_prompts')} "
            f"| {format_pair(metrics, 'neighborhood_prompts')} "
            f"| {format_metric(metrics, 'neighborhood_kl_original_to_edited', 1, 4)} "
            f"| {format_metric(metrics, 'neighborhood_top1_overlap', 100, 2)} "
            f"| {format_metric(metrics, 'neighborhood_top5_overlap', 100, 2)} "
            f"| {format_metric(metrics, 'neighborhood_top10_overlap', 100, 2)} |"
        )


def write_csv(
    output_path: Path,
    rows: Iterable[Tuple[Optional[int], ResultFiles, Dict[str, object]]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "rewrite_success_pct",
        "rewrite_accuracy_pct",
        "paraphrase_success_pct",
        "paraphrase_accuracy_pct",
        "neighborhood_success_pct",
        "neighborhood_accuracy_pct",
        "kl_original_to_edited",
        "top1_overlap_pct",
        "top5_overlap_pct",
        "top10_overlap_pct",
        "result_file",
        "locality_file",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for epoch, files, metrics in rows:
            def percentage(key: str) -> Optional[float]:
                value = metric_as_float(metrics, key)
                return None if value is None else 100 * value

            def pair_percentages(
                prefix: str,
            ) -> Tuple[Optional[float], Optional[float]]:
                success, accuracy = success_accuracy(metrics, prefix)
                return (
                    None if success is None else 100 * success,
                    None if accuracy is None else 100 * accuracy,
                )

            rewrite_success, rewrite_accuracy = pair_percentages(
                "rewrite_prompts"
            )
            paraphrase_success, paraphrase_accuracy = pair_percentages(
                "paraphrase_prompts"
            )
            neighborhood_success, neighborhood_accuracy = pair_percentages(
                "neighborhood_prompts"
            )

            writer.writerow(
                {
                    "epoch": "final" if epoch is None else epoch,
                    "rewrite_success_pct": rewrite_success,
                    "rewrite_accuracy_pct": rewrite_accuracy,
                    "paraphrase_success_pct": paraphrase_success,
                    "paraphrase_accuracy_pct": paraphrase_accuracy,
                    "neighborhood_success_pct": neighborhood_success,
                    "neighborhood_accuracy_pct": neighborhood_accuracy,
                    "kl_original_to_edited": metric_as_float(
                        metrics, "neighborhood_kl_original_to_edited"
                    ),
                    "top1_overlap_pct": percentage("neighborhood_top1_overlap"),
                    "top5_overlap_pct": percentage("neighborhood_top5_overlap"),
                    "top10_overlap_pct": percentage("neighborhood_top10_overlap"),
                    "result_file": files.result_file or "",
                    "locality_file": files.locality_file or "",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect regular and locality evaluation files for every epoch "
            "of one save_name."
        )
    )
    parser.add_argument(
        "result_dir",
        type=Path,
        help="Directory containing files such as memit-<save_name>-epoch-0050",
    )
    parser.add_argument(
        "save_name",
        help="Base save_name, with or without the '<algorithm>-' prefix",
    )
    parser.add_argument(
        "--algorithm",
        default="memit",
        help="Filename algorithm prefix when save_name omits it (default: memit)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        dest="csv_path",
        help="Optionally write raw numeric metrics to this CSV file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_dir = args.result_dir.expanduser().resolve()
    if not result_dir.is_dir():
        print(f"Result directory does not exist: {result_dir}", file=sys.stderr)
        return 2

    prefix = args.save_name
    if not prefix.startswith(f"{args.algorithm}-"):
        prefix = f"{args.algorithm}-{prefix}"

    discovered = discover_result_files(result_dir, prefix)
    if not discovered:
        print(
            f"No result files matching {prefix!r} were found in {result_dir}",
            file=sys.stderr,
        )
        return 1

    rows = [
        (epoch, files, merge_metrics(files))
        for epoch, files in sorted_result_items(discovered)
    ]
    print(f"Results: {prefix} ({result_dir})")
    print_markdown_table(rows)

    if args.csv_path is not None:
        csv_path = args.csv_path.expanduser().resolve()
        write_csv(csv_path, rows)
        print(f"\nSaved CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

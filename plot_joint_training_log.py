#!/usr/bin/env python3
"""Parse MEMIT joint-optimization logs and plot convergence curves.

Supports both the current AdamW log format and historical runs whose joint
objective contained an explicit ``L2 * update_sq_norm`` term.
"""

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean


PARSER_VERSION = "2.0"
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
JOINT_EVENT_PATTERN = re.compile(
    r"joint epoch\s+\d+(?:/\d+)?(?:\s+complete:|\s*,)",
    re.IGNORECASE,
)
EPOCH_CORE_PATTERN = re.compile(
    rf"joint epoch\s+(?P<epoch>\d+)\s+complete:\s*"
    rf"loss\s+(?P<total_loss>{NUMBER})\s*=\s*"
    rf"target\s+(?P<target_loss>{NUMBER})\s*\+\s*"
    rf"alpha\s*\*\s*preserve\s+(?P<alpha>{NUMBER})\s*\*\s*"
    rf"(?P<preserve_loss>{NUMBER})",
    re.IGNORECASE,
)
L2_PATTERN = re.compile(
    rf"L2\s*\*\s*update_sq_norm\s+"
    rf"(?P<l2>{NUMBER})\s*\*\s*(?P<update_sq_norm>{NUMBER})",
    re.IGNORECASE,
)
LAYER_NORM_PATTERN = re.compile(
    rf"beta\s*\*\s*layer_norm\["
    rf"(?P<layer_norm_mode>none|balance|cap)\]\s+"
    rf"(?P<layer_norm_beta>{NUMBER})\s*\*\s*"
    rf"(?P<layer_norm_loss>{NUMBER})",
    re.IGNORECASE,
)
LAYER_RELATIVE_NORM_PATTERN = re.compile(
    rf"layer relative norm min/mean/max/cv\s+"
    rf"(?P<layer_relative_norm_min>{NUMBER})\s*/\s*"
    rf"(?P<layer_relative_norm_mean>{NUMBER})\s*/\s*"
    rf"(?P<layer_relative_norm_max>{NUMBER})\s*/\s*"
    rf"(?P<layer_relative_norm_cv>{NUMBER})",
    re.IGNORECASE,
)
ADAMW_PATTERN = re.compile(
    rf"AdamW\s+weight\s+decay\s+(?P<weight_decay>{NUMBER})\s*"
    rf"\(factor\s+(?P<decay_factor>{NUMBER})\)",
    re.IGNORECASE,
)
GRADIENT_PATTERN = re.compile(
    rf"gradient norm\s+(?P<gradient_norm>{NUMBER})"
    rf"(?P<clipped>\s*\(clipped\))?",
    re.IGNORECASE,
)
UPDATE_PATTERN = re.compile(
    rf"update norm\s+(?P<update_norm>{NUMBER})"
    rf"(?:\s*\(relative norm change\s+"
    rf"(?P<relative_update_norm_change>{NUMBER})\))?",
    re.IGNORECASE,
)
TARGET_STATS_PATTERN = re.compile(
    rf"per-edit target\s+median/p90/max\s+"
    rf"(?P<target_median>{NUMBER})\s*/\s*"
    rf"(?P<target_p90>{NUMBER})\s*/\s*"
    rf"(?P<target_max>{NUMBER})",
    re.IGNORECASE,
)

CSV_FIELDS = (
    "epoch",
    "total_loss",
    "target_loss",
    "alpha",
    "preserve_loss",
    "weighted_preserve_loss",
    "l2",
    "update_sq_norm",
    "weighted_l2_loss",
    "layer_norm_mode",
    "layer_norm_beta",
    "layer_norm_loss",
    "weighted_layer_norm_loss",
    "layer_relative_norm_min",
    "layer_relative_norm_mean",
    "layer_relative_norm_max",
    "layer_relative_norm_cv",
    "weight_decay",
    "decay_factor",
    "adamw_decay_norm_per_step",
    "gradient_norm",
    "gradient_clipped",
    "update_norm",
    "relative_update_norm_change",
    "target_median",
    "target_p90",
    "target_max",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract 'joint epoch ... complete' records from a training log, "
            "then save convergence plots, CSV data, and a text summary."
        )
    )
    parser.add_argument("log_file", type=Path, help="Slurm/stdout training log")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {PARSER_VERSION}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <log_stem>_joint_curves beside log)",
    )
    parser.add_argument(
        "--start-epoch", type=int, default=None, help="First epoch to plot"
    )
    parser.add_argument(
        "--end-epoch", type=int, default=None, help="Last epoch to plot"
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=20,
        help="Moving-average window; use 1 to disable (default: 20)",
    )
    parser.add_argument(
        "--summary-window",
        type=int,
        default=50,
        help="Number of recent epochs used in convergence summary (default: 50)",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--log-target-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use logarithmic y-axis for median/p90/max (default: true)",
    )
    return parser.parse_args()


def _optional_groups(pattern, text):
    match = pattern.search(text)
    return match.groupdict() if match is not None else {}


def _parse_epoch_block(block, line_number):
    """Parse one epoch-complete block without requiring a rigid tail order."""
    core_match = EPOCH_CORE_PATTERN.search(block)
    if core_match is None:
        return None

    values = {
        field: None
        for field in CSV_FIELDS
        if field
        not in {
            "epoch",
            "weighted_preserve_loss",
            "weighted_l2_loss",
            "weighted_layer_norm_loss",
            "adamw_decay_norm_per_step",
            "gradient_clipped",
        }
    }
    values.update(core_match.groupdict())
    for pattern in (
        L2_PATTERN,
        LAYER_NORM_PATTERN,
        LAYER_RELATIVE_NORM_PATTERN,
        ADAMW_PATTERN,
        GRADIENT_PATTERN,
        UPDATE_PATTERN,
        TARGET_STATS_PATTERN,
    ):
        values.update(_optional_groups(pattern, block))

    epoch = int(values.pop("epoch"))
    clipped = bool(values.pop("clipped", None))
    layer_norm_mode = values.pop("layer_norm_mode", None)
    record = {
        key: (float(value) if value is not None else None)
        for key, value in values.items()
    }
    record["epoch"] = epoch
    record["gradient_clipped"] = int(clipped)
    record["layer_norm_mode"] = (
        layer_norm_mode.lower() if layer_norm_mode is not None else None
    )
    record["weighted_preserve_loss"] = (
        record["alpha"] * record["preserve_loss"]
    )
    if record["l2"] is not None and record["update_sq_norm"] is not None:
        record["weighted_l2_loss"] = (
            record["l2"] * record["update_sq_norm"]
        )
    else:
        record["weighted_l2_loss"] = None
    if (
        record["layer_norm_beta"] is not None
        and record["layer_norm_loss"] is not None
    ):
        record["weighted_layer_norm_loss"] = (
            record["layer_norm_beta"] * record["layer_norm_loss"]
        )
    else:
        record["weighted_layer_norm_loss"] = None
    if (
        record["decay_factor"] is not None
        and record["update_norm"] is not None
    ):
        # AdamW weight decay is not part of the differentiable loss. This is
        # only an approximate per-step optimizer-effect diagnostic.
        record["adamw_decay_norm_per_step"] = (
            (1.0 - record["decay_factor"]) * record["update_norm"]
        )
    else:
        record["adamw_decay_norm_per_step"] = None
    record["line_number"] = line_number
    return record


def parse_log(log_file):
    records_by_epoch = {}
    match_count = 0
    raw_text = log_file.read_text(encoding="utf-8", errors="replace")
    text = ANSI_ESCAPE.sub("", raw_text)
    events = list(JOINT_EVENT_PATTERN.finditer(text))
    for event_index, event in enumerate(events):
        if "complete:" not in event.group(0).lower():
            continue
        block_end = (
            events[event_index + 1].start()
            if event_index + 1 < len(events)
            else len(text)
        )
        block = text[event.start() : block_end]
        line_number = text.count("\n", 0, event.start()) + 1
        record = _parse_epoch_block(block, line_number)
        if record is None:
            continue

        # When a resumed run repeats an epoch, its latest entry is the one
        # that corresponds to the latest optimizer/checkpoint state.
        records_by_epoch[record["epoch"]] = record
        match_count += 1

    records = [records_by_epoch[e] for e in sorted(records_by_epoch)]
    duplicate_count = match_count - len(records)
    return records, duplicate_count


def filter_records(records, start_epoch=None, end_epoch=None):
    return [
        record
        for record in records
        if (start_epoch is None or record["epoch"] >= start_epoch)
        and (end_epoch is None or record["epoch"] <= end_epoch)
    ]


def moving_average(values, window):
    if window <= 1:
        return list(values)
    result = []
    for index in range(len(values)):
        window_values = values[max(0, index - window + 1) : index + 1]
        finite_values = [
            value
            for value in window_values
            if value is not None and math.isfinite(value)
        ]
        result.append(mean(finite_values) if finite_values else float("nan"))
    return result


def has_values(values):
    return any(value is not None and math.isfinite(value) for value in values)


def write_csv(records, output_file):
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in CSV_FIELDS})


def add_raw_and_smooth(ax, epochs, values, label, window, color=None):
    (raw_line,) = ax.plot(
        epochs, values, label=f"{label} (raw)", color=color, alpha=0.28, linewidth=0.8
    )
    if window > 1 and len(values) >= 2:
        ax.plot(
            epochs,
            moving_average(values, window),
            label=f"{label} (MA{window})",
            color=raw_line.get_color(),
            linewidth=1.8,
        )
    else:
        raw_line.set_alpha(1.0)
        raw_line.set_linewidth(1.5)


def plot_records(records, output_file, smooth_window, dpi, log_target_stats):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it with "
            "'pip install matplotlib'."
        ) from exc

    epochs = [record["epoch"] for record in records]
    series = {
        field: [record[field] for record in records]
        for field in CSV_FIELDS
        if field not in {"epoch", "gradient_clipped", "layer_norm_mode"}
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    modes = []
    if has_values(series["weighted_l2_loss"]):
        modes.append("explicit L2")
    if has_values(series["weight_decay"]):
        modes.append("AdamW")
    layer_norm_modes = sorted(
        {
            record["layer_norm_mode"]
            for record in records
            if record["layer_norm_mode"] is not None
        }
    )
    if layer_norm_modes:
        modes.append("layer norm: " + ", ".join(layer_norm_modes))
    mode_suffix = f"; {', '.join(modes)}" if modes else ""
    fig.suptitle(
        f"Joint optimization convergence "
        f"(epochs {epochs[0]}-{epochs[-1]}{mode_suffix})",
        fontsize=15,
    )

    ax = axes[0, 0]
    add_raw_and_smooth(
        ax, epochs, series["total_loss"], "total", smooth_window, "tab:blue"
    )
    add_raw_and_smooth(
        ax, epochs, series["target_loss"], "target", smooth_window, "tab:orange"
    )
    ax.set_title("Objective and target loss")
    ax.set_ylabel("Loss")

    ax = axes[0, 1]
    add_raw_and_smooth(
        ax,
        epochs,
        series["weighted_preserve_loss"],
        "alpha * preserve",
        smooth_window,
        "tab:green",
    )
    if has_values(series["weighted_l2_loss"]):
        add_raw_and_smooth(
            ax,
            epochs,
            series["weighted_l2_loss"],
            "L2 * update_sq_norm",
            smooth_window,
            "tab:red",
        )
    if has_values(series["weighted_layer_norm_loss"]):
        add_raw_and_smooth(
            ax,
            epochs,
            series["weighted_layer_norm_loss"],
            "beta * layer norm",
            smooth_window,
            "tab:cyan",
        )
    decay_ax = None
    if has_values(series["adamw_decay_norm_per_step"]):
        decay_ax = ax.twinx()
        add_raw_and_smooth(
            decay_ax,
            epochs,
            series["adamw_decay_norm_per_step"],
            "AdamW decay norm/step",
            smooth_window,
            "tab:purple",
        )
        decay_ax.set_ylabel("Approx. norm removed per step")
        weight_decays = sorted(
            {value for value in series["weight_decay"] if value is not None}
        )
        decay_factors = sorted(
            {value for value in series["decay_factor"] if value is not None}
        )
        ax.text(
            0.98,
            0.97,
            "AdamW wd=" + ", ".join(f"{value:g}" for value in weight_decays)
            + "\nfactor="
            + ", ".join(f"{value:.8f}" for value in decay_factors),
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"},
        )
    ax.set_title("Preservation loss and optimizer decay")
    ax.set_ylabel("Weighted loss")

    ax = axes[1, 0]
    add_raw_and_smooth(
        ax,
        epochs,
        series["gradient_norm"],
        "gradient norm",
        smooth_window,
        "tab:purple",
    )
    add_raw_and_smooth(
        ax,
        epochs,
        series["update_norm"],
        "update norm",
        smooth_window,
        "tab:brown",
    )
    change_ax = None
    if has_values(series["relative_update_norm_change"]):
        relative_changes = list(series["relative_update_norm_change"])
        first_index = next(
            (
                index
                for index, value in enumerate(relative_changes)
                if value is not None
            ),
            None,
        )
        if first_index is not None:
            # The first ratio uses the initial zero update norm as denominator
            # and is not useful for diagnosing convergence.
            relative_changes[first_index] = None
        change_ax = ax.twinx()
        add_raw_and_smooth(
            change_ax,
            epochs,
            relative_changes,
            "relative update-norm change",
            smooth_window,
            "tab:gray",
        )
        change_ax.axhline(0.0, color="0.5", linewidth=0.8, alpha=0.5)
    if has_values(series["layer_relative_norm_cv"]):
        if change_ax is None:
            change_ax = ax.twinx()
        add_raw_and_smooth(
            change_ax,
            epochs,
            series["layer_relative_norm_cv"],
            "layer relative-norm CV",
            smooth_window,
            "tab:olive",
        )
    if change_ax is not None:
        change_ax.set_ylabel("Relative change / layer CV")
    clipped_epochs = [
        record["epoch"] for record in records if record["gradient_clipped"]
    ]
    if clipped_epochs:
        norm_values = [
            value
            for value in series["gradient_norm"] + series["update_norm"]
            if value is not None and math.isfinite(value)
        ]
        y_top = max(norm_values) if norm_values else 0.0
        ax.scatter(
            clipped_epochs,
            [y_top] * len(clipped_epochs),
            marker="x",
            color="black",
            s=24,
            label="gradient clipped",
        )
    ax.set_title("Gradient and update norms")
    ax.set_ylabel("Norm")

    ax = axes[1, 1]
    for field, label, color in (
        ("target_median", "median", "tab:blue"),
        ("target_p90", "p90", "tab:orange"),
        ("target_max", "max", "tab:red"),
    ):
        add_raw_and_smooth(
            ax, epochs, series[field], label, smooth_window, color
        )
    target_stat_values = [
        value
        for field in ("target_median", "target_p90", "target_max")
        for value in series[field]
    ]
    if (
        log_target_stats
        and all(value is not None for value in target_stat_values)
        and all(value > 0 for value in target_stat_values)
    ):
        ax.set_yscale("log")
    ax.set_title("Per-edit target loss distribution")
    ax.set_ylabel("Target loss" + (" (log scale)" if log_target_stats else ""))

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    for primary_ax, secondary_ax in (
        (axes[0, 1], decay_ax),
        (axes[1, 0], change_ax),
    ):
        if secondary_ax is None:
            continue
        primary_handles, primary_labels = primary_ax.get_legend_handles_labels()
        secondary_handles, secondary_labels = secondary_ax.get_legend_handles_labels()
        primary_ax.legend(
            primary_handles + secondary_handles,
            primary_labels + secondary_labels,
            fontsize=8,
        )

    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)


def relative_change(old, new):
    denominator = max(abs(old), 1e-12)
    return (new - old) / denominator


def optional_mean(records, metric):
    values = [
        record[metric]
        for record in records
        if record.get(metric) is not None and math.isfinite(record[metric])
    ]
    return mean(values) if values else None


def build_summary(records, requested_window, duplicate_count):
    if len(records) < 2:
        return (
            f"Parsed {len(records)} epoch; at least 2 are required for a "
            "convergence comparison.\n"
        )

    window = min(requested_window, len(records) // 2)
    window = max(window, 1)
    previous = records[-2 * window : -window]
    recent = records[-window:]
    metrics = [
        "total_loss",
        "target_loss",
        "weighted_preserve_loss",
        "gradient_norm",
        "update_norm",
        "target_median",
        "target_p90",
        "target_max",
    ]
    if any(record["weighted_l2_loss"] is not None for record in records):
        metrics.insert(3, "weighted_l2_loss")
    if any(
        record["weighted_layer_norm_loss"] is not None
        for record in records
    ):
        metrics.insert(3, "weighted_layer_norm_loss")
    if any(record["adamw_decay_norm_per_step"] is not None for record in records):
        metrics.insert(3, "adamw_decay_norm_per_step")
    if any(record["relative_update_norm_change"] is not None for record in records):
        metrics.insert(metrics.index("update_norm") + 1, "relative_update_norm_change")
    if any(record["layer_relative_norm_cv"] is not None for record in records):
        metrics.insert(metrics.index("update_norm") + 1, "layer_relative_norm_cv")

    lines = [
        f"Parsed epochs: {records[0]['epoch']}..{records[-1]['epoch']} "
        f"({len(records)} unique records)",
        f"Repeated epoch entries replaced by latest entry: {duplicate_count}",
        f"Convergence comparison: previous {window} epochs vs latest {window} epochs",
        "",
        f"{'metric':<28} {'previous mean':>15} {'recent mean':>15} {'relative change':>17}",
        "-" * 79,
    ]
    for metric in metrics:
        previous_mean = optional_mean(previous, metric)
        recent_mean = optional_mean(recent, metric)
        if previous_mean is None or recent_mean is None:
            continue
        change = relative_change(previous_mean, recent_mean)
        lines.append(
            f"{metric:<28} {previous_mean:>15.6e} {recent_mean:>15.6e} "
            f"{change:>16.3%}"
        )

    clipped_count = sum(record["gradient_clipped"] for record in recent)
    weight_decays = sorted(
        {
            record["weight_decay"]
            for record in records
            if record["weight_decay"] is not None
        }
    )
    decay_factors = sorted(
        {
            record["decay_factor"]
            for record in records
            if record["decay_factor"] is not None
        }
    )
    lines.extend(
        [
            "",
            f"Gradient clipping events in latest window: {clipped_count}/{window}",
        ]
    )
    if weight_decays:
        lines.append(
            "AdamW weight decay: "
            + ", ".join(f"{value:g}" for value in weight_decays)
            + "; per-step factor: "
            + ", ".join(f"{value:.8f}" for value in decay_factors)
        )
    lines.extend(
        [
            "A flat total/target loss, stable update norm, and near-zero relative",
            "update-norm change together are stronger evidence than target loss alone.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    if args.smooth_window <= 0 or args.summary_window <= 0:
        raise SystemExit("--smooth-window and --summary-window must be positive.")
    if not args.log_file.is_file():
        raise SystemExit(f"Log file does not exist: {args.log_file}")

    all_records, duplicate_count = parse_log(args.log_file)
    if not all_records:
        raw_text = args.log_file.read_text(
            encoding="utf-8", errors="replace"
        )
        clean_text = ANSI_ESCAPE.sub("", raw_text)
        complete_count = len(
            re.findall(
                r"joint epoch\s+\d+\s+complete:",
                clean_text,
                flags=re.IGNORECASE,
            )
        )
        micro_batch_count = len(
            re.findall(
                r"joint epoch\s+\d+/\d+\s*,\s*micro-batch",
                clean_text,
                flags=re.IGNORECASE,
            )
        )
        raise SystemExit(
            "No parseable joint-epoch records found in "
            f"{args.log_file}. Found {complete_count} 'epoch complete' and "
            f"{micro_batch_count} micro-batch markers. Check that this is "
            "the training .out log (not the .err or evaluation log), and "
            "that at least one epoch finished."
        )

    records = filter_records(all_records, args.start_epoch, args.end_epoch)
    if not records:
        raise SystemExit("No records remain after applying the epoch filters.")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.log_file.parent / f"{args.log_file.stem}_joint_curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = output_dir / "joint_training_metrics.csv"
    plot_file = output_dir / "joint_training_curves.png"
    summary_file = output_dir / "convergence_summary.txt"

    write_csv(records, csv_file)
    plot_records(
        records,
        plot_file,
        smooth_window=args.smooth_window,
        dpi=args.dpi,
        log_target_stats=args.log_target_stats,
    )
    summary = build_summary(records, args.summary_window, duplicate_count)
    summary_file.write_text(summary, encoding="utf-8")

    print(summary, end="")
    print(f"CSV:     {csv_file}")
    print(f"Plot:    {plot_file}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()

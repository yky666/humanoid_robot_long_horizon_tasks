#!/usr/bin/env python3
"""Inspect an A1 training run directory, summarize progress, and optionally plot curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TIMESTAMP_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
STEP_RE = re.compile(r"\[step=(\d+)/(\d+)\]")
METRIC_RE = re.compile(
    r"^\s+([^=]+)=([+-]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)
CHECKPOINT_EVENT_RE = re.compile(
    r"(Checkpoint saved to|Unsharded checkpoint saved to|Action head checkpoint saved to)\s+(.+)$"
)
WANDB_RETRY_RE = re.compile(r"retrying error")


@dataclass
class StepRecord:
    step: int
    max_steps: int
    timestamp: datetime | None
    metrics: dict[str, float]


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_RE.search(line)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")


def find_latest_output_log(run_dir: Path) -> Path:
    candidates = sorted(
        run_dir.glob("wandb/wandb/run-*/files/output.log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No output.log found under {run_dir}")
    return candidates[0]


def find_latest_debug_log(run_dir: Path) -> Path | None:
    candidates = sorted(
        run_dir.glob("wandb/wandb/run-*/logs/debug-internal.log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def parse_output_log(path: Path) -> dict[str, Any]:
    step_records: list[StepRecord] = []
    checkpoint_events: list[dict[str, str]] = []
    fit_start_time: datetime | None = None
    latest_timestamp: datetime | None = None
    last_nonempty_line = ""
    current_step: StepRecord | None = None

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = strip_ansi(raw_line.rstrip())
        if not line:
            continue
        last_nonempty_line = line
        ts = parse_timestamp(line)
        if ts is not None:
            latest_timestamp = ts
            if current_step is not None and current_step.timestamp is None:
                current_step.timestamp = ts

        if "fit start" in line and fit_start_time is None:
            fit_start_time = ts

        step_match = STEP_RE.search(line)
        if step_match:
            current_step = StepRecord(
                step=int(step_match.group(1)),
                max_steps=int(step_match.group(2)),
                timestamp=None,
                metrics={},
            )
            step_records.append(current_step)
            continue

        metric_match = METRIC_RE.match(line)
        if metric_match and current_step is not None:
            current_step.metrics[metric_match.group(1)] = float(metric_match.group(2).replace(",", ""))
            continue

        ckpt_match = CHECKPOINT_EVENT_RE.search(line)
        if ckpt_match:
            checkpoint_events.append(
                {
                    "timestamp": latest_timestamp.isoformat(sep=" ") if latest_timestamp else "",
                    "event": ckpt_match.group(1),
                    "path": ckpt_match.group(2).strip(),
                }
            )

    return {
        "fit_start_time": fit_start_time,
        "latest_timestamp": latest_timestamp,
        "last_nonempty_line": last_nonempty_line,
        "step_records": step_records,
        "checkpoint_events": checkpoint_events,
    }


def count_wandb_retries(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    count = 0
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if WANDB_RETRY_RE.search(strip_ansi(raw_line)):
            count += 1
    return count


def get_checkpoint_sizes(run_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(run_dir.iterdir()):
        if not path.name.startswith("step"):
            continue
        if path.name.startswith("step") and path.is_dir():
            size_bytes = sum(
                child.stat().st_size for child in path.rglob("*") if child.is_file()
            )
            entries.append(
                {
                    "name": path.name,
                    "size_gb": round(size_bytes / (1024 ** 3), 3),
                }
            )
    return entries


def estimate_seconds_per_step(
    fit_start_time: datetime | None,
    step_records: list[StepRecord],
) -> float | None:
    if len(step_records) >= 2:
        first = step_records[0]
        last = step_records[-1]
        if first.timestamp is not None and last.timestamp is not None and last.step > first.step:
            return (last.timestamp - first.timestamp).total_seconds() / (last.step - first.step)
    if len(step_records) == 1 and fit_start_time is not None and step_records[0].timestamp is not None:
        first = step_records[0]
        if first.step > 0:
            return (first.timestamp - fit_start_time).total_seconds() / first.step
    return None


def build_summary(run_dir: Path, output_log: Path, debug_log: Path | None) -> dict[str, Any]:
    parsed = parse_output_log(output_log)
    step_records: list[StepRecord] = parsed["step_records"]
    latest_step = step_records[-1] if step_records else None
    seconds_per_step = estimate_seconds_per_step(parsed["fit_start_time"], step_records)
    eta_hours = None
    if latest_step is not None and seconds_per_step is not None and latest_step.max_steps > latest_step.step:
        eta_hours = (latest_step.max_steps - latest_step.step) * seconds_per_step / 3600.0

    summary = {
        "run_dir": str(run_dir),
        "output_log": str(output_log),
        "debug_internal_log": str(debug_log) if debug_log is not None else None,
        "fit_start_time": (
            parsed["fit_start_time"].isoformat(sep=" ") if parsed["fit_start_time"] else None
        ),
        "latest_log_time": (
            parsed["latest_timestamp"].isoformat(sep=" ") if parsed["latest_timestamp"] else None
        ),
        "last_log_line": parsed["last_nonempty_line"],
        "latest_step": latest_step.step if latest_step is not None else None,
        "max_steps": latest_step.max_steps if latest_step is not None else None,
        "progress_percent": (
            round(100.0 * latest_step.step / latest_step.max_steps, 3)
            if latest_step is not None and latest_step.max_steps > 0
            else None
        ),
        "latest_metrics": latest_step.metrics if latest_step is not None else {},
        "observed_seconds_per_step": round(seconds_per_step, 3) if seconds_per_step else None,
        "eta_hours_from_latest_step": round(eta_hours, 3) if eta_hours else None,
        "checkpoint_events": parsed["checkpoint_events"][-6:],
        "checkpoint_sizes_gb": get_checkpoint_sizes(run_dir),
        "wandb_retry_errors": count_wandb_retries(debug_log),
    }
    return summary


def write_csv(step_records: list[StepRecord], csv_path: Path) -> None:
    metric_names = sorted({name for record in step_records for name in record.metrics})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "max_steps", "timestamp", *metric_names])
        for record in step_records:
            writer.writerow(
                [
                    record.step,
                    record.max_steps,
                    record.timestamp.isoformat(sep=" ") if record.timestamp else "",
                    *[record.metrics.get(name, "") for name in metric_names],
                ]
            )


def _write_svg_plot(step_records: list[StepRecord], plot_path: Path, metrics: list[str]) -> list[str]:
    available = [metric for metric in metrics if any(metric in record.metrics for record in step_records)]
    if not available:
        raise RuntimeError("No requested metrics were found in the parsed training log")

    width = 1100
    plot_height = 220
    left = 90
    right = 40
    top = 40
    inner_top = 30
    inner_bottom = 45
    total_height = top + len(available) * plot_height

    steps = [record.step for record in step_records]
    min_step = min(steps)
    max_step = max(steps)
    if max_step == min_step:
        max_step = min_step + 1

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_height}" viewBox="0 0 {width} {total_height}">',
        '<style>text { font-family: monospace; font-size: 12px; } .title { font-size: 16px; font-weight: bold; } .axis { stroke: #333; stroke-width: 1; } .grid { stroke: #ddd; stroke-width: 1; } .line { fill: none; stroke: #1565c0; stroke-width: 2; } .point { fill: #1565c0; }</style>',
        f'<text class="title" x="{left}" y="24">{escape(plot_path.stem)}</text>',
    ]

    for index, metric in enumerate(available):
        values = [record.metrics.get(metric, math.nan) for record in step_records]
        valid_values = [value for value in values if not math.isnan(value)]
        if not valid_values:
            continue
        min_value = min(valid_values)
        max_value = max(valid_values)
        if math.isclose(min_value, max_value):
            delta = 1.0 if math.isclose(min_value, 0.0) else abs(min_value) * 0.1
            min_value -= delta
            max_value += delta

        chart_top = top + index * plot_height
        chart_bottom = chart_top + plot_height - inner_bottom
        chart_left = left
        chart_right = width - right
        chart_height = chart_bottom - (chart_top + inner_top)
        chart_width = chart_right - chart_left

        svg_parts.append(f'<text x="{chart_left}" y="{chart_top + 16}">{escape(metric)}</text>')

        for fraction in (0.0, 0.5, 1.0):
            y = chart_bottom - fraction * chart_height
            label = min_value + fraction * (max_value - min_value)
            svg_parts.append(
                f'<line class="grid" x1="{chart_left}" y1="{y:.2f}" x2="{chart_right}" y2="{y:.2f}" />'
            )
            svg_parts.append(
                f'<text x="10" y="{y + 4:.2f}">{label:.4f}</text>'
            )

        svg_parts.append(
            f'<rect x="{chart_left}" y="{chart_top + inner_top}" width="{chart_width}" height="{chart_height}" fill="none" class="axis" />'
        )
        svg_parts.append(
            f'<text x="{chart_left}" y="{chart_bottom + 24}">step {min_step}</text>'
        )
        svg_parts.append(
            f'<text x="{chart_right - 60}" y="{chart_bottom + 24}">step {max_step}</text>'
        )

        points: list[str] = []
        circles: list[str] = []
        for step, value in zip(steps, values):
            if math.isnan(value):
                continue
            x = chart_left + (step - min_step) / (max_step - min_step) * chart_width
            y = chart_bottom - (value - min_value) / (max_value - min_value) * chart_height
            points.append(f"{x:.2f},{y:.2f}")
            circles.append(f'<circle class="point" cx="{x:.2f}" cy="{y:.2f}" r="3" />')
        svg_parts.append(f'<polyline class="line" points="{" ".join(points)}" />')
        svg_parts.extend(circles)

    svg_parts.append("</svg>")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.write_text("\n".join(svg_parts), encoding="utf-8")
    return available


def plot_metrics(step_records: list[StepRecord], plot_path: Path, metrics: list[str]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        if plot_path.suffix.lower() != ".svg":
            raise RuntimeError(
                "matplotlib is not installed in the active environment. Use a .svg path for --plot-out to enable the built-in zero-dependency plot fallback."
            ) from exc
        return _write_svg_plot(step_records, plot_path, metrics)

    available = [metric for metric in metrics if any(metric in record.metrics for record in step_records)]
    if not available:
        raise RuntimeError("No requested metrics were found in the parsed training log")

    steps = [record.step for record in step_records]
    fig, axes = plt.subplots(len(available), 1, figsize=(10, 3 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]

    for axis, metric in zip(axes, available):
        values = [record.metrics.get(metric, math.nan) for record in step_records]
        axis.plot(steps, values, marker="o", linewidth=1.5)
        axis.set_ylabel(metric)
        axis.grid(alpha=0.3)

    axes[-1].set_xlabel("Step")
    fig.suptitle(plot_path.stem)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Path to the training output directory")
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Optional CSV export path for parsed step metrics",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional JSON export path for the summary",
    )
    parser.add_argument(
        "--plot-out",
        default=None,
        help="Optional PNG path for plotted metrics",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=[
            "train/ActionNoiseL2Loss",
            "optim/total_grad_norm",
            "throughput/device/tokens_per_second",
            "System/Peak GPU Memory (MB)",
        ],
        help="Metric names to plot when --plot-out is provided",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_log = find_latest_output_log(run_dir)
    debug_log = find_latest_debug_log(run_dir)
    parsed = parse_output_log(output_log)
    summary = build_summary(run_dir, output_log, debug_log)

    if args.csv_out:
        write_csv(parsed["step_records"], Path(args.csv_out).expanduser().resolve())

    if args.plot_out:
        used_metrics = plot_metrics(
            parsed["step_records"],
            Path(args.plot_out).expanduser().resolve(),
            args.metrics,
        )
        summary["plotted_metrics"] = used_metrics

    if args.json_out:
        json_path = Path(args.json_out).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

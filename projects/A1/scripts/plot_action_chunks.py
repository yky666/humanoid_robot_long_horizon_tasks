#!/usr/bin/env python3
"""Plot predicted vs ground-truth action chunks from saved A1 eval JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ACTION_GROUPS = {
    "left_arm": list(range(0, 7)),
    "right_arm": list(range(7, 14)),
    "left_hand": list(range(14, 20)),
    "right_hand": list(range(20, 26)),
}

ACTION_NAMES = [
    "L_shoulder_pitch",
    "L_shoulder_roll",
    "L_shoulder_yaw",
    "L_elbow",
    "L_wrist_roll",
    "L_wrist_pitch",
    "L_wrist_yaw",
    "R_shoulder_pitch",
    "R_shoulder_roll",
    "R_shoulder_yaw",
    "R_elbow",
    "R_wrist_roll",
    "R_wrist_pitch",
    "R_wrist_yaw",
    "L_pinky",
    "L_ring",
    "L_middle",
    "L_index",
    "L_thumb_bend",
    "L_thumb_rotation",
    "R_pinky",
    "R_ring",
    "R_middle",
    "R_index",
    "R_thumb_bend",
    "R_thumb_rotation",
]


def load_samples(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        for key in ("samples", "results", "eval_samples"):
            if key in data and isinstance(data[key], list):
                return data[key]
    if isinstance(data, list):
        return data
    raise ValueError(f"Cannot find eval samples in {path}")


def sample_arrays(sample: dict) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(sample["pred_action"], dtype=np.float64)
    gt = np.asarray(sample["gt_action"], dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")
    if pred.ndim != 2 or pred.shape[1] != 26:
        raise ValueError(f"expected action shape (T, 26), got {pred.shape}")
    return pred, gt


def sample_l1(sample: dict) -> float:
    pred, gt = sample_arrays(sample)
    return float(np.mean(np.abs(pred - gt)))


def select_indices(samples: list[dict]) -> list[tuple[str, int]]:
    losses = np.asarray([sample_l1(sample) for sample in samples])
    selected = [
        ("first", 0),
        ("median_l1", int(np.argsort(losses)[len(losses) // 2])),
        ("worst_l1", int(np.argmax(losses))),
    ]
    deduped: list[tuple[str, int]] = []
    used: set[int] = set()
    for label, idx in selected:
        if idx not in used:
            deduped.append((label, idx))
            used.add(idx)
    return deduped


def summarize_sample(sample: dict) -> dict:
    pred, gt = sample_arrays(sample)
    out = {
        "shape": list(pred.shape),
        "l1": float(np.mean(np.abs(pred - gt))),
        "mse": float(np.mean((pred - gt) ** 2)),
        "groups": {},
    }
    for group, dims in ACTION_GROUPS.items():
        pred_g = pred[:, dims]
        gt_g = gt[:, dims]
        pred_std = np.std(pred_g, axis=0)
        gt_std = np.std(gt_g, axis=0)
        pred_delta = np.abs(np.diff(pred_g, axis=0))
        gt_delta = np.abs(np.diff(gt_g, axis=0))
        out["groups"][group] = {
            "pred_temporal_std_mean": float(np.mean(pred_std)),
            "gt_temporal_std_mean": float(np.mean(gt_std)),
            "std_ratio_pred_over_gt": float(np.mean(pred_std) / (np.mean(gt_std) + 1e-12)),
            "pred_step_delta_abs_mean": float(np.mean(pred_delta)),
            "gt_step_delta_abs_mean": float(np.mean(gt_delta)),
            "step_delta_ratio_pred_over_gt": float(
                np.mean(pred_delta) / (np.mean(gt_delta) + 1e-12)
            ),
        }
    return out


def plot_sample(pred: np.ndarray, gt: np.ndarray, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.ravel()
    x = np.arange(pred.shape[0])
    for ax, (group, dims) in zip(axes, ACTION_GROUPS.items()):
        for dim in dims:
            color = f"C{dim % 10}"
            ax.plot(x, gt[:, dim], color=color, linestyle="--", linewidth=1.1, alpha=0.75)
            ax.plot(x, pred[:, dim], color=color, linestyle="-", linewidth=1.1, alpha=0.9)
        labels = [ACTION_NAMES[d] for d in dims]
        ax.set_title(group)
        ax.set_ylabel("action")
        ax.grid(True, alpha=0.25)
        ax.text(
            0.01,
            0.99,
            "\n".join(labels),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            alpha=0.75,
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
        )
    axes[-2].set_xlabel("chunk step")
    axes[-1].set_xlabel("chunk step")
    fig.suptitle(f"{title}\nsolid=pred, dashed=gt", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def iter_eval_files(paths: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.glob("*_step1000_norm_eval_50_actions.json")))
        else:
            files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Saved eval action JSON files or directories containing *_step1000_norm_eval_50_actions.json.",
    )
    parser.add_argument("--out_dir", default="projects/A1/outputs/action_chunk_plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"files": {}}

    for path in iter_eval_files(args.inputs):
        samples = load_samples(path)
        file_summary = {
            "num_samples": len(samples),
            "all_sample_group_means": {},
            "all_sample_group_ratio_of_means": {},
            "selected_samples": {},
        }
        per_sample = [summarize_sample(sample) for sample in samples]
        for group in ACTION_GROUPS:
            file_summary["all_sample_group_means"][group] = {
                key: float(np.mean([item["groups"][group][key] for item in per_sample]))
                for key in per_sample[0]["groups"][group]
            }
            pred_std = file_summary["all_sample_group_means"][group]["pred_temporal_std_mean"]
            gt_std = file_summary["all_sample_group_means"][group]["gt_temporal_std_mean"]
            pred_delta = file_summary["all_sample_group_means"][group]["pred_step_delta_abs_mean"]
            gt_delta = file_summary["all_sample_group_means"][group]["gt_step_delta_abs_mean"]
            file_summary["all_sample_group_ratio_of_means"][group] = {
                "std_ratio_pred_over_gt": float(pred_std / (gt_std + 1e-12)),
                "step_delta_ratio_pred_over_gt": float(pred_delta / (gt_delta + 1e-12)),
            }
        for label, idx in select_indices(samples):
            pred, gt = sample_arrays(samples[idx])
            stem = path.stem.replace("_norm_eval_50_actions", "")
            plot_path = out_dir / f"{stem}_{label}_sample{idx:04d}.png"
            plot_sample(pred, gt, f"{stem} {label} sample={idx}", plot_path)
            file_summary["selected_samples"][label] = {
                "sample_index": idx,
                "plot": str(plot_path),
                **summarize_sample(samples[idx]),
            }
        summary["files"][path.name] = file_summary

    summary_path = out_dir / "action_chunk_collapse_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(summary_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_PHASE_NAMES = ["approach", "handover", "insert"]
DEFAULT_PHASE_BOUNDS = [0.0, 0.33, 0.66, 1.0]
DEFAULT_ACTION_NAMES = [
    "left_arm.shoulder_pitch",
    "left_arm.shoulder_roll",
    "left_arm.shoulder_yaw",
    "left_arm.elbow",
    "left_arm.wrist_roll",
    "left_arm.wrist_pitch",
    "left_arm.wrist_yaw",
    "right_arm.shoulder_pitch",
    "right_arm.shoulder_roll",
    "right_arm.shoulder_yaw",
    "right_arm.elbow",
    "right_arm.wrist_roll",
    "right_arm.wrist_pitch",
    "right_arm.wrist_yaw",
    "left_ee.pinky",
    "left_ee.ring",
    "left_ee.middle",
    "left_ee.index",
    "left_ee.thumb_bend",
    "left_ee.thumb_rotation",
    "right_ee.pinky",
    "right_ee.ring",
    "right_ee.middle",
    "right_ee.index",
    "right_ee.thumb_bend",
    "right_ee.thumb_rotation",
]
DEFAULT_LEFT_DIMS = [0, 1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18, 19]
DEFAULT_RIGHT_DIMS = [7, 8, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze action eval json with per-dim and phase stats.")
    parser.add_argument("--input_json", required=True, help="Path to eval json from robot_experiments.debug")
    parser.add_argument("--output_json", required=True, help="Where to save analysis json")
    parser.add_argument("--left-dims", nargs="+", type=int, default=DEFAULT_LEFT_DIMS)
    parser.add_argument("--right-dims", nargs="+", type=int, default=DEFAULT_RIGHT_DIMS)
    parser.add_argument("--phase-names", nargs="+", default=DEFAULT_PHASE_NAMES)
    parser.add_argument("--phase-bounds", nargs="+", type=float, default=DEFAULT_PHASE_BOUNDS)
    return parser.parse_args()


def _slice_stats(array: np.ndarray, dims: list[int], action_names: list[str]) -> dict:
    view = array[:, :, dims]
    step_delta = np.diff(view, axis=1) if view.shape[1] > 1 else np.zeros_like(view[:, :0, :])
    return {
        "dims": dims,
        "dim_names": [action_names[i] for i in dims],
        "avg_l1": float(np.mean(np.abs(view))),
        "avg_mse": float(np.mean(np.square(view))),
        "per_dim_l1": np.mean(np.abs(view), axis=(0, 1)).astype(np.float32).tolist(),
        "per_dim_mse": np.mean(np.square(view), axis=(0, 1)).astype(np.float32).tolist(),
        "avg_signed_error": float(np.mean(view)),
        "per_dim_signed_error": np.mean(view, axis=(0, 1)).astype(np.float32).tolist(),
        "avg_step_delta_l2": float(np.mean(np.linalg.norm(step_delta, axis=2))) if step_delta.size else 0.0,
        "avg_step_delta_abs": float(np.mean(np.abs(step_delta))) if step_delta.size else 0.0,
    }


def _phase_for_ratio(ratio: float, phase_names: list[str], phase_bounds: list[float]) -> str:
    for i, name in enumerate(phase_names):
        low = phase_bounds[i]
        high = phase_bounds[i + 1]
        if i == len(phase_names) - 1:
            if low <= ratio <= high:
                return name
        elif low <= ratio < high:
            return name
    return phase_names[-1]


def main():
    args = parse_args()
    if len(args.phase_bounds) != len(args.phase_names) + 1:
        raise ValueError("phase_bounds length must equal len(phase_names) + 1")

    with open(args.input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    samples = payload["samples"]
    if not samples:
        raise ValueError("No samples in input_json")
    if "pred_action" not in samples[0] or "gt_action" not in samples[0]:
        raise ValueError("Input json does not contain pred_action/gt_action. Re-run eval with save_actions=true.")

    pred = np.asarray([s["pred_action"] for s in samples], dtype=np.float32)
    gt = np.asarray([s["gt_action"] for s in samples], dtype=np.float32)
    err = pred - gt
    action_names = payload.get("action_names", DEFAULT_ACTION_NAMES)
    if len(action_names) != pred.shape[2]:
        action_names = [f"dim_{i}" for i in range(pred.shape[2])]

    frame_indices = np.asarray([s["metadata"]["frame_index"] for s in samples], dtype=np.int32)
    dataset_indices = np.asarray([s["dataset_index"] for s in samples], dtype=np.int32)
    phase_groups = {name: [] for name in args.phase_names}
    denom = max(int(np.max(frame_indices)), 1)
    for sample_idx, frame_idx in enumerate(frame_indices):
        ratio = float(frame_idx) / float(denom)
        phase_groups[_phase_for_ratio(ratio, args.phase_names, args.phase_bounds)].append(sample_idx)

    analysis = {
        "input_json": str(Path(args.input_json).resolve()),
        "num_samples": int(pred.shape[0]),
        "sequence_length": int(pred.shape[1]),
        "action_dim": int(pred.shape[2]),
        "phase_config": {
            "phase_names": args.phase_names,
            "phase_bounds": args.phase_bounds,
            "phase_assignment": "frame_index / max(frame_index) heuristic",
        },
        "overall": {
            "avg_l1": float(np.mean(np.abs(err))),
            "avg_mse": float(np.mean(np.square(err))),
            "action_names": action_names,
            "per_dim_l1": np.mean(np.abs(err), axis=(0, 1)).astype(np.float32).tolist(),
            "per_dim_mse": np.mean(np.square(err), axis=(0, 1)).astype(np.float32).tolist(),
            "per_dim_signed_error": np.mean(err, axis=(0, 1)).astype(np.float32).tolist(),
        },
        "left_side": _slice_stats(err, args.left_dims, action_names),
        "right_side": _slice_stats(err, args.right_dims, action_names),
        "phases": {},
        "sample_index_summary": {
            "frame_index_min": int(np.min(frame_indices)),
            "frame_index_max": int(np.max(frame_indices)),
            "dataset_index_min": int(np.min(dataset_indices)),
            "dataset_index_max": int(np.max(dataset_indices)),
        },
    }

    for phase_name, indices in phase_groups.items():
        if not indices:
            analysis["phases"][phase_name] = {"num_samples": 0}
            continue
        phase_err = err[indices]
        analysis["phases"][phase_name] = {
            "num_samples": len(indices),
            "sample_indices": indices,
            "avg_l1": float(np.mean(np.abs(phase_err))),
            "avg_mse": float(np.mean(np.square(phase_err))),
            "per_dim_l1": np.mean(np.abs(phase_err), axis=(0, 1)).astype(np.float32).tolist(),
            "per_dim_mse": np.mean(np.square(phase_err), axis=(0, 1)).astype(np.float32).tolist(),
            "per_dim_signed_error": np.mean(phase_err, axis=(0, 1)).astype(np.float32).tolist(),
            "left_side": _slice_stats(phase_err, args.left_dims, action_names),
            "right_side": _slice_stats(phase_err, args.right_dims, action_names),
        }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()

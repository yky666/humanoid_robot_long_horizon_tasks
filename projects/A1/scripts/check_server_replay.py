#!/usr/bin/env python3

import argparse
import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from a1.data.vla.lerobot_datasets import LeRobotDatasetWrapper


LEFT_ARM = list(range(0, 7))
RIGHT_ARM = list(range(7, 14))
LEFT_HAND = list(range(14, 20))
RIGHT_HAND = list(range(20, 26))
SWAP_PERM = RIGHT_ARM + LEFT_ARM + RIGHT_HAND + LEFT_HAND
ACTION_NAMES = [
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


def parse_args():
    parser = argparse.ArgumentParser(description="Replay G1 logs against A1 inference server.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dataset-paths", nargs="+", required=True)
    parser.add_argument("--samples-per-dataset", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-action-dim", type=int, default=26)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def encode_image_to_base64(image_data: np.ndarray) -> str:
    image_bgr = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("Failed to encode image")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def l1(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))


def mse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.square(pred - gt)))


def side_l1(pred: np.ndarray, gt: np.ndarray, dims: list[int]) -> float:
    return float(np.mean(np.abs(pred[:, dims] - gt[:, dims])))


def side_step_l2(pred: np.ndarray, dims: list[int]) -> float:
    view = pred[:, dims]
    if view.shape[0] < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(view, axis=0), axis=1)))


def load_stats(dataset_path: str) -> dict:
    stats_path = Path(dataset_path) / "meta" / "stats.json"
    return json.loads(stats_path.read_text())


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json", "Accept": "application/json"})

    all_samples = []
    shape_failures = []
    out_of_range_counts = np.zeros(args.fixed_action_dim, dtype=np.int64)
    total_pred_steps = 0
    latency_ms = []
    server_processing_ms = []
    direct_l1_values = []
    swapped_l1_values = []

    for dataset_path in args.dataset_paths:
        dataset = LeRobotDatasetWrapper(
            dataset_path=dataset_path,
            fixed_action_dim=args.fixed_action_dim,
            chunk_size=args.chunk_size,
            use_proprio=True,
            use_wrist_image=True,
        )
        stats = load_stats(dataset_path)
        action_min = np.asarray(stats["actions"]["min"], dtype=np.float32)
        action_max = np.asarray(stats["actions"]["max"], dtype=np.float32)
        n = min(len(dataset), args.samples_per_dataset)
        indices = rng.choice(len(dataset), size=n, replace=n > len(dataset))

        for sample_index, idx in enumerate(indices.tolist()):
            item = dataset.get(int(idx), rng)
            payload = {
                "instruction": item["question"],
                "images": [encode_image_to_base64(image) for image in item["images"]],
                "proprio_data": np.asarray(item["proprio"], dtype=np.float32).tolist(),
                "norm_stats_json_path": str(Path(dataset_path) / "meta" / "stats.json"),
            }
            start = time.perf_counter()
            response = session.post(f"{args.base_url.rstrip('/')}/inference", json=payload, timeout=120)
            wall_ms = (time.perf_counter() - start) * 1000.0
            response.raise_for_status()
            result = response.json()
            pred = np.asarray(result["predicted_actions"], dtype=np.float32)
            gt = np.asarray(item["action"], dtype=np.float32)

            latency_ms.append(wall_ms)
            server_processing_ms.append(float(result.get("processing_time_ms", 0.0)))

            if pred.shape != gt.shape:
                shape_failures.append(
                    {
                        "dataset_path": dataset_path,
                        "sample_index": sample_index,
                        "dataset_index": int(idx),
                        "pred_shape": list(pred.shape),
                        "gt_shape": list(gt.shape),
                    }
                )
                continue

            total_pred_steps += pred.shape[0]
            low_violation = pred < action_min[None, :]
            high_violation = pred > action_max[None, :]
            out_of_range_counts += np.sum(low_violation | high_violation, axis=0)

            direct = l1(pred, gt)
            swapped_pred = pred[:, SWAP_PERM]
            swapped = l1(swapped_pred, gt)
            direct_l1_values.append(direct)
            swapped_l1_values.append(swapped)

            all_samples.append(
                {
                    "dataset_path": dataset_path,
                    "sample_index": sample_index,
                    "dataset_index": int(idx),
                    "frame_index": int(item["metadata"]["frame_index"]),
                    "wall_time_ms": wall_ms,
                    "server_processing_time_ms": float(result.get("processing_time_ms", 0.0)),
                    "shape": list(pred.shape),
                    "overall_l1": direct,
                    "overall_mse": mse(pred, gt),
                    "left_arm_l1": side_l1(pred, gt, LEFT_ARM),
                    "right_arm_l1": side_l1(pred, gt, RIGHT_ARM),
                    "left_hand_l1": side_l1(pred, gt, LEFT_HAND),
                    "right_hand_l1": side_l1(pred, gt, RIGHT_HAND),
                    "direct_l1": direct,
                    "swapped_l1": swapped,
                    "swap_ratio": (swapped / direct) if direct > 0 else None,
                    "left_arm_step_delta_l2": side_step_l2(pred, LEFT_ARM),
                    "right_arm_step_delta_l2": side_step_l2(pred, RIGHT_ARM),
                    "left_hand_step_delta_l2": side_step_l2(pred, LEFT_HAND),
                    "right_hand_step_delta_l2": side_step_l2(pred, RIGHT_HAND),
                    "pred_min": float(np.min(pred)),
                    "pred_max": float(np.max(pred)),
                }
            )

    total_samples = len(all_samples)
    effective_hz = total_pred_steps / (sum(latency_ms) / 1000.0) if latency_ms else 0.0
    out_of_range_ratio = (out_of_range_counts / max(total_pred_steps, 1)).tolist()
    worst_dims = []
    for dim, ratio in enumerate(out_of_range_ratio):
        worst_dims.append(
            {
                "dim": dim,
                "name": ACTION_NAMES[dim] if dim < len(ACTION_NAMES) else f"dim_{dim}",
                "out_of_range_ratio": ratio,
            }
        )
    worst_dims.sort(key=lambda x: x["out_of_range_ratio"], reverse=True)

    summary = {
        "base_url": args.base_url,
        "dataset_paths": args.dataset_paths,
        "samples_per_dataset": args.samples_per_dataset,
        "num_successful_samples": total_samples,
        "shape_failures": shape_failures,
        "shape_check": {
            "expected_shape": [args.chunk_size, args.fixed_action_dim],
            "all_ok": len(shape_failures) == 0,
        },
        "range_check": {
            "total_pred_steps": total_pred_steps,
            "overall_out_of_range_ratio": float(np.sum(out_of_range_counts) / max(total_pred_steps * args.fixed_action_dim, 1)),
            "per_dim_out_of_range_ratio": out_of_range_ratio,
            "worst_dims": worst_dims[:8],
        },
        "channel_semantics_check": {
            "avg_direct_l1": float(np.mean(direct_l1_values)) if direct_l1_values else None,
            "avg_swapped_l1": float(np.mean(swapped_l1_values)) if swapped_l1_values else None,
            "swap_suspicious_samples": sum(1 for d, s in zip(direct_l1_values, swapped_l1_values) if s < d),
            "swap_suspicious_ratio": float(np.mean([s < d for d, s in zip(direct_l1_values, swapped_l1_values)])) if direct_l1_values else None,
        },
        "latency_check": {
            "avg_wall_time_ms": float(np.mean(latency_ms)) if latency_ms else None,
            "p95_wall_time_ms": float(np.percentile(latency_ms, 95)) if latency_ms else None,
            "avg_server_processing_ms": float(np.mean(server_processing_ms)) if server_processing_ms else None,
            "p95_server_processing_ms": float(np.percentile(server_processing_ms, 95)) if server_processing_ms else None,
            "effective_hz": effective_hz,
        },
        "samples": all_samples,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output_path)
    print(json.dumps({k: summary[k] for k in ["shape_check", "range_check", "channel_semantics_check", "latency_check"]}, indent=2))


if __name__ == "__main__":
    main()

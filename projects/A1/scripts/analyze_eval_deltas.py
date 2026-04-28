#!/usr/bin/env python3
"""Compare two offline eval JSON files and summarize phase-level behavior."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PHASE_NAMES = ("approach", "grasp_lift", "carry_place_right")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def l2_norm(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(float(value) * float(value) for value in values))


def diff_l2_norm(values_a: list[float], values_b: list[float]) -> float:
    if not values_a or not values_b or len(values_a) != len(values_b):
        return 0.0
    return math.sqrt(
        sum((float(a) - float(b)) * (float(a) - float(b)) for a, b in zip(values_a, values_b))
    )


def round_dict(values: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 6) for key, value in values.items()}


def infer_dataset_length(samples: list[dict[str, Any]], explicit_length: int | None) -> int:
    if explicit_length is not None:
        return explicit_length
    return max(int(sample["dataset_index"]) for sample in samples) + 1


def phase_for_index(dataset_index: int, dataset_length: int) -> str:
    ratio = dataset_index / max(dataset_length - 1, 1)
    if ratio < 1.0 / 3.0:
        return PHASE_NAMES[0]
    if ratio < 2.0 / 3.0:
        return PHASE_NAMES[1]
    return PHASE_NAMES[2]


def image_path_for_index(raw_episode_dir: Path | None, dataset_index: int) -> str | None:
    if raw_episode_dir is None:
        return None
    image_path = raw_episode_dir / "colors" / f"{dataset_index:06d}_color_0.jpg"
    return str(image_path) if image_path.exists() else None


def load_episode_records(raw_episode_dir: Path | None) -> dict[int, dict[str, Any]]:
    if raw_episode_dir is None:
        return {}
    data_json = raw_episode_dir / "data.json"
    if not data_json.exists():
        return {}
    payload = load_json(data_json)
    return {
        int(frame["idx"]): frame
        for frame in payload.get("data", [])
        if isinstance(frame, dict) and "idx" in frame
    }


def enrich_with_episode_stats(record: dict[str, Any], episode_record: dict[str, Any] | None) -> None:
    if not episode_record:
        return

    states = episode_record.get("states", {})
    actions = episode_record.get("actions", {})

    left_arm_state = states.get("left_arm", {}).get("qpos", []) or []
    right_arm_state = states.get("right_arm", {}).get("qpos", []) or []
    left_ee_state = states.get("left_ee", {}).get("qpos", []) or []

    left_arm_action = actions.get("left_arm", {}).get("qpos", []) or []
    right_arm_action = actions.get("right_arm", {}).get("qpos", []) or []
    left_ee_action = actions.get("left_ee", {}).get("qpos", []) or []

    record["pose_stats"] = round_dict(
        {
            "left_arm_state_norm": l2_norm(left_arm_state),
            "right_arm_state_norm": l2_norm(right_arm_state),
            "left_ee_state_norm": l2_norm(left_ee_state),
            "left_arm_motion_norm": diff_l2_norm(left_arm_action, left_arm_state),
            "right_arm_motion_norm": diff_l2_norm(right_arm_action, right_arm_state),
            "left_ee_motion_norm": diff_l2_norm(left_ee_action, left_ee_state),
        }
    )


def summarize_group(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {}
    keys = (
        "baseline_mse",
        "candidate_mse",
        "mse_improvement",
        "baseline_l1",
        "candidate_l1",
        "l1_improvement",
    )
    summary = {
        key: sum(float(record[key]) for record in records) / len(records)
        for key in keys
    }
    return round_dict(summary)


def build_pairwise_records(
    baseline_samples: list[dict[str, Any]],
    candidate_samples: list[dict[str, Any]],
    dataset_length: int,
    raw_episode_dir: Path | None,
) -> list[dict[str, Any]]:
    episode_records = load_episode_records(raw_episode_dir)
    if len(baseline_samples) != len(candidate_samples):
        raise ValueError("Eval files have different numbers of samples")

    pairwise_records: list[dict[str, Any]] = []
    for baseline, candidate in zip(baseline_samples, candidate_samples):
        if (
            baseline.get("sample_index") != candidate.get("sample_index")
            or baseline.get("dataset_index") != candidate.get("dataset_index")
        ):
            raise ValueError("Eval files do not align on sample_index / dataset_index")

        dataset_index = int(baseline["dataset_index"])
        record = {
            "sample_index": int(baseline["sample_index"]),
            "dataset_index": dataset_index,
            "trajectory_ratio": round(dataset_index / max(dataset_length - 1, 1), 6),
            "phase": phase_for_index(dataset_index, dataset_length),
            "question": baseline.get("question", ""),
            "baseline_mse": float(baseline["mse"]),
            "candidate_mse": float(candidate["mse"]),
            "mse_improvement": float(baseline["mse"]) - float(candidate["mse"]),
            "baseline_l1": float(baseline["l1"]),
            "candidate_l1": float(candidate["l1"]),
            "l1_improvement": float(baseline["l1"]) - float(candidate["l1"]),
            "image_path": image_path_for_index(raw_episode_dir, dataset_index),
        }
        enrich_with_episode_stats(record, episode_records.get(dataset_index))
        pairwise_records.append(record)
    return pairwise_records


def top_records(
    records: list[dict[str, Any]],
    metric_key: str,
    top_k: int,
    reverse: bool,
) -> list[dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda record: (float(record[metric_key]), int(record["dataset_index"])),
        reverse=reverse,
    )
    clipped = ordered[:top_k]
    cleaned: list[dict[str, Any]] = []
    for record in clipped:
        subset = {
            "sample_index": record["sample_index"],
            "dataset_index": record["dataset_index"],
            "trajectory_ratio": record["trajectory_ratio"],
            "phase": record["phase"],
            "baseline_mse": round(record["baseline_mse"], 6),
            "candidate_mse": round(record["candidate_mse"], 6),
            "mse_improvement": round(record["mse_improvement"], 6),
            "baseline_l1": round(record["baseline_l1"], 6),
            "candidate_l1": round(record["candidate_l1"], 6),
            "l1_improvement": round(record["l1_improvement"], 6),
            "image_path": record["image_path"],
        }
        if "pose_stats" in record:
            subset["pose_stats"] = record["pose_stats"]
        cleaned.append(subset)
    return cleaned


def phase_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for phase_name in PHASE_NAMES:
        phase_records = [record for record in records if record["phase"] == phase_name]
        better_mse = sum(1 for record in phase_records if record["mse_improvement"] > 0.0)
        better_l1 = sum(1 for record in phase_records if record["l1_improvement"] > 0.0)
        summary[phase_name] = {
            "count": len(phase_records),
            "candidate_better_mse_count": better_mse,
            "candidate_better_l1_count": better_l1,
            "means": summarize_group(phase_records),
        }
    return summary


def pose_aggregate(records: list[dict[str, Any]]) -> dict[str, float]:
    pose_records = [record["pose_stats"] for record in records if "pose_stats" in record]
    if not pose_records:
        return {}
    keys = sorted(pose_records[0].keys())
    return round_dict(
        {
            key: sum(float(record[key]) for record in pose_records) / len(pose_records)
            for key in keys
        }
    )


def overall_summary(
    records: list[dict[str, Any]],
    baseline_name: str,
    candidate_name: str,
) -> dict[str, Any]:
    better_mse = sum(1 for record in records if record["mse_improvement"] > 0.0)
    better_l1 = sum(1 for record in records if record["l1_improvement"] > 0.0)
    return {
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "num_samples": len(records),
        "candidate_better_mse_count": better_mse,
        "candidate_better_l1_count": better_l1,
        "means": summarize_group(records),
        "top_improved_by_mse": top_records(records, "mse_improvement", 8, reverse=True),
        "top_degraded_by_mse": top_records(records, "mse_improvement", 8, reverse=False),
        "top_improved_by_l1": top_records(records, "l1_improvement", 8, reverse=True),
        "top_degraded_by_l1": top_records(records, "l1_improvement", 8, reverse=False),
        "phase_summary": phase_summary(records),
        "improved_pose_aggregate_top_mse": pose_aggregate(
            sorted(records, key=lambda record: record["mse_improvement"], reverse=True)[:8]
        ),
        "degraded_pose_aggregate_top_mse": pose_aggregate(
            sorted(records, key=lambda record: record["mse_improvement"])[:8]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline eval JSON")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate eval JSON")
    parser.add_argument("--baseline-name", type=str, required=True)
    parser.add_argument("--candidate-name", type=str, required=True)
    parser.add_argument("--dataset-length", type=int, default=None)
    parser.add_argument(
        "--raw-episode-dir",
        type=Path,
        default=None,
        help="Original episode directory with data.json and colors/",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_payload = load_json(args.baseline)
    candidate_payload = load_json(args.candidate)

    baseline_samples = baseline_payload.get("samples", [])
    candidate_samples = candidate_payload.get("samples", [])
    dataset_length = infer_dataset_length(baseline_samples, args.dataset_length)
    records = build_pairwise_records(
        baseline_samples=baseline_samples,
        candidate_samples=candidate_samples,
        dataset_length=dataset_length,
        raw_episode_dir=args.raw_episode_dir,
    )

    output = {
        "baseline_eval": str(args.baseline),
        "candidate_eval": str(args.candidate),
        "dataset_length": dataset_length,
        "summary": overall_summary(records, args.baseline_name, args.candidate_name),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    print(json.dumps(output["summary"]["means"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

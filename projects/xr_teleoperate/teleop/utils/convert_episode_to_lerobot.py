#!/usr/bin/env python3
"""
Convert a raw xr_teleoperate episode folder into LeRobot-ready dataset layouts.

This script is intentionally dependency-light:
- required: stdlib, numpy, cv2
- optional: pyarrow (for parquet export)

It supports two targets:
1. psi0  : aligns with the compact Psi0-style dataset schema used by the
           reference `G1WholebodyBendPickMP-v0` dataset.
2. gr00t : aligns field names with the GR00T / decoupled_wbc exporter style.

Because raw teleop episodes do not contain every modality expected by every
downstream consumer, the script also emits a conversion report that explains:
- which fields are present in the raw episode
- which target fields are directly mapped
- which target fields are derived
- which target fields are imputed / still missing
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _try_write_parquet(path: Path, rows: list[dict[str, Any]]) -> tuple[bool, str]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - optional dependency
        return False, f"pyarrow unavailable: {exc}"

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return True, "ok"


def _euler_rpy_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return [qx, qy, qz, qw]


def _flatten_wrist_pose(arm_ee_qpos: list[float]) -> list[float]:
    xyz = [float(x) for x in arm_ee_qpos[:3]]
    rpy = [float(x) for x in arm_ee_qpos[3:6]]
    quat = _euler_rpy_to_quat(rpy[0], rpy[1], rpy[2])
    return xyz + quat


def _sorted_keys_union(frames: list[dict[str, Any]], group: str) -> list[str]:
    keys = set()
    for frame in frames:
        payload = frame.get(group) or {}
        if isinstance(payload, dict):
            keys.update(payload.keys())
    return sorted(keys)


def _image_shape_from_first_frame(episode_dir: Path, rel_path: str) -> tuple[int, int]:
    img = cv2.imread(str(episode_dir / rel_path))
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {episode_dir / rel_path}")
    height, width = img.shape[:2]
    return height, width


def _write_video_from_color_frames(
    episode_dir: Path,
    color_paths: list[str],
    output_path: Path,
    fps: float,
) -> None:
    if not color_paths:
        raise ValueError("No color frames available for video export")

    first = cv2.imread(str(episode_dir / color_paths[0]))
    if first is None:
        raise FileNotFoundError(f"Failed to read image: {episode_dir / color_paths[0]}")
    height, width = first.shape[:2]

    _ensure_dir(output_path.parent)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    try:
        for rel_path in color_paths:
            frame = cv2.imread(str(episode_dir / rel_path))
            if frame is None:
                raise FileNotFoundError(f"Failed to read image: {episode_dir / rel_path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()


def summarize_raw_episode(episode: dict[str, Any], episode_dir: Path) -> dict[str, Any]:
    frames = episode["data"]
    first = frames[0]
    colors = _sorted_keys_union(frames, "colors")
    depths = _sorted_keys_union(frames, "depths")

    summary = {
        "episode_dir": str(episode_dir),
        "num_frames": len(frames),
        "fps": float(episode["info"]["image"]["fps"]),
        "color_streams_present": colors,
        "depth_streams_present": depths,
        "audio_present": any(frame.get("audios") for frame in frames),
        "sim_state_present": any(frame.get("sim_state") is not None for frame in frames),
        "states_keys": sorted(first["states"].keys()),
        "actions_keys": sorted(first["actions"].keys()),
        "tactiles_keys": sorted(first["tactiles"].keys()),
        "source_dimensions": {
            "psi0_state": len(first["states"]["psi0"]["qpos"]),
            "psi0_action": len(first["actions"]["psi0"]["qpos"]),
            "left_arm_qpos": len(first["states"]["left_arm"]["qpos"]),
            "right_arm_qpos": len(first["states"]["right_arm"]["qpos"]),
            "left_hand_qpos": len(first["states"]["left_hand"]["qpos"]),
            "right_hand_qpos": len(first["states"]["right_hand"]["qpos"]),
            "left_touch": len(first["tactiles"]["left_ee"]["touch"]),
            "right_touch": len(first["tactiles"]["right_ee"]["touch"]),
        },
        "missing_color_counts": {
            key: sum(1 for frame in frames if key not in frame.get("colors", {}))
            for key in ["color_0", "color_1", "color_2"]
        },
    }
    return summary


def build_report(
    episode: dict[str, Any],
    episode_dir: Path,
    reference_root: Path | None,
) -> dict[str, Any]:
    summary = summarize_raw_episode(episode, episode_dir)
    fps = summary["fps"]
    report: dict[str, Any] = {
        "source_summary": summary,
        "reference_summary": None,
        "targets": {
            "psi0": {
                "ready": True,
                "direct_mappings": [
                    "states <- states.psi0.qpos",
                    "action <- actions.psi0.qpos",
                    "observation.hand_joints <- left_hand/right_hand qpos",
                    "observation.arm_joints <- left_arm/right_arm qpos",
                    "observation.images.egocentric <- colors.color_0",
                ],
                "derived_fields": [
                    "timestamp <- frame_index / source_fps",
                    "observation.prev_torso_rpy <- previous frame body.rpy",
                    "observation.prev_height <- previous frame body.height",
                    "next.done <- last frame marker",
                ],
                "imputed_fields": [],
                "missing_fields": [],
                "notes": [],
            },
            "gr00t": {
                "ready": True,
                "direct_mappings": [
                    "observation.images.ego_view <- colors.color_0",
                    "teleop.base_height_command <- actions.body.height",
                    "teleop.navigate_command <- actions.body.qpos",
                    "action.eef <- left/right arm_ee action qpos",
                    "observation.eef_state <- left/right arm_ee state qpos",
                ],
                "derived_fields": [
                    "action.eef quaternion <- xyz + rpy converted to quaternion",
                    "observation.eef_state quaternion <- xyz + rpy converted to quaternion",
                ],
                "imputed_fields": [],
                "missing_fields": [],
                "notes": [],
            },
        },
    }

    if "color_0" not in summary["color_streams_present"]:
        report["targets"]["psi0"]["ready"] = False
        report["targets"]["psi0"]["missing_fields"].append("colors.color_0")
        report["targets"]["gr00t"]["ready"] = False
        report["targets"]["gr00t"]["missing_fields"].append("colors.color_0")

    if "color_2" not in summary["color_streams_present"]:
        report["targets"]["psi0"]["notes"].append(
            "Raw episode does not contain right wrist camera color_2; only color_0/color_1 are available."
        )

    report["targets"]["psi0"]["imputed_fields"].append(
        "observation.leg_joints filled with zeros because raw episode has no leg joint observations"
    )
    report["targets"]["gr00t"]["imputed_fields"].extend(
        [
            "observation.state lower-body block filled with zeros because raw episode has no leg joint observations",
            "action lower-body block filled with zeros because raw episode has no lower-body action observations",
            "observation.img_state_delta filled with 0.0",
        ]
    )

    if not summary["audio_present"]:
        report["targets"]["psi0"]["notes"].append("No audio captured in this episode.")
        report["targets"]["gr00t"]["notes"].append("No audio captured in this episode.")
    if not summary["sim_state_present"]:
        report["targets"]["psi0"]["notes"].append("No sim_state captured in this episode.")
        report["targets"]["gr00t"]["notes"].append("No sim_state captured in this episode.")
    if fps != 50:
        report["targets"]["psi0"]["notes"].append(
            f"Source FPS is {fps}, while reference Psi0 dataset uses 50 FPS."
        )

    if reference_root is not None:
        ref_info_path = reference_root / "meta" / "info.json"
        if ref_info_path.exists():
            ref_info = _read_json(ref_info_path)
            report["reference_summary"] = {
                "root": str(reference_root),
                "fps": ref_info.get("fps"),
                "feature_keys": sorted(ref_info.get("features", {}).keys()),
            }

    return report


def build_psi0_rows(episode: dict[str, Any]) -> list[dict[str, Any]]:
    frames = episode["data"]
    fps = float(episode["info"]["image"]["fps"])
    rows: list[dict[str, Any]] = []

    prev_rpy = [float(x) for x in frames[0]["states"]["body"]["rpy"]]
    prev_height = float(frames[0]["states"]["body"]["height"])

    for idx, frame in enumerate(frames):
        hand_joints = (
            [float(x) for x in frame["states"]["left_hand"]["qpos"]]
            + [float(x) for x in frame["states"]["right_hand"]["qpos"]]
        )
        arm_joints = (
            [float(x) for x in frame["states"]["left_arm"]["qpos"]]
            + [float(x) for x in frame["states"]["right_arm"]["qpos"]]
        )

        row = {
            "states": [float(x) for x in frame["states"]["psi0"]["qpos"]],
            "action": [float(x) for x in frame["actions"]["psi0"]["qpos"]],
            "observation.hand_joints": hand_joints,
            "observation.arm_joints": arm_joints,
            "observation.leg_joints": [0.0] * 15,
            "observation.prev_torso_rpy": prev_rpy,
            "observation.prev_height": [prev_height],
            "timestamp": [idx / fps],
            "frame_index": idx,
            "episode_index": 0,
            "index": idx,
            "next.done": idx == len(frames) - 1,
            "task_index": 0,
        }
        rows.append(row)
        prev_rpy = [float(x) for x in frame["states"]["body"]["rpy"]]
        prev_height = float(frame["states"]["body"]["height"])

    return rows


def build_gr00t_rows(episode: dict[str, Any]) -> list[dict[str, Any]]:
    frames = episode["data"]
    fps = float(episode["info"]["image"]["fps"])
    rows: list[dict[str, Any]] = []

    lower_body_dof = 15
    waist_dof = 3

    for idx, frame in enumerate(frames):
        state_vec = (
            [0.0] * lower_body_dof
            + [float(x) for x in frame["states"]["body"]["rpy"][:waist_dof]]
            + [float(x) for x in frame["states"]["left_arm"]["qpos"]]
            + [float(x) for x in frame["states"]["left_hand"]["qpos"]]
            + [float(x) for x in frame["states"]["right_arm"]["qpos"]]
            + [float(x) for x in frame["states"]["right_hand"]["qpos"]]
        )
        action_vec = (
            [0.0] * lower_body_dof
            + [float(x) for x in frame["actions"]["body"]["rpy"][:waist_dof]]
            + [float(x) for x in frame["actions"]["left_arm"]["qpos"]]
            + [float(x) for x in frame["actions"]["left_hand"]["qpos"]]
            + [float(x) for x in frame["actions"]["right_arm"]["qpos"]]
            + [float(x) for x in frame["actions"]["right_hand"]["qpos"]]
        )
        eef_state = _flatten_wrist_pose(frame["states"]["left_arm_ee"]["qpos"]) + _flatten_wrist_pose(
            frame["states"]["right_arm_ee"]["qpos"]
        )
        eef_action = _flatten_wrist_pose(frame["actions"]["left_arm_ee"]["qpos"]) + _flatten_wrist_pose(
            frame["actions"]["right_arm_ee"]["qpos"]
        )

        row = {
            "observation.state": state_vec,
            "observation.eef_state": eef_state,
            "action": action_vec,
            "action.eef": eef_action,
            "teleop.navigate_command": [float(x) for x in frame["actions"]["body"]["qpos"]],
            "teleop.base_height_command": [float(frame["actions"]["body"]["height"])],
            "observation.img_state_delta": [0.0],
            "timestamp": [idx / fps],
            "frame_index": idx,
            "episode_index": 0,
            "index": idx,
            "next.done": idx == len(frames) - 1,
            "task_index": 0,
        }
        rows.append(row)

    return rows


def psi0_features(height: int, width: int, fps: float) -> dict[str, Any]:
    return {
        "observation.images.egocentric": {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "mp4v",
                "video.pix_fmt": "bgr24",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        },
        "observation.hand_joints": {"dtype": "float32", "shape": [14], "names": ["hand_joints"]},
        "observation.arm_joints": {"dtype": "float32", "shape": [14], "names": ["arm_joints"]},
        "observation.leg_joints": {"dtype": "float32", "shape": [15], "names": ["leg_joints"]},
        "observation.prev_torso_rpy": {
            "dtype": "float32",
            "shape": [3],
            "names": ["prev_roll", "prev_pitch", "prev_yaw"],
        },
        "observation.prev_height": {
            "dtype": "float32",
            "shape": [1],
            "names": ["prev_height"],
        },
        "states": {"dtype": "float32", "shape": [32]},
        "action": {"dtype": "float32", "shape": [36]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }


def gr00t_features(height: int, width: int, fps: float) -> dict[str, Any]:
    return {
        "observation.images.ego_view": {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "mp4v",
                "video.pix_fmt": "bgr24",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        },
        "observation.state": {"dtype": "float32", "shape": [46]},
        "observation.eef_state": {"dtype": "float32", "shape": [14]},
        "action": {"dtype": "float32", "shape": [46]},
        "action.eef": {"dtype": "float32", "shape": [14]},
        "teleop.navigate_command": {"dtype": "float32", "shape": [3]},
        "teleop.base_height_command": {"dtype": "float32", "shape": [1]},
        "observation.img_state_delta": {"dtype": "float32", "shape": [1]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }


def psi0_modality() -> dict[str, Any]:
    return {
        "state": {
            "left_hand": {"start": 0, "end": 7, "original_key": "states"},
            "right_hand": {"start": 7, "end": 14, "original_key": "states"},
            "left_arm": {"start": 14, "end": 21, "original_key": "states"},
            "right_arm": {"start": 21, "end": 28, "original_key": "states"},
            "rpy": {"start": 28, "end": 31, "original_key": "states"},
            "height": {"start": 31, "end": 32, "original_key": "states"},
        },
        "action": {
            "left_hand": {"start": 0, "end": 7, "original_key": "action"},
            "right_hand": {"start": 7, "end": 14, "original_key": "action"},
            "left_arm": {"start": 14, "end": 21, "original_key": "action"},
            "right_arm": {"start": 21, "end": 28, "original_key": "action"},
            "rpy": {"start": 28, "end": 31, "original_key": "action"},
            "height": {"start": 31, "end": 32, "original_key": "action"},
            "torso_vx": {"start": 32, "end": 33, "original_key": "action"},
            "torso_vy": {"start": 33, "end": 34, "original_key": "action"},
            "torso_vyaw": {"start": 34, "end": 35, "original_key": "action"},
            "target_yaw": {"start": 35, "end": 36, "original_key": "action"},
        },
        "video": {"rs_view": {"original_key": "observation.images.egocentric"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def gr00t_modality() -> dict[str, Any]:
    return {
        "state": {
            "left_leg": {"start": 0, "end": 6, "original_key": "observation.state"},
            "right_leg": {"start": 6, "end": 12, "original_key": "observation.state"},
            "waist": {"start": 12, "end": 15, "original_key": "observation.state"},
            "left_arm": {"start": 15, "end": 22, "original_key": "observation.state"},
            "left_hand": {"start": 22, "end": 29, "original_key": "observation.state"},
            "right_arm": {"start": 29, "end": 36, "original_key": "observation.state"},
            "right_hand": {"start": 36, "end": 43, "original_key": "observation.state"},
            "left_wrist_pos": {"start": 0, "end": 3, "original_key": "observation.eef_state"},
            "left_wrist_abs_quat": {
                "start": 3,
                "end": 7,
                "original_key": "observation.eef_state",
                "rotation_type": "quaternion",
            },
            "right_wrist_pos": {"start": 7, "end": 10, "original_key": "observation.eef_state"},
            "right_wrist_abs_quat": {
                "start": 10,
                "end": 14,
                "original_key": "observation.eef_state",
                "rotation_type": "quaternion",
            },
        },
        "action": {
            "left_leg": {"start": 0, "end": 6, "original_key": "action"},
            "right_leg": {"start": 6, "end": 12, "original_key": "action"},
            "waist": {"start": 12, "end": 15, "original_key": "action"},
            "left_arm": {"start": 15, "end": 22, "original_key": "action"},
            "left_hand": {"start": 22, "end": 29, "original_key": "action"},
            "right_arm": {"start": 29, "end": 36, "original_key": "action"},
            "right_hand": {"start": 36, "end": 43, "original_key": "action"},
            "left_wrist_pos": {"start": 0, "end": 3, "original_key": "action.eef"},
            "left_wrist_abs_quat": {
                "start": 3,
                "end": 7,
                "original_key": "action.eef",
                "rotation_type": "quaternion",
            },
            "right_wrist_pos": {"start": 7, "end": 10, "original_key": "action.eef"},
            "right_wrist_abs_quat": {
                "start": 10,
                "end": 14,
                "original_key": "action.eef",
                "rotation_type": "quaternion",
            },
            "base_height_command": {
                "start": 0,
                "end": 1,
                "original_key": "teleop.base_height_command",
            },
            "navigate_command": {
                "start": 0,
                "end": 3,
                "original_key": "teleop.navigate_command",
            },
        },
        "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def _write_meta_files(
    root: Path,
    dataset_name: str,
    fps: float,
    num_frames: int,
    features: dict[str, Any],
    modality: dict[str, Any],
    task_description: str,
    data_ext: str,
    video_subdir: str,
) -> None:
    meta_dir = root / "meta"
    _ensure_dir(meta_dir)

    info = {
        "codebase_version": "raw-converter-v1",
        "dataset_name": dataset_name,
        "total_episodes": 1,
        "total_frames": num_frames,
        "total_tasks": 1,
        "total_videos": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "data_path": f"data/chunk-{{episode_chunk:03d}}/episode_{{episode_index:06d}}.{data_ext}",
        "video_path": f"videos/chunk-{{episode_chunk:03d}}/{video_subdir}/episode_{{episode_index:06d}}.mp4",
        "features": features,
    }
    _write_json(meta_dir / "info.json", info)
    _write_json(meta_dir / "modality.json", modality)
    _write_json(meta_dir / "lang_map.json", {"0": task_description})

    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "task_index": 0,
                    "task": task_description,
                    "category": "",
                    "description": task_description,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "episode_index": 0,
                    "tasks": [task_description],
                    "length": num_frames,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def export_target_dataset(
    target: str,
    episode: dict[str, Any],
    episode_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    frames = episode["data"]
    fps = float(episode["info"]["image"]["fps"])
    task_description = episode.get("text", {}).get("goal") or "unspecified task"

    if "color_0" not in frames[0]["colors"]:
        raise ValueError("color_0 is required for video export")

    height, width = _image_shape_from_first_frame(episode_dir, frames[0]["colors"]["color_0"])
    color_paths = [frame["colors"]["color_0"] for frame in frames]

    if target == "psi0":
        rows = build_psi0_rows(episode)
        features = psi0_features(height, width, fps)
        modality = psi0_modality()
        video_subdir = "egocentric"
    elif target == "gr00t":
        rows = build_gr00t_rows(episode)
        features = gr00t_features(height, width, fps)
        modality = gr00t_modality()
        video_subdir = "ego_view"
    else:  # pragma: no cover - protected by argparse choices
        raise ValueError(f"Unsupported target: {target}")

    dataset_root = output_root / target
    data_dir = dataset_root / "data" / "chunk-000"
    video_dir = dataset_root / "videos" / "chunk-000" / video_subdir
    _ensure_dir(data_dir)
    _ensure_dir(video_dir)

    video_path = video_dir / "episode_000000.mp4"
    _write_video_from_color_frames(episode_dir, color_paths, video_path, fps)

    parquet_path = data_dir / "episode_000000.parquet"
    jsonl_path = data_dir / "episode_000000.jsonl"
    wrote_parquet, parquet_msg = _try_write_parquet(parquet_path, rows)
    data_ext = "parquet" if wrote_parquet else "jsonl"
    if not wrote_parquet:
        _write_jsonl(jsonl_path, rows)

    _write_meta_files(
        root=dataset_root,
        dataset_name=target,
        fps=fps,
        num_frames=len(rows),
        features=features,
        modality=modality,
        task_description=task_description,
        data_ext=data_ext,
        video_subdir=video_subdir,
    )

    return {
        "target": target,
        "dataset_root": str(dataset_root),
        "num_rows": len(rows),
        "video_path": str(video_path),
        "data_format": data_ext,
        "parquet_status": parquet_msg,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True, help="Path to raw episode_xxxx directory")
    parser.add_argument("--output-root", required=True, help="Output directory for converted datasets")
    parser.add_argument(
        "--target",
        choices=["psi0", "gr00t", "both"],
        default="both",
        help="Which dataset layout to export",
    )
    parser.add_argument(
        "--reference-root",
        default=None,
        help="Optional reference LeRobot dataset root for reporting",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only write the conversion report, do not export datasets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_dir = Path(args.episode_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    reference_root = (
        Path(args.reference_root).expanduser().resolve() if args.reference_root else None
    )

    episode_json = episode_dir / "data.json"
    if not episode_json.exists():
        raise FileNotFoundError(f"Missing episode json: {episode_json}")

    episode = _read_json(episode_json)
    _ensure_dir(output_root)

    report = build_report(episode, episode_dir, reference_root)
    report_path = output_root / "conversion_report.json"
    _write_json(report_path, report)

    exports: list[dict[str, Any]] = []
    if not args.analyze_only:
        targets = ["psi0", "gr00t"] if args.target == "both" else [args.target]
        for target in targets:
            exports.append(export_target_dataset(target, episode, episode_dir, output_root))

    result = {
        "report_path": str(report_path),
        "exports": exports,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a raw Unitree G1 episode dump into GR00T-flavored LeRobot v2.

Expected raw input layout:

    episode_xxxx/
      data.json
      colors/
        000000_color_0.jpg
        ...

The converted dataset is written as:

    <output>/
      meta/
      data/chunk-000/episode_000000.parquet
      videos/chunk-000/observation.images.ego_view/episode_000000.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gr00t.data.stats import generate_stats


DEFAULT_TASK = "grasp the cup and place it"
DEFAULT_VIDEO_KEY = "ego_view"
DEFAULT_WRIST_VIDEO_KEY = "wrist"
DEFAULT_PSI0_ROBOT_TYPE = "unitree_g1_29dof_psi0"
DEFAULT_LEGACY_ROBOT_TYPE = "unitree_g1_upper_body_custom"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-episode-dir",
        type=Path,
        required=True,
        help="Path to the raw episode directory that contains data.json and colors/.",
    )
    parser.add_argument(
        "--output-dataset-dir",
        type=Path,
        default=None,
        help=(
            "Path to the converted GR00T dataset root. If omitted, --dataset-name is "
            "created under the input episode's parent directory."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset folder name to create under the input episode's parent directory.",
    )
    parser.add_argument(
        "--task-description",
        type=str,
        default=None,
        help="Task description stored in meta/tasks.jsonl. Defaults to raw text.goal or a cup grasp-place fallback.",
    )
    parser.add_argument(
        "--video-key",
        type=str,
        default=DEFAULT_VIDEO_KEY,
        help="Logical video key for color_0, stored in meta/modality.json.",
    )
    parser.add_argument(
        "--wrist-video-key",
        type=str,
        default=DEFAULT_WRIST_VIDEO_KEY,
        help="Logical video key for color_1 when a wrist camera is present.",
    )
    parser.add_argument(
        "--video-keys",
        type=str,
        default=None,
        help=(
            "Comma-separated logical video keys for color_0,color_1,... . "
            "Defaults to --video-key for single-camera data and "
            "--video-key,--wrist-video-key when color_1 exists."
        ),
    )
    parser.add_argument(
        "--state-format",
        choices=("auto", "legacy", "psi0"),
        default="auto",
        help="Vector layout for observation.state/action. Auto uses psi0 when present.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default=None,
        help="Robot type stored in meta/info.json. Defaults based on selected state format.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode index to assign inside the converted dataset.",
    )
    parser.add_argument(
        "--chunks-size",
        type=int,
        default=1000,
        help="Chunk size stored in meta/info.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory first if it already exists.",
    )
    args = parser.parse_args()
    if args.output_dataset_dir is None:
        if not args.dataset_name:
            parser.error("one of --output-dataset-dir or --dataset-name is required")
        args.output_dataset_dir = args.input_episode_dir.parent / args.dataset_name
    return args


def load_raw_episode(input_episode_dir: Path) -> dict[str, Any]:
    data_path = input_episode_dir / "data.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing raw episode json: {data_path}")

    with data_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if "data" not in payload or not isinstance(payload["data"], list) or not payload["data"]:
        raise ValueError(f"Unexpected raw payload format in {data_path}")

    return payload


def validate_raw_episode(payload: dict[str, Any]) -> None:
    frames = payload["data"]
    expected_state_dims = {}
    expected_action_dims = {}

    for frame in frames:
        for key, value in frame["states"].items():
            dims = len(value.get("qpos", []))
            expected_state_dims.setdefault(key, dims)
            if expected_state_dims[key] != dims:
                raise ValueError(f"Inconsistent state dims for {key}: {expected_state_dims[key]} vs {dims}")

        for key, value in frame["actions"].items():
            dims = len(value.get("qpos", []))
            expected_action_dims.setdefault(key, dims)
            if expected_action_dims[key] != dims:
                raise ValueError(f"Inconsistent action dims for {key}: {expected_action_dims[key]} vs {dims}")


def infer_task_description(payload: dict[str, Any], explicit_task: str | None) -> str:
    if explicit_task:
        return explicit_task

    raw_goal = payload.get("text", {}).get("goal")
    if isinstance(raw_goal, str) and raw_goal.strip():
        return raw_goal.strip()

    return DEFAULT_TASK


def has_psi0_vectors(payload: dict[str, Any]) -> bool:
    first_frame = payload["data"][0]
    return "psi0" in first_frame.get("states", {}) and "psi0" in first_frame.get("actions", {})


def select_state_format(payload: dict[str, Any], requested: str) -> str:
    if requested == "auto":
        return "psi0" if has_psi0_vectors(payload) else "legacy"
    if requested == "psi0" and not has_psi0_vectors(payload):
        raise ValueError("Requested --state-format psi0, but states.psi0/actions.psi0 are missing")
    return requested


def build_state_action_vectors(frame: dict[str, Any], state_format: str) -> tuple[np.ndarray, np.ndarray]:
    if state_format == "psi0":
        return (
            np.asarray(frame["states"]["psi0"]["qpos"], dtype=np.float32),
            np.asarray(frame["actions"]["psi0"]["qpos"], dtype=np.float32),
        )

    state_groups = [
        np.asarray(frame["states"]["left_arm"]["qpos"], dtype=np.float32),
        np.asarray(frame["states"]["right_arm"]["qpos"], dtype=np.float32),
        np.asarray(frame["states"]["left_ee"]["qpos"], dtype=np.float32),
        np.asarray(frame["states"]["right_ee"]["qpos"], dtype=np.float32),
        np.asarray(frame["states"]["body"]["qpos"], dtype=np.float32),
    ]
    action_groups = [
        np.asarray(frame["actions"]["left_arm"]["qpos"], dtype=np.float32),
        np.asarray(frame["actions"]["right_arm"]["qpos"], dtype=np.float32),
        np.asarray(frame["actions"]["left_ee"]["qpos"], dtype=np.float32),
        np.asarray(frame["actions"]["right_ee"]["qpos"], dtype=np.float32),
        np.asarray(frame["actions"]["body"]["qpos"], dtype=np.float32),
    ]
    return np.concatenate(state_groups), np.concatenate(action_groups)


def build_episode_dataframe(
    payload: dict[str, Any],
    fps: float,
    episode_index: int,
    state_format: str,
) -> pd.DataFrame:
    records = []

    for frame_index, frame in enumerate(payload["data"]):
        state_vec, action_vec = build_state_action_vectors(frame, state_format)
        records.append(
            {
                "observation.state": state_vec,
                "action": action_vec,
                "timestamp": np.float32(frame_index / fps),
                "frame_index": np.int64(frame_index),
                "episode_index": np.int64(episode_index),
                "index": np.int64(frame_index),
                "task_index": np.int64(0),
            }
        )

    return pd.DataFrame(records)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")


def build_info_json(
    payload: dict[str, Any],
    total_frames: int,
    episode_index: int,
    video_keys: list[str],
    chunks_size: int,
    video_sizes: dict[str, tuple[int, int]],
    state_format: str,
    robot_type: str,
) -> dict[str, Any]:
    image_info = payload.get("info", {}).get("image", {})
    fps = int(round(float(image_info.get("fps", 30.0))))

    joint_names = payload.get("info", {}).get("joint_names", {})
    if state_format == "psi0":
        state_names = [f"{name}.pos" for name in joint_names.get("psi0_state", [])]
        action_names = [f"{name}.cmd" for name in joint_names.get("psi0_action", [])]
        if not state_names:
            state_dim = len(payload["data"][0]["states"]["psi0"]["qpos"])
            state_names = [f"psi0_state_{i}.pos" for i in range(state_dim)]
        if not action_names:
            action_dim = len(payload["data"][0]["actions"]["psi0"]["qpos"])
            action_names = [f"psi0_action_{i}.cmd" for i in range(action_dim)]
    else:
        left_hand_names = joint_names.get("left_ee", [])
        right_hand_names = joint_names.get("right_ee", [])

        state_names = (
            [f"left_arm_{i}.pos" for i in range(5)]
            + [f"right_arm_{i}.pos" for i in range(5)]
            + [f"left_hand.{name}.pos" for name in left_hand_names]
            + [f"right_hand.{name}.pos" for name in right_hand_names]
            + [f"body_state_{i}.pos" for i in range(5)]
        )
        action_names = (
            [f"left_arm_{i}.cmd" for i in range(5)]
            + [f"right_arm_{i}.cmd" for i in range(5)]
            + [f"left_hand.{name}.cmd" for name in left_hand_names]
            + [f"right_hand.{name}.cmd" for name in right_hand_names]
            + [f"body_command_{i}.cmd" for i in range(3)]
        )

    video_features: dict[str, Any] = {}
    for video_key in video_keys:
        width, height = video_sizes[video_key]
        video_features[f"observation.images.{video_key}"] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    return {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": 1,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": chunks_size,
        "fps": fps,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {
                "dtype": "float32",
                "names": action_names,
                "shape": [len(action_names)],
            },
            "observation.state": {
                "dtype": "float32",
                "names": state_names,
                "shape": [len(state_names)],
            },
            **video_features,
            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
        },
        "total_chunks": int(episode_index // chunks_size) + 1,
        "total_videos": len(video_keys),
    }


def build_ranges(groups: list[tuple[str, int]]) -> dict[str, dict[str, int]]:
    ranges = {}
    cursor = 0
    for name, dim in groups:
        ranges[name] = {"start": cursor, "end": cursor + dim}
        cursor += dim
    return ranges


def build_modality_json(video_keys: list[str], state_format: str) -> dict[str, Any]:
    if state_format == "psi0":
        state = build_ranges(
            [
                ("left_hand", 7),
                ("right_hand", 7),
                ("left_arm", 7),
                ("right_arm", 7),
                ("body_state", 4),
            ]
        )
        action = build_ranges(
            [
                ("left_hand", 7),
                ("right_hand", 7),
                ("left_arm", 7),
                ("right_arm", 7),
                ("body_command", 8),
            ]
        )
    else:
        state = {
            "left_arm": {"start": 0, "end": 5},
            "right_arm": {"start": 5, "end": 10},
            "left_hand": {"start": 10, "end": 16},
            "right_hand": {"start": 16, "end": 22},
            "body_state": {"start": 22, "end": 27},
        }
        action = {
            "left_arm": {"start": 0, "end": 5},
            "right_arm": {"start": 5, "end": 10},
            "left_hand": {"start": 10, "end": 16},
            "right_hand": {"start": 16, "end": 22},
            "body_command": {"start": 22, "end": 25},
        }

    return {
        "state": state,
        "action": action,
        "video": {video_key: {"original_key": f"observation.images.{video_key}"} for video_key in video_keys},
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }


def make_video(input_episode_dir: Path, output_mp4: Path, fps: float, color_index: int) -> None:
    colors_dir = input_episode_dir / "colors"
    if not colors_dir.exists():
        raise FileNotFoundError(f"Missing colors directory: {colors_dir}")

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(float(fps)),
        "-i",
        str(colors_dir / f"%06d_color_{color_index}.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def probe_video_size(video_path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    return int(stream["width"]), int(stream["height"])


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def discover_color_indices(input_episode_dir: Path) -> list[int]:
    colors_dir = input_episode_dir / "colors"
    indices = set()
    for path in colors_dir.glob("*_color_*.jpg"):
        try:
            indices.add(int(path.stem.rsplit("_color_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(indices)


def resolve_video_keys(args: argparse.Namespace, color_indices: list[int]) -> list[str]:
    if args.video_keys:
        keys = [key.strip() for key in args.video_keys.split(",") if key.strip()]
    elif len(color_indices) <= 1:
        keys = [args.video_key]
    else:
        keys = [args.video_key, args.wrist_video_key]
        keys.extend(f"camera_{index}" for index in color_indices[2:])

    if len(keys) != len(color_indices):
        raise ValueError(
            f"Video key count ({len(keys)}) must match discovered camera count ({len(color_indices)}): "
            f"{color_indices}"
        )
    if len(set(keys)) != len(keys):
        raise ValueError(f"Video keys must be unique: {keys}")
    return keys


def convert(args: argparse.Namespace) -> None:
    payload = load_raw_episode(args.input_episode_dir)
    validate_raw_episode(payload)

    state_format = select_state_format(payload, args.state_format)
    robot_type = args.robot_type or (
        DEFAULT_PSI0_ROBOT_TYPE if state_format == "psi0" else DEFAULT_LEGACY_ROBOT_TYPE
    )
    task_description = infer_task_description(payload, args.task_description)
    fps = float(payload.get("info", {}).get("image", {}).get("fps", 30.0))
    total_frames = len(payload["data"])
    color_indices = discover_color_indices(args.input_episode_dir)
    if not color_indices:
        raise FileNotFoundError(f"No color images found in: {args.input_episode_dir / 'colors'}")
    video_keys = resolve_video_keys(args, color_indices)

    prepare_output_dir(args.output_dataset_dir, args.overwrite)

    meta_dir = args.output_dataset_dir / "meta"
    data_dir = args.output_dataset_dir / "data" / "chunk-000"

    data_dir.mkdir(parents=True, exist_ok=True)

    df = build_episode_dataframe(
        payload,
        fps=fps,
        episode_index=args.episode_index,
        state_format=state_format,
    )
    parquet_path = data_dir / f"episode_{args.episode_index:06d}.parquet"
    df.to_parquet(parquet_path, index=False)

    write_json(meta_dir / "modality.json", build_modality_json(video_keys, state_format))
    write_jsonl(
        meta_dir / "episodes.jsonl",
        [
            {
                "episode_index": args.episode_index,
                "tasks": [task_description],
                "length": total_frames,
            }
        ],
    )
    write_jsonl(
        meta_dir / "tasks.jsonl",
        [
            {
                "task_index": 0,
                "task": task_description,
            }
        ],
    )

    video_sizes = {}
    output_videos = []
    for color_index, video_key in zip(color_indices, video_keys):
        video_dir = args.output_dataset_dir / "videos" / "chunk-000" / f"observation.images.{video_key}"
        output_mp4 = video_dir / f"episode_{args.episode_index:06d}.mp4"
        make_video(args.input_episode_dir, output_mp4, fps=fps, color_index=color_index)
        video_sizes[video_key] = probe_video_size(output_mp4)
        output_videos.append(output_mp4)

    write_json(
        meta_dir / "info.json",
        build_info_json(
            payload=payload,
            total_frames=total_frames,
            episode_index=args.episode_index,
            video_keys=video_keys,
            chunks_size=args.chunks_size,
            video_sizes=video_sizes,
            state_format=state_format,
            robot_type=robot_type,
        ),
    )

    generate_stats(args.output_dataset_dir)

    print(f"Converted raw episode to GR00T dataset at: {args.output_dataset_dir}")
    print(f"State format: {state_format}")
    print(f"Parquet: {parquet_path}")
    for output_video in output_videos:
        print(f"Video:   {output_video}")


def main() -> None:
    args = parse_args()
    convert(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert Unitree G1 teleoperation `data.json` episodes into a LeRobot dataset.

This script is designed for the A1 training pipeline in this repository.

Input assumptions:
- A single episode is a directory containing `data.json`
- Frame images are referenced from `data.json`, typically via `colors/...jpg`
- `states` and `actions` are nested per arm/body group

Output:
- A LeRobot-format dataset that A1 can load through its `lerobot` dataset path

Example:
    python scripts/convert_g1_to_lerobot.py \
        --src /path/to/g1_dataset_root \
        --dst /path/to/output/g1_pick_cube_lerobot
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG = logging.getLogger("convert_g1_to_lerobot")


@dataclass(frozen=True)
class EpisodeInfo:
    json_path: Path
    episode_root: Path
    task: str
    fps: float
    frame_count: int
    image_width: int | None
    image_height: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Unitree G1 teleoperation data.json episodes into a LeRobot dataset."
    )
    parser.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Source path. Can be a single data.json file, a single episode directory, or a root directory.",
    )
    parser.add_argument(
        "--dst",
        required=True,
        type=Path,
        help="Destination dataset root. Must not already exist.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="LeRobot repo_id metadata. Defaults to the destination folder name.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="unitree_g1",
        help="Robot type written into LeRobot metadata.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override dataset FPS. If omitted, it is inferred from source data and must be consistent.",
    )
    parser.add_argument(
        "--camera-keys",
        type=str,
        default=None,
        help="Comma-separated input camera keys to keep, e.g. 'color_0,color_1'. Defaults to all discovered keys.",
    )
    parser.add_argument(
        "--preserve-camera-names",
        action="store_true",
        help="Keep input camera feature names instead of mapping the first two to image/wrist_image.",
    )
    parser.add_argument(
        "--state-groups",
        type=str,
        default="left_arm,right_arm",
        help="Comma-separated order of source groups to concatenate into `state`.",
    )
    parser.add_argument(
        "--action-groups",
        type=str,
        default="left_arm,right_arm",
        help="Comma-separated order of source groups to concatenate into `actions`.",
    )
    parser.add_argument(
        "--state-field",
        type=str,
        default="qpos",
        help="Field to read inside each state group, e.g. qpos/qvel/torque.",
    )
    parser.add_argument(
        "--action-field",
        type=str,
        default="qpos",
        help="Field to read inside each action group, e.g. qpos/qvel/torque.",
    )
    parser.add_argument(
        "--action-source",
        choices=("actions", "next_state", "auto"),
        default="actions",
        help=(
            "How to build supervision actions. "
            "`actions` uses the per-frame actions field, "
            "`next_state` uses the next frame's state vector, "
            "`auto` falls back to next_state when actions are missing."
        ),
    )
    parser.add_argument(
        "--instruction-source",
        choices=("auto", "goal", "desc", "steps"),
        default="auto",
        help="Which text field to use as the task/instruction.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on the number of source episodes to convert.",
    )
    parser.add_argument(
        "--skip-invalid-frames",
        action="store_true",
        help="Skip frames with malformed state/action/image entries instead of failing fast.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity.",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def natural_sort_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def find_episode_jsons(src: Path) -> list[Path]:
    if src.is_file():
        return [src]

    direct_json = src / "data.json"
    if direct_json.is_file():
        return [direct_json]

    return sorted(src.rglob("data.json"))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_instruction(record: Mapping[str, Any], source: str, fallback: str) -> str:
    text = record.get("text") or {}
    if not isinstance(text, Mapping):
        text = {}

    if source == "goal":
        candidates = [text.get("goal")]
    elif source == "desc":
        candidates = [text.get("desc")]
    elif source == "steps":
        candidates = [text.get("steps")]
    else:
        candidates = [text.get("goal"), text.get("desc"), text.get("steps")]

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return fallback


def vector_from_groups(
    frame_section: Mapping[str, Any],
    groups: Sequence[str],
    field: str,
) -> np.ndarray:
    values: list[float] = []
    for group in groups:
        group_payload = frame_section.get(group) or {}
        if not isinstance(group_payload, Mapping):
            raise ValueError(f"Group '{group}' is not a mapping: {group_payload!r}")
        raw_values = group_payload.get(field)
        if raw_values is None:
            raw_values = []
        if not isinstance(raw_values, Sequence):
            raise ValueError(f"Field '{field}' in group '{group}' is not a sequence: {raw_values!r}")
        values.extend(float(item) for item in raw_values)
    return np.asarray(values, dtype=np.float32)


def frame_state_vector(
    frame: Mapping[str, Any],
    groups: Sequence[str],
    field: str,
) -> np.ndarray:
    section = frame.get("states") or {}
    if not isinstance(section, Mapping):
        raise ValueError(f"Frame states payload is not a mapping: {section!r}")
    return vector_from_groups(section, groups, field)


def frame_action_vector(
    frames: Sequence[Mapping[str, Any]],
    index: int,
    action_groups: Sequence[str],
    action_field: str,
    action_source: str,
    state_groups: Sequence[str],
    state_field: str,
) -> np.ndarray:
    frame = frames[index]
    section = frame.get("actions") or {}
    if action_source in ("actions", "auto") and isinstance(section, Mapping):
        action_vec = vector_from_groups(section, action_groups, action_field)
        if action_vec.size > 0:
            return action_vec
        if action_source == "actions":
            raise ValueError(f"Frame {index} has empty actions vector.")

    if action_source in ("next_state", "auto"):
        next_index = min(index + 1, len(frames) - 1)
        return frame_state_vector(frames[next_index], state_groups, state_field)

    raise ValueError(f"Unsupported action source: {action_source}")


def infer_vector_names(
    info: Mapping[str, Any],
    groups: Sequence[str],
    field: str,
    prefix: str,
    expected_dim: int,
) -> list[str]:
    if field != "qpos":
        return [f"{prefix}_{idx}" for idx in range(expected_dim)]

    joint_names = info.get("joint_names") or {}
    if not isinstance(joint_names, Mapping):
        return [f"{prefix}_{idx}" for idx in range(expected_dim)]

    names: list[str] = []
    for group in groups:
        group_names = joint_names.get(group)
        if not isinstance(group_names, Sequence) or isinstance(group_names, (str, bytes)):
            names = []
            break
        if not group_names:
            names = []
            break
        names.extend(f"{group}.{name}" for name in group_names)

    if len(names) != expected_dim:
        return [f"{prefix}_{idx}" for idx in range(expected_dim)]
    return names


def choose_camera_feature_names(
    input_camera_keys: Sequence[str],
    preserve_camera_names: bool,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for index, key in enumerate(input_camera_keys):
        if preserve_camera_names:
            feature_name = key
        elif index == 0:
            feature_name = "image"
        elif index == 1:
            feature_name = "wrist_image"
        else:
            feature_name = key

        candidate = feature_name
        suffix = 2
        while candidate in used:
            candidate = f"{feature_name}_{suffix}"
            suffix += 1
        used.add(candidate)
        mapping[key] = candidate
    return mapping


def read_image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except ModuleNotFoundError:
        try:
            import cv2
        except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Neither Pillow nor OpenCV is available. Install one of them to read source images."
            ) from exc

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        height, width = image.shape[:2]
        return width, height


def load_rgb_image(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    try:
        from PIL import Image

        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != (target_w, target_h):
                image = image.resize((target_w, target_h), resample=Image.BILINEAR)
            return np.asarray(image, dtype=np.uint8)
    except ModuleNotFoundError:
        try:
            import cv2
        except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Neither Pillow nor OpenCV is available. Install one of them to decode source images."
            ) from exc

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[0] != target_h or image.shape[1] != target_w:
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return image.astype(np.uint8, copy=False)


def make_blank_image(target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    return np.zeros((target_h, target_w, 3), dtype=np.uint8)


def build_stats(array: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": array.mean(axis=0).astype(np.float32).tolist(),
        "std": array.std(axis=0).astype(np.float32).tolist(),
        "min": array.min(axis=0).astype(np.float32).tolist(),
        "max": array.max(axis=0).astype(np.float32).tolist(),
        "q01": np.quantile(array, 0.01, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(array, 0.99, axis=0).astype(np.float32).tolist(),
    }


def coerce_dataset_fps(fps_values: Iterable[float], override: float | None) -> int:
    if override is not None:
        fps = override
    else:
        fps_list = list(fps_values)
        if not fps_list:
            raise ValueError("No FPS values found in source data.")
        first = fps_list[0]
        for value in fps_list[1:]:
            if abs(value - first) > 1e-6:
                raise ValueError(
                    f"Source episodes contain inconsistent FPS values: {fps_list}. "
                    "Please pass --fps to force a dataset FPS."
                )
        fps = first

    rounded = int(round(fps))
    if abs(fps - rounded) > 1e-6:
        raise ValueError(
            f"Dataset FPS must be effectively an integer for this converter, got {fps}. "
            "Please pass --fps with an integer-compatible value."
        )
    return rounded


def get_lerobot_dataset_cls():
    try:
        from a1.data.lerobot_dataset_replace import LeRobotDataset
    except Exception as exc:  # pragma: no cover - import environment issue
        raise RuntimeError(
            "Failed to import A1's bundled LeRobotDataset implementation. "
            "Please run this script from the A1 environment where its dependencies are installed."
        ) from exc
    return LeRobotDataset


def scan_source_episodes(
    json_paths: Sequence[Path],
    camera_keys_filter: set[str] | None,
    state_groups: Sequence[str],
    state_field: str,
    action_groups: Sequence[str],
    action_field: str,
    action_source: str,
    instruction_source: str,
) -> tuple[list[EpisodeInfo], list[str], dict[str, tuple[int, int]], int, int, list[str], list[str]]:
    episodes: list[EpisodeInfo] = []
    discovered_camera_keys: set[str] = set()
    image_sizes: dict[str, tuple[int, int]] = {}
    state_dim: int | None = None
    action_dim: int | None = None
    state_names: list[str] | None = None
    action_names: list[str] | None = None

    for json_path in json_paths:
        record = load_json(json_path)
        frames = record.get("data") or []
        if not isinstance(frames, Sequence) or not frames:
            LOG.warning("Skipping %s because it has no frames.", json_path)
            continue

        episode_root = json_path.parent
        info = record.get("info") or {}
        image_info = info.get("image") or {}
        image_width = int(image_info["width"]) if "width" in image_info else None
        image_height = int(image_info["height"]) if "height" in image_info else None
        fps = float(image_info.get("fps", 30.0))
        task = get_instruction(record, instruction_source, fallback=episode_root.name)

        first_state = frame_state_vector(frames[0], state_groups, state_field)
        first_action = frame_action_vector(
            frames,
            0,
            action_groups=action_groups,
            action_field=action_field,
            action_source=action_source,
            state_groups=state_groups,
            state_field=state_field,
        )
        if first_state.size == 0:
            raise ValueError(f"{json_path} produced an empty state vector.")
        if first_action.size == 0:
            raise ValueError(f"{json_path} produced an empty action vector.")

        if state_dim is None:
            state_dim = int(first_state.shape[0])
            state_names = infer_vector_names(info, state_groups, state_field, "state", state_dim)
        elif state_dim != int(first_state.shape[0]):
            raise ValueError(
                f"Inconsistent state dimension in {json_path}: got {first_state.shape[0]}, expected {state_dim}."
            )

        if action_dim is None:
            action_dim = int(first_action.shape[0])
            action_names = infer_vector_names(info, action_groups, action_field, "action", action_dim)
        elif action_dim != int(first_action.shape[0]):
            raise ValueError(
                f"Inconsistent action dimension in {json_path}: got {first_action.shape[0]}, expected {action_dim}."
            )

        for frame in frames:
            colors = frame.get("colors") or {}
            if not isinstance(colors, Mapping):
                continue
            for camera_key, rel_path in colors.items():
                if camera_keys_filter is not None and camera_key not in camera_keys_filter:
                    continue
                discovered_camera_keys.add(camera_key)
                if camera_key in image_sizes:
                    continue
                if not isinstance(rel_path, str) or not rel_path:
                    continue
                image_path = episode_root / rel_path
                if image_path.is_file():
                    image_sizes[camera_key] = tuple(reversed(read_image_size(image_path)))
            if camera_keys_filter is not None and discovered_camera_keys == camera_keys_filter:
                # We still keep scanning other episodes for missing sizes and dimension checks.
                pass

        episodes.append(
            EpisodeInfo(
                json_path=json_path,
                episode_root=episode_root,
                task=task,
                fps=fps,
                frame_count=len(frames),
                image_width=image_width,
                image_height=image_height,
            )
        )

    if not episodes:
        raise ValueError("No valid episodes found under the provided source path.")

    if camera_keys_filter is not None:
        missing = camera_keys_filter.difference(discovered_camera_keys)
        if missing:
            raise ValueError(f"Requested camera keys not found in source data: {sorted(missing)}")
        camera_keys = sorted(camera_keys_filter, key=natural_sort_key)
    else:
        camera_keys = sorted(discovered_camera_keys, key=natural_sort_key)

    if not camera_keys:
        raise ValueError("No usable color camera keys were found in the source data.")

    for camera_key in camera_keys:
        if camera_key not in image_sizes:
            fallback_hw = None
            for episode in episodes:
                if episode.image_height is not None and episode.image_width is not None:
                    fallback_hw = (episode.image_height, episode.image_width)
                    break
            if fallback_hw is None:
                raise ValueError(
                    f"Could not infer image size for camera '{camera_key}'. "
                    "Ensure at least one referenced image exists or source metadata provides width/height."
                )
            image_sizes[camera_key] = fallback_hw

    assert state_dim is not None
    assert action_dim is not None
    assert state_names is not None
    assert action_names is not None
    return episodes, camera_keys, image_sizes, state_dim, action_dim, state_names, action_names


def convert_dataset(args: argparse.Namespace) -> None:
    lerobot_dataset_cls = get_lerobot_dataset_cls()
    json_paths = find_episode_jsons(args.src)
    if not json_paths:
        raise ValueError(f"No data.json files found under {args.src}")

    if args.max_episodes is not None:
        json_paths = json_paths[: args.max_episodes]

    state_groups = parse_csv(args.state_groups)
    action_groups = parse_csv(args.action_groups)
    camera_keys_filter = set(parse_csv(args.camera_keys)) if args.camera_keys else None

    if not state_groups:
        raise ValueError("--state-groups cannot be empty.")
    if not action_groups:
        raise ValueError("--action-groups cannot be empty.")

    episodes, input_camera_keys, input_image_sizes, state_dim, action_dim, state_names, action_names = (
        scan_source_episodes(
            json_paths=json_paths,
            camera_keys_filter=camera_keys_filter,
            state_groups=state_groups,
            state_field=args.state_field,
            action_groups=action_groups,
            action_field=args.action_field,
            action_source=args.action_source,
            instruction_source=args.instruction_source,
        )
    )
    dataset_fps = coerce_dataset_fps((episode.fps for episode in episodes), args.fps)

    camera_name_map = choose_camera_feature_names(input_camera_keys, args.preserve_camera_names)
    feature_image_sizes = {
        camera_name_map[input_key]: input_image_sizes[input_key] for input_key in input_camera_keys
    }

    features: dict[str, Any] = {}
    for feature_name, (height, width) in feature_image_sizes.items():
        features[feature_name] = {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    features["state"] = {
        "dtype": "float",
        "shape": (state_dim,),
        "names": state_names,
    }
    features["actions"] = {
        "dtype": "float",
        "shape": (action_dim,),
        "names": action_names,
    }

    repo_id = args.repo_id or args.dst.name
    LOG.info("Creating LeRobot dataset at %s", args.dst)
    dataset = lerobot_dataset_cls.create(
        repo_id=repo_id,
        root=args.dst,
        robot_type=args.robot_type,
        fps=dataset_fps,
        features=features,
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
    )

    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    episode_summaries: list[dict[str, Any]] = []

    try:
        for episode_idx, episode in enumerate(episodes):
            record = load_json(episode.json_path)
            frames = record.get("data") or []
            frames_written = 0
            skipped_frames = 0

            for frame_pos, frame in enumerate(frames):
                try:
                    state_vec = frame_state_vector(frame, state_groups, args.state_field)
                    action_vec = frame_action_vector(
                        frames,
                        frame_pos,
                        action_groups=action_groups,
                        action_field=args.action_field,
                        action_source=args.action_source,
                        state_groups=state_groups,
                        state_field=args.state_field,
                    )
                    if state_vec.shape != (state_dim,):
                        raise ValueError(
                            f"State dim mismatch at frame {frame_pos}: got {state_vec.shape}, expected {(state_dim,)}"
                        )
                    if action_vec.shape != (action_dim,):
                        raise ValueError(
                            f"Action dim mismatch at frame {frame_pos}: got {action_vec.shape}, expected {(action_dim,)}"
                        )

                    colors = frame.get("colors") or {}
                    if not isinstance(colors, Mapping):
                        raise ValueError(f"Invalid colors payload at frame {frame_pos}: {colors!r}")

                    frame_payload: dict[str, Any] = {
                        "state": state_vec,
                        "actions": action_vec,
                    }
                    for input_camera_key in input_camera_keys:
                        feature_name = camera_name_map[input_camera_key]
                        target_hw = feature_image_sizes[feature_name]
                        rel_path = colors.get(input_camera_key)
                        if isinstance(rel_path, str) and rel_path:
                            image_path = episode.episode_root / rel_path
                            if image_path.is_file():
                                frame_payload[feature_name] = load_rgb_image(image_path, target_hw)
                            else:
                                LOG.debug(
                                    "Missing image for %s frame %d camera %s: %s",
                                    episode.json_path,
                                    frame_pos,
                                    input_camera_key,
                                    image_path,
                                )
                                frame_payload[feature_name] = make_blank_image(target_hw)
                        else:
                            frame_payload[feature_name] = make_blank_image(target_hw)

                    timestamp = frame.get("timestamp")
                    if timestamp is None:
                        source_index = frame.get("idx", frame_pos)
                        timestamp = float(source_index) / float(dataset_fps)
                    else:
                        timestamp = float(timestamp)

                    dataset.add_frame(frame_payload, task=episode.task, timestamp=timestamp)
                    all_states.append(state_vec)
                    all_actions.append(action_vec)
                    frames_written += 1
                except Exception as exc:
                    if args.skip_invalid_frames:
                        skipped_frames += 1
                        LOG.warning(
                            "Skipping invalid frame %d in %s: %s",
                            frame_pos,
                            episode.json_path,
                            exc,
                        )
                        continue
                    raise

            if frames_written == 0:
                LOG.warning("Skipping episode %s because no valid frames were written.", episode.json_path)
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            episode_summaries.append(
                {
                    "episode_index": episode_idx,
                    "source_json": str(episode.json_path),
                    "task": episode.task,
                    "frames_total": len(frames),
                    "frames_written": frames_written,
                    "frames_skipped": skipped_frames,
                }
            )
            LOG.info(
                "Converted episode %d/%d: %s (%d frames written, %d skipped)",
                episode_idx + 1,
                len(episodes),
                episode.json_path,
                frames_written,
                skipped_frames,
            )
    finally:
        dataset.stop_image_writer()

    if not all_states or not all_actions:
        raise ValueError("No frames were converted successfully; output dataset is empty.")

    states_array = np.stack(all_states, axis=0)
    actions_array = np.stack(all_actions, axis=0)
    stats = {
        "state": build_stats(states_array),
        "actions": build_stats(actions_array),
    }

    stats_path = args.dst / "meta" / "stats.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)

    summary = {
        "source": str(args.src.resolve()),
        "destination": str(args.dst.resolve()),
        "repo_id": repo_id,
        "robot_type": args.robot_type,
        "dataset_fps": dataset_fps,
        "episodes_converted": len(episode_summaries),
        "frames_converted": int(states_array.shape[0]),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "state_groups": state_groups,
        "action_groups": action_groups,
        "state_field": args.state_field,
        "action_field": args.action_field,
        "action_source": args.action_source,
        "instruction_source": args.instruction_source,
        "camera_name_map": camera_name_map,
        "feature_image_sizes": {
            key: {"height": hw[0], "width": hw[1]} for key, hw in feature_image_sizes.items()
        },
        "episodes": episode_summaries,
    }
    summary_path = args.dst / "meta" / "g1_conversion_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    LOG.info("Wrote aggregate stats to %s", stats_path)
    LOG.info("Wrote conversion summary to %s", summary_path)
    LOG.info(
        "Done. Dataset has %d episodes and %d frames.",
        len(episode_summaries),
        int(states_array.shape[0]),
    )


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    if args.dst.exists():
        raise FileExistsError(
            f"Destination already exists: {args.dst}. "
            "Please choose a fresh output directory for the LeRobot dataset."
        )
    convert_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

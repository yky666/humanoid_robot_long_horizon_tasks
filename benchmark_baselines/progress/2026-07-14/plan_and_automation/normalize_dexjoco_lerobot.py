#!/usr/bin/env python3
"""Create schema-compatible DexJoCo task views without duplicating videos."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ACTION_DIM = 44
STATE_DIM = 46
VIDEO_KEYS = ("observation.images.ego", "observation.images.wrist_left", "observation.images.wrist_right")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=4) + "\n", encoding="utf-8")


def pad_stat(stat: dict, old_dim: int, new_dim: int) -> dict:
    out = copy.deepcopy(stat)
    for name, value in out.items():
        if name == "count" or not isinstance(value, list) or len(value) != old_dim:
            continue
        out[name] = value + [0.0] * (new_dim - old_dim)
    return out


def padded_list_column(column: pa.ChunkedArray, target: int) -> pa.FixedSizeListArray:
    values = np.asarray(column.combine_chunks().to_pylist(), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] > target:
        raise ValueError(f"cannot pad shape {values.shape} to {target}")
    values = np.pad(values, ((0, 0), (0, target - values.shape[1])))
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), target)


def mask_column(rows: int, active: int, target: int) -> pa.FixedSizeListArray:
    mask = np.zeros((rows, target), dtype=np.float32)
    mask[:, :active] = 1.0
    return pa.FixedSizeListArray.from_arrays(pa.array(mask.reshape(-1)), target)


def camera_sources(features: dict) -> dict[str, str]:
    existing = {k for k, v in features.items() if v.get("dtype") == "video"}
    ego = next((k for k in ("observation.images.ego", "observation.images.ego_right", "observation.images.front") if k in existing), None)
    left = next((k for k in ("observation.images.wrist_left", "observation.images.wrist") if k in existing), None)
    right = next((k for k in ("observation.images.wrist_right", "observation.images.wrist") if k in existing), None)
    if not all((ego, left, right)):
        raise ValueError(f"unsupported camera schema: {sorted(existing)}")
    return dict(zip(VIDEO_KEYS, (ego, left, right)))


def normalize_task(src: Path, dst: Path) -> dict:
    info = read_json(src / "meta/info.json")
    features = info["features"]
    action_dim = int(features["action"]["shape"][0])
    state_dim = int(features["observation.state"]["shape"][0])
    cams = camera_sources(features)

    if dst.exists():
        raise FileExistsError(dst)
    (dst / "data/chunk-000").mkdir(parents=True)
    shutil.copytree(src / "meta", dst / "meta", dirs_exist_ok=True)

    for parquet in sorted((src / "data").glob("**/*.parquet")):
        table = pq.read_table(parquet)
        for key, target in (("action", ACTION_DIM), ("observation.state", STATE_DIM)):
            idx = table.schema.get_field_index(key)
            table = table.set_column(idx, key, padded_list_column(table[key], target))
        table = table.append_column("action_mask", mask_column(len(table), action_dim, ACTION_DIM))
        table = table.append_column("observation.state_mask", mask_column(len(table), state_dim, STATE_DIM))
        rel = parquet.relative_to(src / "data")
        out = dst / "data" / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out, compression="zstd")

    video_features = {k: copy.deepcopy(features[v]) for k, v in cams.items()}
    info["features"] = {k: v for k, v in features.items() if v.get("dtype") != "video"}
    info["features"].update(video_features)
    info["features"]["action"]["shape"] = [ACTION_DIM]
    info["features"]["observation.state"]["shape"] = [STATE_DIM]
    info["features"]["action_mask"] = {"dtype": "float32", "shape": [ACTION_DIM]}
    info["features"]["observation.state_mask"] = {"dtype": "float32", "shape": [STATE_DIM]}
    write_json(dst / "meta/info.json", info)

    stats = read_json(src / "meta/stats.json")
    stats["action"] = pad_stat(stats["action"], action_dim, ACTION_DIM)
    stats["observation.state"] = pad_stat(stats["observation.state"], state_dim, STATE_DIM)
    for new_key, old_key in cams.items():
        stats[new_key] = copy.deepcopy(stats[old_key])
    for key in list(stats):
        if key.startswith("observation.images.") and key not in VIDEO_KEYS:
            del stats[key]
    mask_stats = {
        "min": [1.0] * action_dim + [0.0] * (ACTION_DIM - action_dim),
        "max": [1.0] * action_dim + [0.0] * (ACTION_DIM - action_dim),
        "mean": [1.0] * action_dim + [0.0] * (ACTION_DIM - action_dim),
        "std": [0.0] * ACTION_DIM,
        "count": [info["total_frames"]],
    }
    stats["action_mask"] = mask_stats
    stats["observation.state_mask"] = {
        **mask_stats,
        "min": [1.0] * state_dim + [0.0] * (STATE_DIM - state_dim),
        "max": [1.0] * state_dim + [0.0] * (STATE_DIM - state_dim),
        "mean": [1.0] * state_dim + [0.0] * (STATE_DIM - state_dim),
        "std": [0.0] * STATE_DIM,
    }
    write_json(dst / "meta/stats.json", stats)

    for new_key, old_key in cams.items():
        link = dst / "videos" / new_key
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink((src / "videos" / old_key).resolve(), link, target_is_directory=True)

    return {"task": src.name, "frames": info["total_frames"], "episodes": info["total_episodes"],
            "source_action_dim": action_dim, "source_state_dim": state_dim, "cameras": cams}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    if any(args.destination.iterdir()):
        raise SystemExit(f"destination must be empty: {args.destination}")
    manifest = [normalize_task(src, args.destination / src.name) for src in sorted(args.source.iterdir()) if (src / "meta/info.json").is_file()]
    if len(manifest) != 11:
        raise SystemExit(f"expected 11 tasks, got {len(manifest)}")
    write_json(args.destination / "manifest.json", manifest)
    print(json.dumps({"tasks": len(manifest), "frames": sum(x["frames"] for x in manifest), "episodes": sum(x["episodes"] for x in manifest)}, indent=2))


if __name__ == "__main__":
    main()

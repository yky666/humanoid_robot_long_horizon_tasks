#!/usr/bin/env python

"""Merge single-episode GR00T LeRobot datasets into one multi-episode dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from gr00t.data.stats import generate_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dataset-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("input_dataset_dirs", type=Path, nargs="+")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=True)


def read_first_jsonl(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def main() -> None:
    args = parse_args()
    output_dir = args.output_dataset_dir
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_dir}")
        shutil.rmtree(output_dir)

    input_dirs = args.input_dataset_dirs
    base_info = read_json(input_dirs[0] / "meta" / "info.json")
    base_modality = read_json(input_dirs[0] / "meta" / "modality.json")
    video_keys = [k for k in base_info["features"] if k.startswith("observation.images.")]

    (output_dir / "meta").mkdir(parents=True)
    (output_dir / "data" / "chunk-000").mkdir(parents=True)

    episodes = []
    tasks_by_text: dict[str, int] = {}
    total_frames = 0

    for episode_index, src in enumerate(input_dirs):
        info = read_json(src / "meta" / "info.json")
        if info["features"]["action"]["shape"] != base_info["features"]["action"]["shape"]:
            raise ValueError(f"Action shape mismatch for {src}")
        if info["features"]["observation.state"]["shape"] != base_info["features"]["observation.state"]["shape"]:
            raise ValueError(f"State shape mismatch for {src}")

        src_episode = read_first_jsonl(src / "meta" / "episodes.jsonl")
        src_task = read_first_jsonl(src / "meta" / "tasks.jsonl")["task"]
        task_index = tasks_by_text.setdefault(src_task, len(tasks_by_text))
        length = int(info["total_frames"])

        df = pd.read_parquet(src / "data" / "chunk-000" / "episode_000000.parquet")
        if len(df) != length:
            raise ValueError(f"Length mismatch for {src}: info={length}, parquet={len(df)}")
        df = df.copy()
        df["episode_index"] = episode_index
        df["task_index"] = task_index
        df["index"] = range(total_frames, total_frames + length)
        df.to_parquet(
            output_dir / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet",
            index=False,
        )

        episodes.append(
            {
                "episode_index": episode_index,
                "tasks": src_episode.get("tasks", [src_task]),
                "length": length,
            }
        )
        total_frames += length

        for video_key in video_keys:
            src_video = src / "videos" / "chunk-000" / video_key / "episode_000000.mp4"
            dst_video = (
                output_dir
                / "videos"
                / "chunk-000"
                / video_key
                / f"episode_{episode_index:06d}.mp4"
            )
            dst_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_video, dst_video)

    info = dict(base_info)
    info["total_episodes"] = len(input_dirs)
    info["total_frames"] = total_frames
    info["total_tasks"] = len(tasks_by_text)
    info["splits"] = {"train": f"0:{len(input_dirs)}"}
    info["total_chunks"] = 1
    info["total_videos"] = len(input_dirs) * len(video_keys)

    write_json(output_dir / "meta" / "info.json", info)
    write_json(output_dir / "meta" / "modality.json", base_modality)

    with (output_dir / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for row in episodes:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    tasks_by_index = sorted((idx, task) for task, idx in tasks_by_text.items())
    with (output_dir / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for task_index, task in tasks_by_index:
            f.write(json.dumps({"task_index": task_index, "task": task}, ensure_ascii=True) + "\n")

    generate_stats(output_dir)
    print(f"Merged {len(input_dirs)} episodes, {total_frames} frames -> {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fast metadata-only audit for the unified multi-task experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit_lerobot_root(root: Path) -> dict:
    records = []
    if not root.is_dir():
        return {"root": str(root), "exists": False, "datasets": []}
    for child in sorted(root.iterdir()):
        info_path = child / "meta" / "info.json"
        tasks_path = child / "meta" / "tasks.jsonl"
        if not info_path.is_file():
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        features = info.get("features", {})
        records.append(
            {
                "name": child.name,
                "episodes": info.get("total_episodes"),
                "frames": info.get("total_frames"),
                "action_dim": features.get("action", {}).get("shape", [None])[0],
                "state_dim": features.get("observation.state", {}).get("shape", [None])[0],
                "camera_keys": sorted(k for k in features if "images" in k),
                "has_tasks": tasks_path.is_file(),
            }
        )
    return {"root": str(root), "exists": True, "datasets": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dexjoco-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_lerobot_root(args.dexjoco_root)
    report["valid_task_count"] = len(report["datasets"])
    report["total_episodes"] = sum(x["episodes"] or 0 for x in report["datasets"])
    report["total_frames"] = sum(x["frames"] or 0 for x in report["datasets"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

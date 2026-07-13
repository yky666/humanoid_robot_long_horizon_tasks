#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
tasks = sorted(p for p in root.iterdir() if (p / "meta/info.json").is_file())
assert len(tasks) == 11, len(tasks)
frames = episodes = 0
for task in tasks:
    info = json.loads((task / "meta/info.json").read_text())
    assert info["features"]["action"]["shape"] == [44]
    assert info["features"]["observation.state"]["shape"] == [46]
    assert info["features"]["action_mask"]["shape"] == [44]
    assert info["features"]["observation.state_mask"]["shape"] == [46]
    for key in ("observation.images.ego", "observation.images.wrist_left", "observation.images.wrist_right"):
        assert key in info["features"]
        assert (task / "videos" / key).resolve().is_dir()
    parquet = next((task / "data").glob("**/*.parquet"))
    table = pq.read_table(parquet, columns=["action", "observation.state", "action_mask", "observation.state_mask"])
    assert len(table["action"][0].as_py()) == 44
    assert len(table["observation.state"][0].as_py()) == 46
    assert len(table["action_mask"][0].as_py()) == 44
    assert len(table["observation.state_mask"][0].as_py()) == 46
    frames += info["total_frames"]
    episodes += info["total_episodes"]
assert frames == 523763, frames
assert episodes == 1100, episodes
print(json.dumps({"status": "valid", "tasks": 11, "frames": frames, "episodes": episodes, "action_dim": 44, "state_dim": 46}))

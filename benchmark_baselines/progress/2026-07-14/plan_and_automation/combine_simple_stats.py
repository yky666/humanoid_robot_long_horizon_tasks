#!/usr/bin/env python3
"""Combine per-task bounds for one SIMPLE multi-task checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.root.glob("*/meta/stats_psi0.json"))
    if len(paths) < 2:
        raise SystemExit(f"expected a multi-task collection, found {len(paths)} stats files")
    docs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    result = {}
    for key in ("action", "states"):
        mins = [doc[key]["min"] for doc in docs]
        maxs = [doc[key]["max"] for doc in docs]
        if len({len(x) for x in mins + maxs}) != 1:
            raise SystemExit(f"incompatible {key} dimensions across SIMPLE tasks")
        result[key] = {
            "min": [min(values) for values in zip(*mins)],
            "max": [max(values) for values in zip(*maxs)],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"combined {len(paths)} task statistics into {args.output}")


if __name__ == "__main__":
    main()


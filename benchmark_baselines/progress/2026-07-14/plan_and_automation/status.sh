#!/usr/bin/env bash
set -u
ROOT=/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/Humanoid/yangky
echo '=== GPU ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader
echo '=== experiment processes ==='
ps -u "$(id -u)" -o pid,etime,pcpu,pmem,cmd |
  grep -E 'uv sync|bootstrap_assets|torchrun|launch_train|train.py' |
  grep -v grep || true
echo '=== bootstrap ==='
cat "$ROOT/baseline_experiments/state/bootstrap_assets.status" 2>/dev/null || echo pending
tail -20 "$ROOT/baseline_experiments/logs/bootstrap_assets.log" 2>/dev/null || true

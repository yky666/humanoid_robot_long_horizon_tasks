#!/usr/bin/env bash
set -uo pipefail
pid="${1:?training pid required}"
ROOT=/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/Humanoid/yangky
STATE="$ROOT/baseline_experiments/state/psi0_simple.status"
LOG="$ROOT/baseline_experiments/logs/psi0_simple_full.log"

while kill -0 "$pid" 2>/dev/null; do
  sleep 60
done

if grep -q "Happy Ending" "$LOG"; then
  echo full_complete >"$STATE"
  exit 0
fi

if grep -q "CUDA out of memory" "$LOG"; then
  echo full_retry_oom >"$STATE"
  retry_log="$ROOT/baseline_experiments/logs/psi0_simple_full_retry_oom.log"
  if BATCH_SIZE=8 GRAD_ACCUM=2 bash "$ROOT/baseline_experiments/run_psi0_simple_multitask.sh" full >>"$retry_log" 2>&1; then
    echo full_complete >"$STATE"
  else
    echo full_failed >"$STATE"
    exit 1
  fi
else
  echo full_failed >"$STATE"
  exit 1
fi

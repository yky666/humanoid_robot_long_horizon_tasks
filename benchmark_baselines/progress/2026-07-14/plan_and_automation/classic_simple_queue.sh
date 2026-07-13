#!/usr/bin/env bash
set -uo pipefail
ROOT=/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/Humanoid/yangky
STATE="$ROOT/baseline_experiments/state"
LOGS="$ROOT/baseline_experiments/logs"
RUN="$ROOT/baseline_experiments/run_classic_simple_multitask.sh"

while [[ "$(cat "$STATE/psi0_simple.status" 2>/dev/null || true)" != full_complete ]]; do
  sleep 60
done

for model in act diffusion-policy; do
  key="${model//-/_}_simple"
  echo smoke_running >"$STATE/$key.status"
  if ! bash "$RUN" "$model" smoke >>"$LOGS/${key}_smoke.log" 2>&1; then
    echo smoke_failed >"$STATE/$key.status"
    continue
  fi
  echo smoke_passed >"$STATE/$key.status"
  echo full_running >"$STATE/$key.status"
  if bash "$RUN" "$model" full >>"$LOGS/${key}_full.log" 2>&1; then
    echo full_complete >"$STATE/$key.status"
  elif grep -q "CUDA out of memory" "$LOGS/${key}_full.log"; then
    echo full_retry_oom >"$STATE/$key.status"
    if BATCH_SIZE=16 GRAD_ACCUM=2 bash "$RUN" "$model" full >>"$LOGS/${key}_full_retry_oom.log" 2>&1; then
      echo full_complete >"$STATE/$key.status"
    else
      echo full_failed >"$STATE/$key.status"
    fi
  else
    echo full_failed >"$STATE/$key.status"
  fi
done

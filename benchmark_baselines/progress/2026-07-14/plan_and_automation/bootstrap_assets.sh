#!/usr/bin/env bash
set -uo pipefail

ROOT=/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/Humanoid/yangky
PSI_ROOT="$ROOT/Psi0"
STATE="$ROOT/baseline_experiments/state"
LOGS="$ROOT/baseline_experiments/logs"
HF=/HOME/sysu_xdliang/sysu_xdliang_1/.local/bin/hf
mkdir -p "$STATE" "$LOGS" "$PSI_ROOT/data"

exec >>"$LOGS/bootstrap_assets.log" 2>&1
echo "[$(date -Iseconds)] bootstrap started"

# Do not compete with an environment build already in progress.
while pgrep -u "$(id -u)" -f 'uv sync' >/dev/null; do
  echo "[$(date -Iseconds)] waiting for uv sync"
  sleep 30
done

proxy="$(git config --global --get http.proxy || true)"
canonical=(
  G1WholebodyBendHandoverTeleop-v0.zip
  G1WholebodyBendPickAndPlaceTeleop-v0.zip
  G1WholebodyBendPickMP-v0.zip
  G1WholebodyBendPickSimToRealTeleop-v0.zip
  G1WholebodyBendPickTeleop-v0.zip
  G1WholebodyCloseDoorTeleop-v0.zip
  G1WholebodyHandoverTeleop-v0.zip
  G1WholebodyLocomotionPickBetweenTablesTeleop-v0.zip
  G1WholebodyOpenFaucetTeleop-v0.zip
  G1WholebodyOpenOvenTeleop-v0.zip
  G1WholebodyOpenTrashCanTeleop-v0.zip
  G1WholebodyPickAndPlaceAndHugContainerTeleop-v0.zip
  G1WholebodyPushOfficeChairTeleop-v0.zip
  G1WholebodyTabletopGraspMP-v0.zip
  G1WholebodyXMoveBendPickMP-v0.zip
  G1WholebodyXMoveBendPickTeleop-v0.zip
  G1WholebodyXMovePickTeleop-v0.zip
)
include_args=()
for archive in "${canonical[@]}"; do
  include_args+=(--include "simple/$archive")
done
download_ok=0
for delay in 0 15 30 60 120 240; do
  sleep "$delay"
  echo "[$(date -Iseconds)] SIMPLE data download attempt after ${delay}s"
  if env HF_ENDPOINT=https://hf-mirror.com HTTP_PROXY="$proxy" HTTPS_PROXY="$proxy" \
      "$HF" download USC-PSI-Lab/psi-data \
      --repo-type dataset "${include_args[@]}" \
      --local-dir "$PSI_ROOT/data"; then
    download_ok=1
    break
  fi
done

if [[ "$download_ok" != 1 ]]; then
  echo "[$(date -Iseconds)] SIMPLE data download exhausted retries"
  echo download_blocked >"$STATE/bootstrap_assets.status"
  exit 1
fi

find "$PSI_ROOT/data/simple" -maxdepth 1 -type f -name '*.zip' -print0 |
  while IFS= read -r -d '' archive; do
    echo "[$(date -Iseconds)] extracting $archive"
    unzip -n -q "$archive" -d "$PSI_ROOT/data/simple"
  done

find "$PSI_ROOT/data/simple" -type f -path '*/meta/info.json' -print |
  sort >"$STATE/simple_datasets.txt"
count="$(wc -l <"$STATE/simple_datasets.txt")"
if (( count < 2 )); then
  echo "[$(date -Iseconds)] refusing single-task setup: only $count datasets found"
  echo invalid_multitask_manifest >"$STATE/bootstrap_assets.status"
  exit 1
fi
echo ready >"$STATE/bootstrap_assets.status"
echo "[$(date -Iseconds)] bootstrap completed"

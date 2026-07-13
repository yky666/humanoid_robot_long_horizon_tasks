#!/usr/bin/env bash
set -uo pipefail
ROOT=/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/Humanoid/yangky
PSI_ROOT="$ROOT/Psi0"
STATE="$ROOT/baseline_experiments/state"
LOGS="$ROOT/baseline_experiments/logs"
mkdir -p "$STATE" "$LOGS" "$PSI_ROOT/cache/checkpoints"
exec >>"$LOGS/bootstrap_models.log" 2>&1

while [[ "$(cat "$STATE/bootstrap_assets.status" 2>/dev/null || true)" != ready ]]; do
  echo "[$(date -Iseconds)] waiting for SIMPLE assets"
  sleep 30
done

proxy="$(git config --global --get http.proxy || true)"
repo=https://hf-mirror.com/USC-PSI-Lab/psi-model/resolve/main
pre=psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k
post=psi0/postpre.1by1.pad36.2601131206.ckpt.he30k
files=(
  "$pre/added_tokens.json"
  "$pre/chat_template.jinja"
  "$pre/config.json"
  "$pre/generation_config.json"
  "$pre/merges.txt"
  "$pre/model.safetensors"
  "$pre/preprocessor_config.json"
  "$pre/special_tokens_map.json"
  "$pre/tokenizer.json"
  "$pre/tokenizer_config.json"
  "$pre/video_preprocessor_config.json"
  "$pre/vocab.json"
  "$post/action_header.safetensors"
)

echo "[$(date -Iseconds)] downloading exact checkpoint files (avoids huge repository tree API)"
for rel in "${files[@]}"; do
  dst="$PSI_ROOT/cache/checkpoints/$rel"
  mkdir -p "$(dirname "$dst")"
  curl --fail --location --retry 12 --retry-all-errors --retry-delay 10 \
    --continue-at - --proxy "$proxy" --output "$dst" "$repo/$rel"
done

[[ "$(stat -c %s "$PSI_ROOT/cache/checkpoints/$pre/model.safetensors")" -eq 4262742488 ]]
[[ "$(stat -c %s "$PSI_ROOT/cache/checkpoints/$post/action_header.safetensors")" -eq 1990811480 ]]
echo ready >"$STATE/bootstrap_models.status"

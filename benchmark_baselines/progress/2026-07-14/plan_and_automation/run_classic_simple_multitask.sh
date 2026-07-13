#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:?act or diffusion-policy required}"
MODE="${2:-smoke}"
ROOT=/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/Humanoid/yangky
PSI_ROOT="$ROOT/Psi0"
DATA_ROOT="$PSI_ROOT/data/simple"
STATS="$ROOT/baseline_experiments/state/simple_multitask_stats.json"
cd "$PSI_ROOT"
source .venv-psi/bin/activate

mapfile -t tasks < <(find "$DATA_ROOT" -mindepth 3 -maxdepth 3 -path '*/meta/info.json' -printf '%h\n' |
  sed 's#/meta$##' | xargs -n1 basename | sort -u)
(( ${#tasks[@]} >= 2 )) || { echo "refusing single-task training" >&2; exit 2; }
python "$ROOT/baseline_experiments/combine_simple_stats.py" "$DATA_ROOT" "$STATS"

if [[ "$MODE" == smoke ]]; then
  steps=20; checkpoint_steps=20; batch=8; accum=4
else
  steps=40000; checkpoint_steps=5000; batch="${BATCH_SIZE:-32}"; accum="${GRAD_ACCUM:-1}"
fi

case "$MODEL" in
  act)
    config=simple_act_config
    exp="simple-multitask-act-$MODE"
    model_args=(
      --data.transform.repack.action_chunk_size=16
      --model.chunk-size=16 --model.n-action-steps=16
      --model.action-dim=36 --model.state-dim=36 --model.use-vae --model.kl-weight=10.0
    )
    ;;
  diffusion-policy)
    config=simple_dp_config
    exp="simple-multitask-dp-$MODE"
    model_args=(
      --data.transform.repack.action_chunk_size=16
      --model.action-chunk-size=16 --model.action-dim=36 --model.obs-dim=36
    )
    ;;
  *) echo "unknown model: $MODEL" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16
torchrun --standalone --nproc_per_node=1 scripts/train.py "$config" \
  --seed=2026 --exp="$exp" --train.name="$MODEL-simple-multitask" \
  --log.report-to=wandb --train.data_parallel=ddp --train.mixed_precision=bf16 \
  --train.train-batch-size="$batch" --train.gradient_accumulation_steps="$accum" \
  --train.warmup-steps=10 --train.warmup-ratio=None \
  --train.checkpointing-steps="$checkpoint_steps" --train.validation_steps="$checkpoint_steps" \
  --train.val_num_batches=2 --train.max-training-steps="$steps" \
  --train.learning-rate="${LEARNING_RATE:-1e-4}" --train.max-grad-norm=1.0 \
  --train.lr_scheduler_kwargs.weight_decay=1e-6 \
  --train.lr_scheduler_kwargs.betas 0.95 0.999 --train.lr_scheduler_type=cosine \
  --data.root_dir="$DATA_ROOT" --data.train-repo-ids "${tasks[@]}" --data.val-repo-ids "${tasks[@]}" \
  --data.transform.repack.pad-action-dim=36 --data.transform.repack.pad-state-dim=36 \
  --data.transform.field.stat-path="$STATS" --data.transform.field.stat-action-key=action \
  --data.transform.field.stat-state-key=states --data.transform.field.normalize-state \
  --data.transform.field.action-norm-type=bounds --data.transform.field.pad-action-dim=36 \
  --data.transform.field.pad-state-dim=36 --data.transform.model.img-aug \
  "${model_args[@]}"

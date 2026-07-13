#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-smoke}"
ROOT=/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/Humanoid/yangky
PSI_ROOT="$ROOT/Psi0"
DATA_ROOT="$PSI_ROOT/data/simple"
STATS="$ROOT/baseline_experiments/state/simple_multitask_stats.json"
cd "$PSI_ROOT"
source .venv-psi/bin/activate

mapfile -t tasks < <(find "$DATA_ROOT" -mindepth 3 -maxdepth 3 -path '*/meta/info.json' -printf '%h\n' |
  sed 's#/meta$##' | xargs -n1 basename | sort -u)
if (( ${#tasks[@]} < 2 )); then
  echo "refusing single-task training: found ${#tasks[@]} SIMPLE tasks" >&2
  exit 2
fi
python "$ROOT/baseline_experiments/combine_simple_stats.py" "$DATA_ROOT" "$STATS"

if [[ "$MODE" == smoke ]]; then
  steps=20
  checkpoint_steps=20
  exp=simple-multitask-psi0-smoke
  batch_size=8
  grad_accum=2
else
  steps=40000
  checkpoint_steps=5000
  exp=simple-multitask-psi0
  batch_size="${BATCH_SIZE:-16}"
  grad_accum="${GRAD_ACCUM:-1}"
fi
learning_rate="${LEARNING_RATE:-1e-4}"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
torchrun --standalone --nproc_per_node=1 scripts/train.py \
  finetune_simple_psi0_config \
  --seed=292285 \
  --exp="$exp" \
  --train.name=finetune \
  --train.data_parallel=ddp \
  --train.mixed_precision=bf16 \
  --train.train_batch_size="$batch_size" \
  --train.max_checkpoints_to_keep=3 \
  --train.gradient_accumulation_steps="$grad_accum" \
  --train.learning_rate="$learning_rate" \
  --train.max_training_steps="$steps" \
  --train.warmup_ratio=None \
  --train.warmup_steps=10 \
  --train.checkpointing_steps="$checkpoint_steps" \
  --train.validation_steps="$checkpoint_steps" \
  --train.val_num_batches=2 \
  --train.max_grad_norm=1.0 \
  --train.lr_scheduler_type=cosine \
  --train.lr_scheduler_kwargs.weight_decay=1e-6 \
  --train.lr_scheduler_kwargs.betas 0.95 0.999 \
  --log.report_to=wandb \
  --data.root_dir="$DATA_ROOT" \
  --data.train-repo-ids "${tasks[@]}" \
  --data.val-repo-ids "${tasks[@]}" \
  --data.transform.repack.pad-action-dim=36 \
  --data.transform.repack.pad-state-dim=36 \
  --data.transform.field.stat-path="$STATS" \
  --data.transform.field.stat-action-key=action \
  --data.transform.field.stat-state-key=states \
  --data.transform.field.action_norm_type=bounds \
  --data.transform.field.no-use-norm-mask \
  --data.transform.field.normalize-state \
  --data.transform.field.pad-action-dim=36 \
  --data.transform.field.pad-state-dim=36 \
  --data.transform.model.img-aug \
  --data.transform.model.resize.size 180 320 \
  --data.transform.model.center_crop.size 180 320 \
  --model.model_name_or_path="$PSI_ROOT/cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k" \
  --model.pretrained-action-header-path="$PSI_ROOT/cache/checkpoints/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k" \
  --model.noise-scheduler=flow \
  --model.train-diffusion-steps=1000 \
  --model.n_conditions=0 \
  --model.action-chunk-size=30 \
  --model.action-dim=36 \
  --model.action-exec-horizon=30 \
  --model.observation-horizon=1 \
  --model.odim=36 \
  --model.view_feature_dim=2048 \
  --model.no-tune-vlm \
  --model.no-use_film \
  --model.no-combined_temb \
  --model.rtc \
  --model.max-delay=8

# Managed Import Notes

This folder is the managed snapshot of the local `A1` workspace on this machine.

## Import Metadata

- Import date: `2026-04-22`
- Imported into: `humanoid_robot_long_horizon_tasks`
- Upstream repository: `https://github.com/ATeam-Research/A1.git`
- Upstream base commit: `2042a8635035481c41ba51ecb41f51d83d8be01b`
- Source local path: `/home/sys01/yangky/test/A1`

## Local Changes Included In This Snapshot

- Added managed `a1/data/` code required by the local training and evaluation entrypoints
- Modified `deploy/api_server.py`
- Modified `deploy/deploy.sh`
- Modified `deploy/infer_vla.py`
- Modified `robot_experiments/debug.py`
- Added `configs/datasets/g1_episode_0013.yaml`
- Added `configs/experiments/g1_episode_0013_finetune.yaml`
- Added and updated `scripts/convert_g1_to_lerobot.py`
- Added `scripts/analyze_eval_deltas.py`

## Excluded From Version Control

- `model/` because it contains local model weights and checkpoints
- local runtime outputs such as `outputs/`, `runs/`, and `wandb/`
- local environments such as `.venv/`

## Managed G1 Workflow On `sys01`

This managed snapshot now includes a runnable G1 teleoperation conversion, smoke finetune,
and offline action-error evaluation workflow for:

- source episode: `/home/sys01/yangky/test/A1/data/episode_0013`
- converted dataset: `/home/sys01/yangky/test/A1/data/g1_episode_0013_lerobot`
- task text: `pick up the water bottle and place it to the right.`

### Environment

Use the managed repo as the working directory and reuse the original workstation data/model paths:

```bash
source /home/sys01/miniconda3/etc/profile.d/conda.sh
conda activate a1
cd /home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1

export DATA_DIR=/home/sys01/yangky/test/A1/data
export HF_HOME=/home/sys01/yangky/.cache/huggingface
export PYTHONPATH=/home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1
```

Shell note:

- when using multi-line bash commands, each trailing `\` must be the very last character on the line
- do not leave spaces after `\`, or the next line will be executed as a new shell command
- `DATA_DIR` is required by the A1 data loaders
- `HF_HOME` is recommended, but the managed training entrypoint now falls back to `~/.cache/huggingface` if it is unset

### 1. Convert G1 Teleoperation Data

The converter expects a fresh `--dst` path.

```bash
python scripts/convert_g1_to_lerobot.py \
  --src /home/sys01/yangky/test/A1/data/episode_0013 \
  --dst /home/sys01/yangky/test/A1/data/g1_episode_0013_lerobot \
  --repo-id g1_episode_0013_lerobot \
  --robot-type unitree_g1 \
  --fps 30 \
  --camera-keys color_0 \
  --state-groups left_arm,right_arm,left_ee \
  --action-groups left_arm,right_arm,left_ee \
  --state-field qpos \
  --action-field qpos \
  --action-source actions \
  --task-override "pick up the water bottle and place it to the right."
```

Current managed assumptions:

- `color_0 -> image`
- `state = left_arm + right_arm + left_ee`, total `16D`
- `actions = left_arm + right_arm + left_ee`, total `16D`
- `right_ee` and `body` are not included in the current managed dataset because this episode uses them as empty or placeholder signals

### 2. Managed Dataset Config

The managed finetune config is:

- dataset config: `configs/datasets/g1_episode_0013.yaml`
- experiment config: `configs/experiments/g1_episode_0013_finetune.yaml`

Important constraint:

- the converted G1 dataset is `16D`, but the local A1 pretrained checkpoint still expects `fixed_action_dim=32` and `num_actions_chunk=50`
- the managed wrapper pads `state` and `actions` to `32D` at train and deploy time to match the checkpoint
- use `normalization_type: bounds` for this dataset; `bounds_q99` is not available from the current LeRobot stats produced here

### 3. Smoke Finetune On Dual RTX 4090

Local pretrained checkpoint used in this workflow:

- `/home/sys01/yangky/test/A1/model/a1-pretrain/latest-unsharded`

Working 1-step smoke finetune command:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m launch_scripts.train_vla qwen2_7b \
  --checkpoint /home/sys01/yangky/test/A1/model/a1-pretrain/latest-unsharded \
  --vla_config_path g1_episode_0013_finetune.yaml \
  --dataset g1_episode_0013_train \
  --global_batch_size 2 \
  --device_train_microbatch_size 1 \
  --train_steps 1 \
  --save_interval 1 \
  --save_interval_unsharded 1 \
  --wandb_debug \
  --num_workers 0 \
  --log_interval 1 \
  --max_crops 1 \
  save_folder=/home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/outputs/g1_episode_0013_smoke_1step
```

Artifacts produced by the successful 1-step smoke run:

- `outputs/g1_episode_0013_smoke_1step/step1/`
- `outputs/g1_episode_0013_smoke_1step/step1-unsharded/`
- `outputs/g1_episode_0013_smoke_1step/step1-action-head/`

Longer 20-step smoke command used for capacity probing:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m launch_scripts.train_vla qwen2_7b \
  --checkpoint /home/sys01/yangky/test/A1/model/a1-pretrain/latest-unsharded \
  --vla_config_path g1_episode_0013_finetune.yaml \
  --dataset g1_episode_0013_train \
  --global_batch_size 2 \
  --device_train_microbatch_size 1 \
  --train_steps 20 \
  --save_interval 10 \
  --save_interval_unsharded 10 \
  --wandb_debug \
  --num_workers 0 \
  --log_interval 1 \
  --max_crops 1 \
  save_folder=/home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/outputs/g1_episode_0013_smoke_2gpu_bounds_cachefix
```

Observed behavior on this workstation:

- dual `RTX 4090` is enough to complete model init, FSDP wrapping, and step `1`
- the `20-step` smoke run still OOMs on step `2` backward with the local 7B checkpoint
- if you want more than a smoke finetune on this machine, the next things to try are more aggressive activation/memory tuning, a smaller checkpoint, or multi-node / higher-memory GPUs

Managed code fixes added after this first smoke pass:

- `launch_scripts/train_vla.py` now supports `--fsdp_precision` and `--disable_float32_attention`
- `launch_scripts/utils.py` no longer crashes when `HF_HOME` is unset during CLI startup
- `scripts/train_for_action.py` now supports filtered checkpoint loading, so compatible backbone weights can still be reused if you experiment with a resized action head

### 3B. Working Multi-Step Finetune On Dual RTX 4090

The most reliable managed command on this workstation keeps the original checkpoint
architecture and only changes the memory behavior:

```bash
export DATA_DIR=/home/sys01/yangky/test/A1/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m launch_scripts.train_vla qwen2_7b \
  --checkpoint /home/sys01/yangky/test/A1/model/a1-pretrain/latest-unsharded \
  --vla_config_path g1_episode_0013_finetune.yaml \
  --dataset g1_episode_0013_train \
  --global_batch_size 2 \
  --device_train_microbatch_size 1 \
  --train_steps 5 \
  --save_interval 5 \
  --save_interval_unsharded 5 \
  --wandb_debug \
  --num_workers 0 \
  --log_interval 1 \
  --max_crops 1 \
  --seq_len 512 \
  --fsdp_precision pure \
  --disable_float32_attention \
  save_folder=/home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/outputs/g1_episode_0013_multistep_original_test_v1
```

Artifacts produced by the successful 5-step run:

- `outputs/g1_episode_0013_multistep_original_test_v1/step5/`
- `outputs/g1_episode_0013_multistep_original_test_v1/step5-unsharded/`
- `outputs/g1_episode_0013_multistep_original_test_v1/step5-action-head/`

Observed behavior of this managed multi-step path:

- the original 7B checkpoint architecture still fits on dual `RTX 4090` when using `--fsdp_precision pure`, `--disable_float32_attention`, and `--seq_len 512`
- the run completed `5/5` training steps and saved all checkpoint variants successfully
- peak GPU memory after FSDP wrapping was about `19.6 GB`
- step losses observed in this 5-step run:
  - `step 1`: `train/ActionNoiseL2Loss=0.6249`
  - `step 2`: `train/ActionNoiseL2Loss=0.4178`
  - `step 3`: `train/ActionNoiseL2Loss=1.5530`
  - `step 4`: `train/ActionNoiseL2Loss=3.2430`
  - `step 5`: `train/ActionNoiseL2Loss=2.1280`

What did not work as a first-choice workaround on this workstation:

- shrinking the flow-matching action head and partially loading the checkpoint is technically supported now
- but on dual `RTX 4090`, those resized-action-head experiments still hit FSDP wrap-time OOM before training starts
- so the recommended managed path remains: keep the original checkpoint shape and change only the training memory settings

### 3C. What The Current 50000-Step Run Is Actually Doing

The active long run on this workstation is:

- run dir: `outputs/g1_episode_0013_multistep_original_test_v1_50000_steps`
- command shape: `train_steps=50000`, `log_interval=100`, `save_interval=100`, `save_interval_unsharded=100`

Important clarification:

- this run did not freeze before training
- it reached `step=100`
- it logged `train/ActionNoiseL2Loss=0.8041`
- it saved `step100`, `step100-unsharded`, and `step100-action-head`

Why it looked stuck:

- with `log_interval=100`, the first train metric does not appear until `step 100`
- on this machine, `100` steps took about `29.5` minutes
- the first checkpoint cycle then spent about another `2.2` minutes saving artifacts
- at the current observed speed, a `50000-step` run is roughly a `10` day job even before network hiccups or manual interruptions

Why this is not the best first path to a real-robot result:

- the dataset is still only one episode, so `50000` steps mostly buys deeper same-episode overfit rather than better task generalization
- saving full checkpoints every `100` steps is extremely expensive on disk and time
- the current `step100` save already produced about `34 GB` sharded + `32 GB` unsharded + `1.7 GB` action-head artifacts

Follow-up observation on `2026-04-27`:

- because `save_num_checkpoints_to_keep=1`, the earlier `step100` checkpoint was automatically rotated out
- the same long run later advanced to `step900`
- the latest saved unsharded checkpoint is now `outputs/g1_episode_0013_multistep_original_test_v1_50000_steps/step900-unsharded`
- if you want to evaluate "the current long run", use the latest retained checkpoint rather than the original `step100`

Recommended long-run monitoring command:

```bash
python scripts/inspect_training_run.py \
  --run-dir /home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/outputs/g1_episode_0013_multistep_original_test_v1_50000_steps \
  --json-out /home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/outputs/g1_episode_0013_multistep_original_test_v1_50000_steps/summary.json \
  --csv-out /home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/outputs/g1_episode_0013_multistep_original_test_v1_50000_steps/metrics.csv \
  --plot-out /home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/outputs/g1_episode_0013_multistep_original_test_v1_50000_steps/training_curve.svg
```

This helper reads the local `output.log` directly, so it still works when WandB only
shows system metrics or is temporarily desynced.

### 3D. Recommended Evolution Route Toward G1 Real-Robot Validation

The most reliable progression on this workstation is staged, not one giant `50000-step`
run on a single episode.

#### Stage A. Same-Episode Overfit Check

Goal:

- verify that the training path is stable
- verify that offline action error continues to go down on the same episode
- select a checkpoint family that is worth carrying forward

Recommended command:

```bash
export DATA_DIR=/home/sys01/yangky/test/A1/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY=...

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m launch_scripts.train_vla qwen2_7b \
  --checkpoint /home/sys01/yangky/test/A1/model/a1-pretrain/latest-unsharded \
  --vla_config_path g1_episode_0013_finetune.yaml \
  --dataset g1_episode_0013_train \
  --global_batch_size 2 \
  --device_train_microbatch_size 1 \
  --train_steps 300 \
  --save_interval 300 \
  --save_interval_unsharded 300 \
  --save_interval_action_head 100 \
  --wandb_entity kaiyuanyang666-sun-yat \
  --wandb_project a1-vla-camd \
  --wandb_run_name g1-episode0013-stageA-overfit300 \
  --num_workers 0 \
  --log_interval 10 \
  --max_crops 1 \
  --seq_len 512 \
  --fsdp_precision pure \
  --disable_float32_attention \
  save_folder=/home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/outputs/g1_episode_0013_stageA_overfit300
```

Why this is the recommended first stage:

- `log_interval=10` surfaces train metrics in the first few minutes instead of after half an hour
- full checkpoints only happen at the end of the stage
- action-head-only checkpoints can still be saved more frequently if you want a lighter resume point

#### Stage B. Same-Task Multi-Episode Finetune

Only after Stage A is stable:

- add more G1 teleoperation episodes for the same bottle-pick-and-place task
- include variation in bottle pose, start pose, and right-side placement target
- hold out a few episodes for offline evaluation instead of training on everything

Suggested first multi-episode target:

- `5-20` episodes of the same task family
- `1000-3000` finetune steps before reassessing

Current same-task bimanual handover dataset update on `2026-05-09`:

- task text: `left hand grasps the water bottle, hands it over to the right hand, then the right hand inserts it into the cup.`
- checked A1 LeRobot episodes:
  - `g1_episode_0030_lerobot`: `102` frames, `state_dim=26`, `action_dim=26`
  - `g1_episode_0031_lerobot`: `973` frames, `state_dim=26`, `action_dim=26`
  - `g1_episode_0038_lerobot`: `647` frames, `state_dim=26`, `action_dim=26`
  - `g1_episode_0040_lerobot`: `854` frames, `state_dim=26`, `action_dim=26`
- checked GR00T psi0 episodes:
  - `g1_episode_0030_psi0_gr00t`: `102` frames, `observation.state=[32]`, `action=[36]`, `3` videos
  - `g1_episode_0031_psi0_gr00t`: `973` frames, `observation.state=[32]`, `action=[36]`, `3` videos
  - `g1_episode_0038_psi0_gr00t`: `647` frames, `observation.state=[32]`, `action=[36]`, `3` videos
  - `g1_episode_0040_psi0_gr00t`: `854` frames, `observation.state=[32]`, `action=[36]`, `3` videos
- `episode_0036` was not converted or trained because
  `/home/sys01/yangky/test/A1/data/episode_0036/data.json` is truncated at
  `786432` bytes and fails JSON parsing near line `35861`.
- `configs/datasets/g1_bimanual_handover.yaml` trains on the four checked A1
  LeRobot datasets above.

Current A1 finetune run completed on `2026-05-12`:

- run dir: `outputs/g1_bimanual_handover_ft1000_20260511_r4`
- W&B run: `g1_bimanual_handover_ft1000_20260511_resize_seq1024`
- W&B URL: `https://wandb.ai/kaiyuanyang666-sun-yat/A1_G1_Finetuning/runs/l3x8smp1`
- dataset: `g1_bimanual_handover_train`
- checkpoint: `/home/sys01/yangky/test/A1/model/a1-pretrain/latest-unsharded`
- stable startup settings:
  - `seq_len=1024` because the multi-episode samples exceeded the earlier
    `512` token limit
  - `crop_mode=resize`, `max_crops=3` because the converted data has three
    camera streams
  - `fsdp_precision=pure`, `disable_float32_attention`, `global_batch_size=2`,
    and `device_train_microbatch_size=1`
  - `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1` to avoid tokenizer startup
    stalls while using the existing local Qwen tokenizer cache
- final result:
  - completed `1000/1000` training steps at `2026-05-12 05:01:33`
  - `step 10`: `train/ActionNoiseL2Loss=1.803`
  - `step 20`: `train/ActionNoiseL2Loss=1.815`
  - `step 30`: `train/ActionNoiseL2Loss=2.007`
  - `step 800`: `train/ActionNoiseL2Loss=0.0968`
  - `step 900`: `train/ActionNoiseL2Loss=0.1074`
  - `step 1000`: `train/ActionNoiseL2Loss=0.1044`
  - final W&B summary `train/ActionNoiseL2Loss=0.10439`
  - peak GPU memory: about `22.5 GB`
- saved checkpoints:
  - `outputs/g1_bimanual_handover_ft1000_20260511_r4/step1000`
  - `outputs/g1_bimanual_handover_ft1000_20260511_r4/step1000-unsharded`
  - `outputs/g1_bimanual_handover_ft1000_20260511_r4/step1000-action-head`

Recommended resume pattern once more data exists:

```bash
export DATA_DIR=/home/sys01/yangky/test/A1/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY=...

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m launch_scripts.train_vla qwen2_7b \
  --checkpoint /home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/outputs/g1_episode_0013_stageA_overfit300/step300-unsharded \
  --vla_config_path g1_episode_0013_finetune.yaml \
  --dataset g1_episode_0013_train \
  --global_batch_size 2 \
  --device_train_microbatch_size 1 \
  --train_steps 1000 \
  --save_interval 1000 \
  --save_interval_unsharded 1000 \
  --save_interval_action_head 200 \
  --wandb_entity kaiyuanyang666-sun-yat \
  --wandb_project a1-vla-camd \
  --wandb_run_name g1-episode0013-stageB-resume1000 \
  --num_workers 0 \
  --log_interval 10 \
  --max_crops 1 \
  --seq_len 512 \
  --fsdp_precision pure \
  --disable_float32_attention \
  save_folder=/home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/outputs/g1_episode_0013_stageB_resume1000
```

#### Stage C. Offline Selection Before Robot Deployment

Before touching the real robot, compare checkpoints with:

- terminal loss trends
- offline action error on held-out G1 logs
- per-sample failure patterns from the eval JSON

Do not pick the checkpoint only by "trained the longest". Pick the checkpoint that:

- keeps action loss stable
- lowers held-out `avg_l1` and `avg_mse`
- does not show obvious late-stage degradation on the same held-out subset

#### Stage D. Guarded Real-Robot Validation

Only after offline selection:

- start with low-speed, guarded execution
- clamp action magnitude and joint deltas
- keep an operator on e-stop
- first validate short horizon fragments before full pick-and-place
- record every rollout back into the managed dataset for the next finetune cycle

The practical rule on this machine is:

- first prove stable overfit
- then prove held-out offline improvement
- then do guarded robot validation
- only after that does larger-scale data collection become the best use of time

### 4. Offline Debug Evaluation

The managed deployment path was patched so local debug evaluation can run on a single 24 GB GPU:

- deployment now follows `config.model.use_proprio` for inference instead of only `config.data.use_proprio`
- deployment disables `float32_attention` for the single-GPU bf16 server path
- deployment wraps prediction in CUDA autocast and aligns floating inputs to model dtype

Start the pretrained server:

```bash
CUDA_VISIBLE_DEVICES=1 bash deploy/deploy.sh \
  --weight /home/sys01/yangky/test/A1/model/a1-pretrain/latest-unsharded \
  --port 18000
```

Start the 1-step finetuned server:

```bash
CUDA_VISIBLE_DEVICES=1 bash deploy/deploy.sh \
  --weight /home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/outputs/g1_episode_0013_smoke_1step/step1-unsharded \
  --port 18000
```

Run the offline debug comparison against whichever server is active:

```bash
python -m robot_experiments.debug \
  --dataset_path /home/sys01/yangky/test/A1/data/g1_episode_0013_lerobot \
  --url http://127.0.0.1:18000 \
  --n_episode 20 \
  --fixed_action_dim 16 \
  --chunk_size 50 \
  --output_json /home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/outputs/g1_episode_0013_eval.json
```

Notes for the debug command:

- keep `fixed_action_dim=16` so the debug wrapper compares the model output against the raw `16D` G1 action labels
- the deployment path returns raw actions for this comparison, so the default `deploy.sh` mode without `--norm` is the intended setting here
- the 20-sample comparison is deterministic because `robot_experiments/debug.py` now uses a fixed seed by default

Managed evaluation summaries already produced:

- pretrained smoke: `outputs/g1_episode_0013_pretrain_eval_smoke.json`
- 1-step finetune smoke: `outputs/g1_episode_0013_step1_eval_smoke.json`
- pretrained 20-sample eval: `outputs/g1_episode_0013_pretrain_eval_20.json`
- pretrained 200-sample eval: `outputs/g1_episode_0013_pretrain_eval_200.json`
- 1-step finetune 20-sample eval: `outputs/g1_episode_0013_step1_eval_20.json`
- 5-step finetune 20-sample eval: `outputs/g1_episode_0013_step5_eval_20.json`
- latest retained long-run checkpoint 20-sample eval: `outputs/g1_episode_0013_step900_eval_20.json`
- 5-step finetune 200-sample eval: `outputs/g1_episode_0013_step5_eval_200.json`
- latest retained long-run checkpoint 200-sample eval: `outputs/g1_episode_0013_step900_eval_200.json`
- pretrain vs 5-step 200-sample pairwise analysis: `outputs/g1_episode_0013_pretrain_vs_step5_200_analysis.json`
- 5-step vs step900 200-sample pairwise analysis: `outputs/g1_episode_0013_step5_vs_step900_200_analysis.json`
- pretrain vs step900 200-sample pairwise analysis: `outputs/g1_episode_0013_pretrain_vs_step900_200_analysis.json`

Current measured action-error comparison:

- `3` samples:
  - pretrained: `avg_l1=0.4433`, `avg_mse=0.4687`
  - 1-step finetune: `avg_l1=0.4254`, `avg_mse=0.4196`
- `20` samples:
  - pretrained: `avg_l1=0.4519`, `avg_mse=0.4916`
  - 1-step finetune: `avg_l1=0.4376`, `avg_mse=0.4500`
  - 5-step finetune: `avg_l1=0.4310`, `avg_mse=0.4478`
  - latest retained long-run checkpoint `step900`: `avg_l1=0.4329`, `avg_mse=0.3130`
- `200` samples on the same fixed-seed protocol:
  - pretrained: `avg_l1=0.4747`, `avg_mse=0.5349`
  - 5-step finetune: `avg_l1=0.4541`, `avg_mse=0.4808`
  - latest retained long-run checkpoint `step900`: `avg_l1=0.3824`, `avg_mse=0.2579`

Concrete 200-sample manual commands already validated on this workstation:

```bash
CUDA_VISIBLE_DEVICES=1 bash deploy/deploy.sh \
  --weight /home/sys01/yangky/test/A1/model/a1-pretrain/latest-unsharded \
  --port 18006
```

```bash
python -m robot_experiments.debug \
  --dataset_path /home/sys01/yangky/test/A1/data/g1_episode_0013_lerobot \
  --url http://127.0.0.1:18006 \
  --n_episode 200 \
  --fixed_action_dim 16 \
  --chunk_size 50 \
  --output_json /home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/outputs/g1_episode_0013_pretrain_eval_200.json
```

Pairwise analysis commands:

```bash
python scripts/analyze_eval_deltas.py \
  --baseline outputs/g1_episode_0013_pretrain_eval_200.json \
  --candidate outputs/g1_episode_0013_step5_eval_200.json \
  --baseline-name pretrain \
  --candidate-name step5 \
  --dataset-length 639 \
  --raw-episode-dir /home/sys01/yangky/test/A1/data/episode_0013 \
  --json-out outputs/g1_episode_0013_pretrain_vs_step5_200_analysis.json
```

```bash
python scripts/analyze_eval_deltas.py \
  --baseline outputs/g1_episode_0013_step5_eval_200.json \
  --candidate outputs/g1_episode_0013_step900_eval_200.json \
  --baseline-name step5 \
  --candidate-name step900 \
  --dataset-length 639 \
  --raw-episode-dir /home/sys01/yangky/test/A1/data/episode_0013 \
  --json-out outputs/g1_episode_0013_step5_vs_step900_200_analysis.json
```

```bash
python scripts/analyze_eval_deltas.py \
  --baseline outputs/g1_episode_0013_pretrain_eval_200.json \
  --candidate outputs/g1_episode_0013_step900_eval_200.json \
  --baseline-name pretrain \
  --candidate-name step900 \
  --dataset-length 639 \
  --raw-episode-dir /home/sys01/yangky/test/A1/data/episode_0013 \
  --json-out outputs/g1_episode_0013_pretrain_vs_step900_200_analysis.json
```

What the pairwise analyzer reports:

- mean `L1` / `MSE` deltas between two checkpoints
- whether the candidate checkpoint wins more often sample-by-sample
- phase splits over the single episode:
  - `approach`
  - `grasp_lift`
  - `carry_place_right`
- top improved and top degraded samples, with raw image paths and motion statistics from the original `data.json`

Interpretation:

- even a single optimization step on this episode already reduced offline action error on the fixed debug sample set
- the verified 5-step run improved the same 20-sample offline metric a bit further compared with the 1-step checkpoint
- the later `step900` checkpoint kept the mean `L1` near the `step5` result, but drove mean `MSE` much lower on the same 20-sample subset
- compared with `step5`, `step900` changed from `avg_l1=0.4310`, `avg_mse=0.4478` to `avg_l1=0.4329`, `avg_mse=0.3130`
- this is not a clean monotonic win across every sample; on the fixed 20-sample subset, only some samples improved and a few large wins dominate the mean `MSE`
- on this fixed subset, `step900` beat `step5` on `9/20` samples by `MSE` and `8/20` samples by `L1`, which is another sign that the later checkpoint is redistributing error rather than uniformly improving everything
- once the sample count was increased to `200`, the earlier small-sample ambiguity mostly disappeared: `step900` beat `step5` on the mean of both metrics
- on the same `200` samples, `step5` also beats the pretrained checkpoint on average: from `avg_l1=0.4747`, `avg_mse=0.5349` down to `avg_l1=0.4541`, `avg_mse=0.4808`
- on the same `200` samples, `step900` beats the pretrained checkpoint by a much larger margin: from `avg_l1=0.4747`, `avg_mse=0.5349` down to `avg_l1=0.3824`, `avg_mse=0.2579`
- the `200`-sample comparison moved from `step5` `avg_l1=0.4541`, `avg_mse=0.4808` to `step900` `avg_l1=0.3824`, `avg_mse=0.2579`
- but even there, `step900` is still not a majority winner on every sample: it beat `step5` on `95/200` samples by `L1` and `100/200` samples by `MSE`
- that means the later checkpoint wins mainly because its improvements are much larger when they happen, not because it is uniformly better on most individual samples
- the phase breakdown makes that pattern much clearer:
  - `step900` is decisively better than `step5` throughout `grasp_lift`: all `75/75` middle-phase samples improved on both `L1` and `MSE`
  - `step900` is worse than `step5` on most `approach` samples: only `7/71` improved on `L1`, and only `10/71` improved on `MSE`
  - `step900` is also unstable in `carry_place_right`: only `13/54` improved on `L1`, and `15/54` improved on `MSE`
- the top `step900` improvements cluster around frames such as `248`, `265`, `278`, and `286`, where the bottle is still near the center and the robot is actively securing or lifting it
- the top `step900` degradations cluster around frames such as `540`, `542`, `545`, and `548`, where the bottle has already moved to the right side and the robot is releasing or retracting after placement
- the motion statistics reinforce the visual pattern:
  - the strongest `step900` wins happen when `left_ee_motion_norm` is active and the left hand is still engaged in grasp control
  - the strongest `step900` failures happen when `left_ee_motion_norm` is `0` and the remaining error is dominated by arm retreat / release behavior near the final placement zone
- for real-robot deployment, that means `step900` currently looks more promising for stabilizing the grasp-and-lift middle phase than for guaranteeing a clean right-side placement finish
- this is still only a smoke result, not a reliable policy-quality conclusion for deployment
- before trusting real-robot behavior, expand to more episodes, add held-out evaluation, and run closed-loop tests in simulation or on a guarded robot setup

### 5. Can We Keep Training On The Current Single Episode?

Yes.

You do not need new teleoperation data just to continue optimizing on the current
`episode_0013`. The current single episode is already useful for:

- validating that the data conversion is correct
- checking whether the model can overfit the task
- checking whether offline action error continues to go down

But the limitation is equally important:

- one episode is enough for smoke finetune and overfit checks
- one episode is not enough to claim generalization
- lower loss or lower offline action error on this same episode only proves the training path works

The current managed resume point is:

- `outputs/g1_episode_0013_smoke_1step/step1-unsharded`

Recommended manual continuation command on this workstation:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m launch_scripts.train_vla qwen2_7b \
  --checkpoint /home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/outputs/g1_episode_0013_smoke_1step/step1-unsharded \
  --vla_config_path g1_episode_0013_finetune.yaml \
  --dataset g1_episode_0013_train \
  --global_batch_size 2 \
  --device_train_microbatch_size 1 \
  --train_steps 1 \
  --save_interval 1 \
  --save_interval_unsharded 1 \
  --wandb_debug \
  --num_workers 0 \
  --log_interval 1 \
  --max_crops 1 \
  save_folder=/home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/outputs/g1_episode_0013_smoke_resume
```

This is best understood as a same-episode continuation run for overfit analysis, not
as a robust policy-validation run.

If you want a continuation command that is more likely to survive for multiple steps on
dual `RTX 4090`, use the same save folder pattern but keep the memory-saving flags from
section `3B`:

```bash
export DATA_DIR=/home/sys01/yangky/test/A1/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m launch_scripts.train_vla qwen2_7b \
  --checkpoint /home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/outputs/g1_episode_0013_multistep_original_test_v1/step5-unsharded \
  --vla_config_path g1_episode_0013_finetune.yaml \
  --dataset g1_episode_0013_train \
  --global_batch_size 2 \
  --device_train_microbatch_size 1 \
  --train_steps 5 \
  --save_interval 5 \
  --save_interval_unsharded 5 \
  --wandb_debug \
  --num_workers 0 \
  --log_interval 1 \
  --max_crops 1 \
  --seq_len 512 \
  --fsdp_precision pure \
  --disable_float32_attention \
  save_folder=/home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/outputs/g1_episode_0013_multistep_original_resume
```

### 6. How To Inspect Training Quality

There are three practical signals in the current managed workflow.

#### A. Console Loss

The training loop prints step-level metrics directly to the terminal. For the current G1
smoke path, the main metric is:

- `train/ActionNoiseL2Loss`

Example from the successful smoke run:

```text
[step=1/1]
    train/ActionNoiseL2Loss=0.4660
```

#### B. Loss Curves With WandB

The current smoke commands use `--wandb_debug`, which disables online wandb logging.

If you want a loss curve, remove `--wandb_debug`, export a valid API key, and set your
entity/project explicitly:

```bash
export WANDB_API_KEY=...
```

Add these flags to the training command:

```bash
--wandb_entity <your_entity> \
--wandb_project a1-vla-camd \
--wandb_run_name g1-episode0013
```

Relevant code paths:

- [launch_scripts/train_vla.py](/home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/launch_scripts/train_vla.py)
- [scripts/train_for_action.py](/home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/scripts/train_for_action.py)
- [a1/train.py](/home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/a1/train.py)

Local fallback when WandB is incomplete:

```bash
python scripts/inspect_training_run.py \
  --run-dir /home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/outputs/g1_episode_0013_multistep_original_test_v1_50000_steps \
  --plot-out /home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/outputs/g1_episode_0013_multistep_original_test_v1_50000_steps/training_curve.svg
```

This is especially useful for the current `50000-step` run because the local WandB
internal log already showed transient `EOF` retries against `api.wandb.ai`, while the
training `output.log` still contains the authoritative step metrics.

#### C. Offline Action Error

This is currently the most useful effect metric for your G1 data, because it directly
compares predicted actions against the recorded teleoperation labels.

Pretrained baseline on 20 fixed samples:

```bash
CUDA_VISIBLE_DEVICES=1 bash deploy/deploy.sh \
  --weight /home/sys01/yangky/test/A1/model/a1-pretrain/latest-unsharded \
  --port 18000

python -m robot_experiments.debug \
  --dataset_path /home/sys01/yangky/test/A1/data/g1_episode_0013_lerobot \
  --url http://127.0.0.1:18000 \
  --n_episode 20 \
  --fixed_action_dim 16 \
  --chunk_size 50 \
  --output_json /home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/outputs/g1_episode_0013_pretrain_eval_20.json
```

Finetuned checkpoint on the same 20 fixed samples:

```bash
CUDA_VISIBLE_DEVICES=1 bash deploy/deploy.sh \
  --weight /home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/outputs/g1_episode_0013_smoke_1step/step1-unsharded \
  --port 18000

python -m robot_experiments.debug \
  --dataset_path /home/sys01/yangky/test/A1/data/g1_episode_0013_lerobot \
  --url http://127.0.0.1:18000 \
  --n_episode 20 \
  --fixed_action_dim 16 \
  --chunk_size 50 \
  --output_json /home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/projects/A1/outputs/g1_episode_0013_step1_eval_20.json
```

Current measured result on this workstation:

- pretrained: `avg_l1=0.4519`, `avg_mse=0.4916`
- 1-step finetune: `avg_l1=0.4376`, `avg_mse=0.4500`

So on the current single episode:

- `avg_l1` improved by about `0.0143`
- `avg_mse` improved by about `0.0416`

### 7. Is There Visual Simulation Right Now?

Not for this managed G1 episode workflow yet.

What exists today:

- supervised finetune on converted G1 teleoperation data
- offline action-error evaluation against recorded action labels

What does not exist yet in this managed path:

- a ready-made G1 simulator rollout loop for this bottle pick-and-place task
- automatic rollout video generation directly from the G1 dataset alone

So the reliable effect checks right now are:

- watch `train/ActionNoiseL2Loss`
- compare `avg_l1` and `avg_mse` with `robot_experiments.debug`
- compare the saved evaluation JSON summaries across checkpoints

If you want visual simulation next, the next engineering step is to connect the
finetuned policy to either:

- a G1-compatible simulator task wrapper
- or a guarded real-robot replay / validation pipeline

---

# 🤖 A1: A Fully Transparent Open-Source, Adaptive and Efficient Truncated Vision-Language-Action Model


<p align="center">
  <a href="https://arxiv.org/abs/2604.05672">
    <img src="https://img.shields.io/badge/arXiv-2604.05672-red?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://github.com/ATeam-Research/A1">
    <img src="https://img.shields.io/badge/Code-GitHub-black?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="http://www.ateam.xin/#/research/A1">
    <img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=internet-explorer&logoColor=white" alt="Project Page">
  </a>
</p>

<!-- <p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10">
  <img src="https://img.shields.io/badge/PyTorch-2.6.0-orange?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.4-green?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA 12.4">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p> -->

---

## 📋 Table of Contents

- [🔧 Installation](#installation)
- [⚙️ Environment Setup](#environment-setup)
- [🚀 Deploy](#deploy)
- [📊 Evaluation](#evaluation)
- [🎓 Training](#training)
- [📦 Model Zoo & Datasets](#model-zoo--datasets)

---

## 🔧 Installation
```
conda create -n a1 python=3.10
conda activate a1
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e .[all]
pip install --no-deps --force-reinstall git+https://github.com/moojink/dlimp_openvla
pip install -r requirements.txt 
```

## ⚙️ Environment Setup

### 🔐 1. Copy Environment Template
```bash
📋 cp .env.example .env.personal
```

### ✏️ 2. Configure Your Settings

Edit `.env.personal` with your personal settings:

```bash
# Example content
CONDA_ROOT=/path/to/conda
CONDA_ENV=a1
WANDB_ENTITY=your_entity
WANDB_PROJECT=your_project
```

> 🔒 This file is Git-ignored and won't be committed!

### 🔄 3. Load Environment
```bash
source .env.personal
```

> **⚠️ Security Note**: `.env.personal` contains sensitive information (paths, API keys, etc.). Do NOT commit it to Git.

---

## 🚀 Deploy

Start the API server for model inference:

```bash
🖥️ bash deploy/deploy.sh --weight /put/checkpoint/here --port <port>
```

**📋 Arguments:**
| Argument | Required | Description |
|:--------:|:--------:|:------------|
| `--weight` | ✅ | Path to model checkpoint |
| `--port` | ❌ | Server port (auto-selected if not provided) |
| `--norm` | ❌ | Enable normalization (0 or 1) |

**✨ Example:**
```bash
bash deploy/deploy.sh --weight ./model/checkpoints/pretrain --port 8000
```

---

## 📊 Evaluation

### 🎮 LIBERO Evaluation

#### 📦 1. LIBERO Installation
```bash
📥 git submodule update --init robot_experiments/libero/LIBERO
📥 pip install -e robot_experiments/libero/LIBERO
```

#### 🚀 2. Run Evaluation

| Mode | Command | Description |
|:----:|:--------|:------------|
| 🎯 **Standard** | `bash eval_libero.sh` | Standard evaluation |
| ⚡ **Early Exit** | `bash eval_libero_exit.sh` | Evaluates all 4 LIBERO task suites |



### 🧪 VLABench Evaluation

VLABench evaluation requires running both a server and a client in separate terminals.

**📋 Setup Steps:**

| Step | Action | Command |
|:----:|:-------|:--------|
| 1️⃣ | **Install VLABench** | `pip install -r ...` |
| 2️⃣ | **Download Assets** | `python scripts/download_assets.py` |
| 3️⃣ | **Start Server** 💻 | `bash deploy/deploy.sh ...` |
| 4️⃣ | **Run Client** 🎮 | `python eval_client.py` |

**🔧 Step 1: Install VLABench**
```bash
pip install -r robot_experiments/vlabench/VLABench/requirements.txt
pip install -e robot_experiments/vlabench/VLABench
```

**📥 Step 2: Download Assets** (if not already downloaded)
```bash
cd robot_experiments/vlabench/VLABench
python scripts/download_assets.py --choice all
```

**🖥️ Step 3: Start the Evaluation Server** (Terminal 1)
```bash
# Load environment and start the API server
bash deploy/deploy.sh --weight <path_to_checkpoint> --port 8000
```

**🎮 Step 4: Run the Evaluation Client** (Terminal 2)
```bash
cd robot_experiments/vlabench
python eval_client.py
```

> **⚠️ Note**: The server and client must run in separate terminals. The server loads the model and waits for client connections, while the client sends evaluation requests and receives results.

### 🏆 RoboChallenge Evaluation

RoboChallenge evaluation is executed through the `run_task.py` script, supporting two modes:

| 🎮 Mode | 📖 Description | 🎯 Purpose |
|:-------:|:--------------|:-----------|
| `mock` | Automatic evaluation using local pre-recorded data | Local testing, debugging |
| `real` | Connect to real robot and submit official evaluation | Official competition evaluation |

**🦾 Supported Robot Types:**
- `ALOHA` 🤖
- `ARX5` 🔧
- `UR5` ⚡
- `FRANKA` 🦿

#### 🧪 Mock Mode (Local Automatic Evaluation)

```bash
# 💻 Terminal 1: Deploy model
bash deploy/deploy.sh --weight <path_to_checkpoint> --port 8000

# 🎮 Terminal 2: Run mock evaluation
cd robot_experiments/RoboChallengeInference
python run_task.py \
    --task_name open_the_drawer \
    --test_type mock \
    --url http://localhost:8000
```

#### 🌍 Real Mode (Official Evaluation)

```bash
# 💻 Terminal 1: Deploy model
bash deploy/deploy.sh --weight <path_to_checkpoint> --port 8000

# 🤖 Terminal 2: Run real robot evaluation
cd robot_experiments/RoboChallengeInference
python run_task.py \
    --task_name open_the_drawer \
    --test_type real \
    --url http://localhost:8000 \
    --user_token <your_token> \
    --run_id <run_id> \
    --action_nums 30
```

> **💡 Tips:**
> - ✅ Mock mode automatically starts the mock server, no manual startup required
> - 🔑 Real mode requires valid `user_token` and `run_id` for official evaluation
> - 📋 `task_name` must be defined in `task_config.ROBO_CHALLENGE_TASKS`


---

## 🎓 Training

### 🌟 Pretraining

Pretraining trains the model from scratch using large-scale VLA datasets, supporting distributed training on Slurm clusters.

**📁 Configuration files:**
- 📄 `configs/experiments/pretrain.yaml` - Pretraining experiment configuration
- 📄 `configs/datasets/pretrain.yaml` - Pretraining dataset configuration
- 🖥️ `scripts/slurms/pretrain.sh` - Pretraining script (runs on Slurm cluster)

**🖥️ Slurm Cluster Training:**

1️⃣ **Configure Slurm submission script** `scripts/slurms/submit_job.sh`:
   - 🔧 `nnodes`: Number of nodes required (default 8 nodes)
   - 🔧 `gpus_per_node`: GPUs per node (default 8)
   - 🔧 `partition` and `quotatype`: Partition name and QOS type

2️⃣ **Submit pretraining job:**
```bash
🚀 bash scripts/slurms/submit_job.sh
```

**💻 Single-node multi-GPU training (non-Slurm):**
```bash
🚀 bash scripts/slurms/pretrain.sh
```

> **⚠️ Note:** Pretraining requires significant computational resources. Distributed training on Slurm clusters is recommended. Global batch size = 128 × number of nodes.

---

### 📚 LIBERO Training

LIBERO training fine-tunes on simulation data, supporting single-node multi-GPU training.

**📁 Configuration files:**
- 📄 `configs/experiments/libero_simulation.yaml` - LIBERO training configuration
- 📄 `configs/datasets/libero_4_tasks.yaml` - LIBERO 4-task dataset configuration

**🚀 Run training:**
```bash
bash train_libero.sh
```



---

### 🧪 VLAbench Training

VLAbench training fine-tunes in the VLAbench simulation environment.

**📁 Configuration files:**
- 📄 `configs/experiments/vlabench.yaml` - VLAbench training configuration
- 📄 `configs/datasets/vlabench.yaml` - VLAbench dataset configuration

**🚀 Run training:**
```bash
bash train_vlabench.sh
```


---

### 🏆 RoboChallenge Training

RoboChallenge training uses the `train_rc.sh` script to fine-tune on specific tasks (e.g., open_the_drawer, put_cup_on_coaster).

**📁 Configuration files:**
| File | Description |
|:----:|:------------|
| `configs/experiments/rc_open_the_drawer.yaml` | 🗄️ Open the drawer task config |
| `configs/experiments/rc_put_cup_on_coaster.yaml` | ☕ Put cup on coaster task config |
| `configs/datasets/rc_*.yaml` | 🤖 Dataset configs (ARX5, etc.) |

**🚀 Run training:**
```bash
bash train_rc.sh
```

> **💡 Tip:** Modify the `vla_config_path` variable in the script to switch between different task configurations.

---

## 📦 Model Zoo & Datasets

### 🤖 Pretrained Models

| Model | Description | Checkpoint |
|:------|:------------|:----------:|
| **pretrain** | Pretrained model on large-scale VLA datasets | [Link](https://huggingface.co/spatialtemporal-ai/a1-pretrain) |
| **libero** | Fine-tuned on LIBERO simulation tasks | [Link](https://huggingface.co/spatialtemporal-ai/a1-libero) |
| **libero_exit** | LIBERO model with early exit mechanism | [Link](https://huggingface.co/spatialtemporal-ai/a1-libero-exit) |
| **rc_put_cup_on_coaster** | Fine-tuned on RoboChallenge put cup task | [Link](https://huggingface.co/spatialtemporal-ai/a1-rc-put-cup-on-coaster) |
| **rc_open_the_drawer** | Fine-tuned on RoboChallenge open drawer task | [Link](https://huggingface.co/spatialtemporal-ai/a1-rc-open-the-drawer) |

### 📊 Training Datasets

| Dataset | Description | Download |
|:--------|:------------|:--------:|
| **Droid** | DROID dataset for robotic manipulation | [Link](https://huggingface.co/datasets/IPEC-COMMUNITY/droid_lerobot) |
| **RoboChallenge** | RoboChallenge competition data | [Link](https://huggingface.co/datasets/RoboChallenge/Table30) |
| **RoboCOIN** | RoboCOIN dataset | [Link](https://huggingface.co/RoboCOIN) |
| **RoboMIND** | RoboMIND benchmark dataset | [Link](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND) |
| **AgiBot** | AgiBot dataset | [Link](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) |
| **LIBERO** | LIBERO simulation tasks | [Link](https://huggingface.co/datasets/spatialtemporal-ai/libero_rlds) |
| **VlaBench** | VLABench simulation environment | [Link]() |

**📁 Storage Paths:**
- **Model weights**: Place downloaded model weights in the `model/` directory
- **Training data**: Place downloaded datasets in the `data/` directory

> **⚠️ Important Notes:**
> 1. **RoboMIND dataset preprocessing**: Before using RoboMIND dataset, you need to run the indexing script:
>    ```bash
>    bash scripts/robomind_build_index.sh
>    ```
> 2. **LeRobot dataset patch for pretraining**: Before pretraining, you must replace the LeRobot dataset file:
>    ```bash
>    cp a1/data/vla/lerobot_datasets_replace.py <CONDA_ENV_PATH>/lib/python3.10/site-packages/lerobot/datasets/lerobot_dataset.py
>    ```
>    Replace `<CONDA_ENV_PATH>` with your actual conda environment path (e.g., `/path/to/conda/envs/a1`)

> **📝 Note:** Please fill in the actual download links for models and datasets in the table above.

---

## 🙏 Acknowledgements

This project is built upon the [Molmo](https://github.com/allenai/molmo) project. We thank the Allen Institute for AI for their excellent open-source work.

---

## 📚 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@misc{zhang2026a1fullytransparentopensource,
      title={A1: A Fully Transparent Open-Source, Adaptive and Efficient Truncated Vision-Language-Action Model}, 
      author={Kaidong Zhang and Jian Zhang and Rongtao Xu and Yu Sun and Shuoshuo Xue and Youpeng Wen and Xiaoyu Guo and Minghao Guo and Weijia Liufu and Liu Zihou and Kangyi Ji and Yangsong Zhang and Jiarun Zhu and Jingzhi Liu and Zihang Li and Ruiyi Chen and Meng Cao and Jingming Zhang and Shen Zhao and Xiaojun Chang and Feng Zheng and Ivan Laptev and Xiaodan Liang},
      year={2026},
      eprint={2604.05672},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2604.05672}, 
}
```

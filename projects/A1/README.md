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
- 1-step finetune 20-sample eval: `outputs/g1_episode_0013_step1_eval_20.json`
- 5-step finetune 20-sample eval: `outputs/g1_episode_0013_step5_eval_20.json`

Current measured action-error comparison:

- `3` samples:
  - pretrained: `avg_l1=0.4433`, `avg_mse=0.4687`
  - 1-step finetune: `avg_l1=0.4254`, `avg_mse=0.4196`
- `20` samples:
  - pretrained: `avg_l1=0.4519`, `avg_mse=0.4916`
  - 1-step finetune: `avg_l1=0.4376`, `avg_mse=0.4500`
  - 5-step finetune: `avg_l1=0.4310`, `avg_mse=0.4478`

Interpretation:

- even a single optimization step on this episode already reduced offline action error on the fixed debug sample set
- the verified 5-step run improved the same 20-sample offline metric a bit further compared with the 1-step checkpoint
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

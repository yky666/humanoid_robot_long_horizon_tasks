# Managed Import Notes

This folder is the managed snapshot of the local `Isaac-GR00T` workspace on this
machine.

## Import Metadata

- Import date: `2026-05-31`
- Imported into: `humanoid_robot_long_horizon_tasks`
- Upstream repository: `https://github.com/NVIDIA/Isaac-GR00T`
- Source local path: `/home/sys01/yangky/test/Isaac-GR00T`

## Local Changes Included In This Snapshot

- Added `scripts/convert_g1_raw_episode_to_gr00t.py` for converting raw Unitree
  G1 episode dumps into GR00T-flavored LeRobot v2 datasets.
- Added `examples/G1/g1_upper_body_config.py` for custom G1 upper-body
  fine-tuning experiments under `NEW_EMBODIMENT`.
- Added local converted sample dataset
  `data/g1_episode_0015_psi0_gr00t/`, generated from local G1 recordings.
- Exposed fine-tuning knobs for `tune_vlln`, gradient checkpointing, optimizer
  selection, and bf16 model loading in the finetune CLI config.

## G1 Baseline Guidance

For checking GR00T-N1.7 on humanoid G1, keep the baseline tied to GR00T
itself:

- Use `nvidia/GR00T-N1.7-3B` as the base model checkpoint.
- Use `--embodiment-tag REAL_G1` for zero-shot/base-model inference.
- Use the checkpoint snapshot directory, not the HuggingFace cache root.

```bash
source .venv/bin/activate
python gr00t/eval/run_gr00t_server.py \
    --model-path /home/sys01/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495 \
    --embodiment-tag REAL_G1 \
    --device cuda:0
```

The cache root
`/home/sys01/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/` is not a
loadable model directory. The loadable directory is the snapshot containing
`model-00001-of-00002.safetensors`, `model.safetensors.index.json`,
`processor_config.json`, and `statistics.json`.

`nvidia/GR00T-N1.7-DROID` is a DROID robot finetuned checkpoint. It is useful
for DROID examples, but it is not the correct baseline for a Unitree G1
humanoid simulation because the state/action semantics and embodiment tag are
DROID-specific.

### Convert local G1 recordings

Raw local G1 episodes must be converted to a LeRobot dataset whose modality
groups match the GR00T checkpoint. For the base `REAL_G1` model, convert with
`--state-format real_g1`:

```bash
source .venv/bin/activate
python scripts/convert_g1_raw_episode_to_gr00t.py \
    --input-episode-dir /home/sys01/yangky/test/A1/data/episode_0030 \
    --output-dataset-dir data/g1_episode_0030_real_g1_gr00t \
    --state-format real_g1 \
    --video-keys ego_view,wrist,camera_2 \
    --overwrite
```

The `real_g1` layout writes these GR00T groups:

- state: `left_wrist_eef_9d`, `right_wrist_eef_9d`, `left_hand`, `right_hand`,
  `left_arm`, `right_arm`, `waist`
- action: the same groups plus `base_height_command` and `navigate_command`

The wrist EEF groups are converted from raw `left_arm_ee` and `right_arm_ee`
xyz+rpy poses into xyz+rot6d. Arm actions are stored as absolute targets; the
GR00T processor converts configured groups to relative action chunks internally
when `use_relative_action` is enabled by the checkpoint.

`left_wrist_eef_9d` and `right_wrist_eef_9d` are 9D end-effector pose groups,
not 9-DOF arm joint groups. Each one is `xyz` translation (3D) plus a 6D
rotation representation (`rot6d`). The actual arm joint groups are separate
`left_arm` and `right_arm` 7D joint vectors.

### Open-loop GR00T baseline

Run the open-loop evaluator against the converted dataset to compare full
ground-truth action trajectories with GR00T predictions:

```bash
source .venv/bin/activate
python gr00t/eval/open_loop_eval.py \
    --model-path /home/sys01/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495 \
    --dataset-path data/g1_episode_0030_real_g1_gr00t \
    --embodiment-tag REAL_G1 \
    --traj-ids 0 \
    --steps 100000 \
    --action-horizon 40 \
    --save-plot-path outputs/g1_real_g1_baseline/episode_0030_open_loop.jpeg
```

This evaluates GR00T predictions, not any A1 model output. Ground truth comes
from the converted dataset action columns. Prediction comes directly from the
GR00T action head through `Gr00tPolicy`.

### Local G1 data audit and results

Current local raw recordings under `/home/sys01/yangky/test/A1/data` were
checked for JSON readability, camera frame coverage, empty image files, and
`REAL_G1` state/action key compatibility.

| Episode | Raw status | `REAL_G1` compatible | Converted dataset | Base open-loop result |
| --- | --- | --- | --- | --- |
| `episode_0013` | 639/639 frames, single camera complete | No; legacy keys only | Not used for base `REAL_G1` | Not run |
| `episode_0030` | 738 JSON frames, but many empty camera frames; `color_0` first empty frame is 102 | Yes | `data/g1_episode_0030_real_g1_gr00t_smoke102` | 40 steps: MSE `0.1436`, MAE `0.2113` |
| `episode_0031` | 973/973 frames, 3 cameras complete | Yes | `data/g1_episode_0031_real_g1_gr00t` | 120 steps: MSE `0.2991`, MAE `0.3433` |
| `episode_0036` | Camera images complete, but `data.json` is truncated/invalid JSON | Unknown until JSON is recovered | Not converted | Not run |
| `episode_0038` | 647/647 frames, 3 cameras complete | Yes | `data/g1_episode_0038_real_g1_gr00t` | 120 steps: MSE `0.1117`, MAE `0.1791` |
| `episode_0040` | 854/854 frames, 3 cameras complete | Yes | `data/g1_episode_0040_real_g1_gr00t` | 120 steps: MSE `0.2130`, MAE `0.2640` |

All converted complete datasets above were verified to have matching metadata
frame counts and encoded video frame counts for `ego_view`, `wrist`, and
`camera_2`. The GR00T base open-loop plots are stored under
`outputs/g1_real_g1_baseline/` and are intentionally ignored by git. In those
plots, GT is the converted local demonstration action trajectory and prediction
is generated by `Gr00tPolicy` from `nvidia/GR00T-N1.7-3B`.

The MuJoCo side-by-side videos are a separate A1 action replay diagnostic, not
GR00T output. The old `outputs/g1_visual_replay/` directory was renamed to
`outputs/a1_action_replay_ep0030_base_norm_eval50/` because its `pred_action`
comes from saved A1 eval JSON files such as
`projects/A1/outputs/g1_bimanual_handover_ep0030_base_norm_eval_50_actions.json`.
In those A1 JSON files, `gt_action` is the locally collected 26D action chunk
and `pred_action` is the A1 model's saved prediction. It is not a GR00T action
sequence.

Full-length A1 replay videos were generated from raw episode GT actions with
sparse A1 eval-50 prediction coverage:

| Episode | Video frames | Duration at 10 FPS | A1 pred coverage | Output pattern |
| --- | ---: | ---: | ---: | --- |
| `episode_0031` | 973 | 97.3 s | 944/973 (`97.0%`) | `outputs/a1_action_replay_full_episode_sparse_pred/a1_ep0031_{base,step1000}_eval50_full_episode_sparse_pred_gt_vs_pred.mp4` |
| `episode_0038` | 647 | 64.7 s | 644/647 (`99.5%`) | `outputs/a1_action_replay_full_episode_sparse_pred/a1_ep0038_{base,step1000}_eval50_full_episode_sparse_pred_gt_vs_pred.mp4` |
| `episode_0040` | 854 | 85.4 s | 782/854 (`91.6%`) | `outputs/a1_action_replay_full_episode_sparse_pred/a1_ep0040_{base,step1000}_eval50_full_episode_sparse_pred_gt_vs_pred.mp4` |

These videos render the full GT episode timeline. Frames without saved A1
prediction coverage are labeled `Pred missing; showing GT pose`, so they should
not be interpreted as model predictions. To get a strict full-prediction replay,
rerun A1 or GR00T inference at every frame and save dense action chunks.

For fine-tuning, `examples/G1/real_g1_config.py` registers the base checkpoint's
`REAL_G1` modality config for the training entry point. A two-GPU RTX 4090
smoke run on `data/g1_episode_0030_real_g1_gr00t_smoke102` completed when
training only the action-head projector:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1 USE_WANDB=0 NUM_GPUS=2 MAX_STEPS=2 SAVE_STEPS=2 \
  GLOBAL_BATCH_SIZE=2 DATALOADER_NUM_WORKERS=0 SHARD_SIZE=64 \
  NUM_SHARDS_PER_EPOCH=2 EPISODE_SAMPLING_RATE=1 MASTER_PORT=29503 \
  bash examples/finetune.sh \
    --base-model-path /home/sys01/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495 \
    --dataset-path data/g1_episode_0030_real_g1_gr00t_smoke102 \
    --embodiment-tag REAL_G1 \
    --modality-config-path examples/G1/real_g1_config.py \
    --output-dir outputs/g1_real_g1_finetune_smoke102_projector_2gpu_2steps \
    --save-only-model \
    -- --load-bf16 --gradient-checkpointing --no-tune-diffusion-model --no-tune-vlln
```

That run saved `checkpoint-2` and reported train loss `1.4492`. The default
training setup with projector + diffusion + VLLN reached the optimizer step but
ran out of memory on dual 24 GB 4090s because it made about 1.62B parameters
trainable and Adam needed additional optimizer state. For longer runs on the
full complete episodes, use the complete converted datasets above and keep the
trainable module set constrained unless ZeRO-3, an 8-bit optimizer, or another
lower-memory optimizer setup is added.

The alternate upper-body/`NEW_EMBODIMENT` path was also smoke-tested with
`examples/G1/g1_upper_body_config.py` on the complete
`data/g1_episode_0031_psi0_gr00t` dataset. This path uses the older upper-body
layout (`left_arm`, `right_arm`, `left_hand`, `right_hand`, `body_state`) rather
than the base checkpoint's `REAL_G1` EEF layout. A projector-only 2-step run
completed on one RTX 4090 with `SHARD_SIZE=2048`, `NUM_SHARDS_PER_EPOCH=1`, and
reported train loss `1.5166`:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 USE_WANDB=0 NUM_GPUS=1 MAX_STEPS=2 SAVE_STEPS=2 \
  GLOBAL_BATCH_SIZE=1 DATALOADER_NUM_WORKERS=0 SHARD_SIZE=2048 \
  NUM_SHARDS_PER_EPOCH=1 EPISODE_SAMPLING_RATE=1 \
  bash examples/finetune.sh \
    --base-model-path /home/sys01/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495 \
    --dataset-path data/g1_episode_0031_psi0_gr00t \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/G1/g1_upper_body_config.py \
    --output-dir outputs/g1_upper_body_0031_projector_1gpu_2steps \
    --save-only-model \
    -- --load-bf16 --gradient-checkpointing --no-tune-diffusion-model --no-tune-vlln
```

Using a small `SHARD_SIZE=64` for this single-episode upper-body dataset creates
empty shards and fails with `All shards must have length greater than 0`; use a
large shard size or merge multiple episodes before multi-GPU training.

### 1000-step GR00T projector fine-tune

The complete `REAL_G1` episodes `0031`, `0038`, and `0040` were merged into
`data/g1_real_g1_complete_0031_0038_0040_gr00t` with
`scripts/merge_lerobot_episodes.py`. This merged set has 3 episodes and 2474
frames.

A dual RTX 4090 projector-only GR00T fine-tune completed for 1000 steps:

- output directory:
  `outputs/gr00t_real_g1_0031_0038_0040_projector_2gpu_1000steps/gr00t-real-g1-0031-0038-0040-projector-1000`
- checkpoints: `checkpoint-250`, `checkpoint-500`, `checkpoint-750`,
  `checkpoint-1000`
- final trainer summary: `train_loss=0.1649`,
  `train_runtime=1130.764s`, `train_steps_per_second=0.884`
- last logged loss at step 1000: `0.0718`
- W&B run:
  `https://wandb.ai/kaiyuanyang666-sun-yat/gr00t-g1-local/runs/vve44jrv`

The run used W&B offline logging during training and was synced afterward to
the `kaiyuanyang666-sun-yat/gr00t-g1-local` project. The command shape was:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1 USE_WANDB=1 NUM_GPUS=2 MAX_STEPS=1000 SAVE_STEPS=250 \
  GLOBAL_BATCH_SIZE=2 DATALOADER_NUM_WORKERS=0 SHARD_SIZE=2048 \
  NUM_SHARDS_PER_EPOCH=2 EPISODE_SAMPLING_RATE=1 MASTER_PORT=29505 \
  bash examples/finetune.sh \
    --base-model-path /home/sys01/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495 \
    --dataset-path data/g1_real_g1_complete_0031_0038_0040_gr00t \
    --embodiment-tag REAL_G1 \
    --modality-config-path examples/G1/real_g1_config.py \
    --output-dir outputs/gr00t_real_g1_0031_0038_0040_projector_2gpu_1000steps \
    --save-only-model \
    -- --load-bf16 --gradient-checkpointing --no-tune-diffusion-model --no-tune-vlln
```

`scripts/export_gr00t_open_loop_actions.py` exports dense GR00T open-loop
action chunks from a checkpoint. It writes a native `REAL_G1` 53D JSON for
metrics and an optional `a1_26d_approx_from_real_g1` JSON for the existing
MuJoCo replay utility. The 26D projection is only for visualization: it maps
`left_arm` and `right_arm` directly and uses the first 6 of 7 hand channels per
side; the REAL_G1 EEF xyz+rot6d commands are not visualized in that replay.

Checkpoint-1000 dense open-loop export results:

| Episode | Evaluable GR00T steps | Native 53D L1 | Native 53D MSE | Native JSON | A1-style replay JSON |
| --- | ---: | ---: | ---: | --- | --- |
| `episode_0031` | 953 | `0.1275` | `0.0630` | `outputs/gr00t_open_loop_actions/gr00t_real_g1_ep0031_checkpoint1000_openloop_actions_53d.json` | `outputs/gr00t_open_loop_actions/gr00t_real_g1_to_a1_26d_approx_ep0031_checkpoint1000_openloop_actions.json` |
| `episode_0038` | 627 | `0.1392` | `0.0724` | `outputs/gr00t_open_loop_actions/gr00t_real_g1_ep0038_checkpoint1000_openloop_actions_53d.json` | `outputs/gr00t_open_loop_actions/gr00t_real_g1_to_a1_26d_approx_ep0038_checkpoint1000_openloop_actions.json` |
| `episode_0040` | 834 | `0.1258` | `0.0595` | `outputs/gr00t_open_loop_actions/gr00t_real_g1_ep0040_checkpoint1000_openloop_actions_53d.json` | `outputs/gr00t_open_loop_actions/gr00t_real_g1_to_a1_26d_approx_ep0040_checkpoint1000_openloop_actions.json` |

Full-episode GR00T replay videos were generated from the approximate A1-style
JSON files:

| Episode | Video frames | Duration at 10 FPS | Output |
| --- | ---: | ---: | --- |
| `episode_0031` | 973 | 97.3 s | `outputs/gr00t_action_replay_full_episode_approx/gr00t_real_g1_to_a1_26d_approx_ep0031_checkpoint1000_full_episode_gt_vs_pred.mp4` |
| `episode_0038` | 647 | 64.7 s | `outputs/gr00t_action_replay_full_episode_approx/gr00t_real_g1_to_a1_26d_approx_ep0038_checkpoint1000_full_episode_gt_vs_pred.mp4` |
| `episode_0040` | 854 | 85.4 s | `outputs/gr00t_action_replay_full_episode_approx/gr00t_real_g1_to_a1_26d_approx_ep0040_checkpoint1000_full_episode_gt_vs_pred.mp4` |

These replay videos are now full raw-episode timelines, not the earlier
102-frame `episode_0030` smoke prefix. The first frames before GR00T has enough
visual history are labeled as missing prediction coverage and rendered with the
GT pose by the replay utility.

### Fine-tune smoke path

For a first wiring check, fine-tune from the base snapshot on the converted
`REAL_G1` dataset with a short run:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 MAX_STEPS=20 SAVE_STEPS=20 \
  bash examples/finetune.sh \
    --base-model-path /home/sys01/.cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495 \
    --dataset-path data/g1_episode_0030_real_g1_gr00t \
    --embodiment-tag REAL_G1 \
    --output-dir outputs/g1_real_g1_finetune_smoke
```

For the older custom upper-body dataset layout, use
`examples/G1/g1_upper_body_config.py` and `--embodiment-tag NEW_EMBODIMENT`.
Do not use that path for evaluating the base `REAL_G1` checkpoint.

### Closed-loop G1 simulation

SimplerEnv is not the right simulator for Unitree G1/SMPLX humanoid evaluation.
Closed-loop visual evaluation needs a G1 simulator adapter that:

- renders/provides `ego_view`
- publishes the `REAL_G1` state groups listed above
- calls `PolicyClient.get_action` or `Gr00tPolicy.get_action`
- applies returned absolute action targets to the G1 simulator
- records rollout video and metrics

Keep A1 model plots, A1 action replay JSONs, and A1 visualization scripts under
`projects/A1/outputs` or `projects/A1/scripts`. This project directory is for
GR00T conversion, inference, fine-tuning, and G1 deployment wiring.

## Excluded From Version Control

- `.venv/` because it is a local environment.
- Most local `data/` content because raw recordings and generated datasets are
  workstation artifacts; `data/g1_episode_0015_psi0_gr00t/` is kept as a compact
  converted sample.
- `media/` and `external_dependencies/` because they are upstream asset and
  dependency payloads, not local source changes.
- `pyproject copy.toml` and local wheel artifacts because they are local backups
  or bulky deployment artifacts.

---

<div align="center">

  <img src="media/header_compress.png" width="800" alt="NVIDIA Isaac GR00T N1.7 Header">

  <!-- --- -->

  <p style="font-size: 1.2em;">
    <a href="https://developer.nvidia.com/isaac/gr00t"><strong>Website</strong></a> |
    <a href="https://huggingface.co/collections/nvidia/gr00t-n17"><strong>Model</strong></a> |
    <a href="https://huggingface.co/collections/nvidia/physical-ai"><strong>Dataset</strong></a> |
    <a href="https://arxiv.org/abs/2503.14734"><strong>Paper</strong></a> |
    <a href="https://developer.nvidia.com/isaac"><strong>NVIDIA Isaac</strong></a> |
    <a href="FAQ.md"><strong>FAQ</strong></a>
  </p>
</div>

## Table of Contents

- [NVIDIA Isaac GR00T](#nvidia-isaac-gr00t)
- [What's New in GR00T N1.7](#whats-new-in-gr00t-n17)
- [Installation](#installation)
- [Model Checkpoints & Embodiment Tags](#model-checkpoints--embodiment-tags)
- [Data Format](#data-format)
- [Inference](#inference)
- [Fine-tuning](#fine-tuning)
- [Evaluation](#evaluation)
- [Contributions](#contributions)
- [License](#license)
- [Citation](#citation)

---

## NVIDIA Isaac GR00T

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td style="width:33.33%; text-align:center;">
      <img src="media/unitree_g1.gif" style="max-width:100%; height:auto;">
    </td>
    <td style="width:33.33%; text-align:center;">
      <img src="media/agibot_g1.gif" style="max-width:100%; height:auto;">
    </td>
    <td style="width:33.33%; text-align:center;">
      <img src="media/yam.gif" style="max-width:100%; height:auto;">
    </td>
  </tr>
</table>

> We just released GR00T N1.7 Early Access, the latest version of GR00T N1 with a new VLM backbone (Cosmos-Reason2-2B / Qwen3-VL) and improved performance.

> **This is an Early Access (EA) release.** You are welcome to download the model, explore the codebase, and begin building on the stack, with the understanding that support and stability guarantees are limited until the GA release.
>
> **What's available:**
> - Pre-trained GR00T N1.7 model weights and reference code
> - Fine-tuning and inference with custom robot data or demonstrations
> - Experimentation, prototyping, and research use cases
>
> **Available at GA:**
> - Production deployment with commercial support
> - Complete benchmarks and a fully validated, stable feature set
> - Pull request contributions
>
> We welcome feedback - please feel free to raise issues in this repository.

> To use older versions: [N1.6](https://github.com/NVIDIA/Isaac-GR00T/releases/tag/n1.6-release) | [N1.5](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release)

NVIDIA Isaac GR00T N1.7 is an open vision-language-action (VLA) model for generalized humanoid robot skills. This cross-embodiment model takes multimodal input, including language and images, to perform manipulation tasks in diverse environments.

GR00T N1.7 is trained on a diverse mixture of robot data including bimanual, semi-humanoid and an expansive humanoid dataset. It is adaptable through post-training for specific embodiments, tasks and environments.

GR00T N1.7 is fully commercially licensable under Apache 2.0. It delivers comparable performance to N1.6, with improved generalization and language-following capabilities driven by the inclusion of 20K hours of EgoScale human video data in pretraining.

The neural network architecture of GR00T N1.7 is a combination of vision-language foundation model and diffusion transformer head that denoises continuous actions. Here is a schematic diagram of the architecture:

<div align="center">
<img src="media/model-architecture.png" width="800" alt="model-architecture">
</div>

### Workflow Overview

1. **Prepare data** — Collect robot demonstrations (video, state, action) and convert them to the [GR00T LeRobot format](#data-format). Demo datasets are included for quick testing.
2. **Run inference** — Try zero-shot inference with the base model on [pretrain embodiments](#embodiment-tags), or use a [finetuned checkpoint](#checkpoints) for benchmark tasks.
3. **Fine-tune** — Adapt the model to your robot using [`launch_finetune.py`](#fine-tuning) with your own data and modality config.
4. **Evaluate** — Validate with [open-loop evaluation](#open-loop-evaluation), then test in [simulation benchmarks](#benchmark-examples) or on real hardware via the [Policy API](getting_started/policy.md).
5. **Deploy** — Connect `Gr00tPolicy` to your robot controller, optionally accelerated with [TensorRT](scripts/deployment/README.md).

## What's New in GR00T N1.7

GR00T N1.7 builds on N1.6 with a new VLM backbone and code-level improvements.

### Key Changes from N1.6

- **New VLM backbone:** Cosmos-Reason2-2B (Qwen3-VL architecture), replacing the Eagle backbone used in N1.6. Supports flexible resolution and encodes images in their native aspect ratio without padding.
- Simplified data processing pipeline (`processing_gr00t_n1d7.py`).
- Added full pipeline export to ONNX and TensorRT with improved frequency.

---

## Installation

### Hardware Requirements

**Inference:** 1 GPU with 16 GB+ VRAM (e.g., RTX 4090, L40, H100, Jetson AGX Thor/Orin, DGX Spark).

**Fine-tuning:** 1 or more GPUs with 40 GB+ VRAM recommended. We recommend H100 or L40 nodes for optimal performance. Other hardware (e.g., A6000) works but may require longer training time. See the [Hardware Recommendation Guide](getting_started/hardware_recommendation.md) for detailed specs.

**CUDA / Python per platform:** dGPU on CUDA 12.8 with Python 3.10; Jetson Orin on CUDA 12.6 with Python 3.10; Jetson Thor and DGX Spark on CUDA 13.0 with Python 3.12. The per-platform install scripts and Dockerfiles live under `scripts/deployment/`; see the [Deployment & Inference Guide](scripts/deployment/README.md) for the full matrix.

### Clone the Repository

GR00T relies on submodules for certain dependencies. Include them when cloning:

**Note:** `git-lfs` is **required** to download parquet data files in `/demo_data`. Install it before cloning: `sudo apt install git-lfs && git lfs install`.
```sh
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
```

If you've already cloned without submodules, initialize them separately:

```sh
git submodule update --init --recursive
```

### Set Up the Environment

GR00T uses [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency management. Install uv first:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### dGPU (x86_64) — Default

Install FFmpeg (required by `torchcodec`, the default video backend):
```sh
sudo apt-get update && sudo apt-get install -y ffmpeg
```

Create the environment and install GR00T:
```sh
uv sync --python 3.10
```
GPU dependencies (flash-attn, TensorRT, etc.) are included in the default install.

Verify the installation:
```sh
uv run python -c "import gr00t; print('GR00T installed successfully')"
```

> **`flash-attn` message on every `uv run`:** You may see `Installing flash-attn...` each time you run `uv run`. This is a known `uv` behavior with URL-pinned wheel sources — `uv` re-validates the cached wheel against the source URL on each invocation. It is **not** rebuilding from source; the wheel is already cached locally and the operation takes 2-3 seconds. This only affects x86_64 platforms. 
> To suppress it, remove the `flash-attn` entries under `[tool.uv.sources]` in your local `pyproject.toml` after the initial install. But that will break `uv lock` and cause flash-attn to build from source on next lock regeneration.

<details>
<summary><strong>Alternative: pip install (without uv)</strong></summary>

If you prefer pip/conda over uv, create a Python 3.10 virtualenv and install:
```sh
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e .
```
Note: GPU dependencies (flash-attn, TensorRT) may require manual installation with pip. The `uv` workflow handles these automatically.
</details>

> **If fine-tuning fails with `CUDA_HOME is unset`:** Run `bash scripts/deployment/dgpu/install_deps.sh` once to configure CUDA paths, or manually `export CUDA_HOME=/usr/local/cuda`.

> **CUDA 13.x Users (Thor, Spark, and other CUDA 13+ platforms):** PyTorch 2.7 pins Triton to 3.3.1, which does not recognize CUDA major version 13+. This causes a `RuntimeError` in Triton's `ptx_get_version()`. Run the patch script to fix:
> ```sh
> uv run bash scripts/patch_triton_cuda13.sh
> ```

> **GB300 (sm_103) Users:** Triton 3.3.1 (pinned by PyTorch 2.7) does not support the GB300 GPU architecture (sm_103). `torch.compile` will fail on GB300. Use PyTorch eager mode or TensorRT inference instead. Triton 3.5.1+ adds sm_103 support but is not yet compatible with the pinned PyTorch version.

> **aarch64 Video Backend:** On aarch64 platforms (Thor, Orin, Spark), `torchcodec` is the required video backend. `install_deps.sh` prefers the prebuilt aarch64 wheel under `scripts/deployment/dgpu/wheels/` (shared by Thor/Spark against FFmpeg 6; Orin uses a matching build against FFmpeg 4) and falls back to a source build only if the wheel is missing. If you encounter `NotImplementedError` from the video backend, ensure `torchcodec` was installed successfully during setup. Other backends (decord, pyav) are not supported on aarch64.

<details>
<summary><strong>DGX Spark</strong> (tested with DGX Spark GB10)</summary>

```bash
bash scripts/deployment/spark/install_deps.sh
source .venv/bin/activate
source scripts/activate_spark.sh
```

See the [Spark setup guide](scripts/deployment/README.md#dgx-spark-setup) for Docker and bare metal details.
</details>

<details>
<summary><strong>Jetson AGX Thor</strong> (tested with JetPack 7.1)</summary>

> **flash-attn on older systems (e.g., Ubuntu 20.04 with glibc < 2.35):** The pre-built `flash-attn` wheel may fail with `ImportError: glibc_compat.so: cannot open shared object file`. To fix this, build from source:
> ```sh
> uv pip install flash-attn==2.7.4.post1 --no-binary flash-attn --no-cache
> ```
> This compiles locally (~10-30 minutes) and avoids the glibc compatibility issue.

```bash
bash scripts/deployment/thor/install_deps.sh
source .venv/bin/activate
source scripts/activate_thor.sh
```

See the [Thor setup guide](scripts/deployment/README.md#jetson-thor-setup) for Docker and bare metal details.
</details>

<details>
<summary><strong>Jetson Orin</strong> (tested with JetPack 6.2)</summary>

```bash
bash scripts/deployment/orin/install_deps.sh
source .venv/bin/activate
source scripts/activate_orin.sh
```

See the [Orin setup guide](scripts/deployment/README.md#jetson-orin-setup) for Docker and bare metal details.
</details>

For a containerized setup that avoids system-level dependency conflicts, see our [Docker Setup Guide](docker/README.md).

---

## Model Checkpoints & Embodiment Tags

### Checkpoints

| Checkpoint | Type | Embodiment Tag | Description |
|------------|------|---------------|-------------|
| [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) | Base | See [pretrain tags](getting_started/policy.md#--embodiment-tag) | Base model (3B params) — zero-shot inference on pretrain embodiments, or finetune for new tasks |
| [`nvidia/GR00T-N1.7-LIBERO`](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO) | Finetuned | `LIBERO_PANDA` | Finetuned on [LIBERO](https://libero-project.github.io/) benchmark (Franka Panda) |
| [`nvidia/GR00T-N1.7-DROID`](https://huggingface.co/nvidia/GR00T-N1.7-DROID) | Finetuned | `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT` | Finetuned on [DROID](https://droid-dataset.github.io/) dataset |
| [`nvidia/GR00T-N1.7-SimplerEnv-Bridge`](https://huggingface.co/nvidia/GR00T-N1.7-SimplerEnv-Bridge) | Finetuned | `SIMPLER_ENV_WIDOWX` | Finetuned on SimplerEnv Bridge (WidowX) |
| [`nvidia/GR00T-N1.7-SimplerEnv-Fractal`](https://huggingface.co/nvidia/GR00T-N1.7-SimplerEnv-Fractal) | Finetuned | `SIMPLER_ENV_GOOGLE` | Finetuned on SimplerEnv Fractal (Google Robot) |

> Older versions: [N1.6 checkpoints](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.6-release) | [N1.5 checkpoints](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release)

### Embodiment Tags

Every inference or finetuning command requires an `--embodiment-tag`. The tag determines which modality config (state/action keys, normalization) the model uses. Tags are **case-insensitive**.

For the full list of pretrain and posttrain tags, see the [Policy API Guide — Embodiment Tags](getting_started/policy.md#--embodiment-tag).

---

## Data Format

GR00T uses a flavor of the [LeRobot v2 dataset format](https://github.com/huggingface/lerobot) with an additional `meta/modality.json` file that describes state/action/video structure. A dataset looks like:

```
my_dataset/
  meta/
    info.json            # dataset metadata
    episodes.jsonl       # episode index and lengths
    tasks.jsonl          # language task descriptions
    modality.json        # state/action/video key mapping (GR00T-specific)
  data/chunk-000/        # parquet files (state, action per timestep)
  videos/chunk-000/      # mp4 video files per episode
```

The `modality.json` maps how the concatenated state/action arrays split into named fields (e.g., `x`, `y`, `z`, `gripper`) and which video keys are available. This is what the embodiment tag uses to interpret the data.

**Included demo datasets** (ready to use, no download needed):

| Dataset | Robot | Embodiment Tag | Use Case |
|---------|-------|---------------|----------|
| `demo_data/droid_sample` | DROID (3 episodes) | `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT` | Zero-shot or finetuned inference (DROID) |
| `demo_data/libero_demo` | LIBERO Panda (5 episodes) | `LIBERO_PANDA` | Inference with finetuned checkpoint |
| `demo_data/simplerenv_bridge_sample` | WidowX (SimplerEnv Bridge) | `SIMPLER_ENV_WIDOWX` | Inference with finetuned SimplerEnv Bridge checkpoint |
| `demo_data/simplerenv_fractal_sample` | Google Robot (SimplerEnv Fractal) | `SIMPLER_ENV_GOOGLE` | Inference with finetuned SimplerEnv Fractal checkpoint |
| `demo_data/cube_to_bowl_5` | SO100 arm (5 episodes) | `NEW_EMBODIMENT` | Fine-tuning custom embodiment example |
| `demo_data/cube_to_bowl_5_with_mask` | SO100 arm + per-frame masks | `NEW_EMBODIMENT` | [Mask-guided background suppression](examples/mask-guided-background-suppression/README.md) example |

> To generate more DROID episodes: `python scripts/download_droid_sample.py --num-episodes 10`

**Using your own data:** Convert your demonstrations to the format above. If coming from LeRobot v3, use the conversion script: `python scripts/lerobot_conversion/convert_v3_to_v2.py`. See the full [Data Preparation Guide](getting_started/data_preparation.md) for schema details and examples.

---

## Inference

### Zero-Shot Inference (Base Model)

The included `demo_data/droid_sample` dataset works with the base model out of the box — no finetuning or checkpoint download needed:

```bash
uv run python scripts/deployment/standalone_inference_script.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --traj-ids 1 2 \
    --inference-mode pytorch \
    --action-horizon 8
```

This runs open-loop inference on 2 DROID episodes, comparing predicted actions against ground truth. The base model downloads automatically from HuggingFace on first run (~6 GB).

### Finetuned Inference

For posttrain embodiments, use a finetuned checkpoint. Most finetuned checkpoints (e.g., DROID, SimplerEnv) have a flat file structure and can be passed directly as a HuggingFace model ID — no manual download needed:

```bash
uv run python scripts/deployment/standalone_inference_script.py \
    --model-path nvidia/GR00T-N1.7-DROID \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --traj-ids 1 2 \
    --inference-mode pytorch \
    --action-horizon 8
```

Some checkpoints (e.g., LIBERO) use a nested folder structure with model files under a subfolder. HuggingFace does not support nested repo paths in `--model-path`, so you must download first:

```bash
uv run hf download nvidia/GR00T-N1.7-LIBERO \
    --include "libero_10/config.json" "libero_10/embodiment_id.json" \
    "libero_10/model-*.safetensors" "libero_10/model.safetensors.index.json" \
    "libero_10/processor_config.json" "libero_10/statistics.json" \
    --local-dir checkpoints/GR00T-N1.7-LIBERO
```

```bash
uv run python scripts/deployment/standalone_inference_script.py \
    --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
    --dataset-path demo_data/libero_demo \
    --embodiment-tag LIBERO_PANDA \
    --traj-ids 0 1 2 \
    --inference-mode pytorch \
    --action-horizon 8
```

### Server-Client Inference (for Deployment)

For real-world deployment or simulation evaluation, use the server-client architecture. The policy runs on a GPU server; a lightweight client sends observations and receives actions over ZMQ.

**Terminal 1 — Start the policy server:**
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --device cuda:0
```

**Terminal 2 — Run open-loop evaluation as a client:**
```bash
uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --host 127.0.0.1 \
    --port 5555 \
    --traj-ids 1 2 \
    --action-horizon 8
```

> **Tip:** If you get `ZMQError: Address already in use`, the default port 5555 is occupied. Use `--port <other_port>`.

For connecting to a real robot (e.g., DROID hardware), see [examples/DROID/README.md](examples/DROID/README.md). For faster inference with TensorRT, see the [Deployment & Inference Guide](scripts/deployment/README.md).

See the complete [Policy API Guide](getting_started/policy.md) for documentation on observation/action formats, batched inference, and troubleshooting.

---

## Fine-tuning

### Reproducing Benchmark Results

Each benchmark has a self-contained README with dataset download, finetune, and evaluation commands:

| Benchmark | Embodiment | Guide |
|-----------|-----------|-------|
| LIBERO | `LIBERO_PANDA` | [examples/LIBERO/README.md](examples/LIBERO/README.md) |
| SimplerEnv (Fractal) | `SIMPLER_ENV_GOOGLE` | [examples/SimplerEnv/README.md](examples/SimplerEnv/README.md) |
| SimplerEnv (Bridge) | `SIMPLER_ENV_WIDOWX` | [examples/SimplerEnv/README.md](examples/SimplerEnv/README.md) |
| SO100 | `NEW_EMBODIMENT` | [examples/SO100/README.md](examples/SO100/README.md) |

### Fine-tune on Your Own Robot ("NEW_EMBODIMENT")

To finetune GR00T on your own robot data and configuration, follow the detailed tutorial at [`getting_started/finetune_new_embodiment.md`](getting_started/finetune_new_embodiment.md).

Ensure your input data follows the [GR00T LeRobot format](#data-format), and specify your modality configuration via `--modality-config-path`.

**Single GPU:**
```bash
CUDA_VISIBLE_DEVICES=0 uv run python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/cube_to_bowl_5 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/SO100/so100_config.py \
    --num-gpus 1 \
    --output-dir /tmp/test_finetune \
    --max-steps 2000 \
    --global-batch-size 32 \
    --dataloader-num-workers 4
```

**Multi-GPU (e.g., 8xH100):**
```bash
uv run torchrun --nproc_per_node=8 --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/cube_to_bowl_5 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/SO100/so100_config.py \
    --num-gpus 8 \
    --output-dir /tmp/test_finetune_8gpu \
    --max-steps 2000 \
    --global-batch-size 32 \
    --dataloader-num-workers 4
```

Replace `demo_data/cube_to_bowl_5` and `examples/SO100/so100_config.py` with your own dataset and modality config. See [`examples/SO100`](examples/SO100/README.md) for a complete walkthrough.

> **Note:** Use `uv run torchrun` (not bare `torchrun`) to ensure the correct virtual environment is used. Add `--use-wandb` to enable Weights & Biases logging. For more extensive configuration, use `gr00t/experiment/launch_train.py`.

### Training Tips

- Maximize batch size for your hardware and train for a few thousand steps.
- Users may observe 5-6% variance between runs due to non-deterministic image augmentations. Keep this in mind when comparing to reported benchmarks.
- **`--state_dropout_prob`** (model config default: 0.8; finetune CLI default: 0.2; see `gr00t/configs/finetune_config.py`): Randomly drops state inputs during training to improve generalization and reduce state-dependency. The shipped benchmark scripts override the CLI default per suite: LIBERO 10-Long uses 0.2 (the CLI default), SimplerEnv Bridge uses 0.8, SimplerEnv Fractal uses 0.5. If your task relies heavily on proprioceptive state, lower this value.

---

## Evaluation

### Open-Loop Evaluation

Compare predicted actions against ground truth from your dataset:

```bash
uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path <DATASET_PATH> \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path <CHECKPOINT_PATH> \
    --traj-ids 0 \
    --action-horizon 16
```

This generates a visualization at `/tmp/open_loop_eval/traj_{traj_id}.jpeg` with ground truth vs. predicted actions and MSE metrics. Use `--save-plot-path <dir>` to save plots to a custom location.

### Closed-Loop Evaluation

Test your model in simulation or on real hardware using the server-client architecture:

```bash
# Start the policy server
uv run python gr00t/eval/run_gr00t_server.py \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path <CHECKPOINT_PATH> \
    --device cuda:0 \
    --host 0.0.0.0 --port 5555
```

```python
from gr00t.policy.server_client import PolicyClient

policy = PolicyClient(host="localhost", port=5555)
env = YourEnvironment()
obs, info = env.reset()
action, info = policy.get_action(obs)
obs, reward, done, truncated, info = env.step(action)
```

**Debugging with ReplayPolicy:** To verify your environment setup without a trained model, start the server with `--dataset-path <DATASET_PATH>` (omit `--model-path`) to replay recorded actions from the dataset.

See the complete [Policy API Guide](getting_started/policy.md) for observation/action formats, batched inference, and troubleshooting.

### Benchmark Examples

We support evaluation on public benchmarks using a server-client architecture. The policy server reuses the project root's uv environment; simulation clients have individual setup scripts.

You can use [the verification script](scripts/eval/check_sim_eval_ready.py) to verify that all dependencies are properly configured.

**Zero-shot** (evaluate with the base model, no finetuning):
- [DROID](examples/DROID/README.md) — real-world DROID robot (also available as the finetuned `nvidia/GR00T-N1.7-DROID` checkpoint; `examples/DROID/README.md` covers both paths)

**Finetuned** (evaluate with finetuned checkpoints):
- [DROID](examples/DROID/README.md) — real-world DROID robot via `nvidia/GR00T-N1.7-DROID`
- [LIBERO](examples/LIBERO/README.md) — LIBERO benchmark (Franka Panda)
- [SimplerEnv](examples/SimplerEnv/README.md) — Google Robot (Fractal) and WidowX (Bridge)
- [SO100](examples/SO100/README.md) — SO100 custom embodiment workflow

<details>
<summary><strong>Adding a New Sim Benchmark</strong></summary>

Each sim benchmark registers its environments under a gym env_name with the format `{prefix}/{task_name}` (e.g., `libero_sim/LIVING_ROOM_SCENE2_put_soup_in_basket`). The evaluation framework uses the prefix to look up the corresponding `EmbodimentTag` via a mapping in [`gr00t/eval/sim/env_utils.py`](gr00t/eval/sim/env_utils.py).

> **Important:** The env_name prefix and the `EmbodimentTag` value are often different. For example, `libero_sim` maps to `EmbodimentTag.LIBERO_PANDA` (`"libero_sim"`). Do not assume they match.

To add a new benchmark:

1. Add an entry to `ENV_PREFIX_TO_EMBODIMENT_TAG` in `gr00t/eval/sim/env_utils.py`:
   ```python
   ENV_PREFIX_TO_EMBODIMENT_TAG = {
       ...
       "my_new_benchmark": EmbodimentTag.MY_ROBOT,
   }
   ```
2. If the benchmark has multiple env_name prefixes (e.g., `my_benchmark_v1`, `my_benchmark_v2`), all related prefixes **must** map to the same `EmbodimentTag`.
3. Add corresponding test cases in `tests/gr00t/eval/sim/test_env_utils.py` and update the `test_all_known_prefixes_present` test.
</details>



---

# Contributions

During Early Access we are not accepting pull requests while the codebase stabilizes. If you encounter issues or have suggestions, please open an [Issue](https://github.com/NVIDIA/Isaac-GR00T/issues) in this repository.

# Support

Support during Early Access is best-effort. We will continue iterating toward a more stable General Availability (GA) release.


## License

- **Code:** Apache 2.0 — see [LICENSE](LICENSE)
- **Model weights:** [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)

```
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```


## Citation

[Paper Site](https://research.nvidia.com/labs/lpr/publication/gr00tn1_2025/)
```bibtex
@inproceedings{gr00tn1_2025,
  archivePrefix = {arxiv},
  eprint     = {2503.14734},
  title      = {{GR00T} {N1}: An Open Foundation Model for Generalist Humanoid Robots},
  author     = {NVIDIA and Johan Bjorck and Fernando Castañeda, Nikita Cherniadev and Xingye Da and Runyu Ding and Linxi "Jim" Fan and Yu Fang and Dieter Fox and Fengyuan Hu and Spencer Huang and Joel Jang and Zhenyu Jiang and Jan Kautz and Kaushil Kundalia and Lawrence Lao and Zhiqi Li and Zongyu Lin and Kevin Lin and Guilin Liu and Edith Llontop and Loic Magne and Ajay Mandlekar and Avnish Narayan and Soroush Nasiriany and Scott Reed and You Liang Tan and Guanzhi Wang and Zu Wang and Jing Wang and Qi Wang and Jiannan Xiang and Yuqi Xie and Yinzhen Xu and Zhenjia Xu and Seonghyeon Ye and Zhiding Yu and Ao Zhang and Hao Zhang and Yizhou Zhao and Ruijie Zheng and Yuke Zhu},
  month      = {March},
  year       = {2025},
  booktitle  = {ArXiv Preprint},
}
```

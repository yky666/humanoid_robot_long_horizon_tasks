# Import Log

This file tracks each managed-repo import/update step.

## 2026-04-22

### Step 1

- Commit message: `repo: initialize managed workspace structure`
- Scope:
  - replace placeholder root README with workspace governance documentation
  - add root `.gitignore`
  - add import log and repository workflow docs
  - reserve project directories under `projects/`
- README updates:
  - root `README.md` now explains repository policy, layout, and migration plan

### Step 2

- Commit message: `import: add A1 working snapshot`
- Scope:
  - import the current local `A1` working tree into `projects/A1`
  - keep local code edits that are not yet committed upstream
  - exclude heavy local-only artifacts such as `model/`, outputs, and local envs
- README updates:
  - root `README.md` marks `A1` as imported
  - `projects/A1/README.md` records upstream origin, base commit, local diffs, and excluded paths

### Step 3

- Commit message: `import: add dex-retargeting working snapshot`
- Scope:
  - import the current local `dex-retargeting` working tree into `projects/dex-retargeting`
  - preserve local teleoperation and hand-specific customizations from the workstation
  - exclude local build and environment artifacts only
- README updates:
  - root `README.md` marks `dex-retargeting` as imported
  - `projects/dex-retargeting/README.md` records upstream origin, base commit, local diffs, and excluded paths

### Step 4

- Commit message: `import: add Isaac-GR00T working snapshot`
- Scope:
  - import a trimmed local `Isaac-GR00T` working tree into `projects/Isaac-GR00T`
  - preserve the local `pyproject.toml` modification
  - exclude workstation-only heavy artifacts including `.venv/`, demo data, media, large deployment wheels, and checked-out submodule payloads
- README updates:
  - root `README.md` marks `Isaac-GR00T` as imported
  - `projects/Isaac-GR00T/README.md` records upstream origin, base commit, local diffs, excluded paths, and the trimmed-import caveat

## 2026-04-24

### Step 5

- Commit message: `a1: add managed g1 smoke finetune workflow`
- Scope:
  - sync the missing managed `projects/A1/a1/data/` code required by the local A1 train and eval entrypoints
  - add managed G1 dataset and experiment configs for `episode_0013`
  - update the G1 converter, deploy server, deploy inference path, and offline debug evaluator to support this workstation workflow
  - run a dual-4090 smoke finetune on the local A1 7B checkpoint and save a 1-step checkpoint
  - run offline pretrained-vs-finetuned action-error evaluation on the converted G1 dataset
- README updates:
  - `projects/A1/README.md` now records the exact conversion, smoke finetune, and offline evaluation commands for `episode_0013`
  - root `README.md` marks `A1` as having a managed G1 workflow update on 2026-04-24
- Key observed results:
  - the 20-step dual-4090 smoke finetune still OOMs on step 2 backward with the local 7B checkpoint
  - the 1-step dual-4090 smoke finetune completes and saves `step1`, `step1-unsharded`, and `step1-action-head`
  - on the same 20 debug samples, offline action error improved from `avg_l1=0.4519`, `avg_mse=0.4916` to `avg_l1=0.4376`, `avg_mse=0.4500` after the 1-step smoke finetune

## 2026-04-26

### Step 6

- Commit message: `a1: document g1 continuation and training inspection`
- Scope:
  - clarify that the current single G1 episode can still be used for continued same-episode finetuning
  - document the exact manual continuation command from the saved `step1-unsharded` checkpoint
  - document how to inspect training quality through terminal loss, optional WandB curves, and offline action-error evaluation
  - document the current limitation that this managed workflow does not yet provide a ready-made G1 visual simulation or rollout video path
- README updates:
  - `projects/A1/README.md` now includes a dedicated section on continuing finetune with the current dataset and a dedicated section on how to inspect training results and limitations

### Step 7

- Commit message: `a1: stabilize dual-4090 multistep finetune`
- Scope:
  - patch `launch_scripts/utils.py` so managed CLI startup no longer crashes when `HF_HOME` is unset
  - patch `launch_scripts/train_vla.py` to expose `--fsdp_precision` and `--disable_float32_attention` as explicit manual controls
  - patch `scripts/train_for_action.py` to support filtered checkpoint loading when experimental action-head shapes do not match the pretrained checkpoint exactly
  - verify that the original 7B checkpoint architecture can complete a managed `5-step` dual-4090 finetune when using `--fsdp_precision pure`, `--disable_float32_attention`, and `--seq_len 512`
  - run offline debug evaluation for the new `step5-unsharded` checkpoint on the same fixed 20-sample G1 subset
- README updates:
  - `projects/A1/README.md` now records the exact dual-4090 multi-step finetune command, the shell-formatting caveat for multi-line commands, the new CLI memory flags, and the `step5` offline evaluation result
- Key observed results:
  - the original managed 7B path completed `5/5` training steps and saved `step5`, `step5-unsharded`, and `step5-action-head`
  - on the same fixed 20 debug samples, offline action error reached `avg_l1=0.4310`, `avg_mse=0.4478`
  - resized-action-head experiments are now load-compatible, but they still are not the recommended first choice on dual `RTX 4090` because FSDP wrap-time OOM remains possible before training starts

### Step 8

- Commit message: `a1: add staged g1 run inspection workflow`
- Scope:
  - inspect the live `50000-step` dual-4090 G1 run and confirm it is not stuck before training, but instead is progressing slowly with expensive full-checkpoint saves
  - patch `launch_scripts/train_vla.py` to expose a separate `--save_interval_action_head` control so long runs can save lighter action-head checkpoints more often than full model checkpoints
  - add `projects/A1/scripts/inspect_training_run.py` to parse the local WandB `output.log`, summarize progress and ETA, export CSV and JSON summaries, and render local training curves without depending on the online WandB UI
  - document a staged path from same-episode overfit to multi-episode finetune and finally guarded G1 real-robot validation
- README updates:
  - `projects/A1/README.md` now documents why the current `50000-step` run looks slow, the estimated runtime implications, the new local run-inspection command, the new `--save_interval_action_head` flag, and the recommended staged real-robot finetune path
- Key observed results:
  - the current `50000-step` run reached `step100`, logged `train/ActionNoiseL2Loss=0.8041`, and successfully saved `step100`, `step100-unsharded`, and `step100-action-head`
  - the apparent "hang" was mainly caused by `log_interval=100` plus heavy checkpoint I/O, not by a pre-training crash
  - at the observed speed on this workstation, a single-episode `50000-step` run is a roughly `10` day path and is not the recommended first route to a meaningful G1 deployment result

## 2026-04-27

### Step 9

- Commit message: `a1: record latest long-run eval comparison`
- Scope:
  - re-inspect the long-running single-episode G1 finetune and confirm the retained latest checkpoint had advanced from the original `step100` state to `step900`
  - launch a temporary single-GPU inference server for `step900-unsharded`
  - run the same fixed 20-sample offline debug evaluation used for the earlier pretrained and `step5` comparisons
  - document the resulting `step900` metrics and the interpretation caveat that average improvement is not uniform sample-by-sample
- README updates:
  - `projects/A1/README.md` now records that `step100` was rotated out by checkpoint retention, points to the retained `step900-unsharded` checkpoint, and adds the `step900` offline comparison result
- Key observed results:
  - the retained latest checkpoint from the long run is `step900-unsharded`
  - on the same fixed 20-sample subset, `step900` reached `avg_l1=0.4329`, `avg_mse=0.3130`
  - compared with `step5`, `step900` produced nearly unchanged mean `L1` but much lower mean `MSE`
  - the `step900` improvement is not uniform across all samples, so it should be treated as evidence of stronger overfit on some cases rather than as a clean deployment-ready win

### Step 10

- Commit message: `a1: record 200-sample eval comparison`
- Scope:
  - launch temporary single-GPU inference servers for `step5-unsharded` and `step900-unsharded`
  - run larger fixed-seed offline debug evaluations with `200` samples for both checkpoints
  - compare the larger-sample results against the earlier `20`-sample conclusion to check whether the apparent `MSE`-only improvement persists
- README updates:
  - `projects/A1/README.md` now records the new `step5` and `step900` `200`-sample evaluation artifacts and explains the larger-sample interpretation
- Key observed results:
  - on `200` fixed samples, `step900` beat `step5` on the mean of both metrics: `avg_l1=0.3824` vs `0.4541`, and `avg_mse=0.2579` vs `0.4808`
  - the earlier `20`-sample ambiguity did not hold up under a larger sample count; the long-run checkpoint is better on average by both metrics
- however, `step900` is still not a majority winner sample-by-sample: it beat `step5` on `95/200` samples by `L1` and `100/200` samples by `MSE`
- this suggests the long-run checkpoint is improving average quality through larger wins on some cases rather than through uniformly better behavior on most individual examples

## 2026-04-28

### Step 11

- Commit message: `a1: record pretrain 200-sample baseline and phase analysis`
- Scope:
  - launch a temporary single-GPU inference server for the original pretrained checkpoint and run the same fixed-seed `200`-sample offline evaluation used for the `step5` and `step900` checkpoints
  - add a reusable `scripts/analyze_eval_deltas.py` helper for pairwise checkpoint comparison, per-phase aggregation, and top improved / degraded sample extraction
  - compare `pretrain -> step5`, `step5 -> step900`, and `pretrain -> step900` on exactly the same `200` sampled frames
  - inspect the strongest `step900` improvements and degradations against raw episode frames to connect metric shifts to concrete task stages
- README updates:
  - `projects/A1/README.md` now records the exact `pretrain` `200`-sample eval command, the pairwise analysis commands, and the stage-level interpretation for `step900`
  - root `README.md` now marks `A1` as having a `pretrain` `200`-sample baseline and phase analysis update on 2026-04-28
- Key observed results:
  - on `200` fixed-seed samples, the pretrained checkpoint reached `avg_l1=0.4747`, `avg_mse=0.5349`
  - `step5` improves on that baseline modestly, reaching `avg_l1=0.4541`, `avg_mse=0.4808`
  - `step900` improves on both earlier checkpoints much more strongly on the mean, reaching `avg_l1=0.3824`, `avg_mse=0.2579`
  - the phase split shows where the gain comes from: `step900` beats `step5` on all `75/75` `grasp_lift` samples, but still loses on most `approach` and `carry_place_right` samples
  - the largest `step900` wins occur while the bottle is still central and the robot is actively grasping or lifting it, while the largest regressions occur after the bottle has already moved right and the robot is releasing or retreating

## 2026-04-29

### Step 12

- Commit message: `gr00t: add g1 psi0 episode conversion sample`
- Scope:
  - add `projects/Isaac-GR00T/scripts/convert_g1_raw_episode_to_gr00t.py`
    for converting raw Unitree G1 episode dumps into GR00T LeRobot v2 format
  - support both the legacy upper-body vector layout and the newer `psi0`
    layout exposed by `states.psi0` and `actions.psi0`
  - support multi-camera color streams, including `color_0 -> ego_view` and
    `color_1 -> wrist`
  - add compact converted sample dataset
    `projects/Isaac-GR00T/data/g1_episode_0015_psi0_gr00t/`
- README updates:
  - root `README.md` now marks `Isaac-GR00T` as having the local G1 psi0
    conversion sample update on 2026-04-29
  - `projects/Isaac-GR00T/README.md` records the conversion script, included
    sample dataset, and why this specific generated data artifact is tracked
- Key observed results:
  - `episode_0015` converts to `385` frames
  - `observation.state` has shape `[32]`
  - `action` has shape `[36]`
  - the dataset includes `observation.images.ego_view` and
    `observation.images.wrist`

### Step 13

- Commit message: `archive: add xr_teleoperate and teleimager working snapshots`
- Scope:
  - archive the current local `/home/unitree/xr_teleoperate` working tree under `projects/xr_teleoperate`
  - archive the current local `/home/unitree/teleimager` working tree under `projects/teleimager`
  - preserve local bridge, replay, recording, image server/client, config, README, and changelog updates
  - keep the monorepo layout consistent by placing both teleoperation-related codebases under `projects/`
  - exclude heavy local datasets and sensitive certificate material from version control
- README updates:
  - root `README.md` now lists `xr_teleoperate` and `teleimager` as managed archived projects
  - `projects/README.md` now references the archived teleoperation projects
  - `projects/xr_teleoperate/README_ARCHIVE.md` and `projects/teleimager/README_ARCHIVE.md` record upstream sources, local scope, and exclusions
- Key observed results:
  - `projects/xr_teleoperate` captures the local bridge/replay workflow including `teleop_hand_and_arm_bridge.py`, `replay_episode.py`, `replay_episode_player.py`, and `convert_episode_to_lerobot.py`
  - `projects/teleimager` captures the local image stack updates including `image_client.py`, `image_server.py`, `cam_config_server.yaml`, and `psi0_bridge.py`
  - `projects/xr_teleoperate/teleop/utils/data` was intentionally excluded because it is large local data rather than repository source

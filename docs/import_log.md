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

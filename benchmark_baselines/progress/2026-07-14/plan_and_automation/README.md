# Unified multi-task baseline experiments

Remote root: `/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/Humanoid/yangky`

This directory records the reproducible plan and automation used for the
SIMPLE and DexJoCo baseline study.  A run is valid only if one checkpoint is
trained from a benchmark-level task mixture.  Per-task checkpoints are not
part of this study.

## Current progress and archives

At the 2026-07-14 (Asia/Shanghai) snapshot, Psi0/SIMPLE full training had
reached step 23,956/40,000 (60%). Checkpoints at steps 10k, 15k, and 20k were
present, recent losses were mainly 0.15--1.0, and GPU memory use was about
17.2/80 GiB. The completion watcher and the subsequent ACT/Diffusion Policy
queue were alive.

Progress snapshots are stored remotely under `baseline_experiments/archives/`;
`archives/LATEST` identifies the canonical sanitized snapshot. Run
`archive_progress.py` to capture a new snapshot. It archives full logs,
resolved launch arguments/configuration, data manifests, repository revisions,
status files, and SHA-256 checksums, while intentionally excluding environment
files that might contain credentials.

## Frozen scope

Models: Psi0, pi0.5, RLDX-1, EgoVLA, GR00T N1.6, GR00T N1.7, ABot-M0.5,
ACT, and Diffusion Policy.

Benchmarks:

- SIMPLE: all released humanoid tasks in one balanced training mixture.
- DexJoCo: all 11 tasks in one balanced training mixture.  Single-arm samples
  are padded and masked into the common bimanual 44-D action / 46-D state
  schema.  The normalized view lives at `data/dexjoco_normalized`; videos are
  referenced by symlink rather than copied.  Following the official converter,
  a single-arm wrist stream is explicitly aliased to both wrist camera slots,
  and this mapping is recorded in `manifest.json`.

RoboCasa365 is an auxiliary source and compatibility check for RLDX-1 and
ABot-M0.5.  It is not silently mixed into SIMPLE or DexJoCo training.

## Execution stages

1. Inventory code commits, dataset manifests, GPU, driver, and seeds.
2. Validate every LeRobot dataset and build balanced benchmark mixtures.
3. Run one-batch loader tests and 20-step smoke training for every model.
4. Run full multi-task training sequentially on the single A800 80GB.
5. Run open-loop checks, then simulator rollouts per task and randomization
   level with seeds 0, 1, and 2.
6. Aggregate success rate, progress score, inference latency, peak VRAM, and
   wall-clock training time.  Preserve raw logs and resolved configs.

## Automatic recovery policy

- Network failure: exponential-backoff retry, without changing artifacts.
- CUDA OOM: halve micro-batch and double gradient accumulation, preserving the
  effective batch size where possible.
- NaN/Inf: restart from the last finite checkpoint at half learning rate.
- Repeated failure: mark the cell blocked with the exact log; never substitute
  a different model or a single-task run.

The machine-readable matrix is in `matrix.json`; `validate_data.py` produces
the initial data audit used by the experiment supervisor.

## Upstream availability note

As of 2026-07-13, the official ABot-M0.5 repository explicitly labels both
code and weights as coming soon.  Public ABot-M0 code is a different model and
must not be reported as M0.5.  The ABot-M0.5 cells therefore remain registered
but blocked until the promised implementation is public.  RLDX-1 and GR00T
N1.7 both expose genuine multi-dataset training interfaces and can proceed.

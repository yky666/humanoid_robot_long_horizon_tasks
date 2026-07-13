# Unified multi-task baseline experiments

This directory is the publication bundle for the SIMPLE and DexJoCo baseline
study. A reported run is valid only when one checkpoint is trained from the
complete benchmark-level task mixture; per-task checkpoints are excluded.

## Scope

- Benchmarks: SIMPLE (17 canonical tasks) and DexJoCo (11 tasks).
- Models: Psi0, pi0.5, RLDX-1, EgoVLA, GR00T N1.6/N1.7, ABot-M0.5, ACT, and
  Diffusion Policy.
- DexJoCo schema: 44-D action and 46-D state with explicit valid-dimension masks.
- Evaluation seeds: 0, 1, and 2.

## Latest archived progress

The historical snapshot under `progress/2026-07-14/` keeps the earlier launch
configuration, repository revisions, data manifests, checksums, and raw logs.

Current live metrics (parsed from remote training logs) live under
[`meeting_summary/`](meeting_summary/00_%E7%BB%84%E4%BC%9A%E6%B1%87%E6%8A%A5%E9%80%9F%E8%A7%88.md):

- **Psi0 / SIMPLE**: full complete (`40000/40000`), latest loss `0.375`,
  checkpoints through `40k`. Curves/CSV/JSON under
  `meeting_summary/artifacts/psi0_simple/`.
- **Diffusion Policy / SIMPLE**: full running (`2089/40000`, ~5.22% at last
  parse), latest loss `0.0931`. Curves/CSV/JSON under
  `meeting_summary/artifacts/diffusion_policy_simple/`.
- **ACT / SIMPLE**: smoke failed earlier on ResNet download; weight cached and
  queued to retry after DP finishes.
- Overview JSON: `meeting_summary/artifacts/overview.json`.

Refresh helper: `update_training_metrics.py` (parse remote tqdm logs → CSV /
summary / plots). These are training metrics only; simulator success rates
require SIMPLE closed-loop rollouts.

Large model checkpoints and datasets are intentionally not committed. Their
remote paths and reproducibility metadata are recorded in the snapshot.

ABot-M0.5 remains a registered blocked cell until its official implementation
and weights are released; ABot-M0 is not substituted.

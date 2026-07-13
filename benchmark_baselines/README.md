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

The snapshot under `progress/2026-07-14/` records the resolved launch
configuration, repository revisions, data manifests, checksums, status files,
and raw logs. At snapshot time, Psi0/SIMPLE was at step 25,160/40,000 (63%)
with checkpoints through step 25,000. The training process and the automatic
ACT/Diffusion Policy continuation queue were alive.

For group-meeting reporting, see
[`meeting_summary/00_组会汇报速览.md`](meeting_summary/00_%E7%BB%84%E4%BC%9A%E6%B1%87%E6%8A%A5%E9%80%9F%E8%A7%88.md).
It includes the latest parsed live-log metrics, training curves, CSV/JSON
metric exports, and selected W&B offline media samples. The latest parsed
Psi0/SIMPLE full-run progress in that summary is 33,495/40,000 steps (83.74%).

Large model checkpoints and datasets are intentionally not committed. Their
remote paths and reproducibility metadata are recorded in the snapshot.

ABot-M0.5 remains a registered blocked cell until its official implementation
and weights are released; ABot-M0 is not substituted.

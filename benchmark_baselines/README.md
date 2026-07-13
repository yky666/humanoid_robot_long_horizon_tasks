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

Large model checkpoints and datasets are intentionally not committed. Their
remote paths and reproducibility metadata are recorded in the snapshot.

ABot-M0.5 remains a registered blocked cell until its official implementation
and weights are released; ABot-M0 is not substituted.

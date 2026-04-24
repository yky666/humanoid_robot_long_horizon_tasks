# humanoid_robot_long_horizon_tasks

Unified monorepo for our humanoid long-horizon task stack on this workstation.

This repository is the single version-managed entry point for the active codebases
we are using together:

- `A1`
- `dex-retargeting`
- `Isaac-GR00T`

The goal is to keep code, configuration, integration notes, and migration history
in one place so later updates do not drift across multiple local folders.

## Repository Policy

- Track source code, configs, scripts, and project documentation.
- Do not track model weights, local virtual environments, caches, logs, or large
  generated artifacts.
- Every structural import or update must include:
  - a clear git commit message
  - a matching `README.md` update in the affected project folder
  - a short entry in `docs/import_log.md`

## Layout

```text
.
|-- docs/
|   |-- import_log.md
|   `-- repository_workflow.md
|-- projects/
|   |-- README.md
|   |-- A1/
|   |-- dex-retargeting/
|   `-- Isaac-GR00T/
|-- .gitignore
|-- LICENSE
`-- README.md
```

## Managed Projects

| Project | Upstream | Local Status |
|---|---|---|
| `A1` | `https://github.com/ATeam-Research/A1.git` | Imported from local working snapshot on 2026-04-22; managed G1 conversion + smoke finetune workflow added on 2026-04-24 |
| `dex-retargeting` | `https://github.com/dexsuite/dex-retargeting.git` | Imported from local working snapshot on 2026-04-22 |
| `Isaac-GR00T` | `https://github.com/NVIDIA/Isaac-GR00T` | Imported as a trimmed working snapshot on 2026-04-22 |

## Current Migration Notes

- This repo started as a minimal placeholder repository.
- We are converting it into the main workspace management repo for the local
  humanoid robotics stack.
- Initial consolidation excludes heavy local-only assets such as:
  - model checkpoints
  - local Python environments
  - cached deployment wheels
  - sample datasets and generated media unless explicitly needed

## Workflow

1. Import one project at a time from the workstation into `projects/`.
2. Record the upstream source repo and source commit in that project's
   `README.md`.
3. Record local deviations from upstream in the same `README.md`.
4. Commit each import as an isolated step.
5. Push to `main` only after the corresponding README and import log are updated.

The detailed workflow is documented in [docs/repository_workflow.md](/home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/docs/repository_workflow.md).

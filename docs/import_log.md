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

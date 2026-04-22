# Repository Workflow

## Purpose

This repository is the stable coordination point for the humanoid long-horizon
task stack on this machine.

## Rules

- Import code into `projects/<name>/`.
- Preserve upstream reference information in each project `README.md`.
- Describe local workstation-only modifications before pushing.
- Exclude large local artifacts that should not be versioned.

## Required Metadata For Each Project Import

- upstream repository URL
- upstream base commit
- local import date
- local modifications included in the import
- files or directories intentionally excluded from version control

## Commit Style

- `repo: ...` for monorepo-level structure or policy changes
- `import: ...` for first-time project imports
- `update: ...` for later synced changes
- `docs: ...` for documentation-only updates

## Push Discipline

- Do not push an import unless the matching project `README.md` was updated.
- Do not push heavy local artifacts just because they exist in the working tree.
- Prefer small, isolated commits that map to one clear migration step.

# humanoid_robot_long_horizon_tasks

Unified monorepo for the humanoid long-horizon task stack on this workstation.

This repository now also archives the local `xr_teleoperate` and `teleimager`
working trees under the managed `projects/` layout so the teleoperation stack
is versioned alongside the rest of the workspace.

## Layout

- `projects/xr_teleoperate/`: archived working tree from `/home/unitree/xr_teleoperate`
- `projects/teleimager/`: archived working tree from `/home/unitree/teleimager`
- `ARCHIVE_NOTES.md`: snapshot notes, exclusions, and source pointers
- `docs/`: repository workflow and import log

## What Is Included

| Project | Upstream | Local Status |
|---|---|---|
| `A1` | `https://github.com/ATeam-Research/A1.git` | Imported from local working snapshot on 2026-04-22; managed G1 conversion + smoke finetune workflow added on 2026-04-24; staged G1 run inspection and real-robot finetune notes added on 2026-04-26; latest long-run and 200-sample eval comparisons added on 2026-04-27; pretrained 200-sample baseline and phase analysis added on 2026-04-28 |
| `dex-retargeting` | `https://github.com/dexsuite/dex-retargeting.git` | Imported from local working snapshot on 2026-04-22 |
| `Isaac-GR00T` | `https://github.com/NVIDIA/Isaac-GR00T` | Imported as a trimmed working snapshot on 2026-04-22; local G1 psi0 conversion script and compact `episode_0015` GR00T dataset sample added on 2026-04-29 |

The current archive snapshot also includes:

- README files and changelogs
- local code updates
- replay and bridge utilities
- teleimager config and server/client changes

## What Is Excluded

- large recorded datasets such as `projects/xr_teleoperate/teleop/utils/data`
- git metadata from the source repositories
- private certificate material such as `cert.pem` and `key.pem`
- backup artifacts such as `*.bak`

## Managed Projects

| Project | Upstream | Local Status |
|---|---|---|
| `A1` | `https://github.com/ATeam-Research/A1.git` | Imported and iterated in prior repository steps |
| `dex-retargeting` | `https://github.com/dexsuite/dex-retargeting.git` | Imported and iterated in prior repository steps |
| `Isaac-GR00T` | `https://github.com/NVIDIA/Isaac-GR00T` | Imported and iterated in prior repository steps |
| `xr_teleoperate` | `https://github.com/unitreerobotics/xr_teleoperate.git` | Archived local working snapshot on 2026-04-29 with bridge, replay, recording, and documentation updates |
| `teleimager` | `https://github.com/silencht/teleimager` | Archived local working snapshot on 2026-04-29 with image server/client and config updates |

See [projects/README.md](projects/README.md) and [docs/import_log.md](docs/import_log.md) for per-project archive notes.

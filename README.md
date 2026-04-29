# humanoid_robot_long_horizon_tasks

Archive repository for the local `xr_teleoperate` and `teleimager` workspaces used in the humanoid long-horizon task setup.

## Layout

- `xr_teleoperate/`: archived working tree from `/home/unitree/xr_teleoperate`
- `teleimager/`: archived working tree from `/home/unitree/teleimager`
- `ARCHIVE_NOTES.md`: snapshot notes, exclusions, and source pointers

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

- large recorded datasets such as `xr_teleoperate/teleop/utils/data`
- git metadata from the source repositories
- private certificate material such as `cert.pem` and `key.pem`
- backup artifacts such as `*.bak`

## Source Repositories

- `xr_teleoperate` upstream: `https://github.com/unitreerobotics/xr_teleoperate.git`
- `teleimager` upstream: `https://github.com/silencht/teleimager`

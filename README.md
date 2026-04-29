# humanoid_robot_long_horizon_tasks

Unified monorepo for the local humanoid long-horizon manipulation stack.

This repository brings the workstation's teleoperation, sensing, dataset, and
training code into one place so the full pipeline is easier to trace, version,
and reproduce.

## What This Repo Is For

This monorepo is meant to answer one practical question:

How do we go from live humanoid teleoperation on G1 to reusable datasets, then
to GR00T/A1-style training and evaluation?

The current repository is organized around that full workflow rather than around
individual upstream projects.

## System Map

```text
operator / controller / planner
            |
            v
projects/xr_teleoperate
  - teleop bridge
  - live replay
  - episode recording
            |
            +------------------------+
            |                        |
            v                        v
projects/teleimager          raw recorded episodes
  - camera config            - data.json
  - image server/client      - color streams
  - image transport          - tactile / arm / hand states
                                     |
                                     v
projects/Isaac-GR00T
  - raw episode conversion
  - LeRobot / GR00T dataset prep
  - compact converted samples
                                     |
                                     v
projects/A1
  - training / finetuning
  - offline evaluation
  - checkpoint comparison
  - deployment-oriented experiments
```

## Project Roles

### `projects/xr_teleoperate`

The runtime control and recording side of the stack.

- Owns teleoperation entrypoints, bridges, replay tools, and episode writers.
- Contains the local G1-oriented workflow updates, including bridge fixes,
  G1_29 replay compatibility, and dataset export helpers.
- Produces the raw episodes that later get converted into training-ready data.

Key examples:

- `teleop/teleop_hand_and_arm_bridge.py`
- `teleop/replay_episode.py`
- `teleop/replay_episode_player.py`
- `teleop/utils/convert_episode_to_lerobot.py`

### `projects/teleimager`

The camera and image transport side of the stack.

- Owns camera configuration, image serving, and client-side image retrieval.
- Supports the vision inputs consumed during live teleoperation and recording.
- Lives as its own archived project because it is operationally important and
  evolves on a different cadence from the control stack.

Key examples:

- `cam_config_server.yaml`
- `src/teleimager/image_server.py`
- `src/teleimager/image_client.py`
- `psi0_bridge.py`

### `projects/Isaac-GR00T`

The dataset shaping layer between raw robot logs and model training.

- Converts raw G1 episodes into GR00T / LeRobot-style datasets.
- Holds compact converted samples that make the data contract easier to inspect.
- Serves as the main bridge from robot-native logging formats to learning-ready
  dataset formats.

In this repo, `Isaac-GR00T` is where the teleop data starts becoming ML data.

### `projects/A1`

The training, evaluation, and deployment-experiment side of the stack.

- Consumes converted datasets and runs finetuning or offline evaluation.
- Tracks checkpoint comparisons, phase analysis, and managed training notes.
- Represents the current downstream learning workflow once data is ready.

In this repo, `A1` is where converted demonstrations turn into model behavior.

### `projects/dex-retargeting`

A supporting dependency for hand and embodiment mapping.

- Provides retargeting logic used by the teleoperation side of the stack.
- Exists here because it is part of the practical local workflow, even though it
  is not the main project entrypoint for most experiments.

## How The Pieces Fit Together

If you want the shortest mental model, it is this:

1. `xr_teleoperate` runs the robot-facing teleop and records raw episodes.
2. `teleimager` handles the camera/image plumbing used by that teleop loop.
3. `Isaac-GR00T` converts the raw recordings into normalized dataset layouts.
4. `A1` trains on or evaluates those converted datasets.

That means the main dependency flow is:

`teleimager -> xr_teleoperate -> Isaac-GR00T -> A1`

Operationally, `teleimager` and `xr_teleoperate` are the online stack, while
`Isaac-GR00T` and `A1` are the offline data-and-learning stack.

## Quick Start

If you are new to this repository, use one of these entry paths:

1. Want to understand live robot teleoperation:
   Open `projects/xr_teleoperate` first, then `projects/teleimager`.
2. Want to understand how raw episodes become training data:
   Open `projects/Isaac-GR00T` after checking the episode structure in
   `projects/xr_teleoperate`.
3. Want to understand finetuning or checkpoint evaluation:
   Open `projects/A1` after you understand the converted dataset format.

Recommended first-read order:

1. [projects/README.md](projects/README.md)
2. `projects/xr_teleoperate`
3. `projects/Isaac-GR00T`
4. `projects/A1`
5. [docs/import_log.md](docs/import_log.md)

## Typical Workflows

### 1. Teleoperate And Record

Use this path when the goal is to collect new real-robot data.

- `projects/teleimager` provides the image transport and camera configuration.
- `projects/xr_teleoperate` runs the bridge, replay, and recording utilities.
- Output is a raw episode, typically including `data.json` plus recorded image
  streams.

### 2. Convert Raw Episodes Into Training Data

Use this path when the goal is to normalize robot-native recordings into a
model-facing dataset layout.

- Inspect the raw episode generated by `projects/xr_teleoperate`.
- Convert it in `projects/Isaac-GR00T` into GR00T / LeRobot-style structure.
- Validate field coverage, modality mapping, and camera outputs before training.

### 3. Train, Evaluate, And Compare Checkpoints

Use this path when the goal is model iteration rather than data collection.

- Start from the converted dataset artifacts in `projects/Isaac-GR00T`.
- Use `projects/A1` for finetuning, offline evaluation, and checkpoint
  comparison.
- Record outcomes and caveats in the managed workflow notes and import log.

## Current Status

The repository is already strong in these areas:

- G1 teleoperation and recording are archived with local bridge and replay
  updates.
- The image transport stack is archived separately and linked clearly to the
  teleop workflow.
- Raw-to-LeRobot / GR00T conversion is now represented in the monorepo.
- A1-side training and evaluation history is documented with concrete local
  results.

Current practical limitations:

- Large local datasets are intentionally not tracked here.
- Some workflows are documented and archived rather than packaged into a single
  polished automation layer.
- The online robot stack and offline learning stack are now connected by repo
  structure, but still live across multiple imported upstream codebases.

## Roadmap

Near-term documentation and workflow improvements that would add the most value:

- add a concise end-to-end walkthrough from teleop recording to A1 evaluation
- add per-project quick links for the most important scripts and entrypoints
- add a standard dataset-contract note covering raw episode fields versus
  converted LeRobot / GR00T fields
- add a small set of reproducible command recipes for common operator tasks
- continue tightening the interface between `xr_teleoperate`,
  `Isaac-GR00T`, and `A1`

## Managed Projects

| Project | Upstream | Role In This Monorepo |
|---|---|---|
| `A1` | `https://github.com/ATeam-Research/A1.git` | Training, finetuning, evaluation, deployment-oriented experimentation |
| `dex-retargeting` | `https://github.com/dexsuite/dex-retargeting.git` | Retargeting support for teleoperation and embodiment mapping |
| `Isaac-GR00T` | `https://github.com/NVIDIA/Isaac-GR00T` | Dataset conversion and GR00T / LeRobot-facing data preparation |
| `xr_teleoperate` | `https://github.com/unitreerobotics/xr_teleoperate.git` | Teleoperation runtime, replay, recording, and raw episode generation |
| `teleimager` | `https://github.com/silencht/teleimager` | Camera configuration, image transport, and teleop vision support |

## Included And Excluded

Included in this monorepo:

- local code updates
- local README and changelog updates
- teleoperation bridge and replay utilities
- image server/client and camera config updates
- dataset conversion scripts and compact generated samples already committed in
  managed projects

Intentionally excluded:

- large recorded datasets such as `projects/xr_teleoperate/teleop/utils/data`
- source-repository `.git` metadata
- private certificate material such as `cert.pem` and `key.pem`
- backup artifacts such as `*.bak`

## Where To Start

- Start with [projects/README.md](projects/README.md) for the project inventory.
- Read [docs/import_log.md](docs/import_log.md) for the change history of this
  managed workspace.
- Open `projects/xr_teleoperate` if you are tracing live robot behavior.
- Open `projects/Isaac-GR00T` if you are tracing dataset conversion.
- Open `projects/A1` if you are tracing training or checkpoint evaluation.

# Projects

Managed working snapshots live here.

The top-level workflow in this monorepo is:

`teleimager -> xr_teleoperate -> Isaac-GR00T -> A1`

Use that flow as the default way to navigate the repository:

- `projects/teleimager`
  Camera configuration, image server/client, and teleop vision transport.
- `projects/xr_teleoperate`
  Live teleoperation, replay, recording, and raw episode generation.
- `projects/Isaac-GR00T`
  Conversion of raw G1 episodes into GR00T / LeRobot-style datasets.
- `projects/A1`
  Finetuning, offline evaluation, checkpoint comparison, and deployment-side experiments.
- `projects/dex-retargeting`
  Retargeting support used by the teleoperation stack.

Each project folder should keep its own local context:

- upstream origin
- import date or archive date
- notable local changes included in the snapshot
- excluded heavy artifacts when relevant

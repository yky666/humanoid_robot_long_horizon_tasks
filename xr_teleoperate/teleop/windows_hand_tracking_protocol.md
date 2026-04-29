# Windows Hand Tracking Producer Protocol

This document fixes the JSON protocol used by a Windows-side hand-tracking producer that sends data to
`windows_dual_udp_bridge.py` on local UDP port `5100`.

The bridge then normalizes and forwards the packet to the robot on UDP port `5005`.

## Transport

- Protocol: UDP
- Producer destination: `127.0.0.1:5100`
- Encoding: UTF-8 JSON object
- Recommended rate: `30-90 Hz`

## Required shape

At least one of the following should be present in each packet:

- `left_hand_pos`
- `right_hand_pos`
- `left`
- `right`
- `left_trigger`
- `right_trigger`

## Canonical fields

### Wrist pose

- `left`: 3x4 pose matrix
- `right`: 3x4 pose matrix

Example:

```json
"left": [
  [1.0, 0.0, 0.0, 0.12],
  [0.0, 1.0, 0.0, 0.08],
  [0.0, 0.0, 1.0, -0.33]
]
```

### Hand landmarks

- `left_hand_pos`: `25 x 3` landmark array
- `right_hand_pos`: `25 x 3` landmark array

The bridge also accepts these aliases:

- `left_hand_positions`
- `right_hand_positions`
- `left_hand_landmarks`
- `right_hand_landmarks`

You may also send a flattened `75`-element array instead of `25 x 3`.

### Analog close / grab values

Normalized range: `0.0 -> open`, `1.0 -> fully closed`

- `left_trigger`
- `right_trigger`

Accepted aliases:

- `left_grab`, `right_grab`
- `left_grip`, `right_grip`
- `left_grab_strength`, `right_grab_strength`
- `left_pinch_value`, `right_pinch_value`
- `left_pinch_strength`, `right_pinch_strength`
- `left_middle`, `right_middle`
- `left_middle_button`, `right_middle_button`

### Gesture booleans

- `left_pinch`, `right_pinch`
- `left_grab_button`, `right_grab_button`
- `left_middle_button`, `right_middle_button`

Accepted values:

- `true / false`
- `1 / 0`
- `"true" / "false"`

### Gesture labels

- `left_gesture`
- `right_gesture`

Accepted alias:

- `left_gesture_name`
- `right_gesture_name`

Recommended label set:

- `open`
- `pinch`
- `grab`
- `fist`
- `point`
- `victory`
- `release`

Bridge behavior:

- `pinch`, `grab`, `fist`, `close` imply trigger close if no stronger analog value is present
- `open`, `release` imply trigger open if no analog value is present

## Recommended packet

```json
{
  "timestamp_ms": 1710000000123,
  "source": "pico_hand_tracking",
  "left": [[1,0,0,0.10],[0,1,0,0.20],[0,0,1,-0.30]],
  "right": [[1,0,0,-0.10],[0,1,0,0.20],[0,0,1,-0.30]],
  "left_hand_pos": [[0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3],
                    [0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3],
                    [0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3],
                    [0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3],
                    [0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3], [0.1,0.2,0.3]],
  "right_hand_pos": [[-0.1,0.2,0.3], [-0.1,0.2,0.3], [-0.1,0.2,0.3], [-0.1,0.2,0.3], [-0.1,0.2,0.3],
                     [-0.1,0.2,0.3], [-0.1,0.2,0.3], [-0.1,0.2,0.3], [-0.1,0.2,0.3], [-0.1,0.2,0.3],
                     [-0.1,0.2,0.3], [-0.1,0.2,0.3], [-0.1,0.2,0.3], [-0.1,0.2,0.3], [-0.1,0.2,0.3],
                     [-0.1,0.2,0.3], [-0.1,0.2,0.3], [-0.1,0.2,0.3], [-0.1,0.2,0.3], [-0.1,0.2,0.3],
                     [-0.1,0.2,0.3], [-0.1,0.2,0.3], [-0.1,0.2,0.3], [-0.1,0.2,0.3], [-0.1,0.2,0.3]],
  "left_pinch": true,
  "left_pinch_value": 0.82,
  "left_gesture": "pinch",
  "right_pinch": false,
  "right_pinch_value": 0.03,
  "right_gesture": "open"
}
```

## Minimum viable packet

If your SDK can only provide landmarks and a pinch strength:

```json
{
  "left_hand_pos": [[...25x3...]],
  "right_hand_pos": [[...25x3...]],
  "left_pinch_value": 0.75,
  "right_pinch_value": 0.10,
  "left_gesture": "pinch",
  "right_gesture": "open"
}
```

## Producer guidance

- Always prefer sending `left_hand_pos/right_hand_pos` over only gesture labels.
- If your SDK provides wrist pose, send `left/right` too.
- If your SDK provides both a gesture label and an analog pinch/grab value, send both.
- Keep values normalized to `[0, 1]` when possible.

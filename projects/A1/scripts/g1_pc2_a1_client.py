#!/usr/bin/env python3
"""PC2-side A1 inference client for guarded G1 real-robot validation.

The script aligns the G1 teleoperation recording interface with the A1 deploy
API:

    state/action layout: left_arm(7) + right_arm(7) + left_ee(6) + right_ee(6)

By default it is a dry run: it reads robot/camera state, calls the A1 server,
and prints the first predicted action. Pass --execute only after validating the
server response and keeping a human on the e-stop.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import requests


A1_ROOT = Path(__file__).resolve().parents[1]
XR_TELEOP_ROOT = Path("/home/unitree/xr_teleoperate")
TELEIMAGER_SRC = Path("/home/unitree/teleimager/src")

for path in (A1_ROOT, XR_TELEOP_ROOT, TELEIMAGER_SRC):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: E402

from teleop.teleop_hand_and_arm_bridge import (  # noqa: E402
    InspireFTPControllerBridge,
    InspireFTPStateRecorder,
    ROBOT_ARM_CONFIGS,
    ReadOnlyG1ArmState,
)
from teleimager.image_client import ImageClient  # noqa: E402
from teleop.robot_control.robot_arm import (  # noqa: E402
    G1_23_ArmController,
    G1_29_ArmController,
)


DEFAULT_INSTRUCTION = (
    "left hand grasps the water bottle, hands it over to the right hand, "
    "then the right hand inserts it into the cup."
)

ARM_JOINT_NAMES = [
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A1 inference from G1 PC2.")
    parser.add_argument("--server-url", default="http://127.0.0.1:18000", help="A1 deploy server URL")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="Task instruction")
    parser.add_argument("--network-interface", default="enP8p1s0", help="DDS network interface on PC2")
    parser.add_argument("--robot", choices=("G1_29", "G1_23"), default="G1_29")
    parser.add_argument("--img-server-ip", default="127.0.0.1", help="Teleimager image server host")
    parser.add_argument("--camera-count", type=int, default=3, choices=(1, 2, 3), help="Images sent to A1")
    parser.add_argument(
        "--camera-order",
        default="head,left_wrist,right_wrist",
        help="Comma-separated camera order, any of: head,left_wrist,right_wrist",
    )
    parser.add_argument("--image-wait-timeout", type=float, default=5.0, help="Seconds to wait for ZMQ frames")
    parser.add_argument("--image-poll-period", type=float, default=0.1, help="Seconds between image polls")
    parser.add_argument("--period", type=float, default=0.2, help="Policy request period in seconds")
    parser.add_argument("--iterations", type=int, default=1, help="Number of policy requests, <=0 runs forever")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP request timeout")
    parser.add_argument("--execute", action="store_true", help="Actually send actions to robot/hand")
    parser.add_argument("--chunk-execute", action="store_true", help="Execute the returned action chunk step by step")
    parser.add_argument("--chunk-steps", type=int, default=10, help="Max action steps to play from each chunk")
    parser.add_argument("--chunk-rate", type=float, default=10.0, help="Chunk playback rate in Hz")
    parser.add_argument("--motion-mode", action="store_true", help="Use rt/arm_sdk instead of rt/lowcmd")
    parser.add_argument("--arm-delta-limit", type=float, default=0.05, help="Max per-request arm joint delta")
    parser.add_argument(
        "--controller-velocity-limit",
        type=float,
        default=1.0,
        help="Internal arm controller velocity limit in rad/s",
    )
    parser.add_argument(
        "--max-arm-target-delta",
        type=float,
        default=2.0,
        help="Skip arm execution if model target is farther than this from current arm q",
    )
    parser.add_argument("--hand-delta-limit", type=float, default=0.05, help="Max per-request Inspire open-target delta")
    parser.add_argument("--action-scale", type=float, default=0.35, help="Scale displacement from current state")
    parser.add_argument("--disable-hand", action="store_true", help="Do not publish Inspire hand actions")
    parser.add_argument(
        "--relative-to-start",
        action="store_true",
        help="Offset model arm targets so the first predicted arm action matches the current arm q",
    )
    parser.add_argument(
        "--relative-offset-max",
        type=float,
        default=2.5,
        help="Refuse relative-to-start if any arm offset exceeds this many radians",
    )
    parser.add_argument(
        "--open-hand-on-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open both Inspire hands when the script exits",
    )
    parser.add_argument("--print-joint-diff", action="store_true", help="Print per-joint current/action deltas")
    parser.add_argument("--print-action-range", action="store_true", help="Print per-dimension min/max/range for the returned action chunk")
    parser.add_argument("--print-json", action="store_true", help="Print raw server response")
    return parser.parse_args()


def _frame_to_bgr(frame) -> np.ndarray | None:
    if frame is None:
        return None
    if isinstance(frame, tuple):
        if not frame:
            return None
        return _frame_to_bgr(frame[0])
    bgr = getattr(frame, "bgr", None)
    if bgr is not None:
        return bgr
    if isinstance(frame, np.ndarray):
        return frame
    return None


def _encode_bgr_to_base64(bgr: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("failed to JPEG-encode camera frame")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _parse_camera_order(value: str) -> list[str]:
    cameras = [item.strip() for item in value.split(",") if item.strip()]
    valid = {"head", "left_wrist", "right_wrist"}
    unknown = [camera for camera in cameras if camera not in valid]
    if unknown:
        raise ValueError(f"unknown camera names in --camera-order: {unknown}")
    return cameras


def _get_camera_frame(client: ImageClient, camera_name: str):
    if camera_name == "head":
        return client.get_head_frame()
    if camera_name == "left_wrist":
        return client.get_left_wrist_frame()
    if camera_name == "right_wrist":
        return client.get_right_wrist_frame()
    raise ValueError(f"unknown camera: {camera_name}")


def collect_images(
    client: ImageClient,
    camera_names: list[str],
    wait_timeout: float,
    poll_period: float,
) -> tuple[list[str], dict[str, tuple[int, ...] | None]]:
    deadline = time.time() + wait_timeout
    latest_shapes: dict[str, tuple[int, ...] | None] = {}
    selected: list[tuple[str, np.ndarray]] = []

    while time.time() <= deadline:
        selected = []
        latest_shapes = {}
        for camera_name in camera_names:
            frame = _frame_to_bgr(_get_camera_frame(client, camera_name))
            latest_shapes[camera_name] = None if frame is None else tuple(frame.shape)
            if frame is not None:
                selected.append((camera_name, frame))
        if len(selected) == len(camera_names):
            break
        time.sleep(poll_period)

    if not selected:
        raise RuntimeError(
            "no camera frames available from teleimager; "
            f"last_shapes={latest_shapes}. Check --img-server-ip and ZMQ ports."
        )

    if len(selected) < len(camera_names):
        missing = [name for name, shape in latest_shapes.items() if shape is None]
        print(f"[camera] Warning: missing frames from {missing}; using {[name for name, _ in selected]}")
    else:
        print(f"[camera] frames ready: {latest_shapes}")

    return [_encode_bgr_to_base64(frame) for _, frame in selected], latest_shapes


def pad_or_trim(values: Iterable[float], size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.float32)
    arr = np.asarray(list(values), dtype=np.float32).reshape(-1)
    out[: min(size, arr.size)] = arr[:size]
    return out


def read_proprio(
    arm_reader: ReadOnlyG1ArmState,
    hand_reader: InspireFTPStateRecorder,
) -> np.ndarray:
    arm_state = arm_reader.get_arm_state()
    left_arm = pad_or_trim(arm_state.q7[:7], 7)
    right_arm = pad_or_trim(arm_state.q7[7:], 7)
    left_hand, right_hand, *_ = hand_reader.poll()
    left_ee = pad_or_trim(left_hand, 6)
    right_ee = pad_or_trim(right_hand, 6)
    return np.concatenate([left_arm, right_arm, left_ee, right_ee]).astype(np.float32)


def call_policy(
    session: requests.Session,
    server_url: str,
    instruction: str,
    images: list[str],
    proprio: np.ndarray,
    timeout: float,
) -> dict:
    payload = {
        "instruction": instruction,
        "images": images,
        "proprio_data": [proprio.astype(float).tolist()],
    }
    response = session.post(f"{server_url.rstrip('/')}/inference", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def first_action(result: dict) -> np.ndarray:
    actions = np.asarray(result.get("predicted_actions"), dtype=np.float32)
    if actions.ndim == 1:
        action = actions
    elif actions.ndim == 2:
        action = actions[0]
    else:
        action = actions.reshape(-1, actions.shape[-1])[0]
    if action.size < 26:
        raise RuntimeError(f"expected at least 26 action dims, got {action.size}")
    return action[:26]


def action_chunk(result: dict) -> np.ndarray:
    actions = np.asarray(result.get("predicted_actions"), dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    elif actions.ndim > 2:
        actions = actions.reshape(-1, actions.shape[-1])
    if actions.shape[-1] < 26:
        raise RuntimeError(f"expected at least 26 action dims, got {actions.shape[-1]}")
    return actions[:, :26]


def apply_arm_start_offset(actions: np.ndarray, offset: np.ndarray) -> np.ndarray:
    shifted = actions.copy()
    shifted[:, :14] = shifted[:, :14] + offset[None, :]
    return shifted


def guarded_target(current: np.ndarray, desired: np.ndarray, limit: float, scale: float) -> np.ndarray:
    desired = current + float(scale) * (desired - current)
    return current + np.clip(desired - current, -float(limit), float(limit))


def format_joint_diff(current: np.ndarray, target: np.ndarray) -> str:
    rows = []
    for idx, name in enumerate(ARM_JOINT_NAMES):
        rows.append(
            f"{idx:02d} {name}: current={current[idx]: .4f} "
            f"target={target[idx]: .4f} delta={target[idx] - current[idx]: .4f}"
        )
    max_idx = int(np.argmax(np.abs(target[:14] - current[:14])))
    rows.append(f"max_abs_delta_joint={max_idx:02d} {ARM_JOINT_NAMES[max_idx]}")
    return "\n".join(rows)


def format_action_range(actions: np.ndarray) -> str:
    rows = []
    mins = np.min(actions, axis=0)
    maxs = np.max(actions, axis=0)
    ranges = maxs - mins
    for idx, name in enumerate(ARM_JOINT_NAMES):
        rows.append(
            f"{idx:02d} {name}: min={mins[idx]: .4f} max={maxs[idx]: .4f} range={ranges[idx]: .4f}"
        )
    hand_ranges = ranges[14:26]
    rows.append(
        f"arm_range_max={float(np.max(ranges[:14])):.4f} "
        f"hand_range_max={float(np.max(hand_ranges)):.4f}"
    )
    return "\n".join(rows)


def apply_action(
    robot: str,
    arm_controller,
    hand_controller: InspireFTPControllerBridge,
    current: np.ndarray,
    action: np.ndarray,
    arm_delta_limit: float,
    hand_delta_limit: float,
    action_scale: float,
    publish_hand: bool = True,
    max_arm_target_delta: float | None = None,
    print_joint_diff: bool = False,
) -> np.ndarray:
    target = np.asarray(action[:26], dtype=np.float32)
    arm_target_delta = float(np.max(np.abs(target[:14] - current[:14])))
    if print_joint_diff:
        print("[joint-diff]\n" + format_joint_diff(current[:14], target[:14]))
    skip_arm = max_arm_target_delta is not None and arm_target_delta > max_arm_target_delta
    safe = guarded_target(current, target, arm_delta_limit, action_scale)
    safe[14:26] = guarded_target(current[14:26], target[14:26], hand_delta_limit, action_scale)

    if skip_arm:
        safe[:14] = current[:14]
        print(
            f"[safety] skip arm command: arm_target_delta_max={arm_target_delta:.4f} "
            f"> max_arm_target_delta={max_arm_target_delta:.4f}"
        )
    else:
        left_arm = safe[:7]
        right_arm = safe[7:14]
        if robot == "G1_23":
            arm_q = np.concatenate([left_arm[:5], right_arm[:5]]).astype(np.float64)
        else:
            arm_q = np.concatenate([left_arm, right_arm]).astype(np.float64)
        arm_controller.ctrl_dual_arm(arm_q, np.zeros_like(arm_q))

    if publish_hand:
        publish_hand_open_targets(hand_controller, safe[14:20], safe[20:26])
    return safe


def publish_hand_open_targets(
    hand_controller: InspireFTPControllerBridge,
    left_open: Iterable[float],
    right_open: Iterable[float],
) -> None:
    left_msg = hand_controller.inspire_hand_defaut.get_inspire_hand_ctrl()
    left_msg.angle_set = hand_controller._normalize_to_scaled(list(left_open))
    left_msg.mode = 0b0001
    hand_controller.left_pub.Write(left_msg)

    right_msg = hand_controller.inspire_hand_defaut.get_inspire_hand_ctrl()
    right_msg.angle_set = hand_controller._normalize_to_scaled(list(right_open))
    right_msg.mode = 0b0001
    hand_controller.right_pub.Write(right_msg)


def open_hands(hand_controller: InspireFTPControllerBridge, hold_s: float = 0.5) -> None:
    open_targets = [1.0] * 6
    end_time = time.time() + max(hold_s, 0.0)
    while time.time() <= end_time:
        publish_hand_open_targets(hand_controller, open_targets, open_targets)
        time.sleep(0.05)


def execute_action_chunk(
    robot: str,
    arm_controller,
    hand_controller: InspireFTPControllerBridge,
    arm_reader: ReadOnlyG1ArmState,
    hand_reader: InspireFTPStateRecorder,
    actions: np.ndarray,
    max_steps: int,
    rate_hz: float,
    arm_delta_limit: float,
    hand_delta_limit: float,
    action_scale: float,
    publish_hand: bool,
    max_arm_target_delta: float | None,
    print_joint_diff: bool,
    relative_arm_offset: np.ndarray | None = None,
) -> None:
    period = 1.0 / max(float(rate_hz), 1e-6)
    if relative_arm_offset is not None:
        actions = apply_arm_start_offset(actions, relative_arm_offset)
    steps = min(max_steps, actions.shape[0])
    for step_idx in range(steps):
        step_start = time.time()
        current = read_proprio(arm_reader, hand_reader)
        safe = apply_action(
            robot,
            arm_controller,
            hand_controller,
            current,
            actions[step_idx],
            arm_delta_limit,
            hand_delta_limit,
            action_scale,
            publish_hand=publish_hand,
            max_arm_target_delta=max_arm_target_delta,
            print_joint_diff=print_joint_diff and step_idx == 0,
        )
        print(
            f"[chunk] step={step_idx + 1}/{steps} "
            f"arm_target_delta_max={np.max(np.abs(actions[step_idx, :14] - current[:14])):.4f} "
            f"arm_delta_max={np.max(np.abs(safe[:14] - current[:14])):.4f} "
            f"hand_target_delta_max={np.max(np.abs(actions[step_idx, 14:26] - current[14:26])):.4f} "
            f"hand_delta_max={np.max(np.abs(safe[14:26] - current[14:26])):.4f}"
        )
        sleep_s = period - (time.time() - step_start)
        if sleep_s > 0:
            time.sleep(sleep_s)


def main() -> None:
    args = parse_args()
    ChannelFactoryInitialize(0, networkInterface=args.network_interface)

    robot_config = ROBOT_ARM_CONFIGS[args.robot]
    arm_reader = ReadOnlyG1ArmState(robot_config)
    hand_reader = InspireFTPStateRecorder()
    image_client = ImageClient(host=args.img_server_ip)
    session = requests.Session()
    camera_names = _parse_camera_order(args.camera_order)[: args.camera_count]

    arm_controller = None
    hand_controller = None
    relative_arm_offset = None
    if args.execute:
        controller_cls = G1_29_ArmController if args.robot == "G1_29" else G1_23_ArmController
        arm_controller = controller_cls(motion_mode=args.motion_mode, simulation_mode=False)
        arm_controller.arm_velocity_limit = args.controller_velocity_limit
        warm_start = read_proprio(arm_reader, hand_reader)
        if args.robot == "G1_23":
            warm_arm = np.concatenate([warm_start[:5], warm_start[7:12]]).astype(np.float64)
        else:
            warm_arm = warm_start[:14].astype(np.float64)
        arm_controller.ctrl_dual_arm(warm_arm, np.zeros_like(warm_arm))
        hand_controller = InspireFTPControllerBridge()
        print("[execute] Robot action publishing is ENABLED. Keep e-stop ready.")
    else:
        print("[dry-run] No robot commands will be published. Add --execute to actuate.")

    try:
        count = 0
        while args.iterations <= 0 or count < args.iterations:
            loop_start = time.time()
            proprio = read_proprio(arm_reader, hand_reader)
            images, image_shapes = collect_images(
                image_client,
                camera_names,
                args.image_wait_timeout,
                args.image_poll_period,
            )
            result = call_policy(
                session,
                args.server_url,
                args.instruction,
                images,
                proprio,
                args.timeout,
            )
            actions = action_chunk(result)
            if args.print_action_range:
                print("[action-range]\n" + format_action_range(actions))
            if args.relative_to_start and relative_arm_offset is None:
                relative_arm_offset = proprio[:14] - actions[0, :14]
                offset_abs_max = float(np.max(np.abs(relative_arm_offset)))
                print(
                    f"[relative-to-start] arm_offset={np.round(relative_arm_offset, 4).tolist()} "
                    f"max_abs={offset_abs_max:.4f}"
                )
                if offset_abs_max > args.relative_offset_max:
                    raise RuntimeError(
                        f"relative arm offset {offset_abs_max:.4f} exceeds "
                        f"--relative-offset-max {args.relative_offset_max:.4f}"
                    )
            actions_to_execute = (
                apply_arm_start_offset(actions, relative_arm_offset)
                if relative_arm_offset is not None
                else actions
            )
            action = actions[0]
            if args.print_json:
                print(json.dumps(result, ensure_ascii=False, indent=2)[:4000])
            print(
                f"iter={count} proprio_dim={proprio.size} action0={np.round(action[:26], 4).tolist()} "
                f"images={image_shapes} latency_ms={result.get('processing_time_ms')}"
            )

            if args.execute:
                if args.chunk_execute:
                    execute_action_chunk(
                        args.robot,
                        arm_controller,
                        hand_controller,
                        arm_reader,
                        hand_reader,
                        actions,
                        args.chunk_steps,
                        args.chunk_rate,
                        args.arm_delta_limit,
                        args.hand_delta_limit,
                        args.action_scale,
                        publish_hand=not args.disable_hand,
                        max_arm_target_delta=args.max_arm_target_delta,
                        print_joint_diff=args.print_joint_diff,
                        relative_arm_offset=relative_arm_offset,
                    )
                else:
                    safe = apply_action(
                        args.robot,
                        arm_controller,
                        hand_controller,
                        proprio,
                        actions_to_execute[0],
                        args.arm_delta_limit,
                        args.hand_delta_limit,
                        args.action_scale,
                        publish_hand=not args.disable_hand,
                        max_arm_target_delta=args.max_arm_target_delta,
                        print_joint_diff=args.print_joint_diff,
                    )
                    print(
                        f"[execute] arm_target_delta_max={np.max(np.abs(actions_to_execute[0, :14] - proprio[:14])):.4f} "
                        f"arm_delta_max={np.max(np.abs(safe[:14] - proprio[:14])):.4f} "
                        f"hand_target_delta_max={np.max(np.abs(actions_to_execute[0, 14:26] - proprio[14:26])):.4f} "
                        f"hand_delta_max={np.max(np.abs(safe[14:26] - proprio[14:26])):.4f}"
                    )

            count += 1
            sleep_s = args.period - (time.time() - loop_start)
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        if hand_controller is not None and args.open_hand_on_exit:
            print("[exit] Opening Inspire hands.")
            open_hands(hand_controller)
        image_client.close()


if __name__ == "__main__":
    main()

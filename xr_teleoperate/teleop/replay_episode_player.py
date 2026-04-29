import argparse
import json
import logging
import os
import sys
import time
from typing import Any

import numpy as np

try:
    import logging_mp  # type: ignore
except ImportError:
    logging_mp = None


def _build_logger():
    if logging_mp is not None:
        if hasattr(logging_mp, "basicConfig"):
            logging_mp.basicConfig(level=logging_mp.INFO)
        return logging_mp.getLogger(__name__)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return logging.getLogger(__name__)


logger_mp = _build_logger()

GR00T_ROOT = "/home/unitree/GR00T-WholeBodyControl"
if GR00T_ROOT not in sys.path:
    sys.path.append(GR00T_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay recorded episode actions to the robot for data quality checks."
    )
    parser.add_argument(
        "--episode-json",
        type=str,
        required=True,
        help="Absolute or relative path to episode data.json",
    )
    parser.add_argument(
        "--arm",
        choices=("G1_29", "G1_23"),
        default="G1_29",
        help="Robot arm type for live replay. Defaults to G1_29.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=30.0,
        help="Playback rate in Hz",
    )
    parser.add_argument(
        "--network-interface",
        type=str,
        default=None,
        help="DDS network interface passed to ChannelFactoryInitialize",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print decoded actions without sending robot commands",
    )
    parser.add_argument(
        "--confirm-live",
        type=str,
        default=None,
        help="Required for real robot replay. Pass exactly REPLAY_LIVE to enable command sending.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start replay from this frame index inside the episode data array",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to replay",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "arms-only", "hands-only", "body-only"),
        default="all",
        help="Select which channels to replay.",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=2.0,
        help="Ramp from the first recorded state to the first replay action before live playback.",
    )
    parser.add_argument(
        "--include-body",
        action="store_true",
        help="Enable live body/planner replay in addition to the selected mode.",
    )
    parser.add_argument(
        "--planner-zmq-host",
        type=str,
        default="127.0.0.1",
        help="Sonic planner ZMQ host. Use 127.0.0.1 when replay runs on the robot itself.",
    )
    parser.add_argument(
        "--planner-zmq-port",
        type=int,
        default=5556,
        help="ZMQ port for Sonic planner replay publisher.",
    )
    parser.add_argument(
        "--planner-keepalive-sec",
        type=float,
        default=1.0,
        help="Keepalive interval for planner start messages.",
    )
    parser.add_argument(
        "--planner-zmq-mode",
        choices=("connect", "bind"),
        default="bind",
        help="ZMQ mode for the replay publisher. Sonic's planner subscriber connects, so bind matches the normal teleop topology.",
    )
    parser.add_argument(
        "--planner-command-preroll-seconds",
        type=float,
        default=0.6,
        help="Before body replay starts, repeatedly send planner start for this long to avoid PUB/SUB cold-start drops.",
    )
    parser.add_argument(
        "--planner-command-refresh-hz",
        type=float,
        default=10.0,
        help="Frequency used while sending planner start preroll packets.",
    )
    parser.add_argument(
        "--hand-action-space",
        choices=("auto", "open", "close"),
        default="auto",
        help="Interpret recorded hand actions as already-open values or controller close-values.",
    )
    parser.add_argument(
        "--arm-control-backend",
        choices=("sonic", "direct"),
        default="sonic",
        help="Replay arms through Sonic planner like teleop, or directly through the low-level arm DDS controller.",
    )
    parser.add_argument(
        "--keep-planner-running",
        action="store_true",
        help="Do not send planner stop when replaying without body.",
    )
    parser.add_argument(
        "--stop-planner-first",
        action="store_true",
        help="Best-effort stop for an existing planner publisher before replay. Disabled by default to avoid ZMQ bind conflicts.",
    )
    return parser.parse_args()


class InspireHandActionPublisher:
    def __init__(self):
        from unitree_sdk2py.core.channel import ChannelPublisher
        from inspire_sdkpy import inspire_dds
        import inspire_sdkpy.inspire_hand_defaut as inspire_hand_defaut

        self._default = inspire_hand_defaut
        self.left_pub = ChannelPublisher("rt/inspire_hand/ctrl/l", inspire_dds.inspire_hand_ctrl)
        self.left_pub.Init()
        self.right_pub = ChannelPublisher("rt/inspire_hand/ctrl/r", inspire_dds.inspire_hand_ctrl)
        self.right_pub.Init()
        logger_mp.info("[ReplayPlayer] Inspire hand DDS publishers ready")

    @staticmethod
    def _to_scaled(values):
        return [int(np.clip(float(v) * 1000.0, 0.0, 1000.0)) for v in values]

    def publish(self, left_qpos, right_qpos):
        left_msg = self._default.get_inspire_hand_ctrl()
        left_msg.angle_set = self._to_scaled(left_qpos)
        left_msg.mode = 0b0001
        self.left_pub.Write(left_msg)

        right_msg = self._default.get_inspire_hand_ctrl()
        right_msg.angle_set = self._to_scaled(right_qpos)
        right_msg.mode = 0b0001
        self.right_pub.Write(right_msg)


class PlannerReplayPublisher:
    def __init__(self, host: str, port: int, keepalive_sec: float, zmq_mode: str = "connect"):
        import zmq
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
            build_command_message,
            build_planner_message,
        )

        self._build_command_message = build_command_message
        self._build_planner_message = build_planner_message
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        endpoint = f"tcp://{host}:{port}"
        self._zmq_mode = zmq_mode
        if zmq_mode == "bind":
            self._socket.bind(endpoint)
        else:
            self._socket.connect(endpoint)
        self._keepalive_sec = keepalive_sec
        self._last_keepalive = 0.0
        self._last_upper_body_position = None
        self._last_upper_body_time = None
        time.sleep(0.1)
        logger_mp.info(f"[ReplayPlayer] Planner ZMQ PUB {zmq_mode} tcp://{host}:{port}")

    def close(self):
        self._socket.close()
        self._context.term()

    def send_start(self):
        self._socket.send(self._build_command_message(start=True, stop=False, planner=True))
        self._last_keepalive = time.monotonic()

    def send_stop(self):
        self._socket.send(self._build_command_message(start=False, stop=True, planner=True))

    def maybe_keepalive(self):
        now = time.monotonic()
        if now - self._last_keepalive >= self._keepalive_sec:
            self.send_start()

    @staticmethod
    def arm_q_to_upper_body_position(arm_q: np.ndarray) -> np.ndarray:
        arm_q = np.asarray(arm_q, dtype=np.float32)
        upper = np.zeros(17, dtype=np.float32)
        upper[3] = arm_q[0]
        right_offset = 7 if arm_q.size >= 14 else 5
        upper[4] = arm_q[right_offset]
        upper[5] = arm_q[1]
        upper[6] = arm_q[right_offset + 1]
        upper[7] = arm_q[2]
        upper[8] = arm_q[right_offset + 2]
        upper[9] = arm_q[3]
        upper[10] = arm_q[right_offset + 3]
        upper[11] = arm_q[4]
        upper[12] = arm_q[right_offset + 4]
        if arm_q.size >= 14:
            upper[13] = arm_q[5]
            upper[14] = arm_q[12]
            upper[15] = arm_q[6]
            upper[16] = arm_q[13]
        return upper

    def _compute_upper_body_velocity(self, upper_pos: np.ndarray) -> np.ndarray:
        now = time.monotonic()
        if self._last_upper_body_position is None or self._last_upper_body_time is None:
            vel = np.zeros_like(upper_pos)
        else:
            dt = max(1e-3, now - self._last_upper_body_time)
            vel = (upper_pos - self._last_upper_body_position) / dt
        self._last_upper_body_position = upper_pos.copy()
        self._last_upper_body_time = now
        return vel.astype(np.float32)

    def publish(
        self,
        mode: int,
        movement: list[float],
        facing_xy: list[float],
        speed: float,
        height: float,
        arm_q: np.ndarray | None = None,
    ):
        facing = [float(facing_xy[0]), float(facing_xy[1]), 0.0]
        payload_kwargs = {
            "mode": mode,
            "movement": movement,
            "facing": facing,
            "speed": speed,
            "height": height,
        }
        if arm_q is not None:
            upper_pos = self.arm_q_to_upper_body_position(arm_q)
            upper_vel = self._compute_upper_body_velocity(upper_pos)
            payload_kwargs["upper_body_position"] = upper_pos.tolist()
            payload_kwargs["upper_body_velocity"] = upper_vel.tolist()
        payload = self._build_planner_message(**payload_kwargs)
        self._socket.send(payload)


def _send_planner_stop(host: str, port: int, zmq_mode: str):
    stop_pub = PlannerReplayPublisher(host=host, port=port, keepalive_sec=1.0, zmq_mode=zmq_mode)
    try:
        for _ in range(3):
            stop_pub.send_stop()
            time.sleep(0.05)
        logger_mp.info(f"[ReplayPlayer] sent planner stop on tcp://{host}:{port}")
    finally:
        stop_pub.close()


def _planner_command_preroll(body_ctrl, duration_sec: float, refresh_hz: float):
    if body_ctrl is None or duration_sec <= 0.0:
        return

    refresh_hz = max(refresh_hz, 1.0)
    interval = 1.0 / refresh_hz
    count = max(int(round(duration_sec * refresh_hz)), 1)
    logger_mp.info(
        f"[ReplayPlayer] planner preroll start packets={count} "
        f"duration={duration_sec:.2f}s rate={refresh_hz:.1f}Hz"
    )
    for _ in range(count):
        body_ctrl.send_start()
        time.sleep(interval)


def load_episode(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_frames(data_items, start_frame: int, max_frames: int | None):
    sliced = data_items[start_frame:]
    if max_frames is not None:
        sliced = sliced[:max_frames]
    return sliced


def _get_qpos(block: dict[str, Any] | None, key: str) -> list[float]:
    if not block:
        return []
    return block.get(key, {}).get("qpos", []) or []


def _validate_mode_flags(args):
    if args.mode == "body-only" and not args.include_body:
        raise ValueError("--mode body-only requires --include-body.")


def _validate_live_confirmation(args):
    if args.dry_run:
        return
    if args.confirm_live != "REPLAY_LIVE":
        raise ValueError(
            "Live replay is blocked by default. Re-run with --confirm-live REPLAY_LIVE after verifying the robot is safe."
        )


def _mode_enabled(mode: str):
    return {
        "arms": mode in ("all", "arms-only"),
        "hands": mode in ("all", "hands-only"),
        "body": mode in ("all", "body-only"),
    }


def _use_sonic_arm_backend(args, enabled: dict[str, bool]) -> bool:
    return enabled["arms"] and args.arm_control_backend == "sonic"


def _get_direct_arm_dofs_per_side(arm: str) -> int:
    return 7 if arm == "G1_29" else 5


def _get_sonic_arm_dofs_per_side(arm: str) -> int:
    return 7 if arm == "G1_29" else 5


def _interpolate(start_vals, end_vals, alpha: float):
    start = np.array(start_vals, dtype=np.float64)
    end = np.array(end_vals, dtype=np.float64)
    if start.shape != end.shape or start.size == 0:
        return end
    return (1.0 - alpha) * start + alpha * end


def _parse_body_state(body_state: list[float]):
    if len(body_state) < 5:
        return 0, -1.0, -1.0, [1.0, 0.0]
    mode = int(round(float(body_state[0])))
    speed = float(body_state[1])
    height = float(body_state[2])
    facing_xy = [float(body_state[3]), float(body_state[4])]
    return mode, speed, height, facing_xy


def _resolve_body_arm_q(item, actions, expected_dofs_per_side: int) -> np.ndarray:
    action_left = np.array(_get_qpos(actions, "left_arm"), dtype=np.float64)
    action_right = np.array(_get_qpos(actions, "right_arm"), dtype=np.float64)
    if action_left.size > 0 and action_left.size == action_right.size:
        return np.concatenate(
            (
                _normalize_arm_side(action_left.tolist(), expected_dofs_per_side),
                _normalize_arm_side(action_right.tolist(), expected_dofs_per_side),
            )
        )

    states = item.get("states", {}) or {}
    state_left = np.array(_get_qpos(states, "left_arm"), dtype=np.float64)
    state_right = np.array(_get_qpos(states, "right_arm"), dtype=np.float64)
    if state_left.size > 0 and state_left.size == state_right.size:
        return np.concatenate(
            (
                _normalize_arm_side(state_left.tolist(), expected_dofs_per_side),
                _normalize_arm_side(state_right.tolist(), expected_dofs_per_side),
            )
        )

    return np.zeros(expected_dofs_per_side * 2, dtype=np.float64)


def _normalize_arm_side(values: list[float], expected_dof: int) -> np.ndarray:
    arr = np.array(values or [], dtype=np.float64)
    if arr.size == expected_dof:
        return arr
    if arr.size > expected_dof:
        return arr[:expected_dof]
    return np.array([], dtype=np.float64)


def _resolve_hand_command(values: list[float], action_space: str) -> list[float]:
    if len(values) != 6:
        return values

    arr = np.clip(np.array(values, dtype=np.float64), 0.0, 1.0)
    if action_space == "open":
        return arr.tolist()
    if action_space == "close":
        return (1.0 - arr).tolist()

    # auto: controller-trigger recordings are typically uniform across all 6 joints.
    if float(np.std(arr)) < 0.02:
        return (1.0 - arr).tolist()
    return arr.tolist()


def _run_warmup(
    arm_ctrl,
    hand_ctrl,
    body_ctrl,
    first_item,
    enabled,
    rate: float,
    warmup_seconds: float,
    hand_action_space: str,
    sonic_arm_backend: bool,
    direct_arm_dofs_per_side: int,
    sonic_arm_dofs_per_side: int,
):
    if warmup_seconds <= 0.0:
        return

    steps = max(int(rate * warmup_seconds), 1)
    states = first_item.get("states", {}) or {}
    actions = first_item.get("actions", {}) or {}
    tauff_target = np.zeros(direct_arm_dofs_per_side * 2, dtype=np.float64)
    period = 1.0 / max(rate, 1.0)

    start_left_arm = _get_qpos(states, "left_arm")
    start_right_arm = _get_qpos(states, "right_arm")
    if enabled["arms"] and arm_ctrl is not None:
        current_arm_q = arm_ctrl.get_current_dual_arm_q().tolist()
        direct_total_dofs = direct_arm_dofs_per_side * 2
        if len(current_arm_q) == direct_total_dofs:
            start_left_arm = current_arm_q[:direct_arm_dofs_per_side]
            start_right_arm = current_arm_q[direct_arm_dofs_per_side:]
    arm_dofs_per_side = sonic_arm_dofs_per_side if sonic_arm_backend else direct_arm_dofs_per_side
    start_left_arm = _normalize_arm_side(start_left_arm, arm_dofs_per_side).tolist()
    start_right_arm = _normalize_arm_side(start_right_arm, arm_dofs_per_side).tolist()
    target_left_arm = _normalize_arm_side(_get_qpos(actions, "left_arm"), arm_dofs_per_side).tolist()
    target_right_arm = _normalize_arm_side(_get_qpos(actions, "right_arm"), arm_dofs_per_side).tolist()

    start_left_ee = _get_qpos(states, "left_ee")
    start_right_ee = _get_qpos(states, "right_ee")
    target_left_ee = _get_qpos(actions, "left_ee")
    target_right_ee = _get_qpos(actions, "right_ee")
    body_state = _get_qpos(states, "body")
    body_action = _get_qpos(actions, "body")
    mode, speed, height, facing_xy = _parse_body_state(body_state)

    logger_mp.info(f"[ReplayPlayer] warmup_steps={steps} warmup_seconds={warmup_seconds:.2f}")
    for step in range(steps):
        alpha = float(step + 1) / float(steps)

        if (
            enabled["arms"]
            and not sonic_arm_backend
            and len(start_left_arm) == direct_arm_dofs_per_side
            and len(start_right_arm) == direct_arm_dofs_per_side
        ):
            arm_q_target = np.concatenate(
                (
                    _interpolate(start_left_arm, target_left_arm, alpha),
                    _interpolate(start_right_arm, target_right_arm, alpha),
                )
            )
            arm_ctrl.ctrl_dual_arm(arm_q_target, tauff_target)

        if enabled["hands"] and len(start_left_ee) == 6 and len(start_right_ee) == 6:
            left_hand = _resolve_hand_command(
                _interpolate(start_left_ee, target_left_ee, alpha).tolist(),
                hand_action_space,
            )
            right_hand = _resolve_hand_command(
                _interpolate(start_right_ee, target_right_ee, alpha).tolist(),
                hand_action_space,
            )
            hand_ctrl.publish(left_hand, right_hand)

        if body_ctrl is not None and ((enabled["body"] and len(body_action) >= 3) or sonic_arm_backend):
            if (
                sonic_arm_backend
                and len(start_left_arm) == sonic_arm_dofs_per_side
                and len(start_right_arm) == sonic_arm_dofs_per_side
            ):
                arm_q = np.concatenate(
                    (
                        _interpolate(start_left_arm, target_left_arm, alpha),
                        _interpolate(start_right_arm, target_right_arm, alpha),
                    )
                )
            elif not enabled["arms"]:
                arm_q = _resolve_body_arm_q(first_item, actions, sonic_arm_dofs_per_side)
            else:
                arm_q = None

            movement = (
                _interpolate([0.0, 0.0, 0.0], body_action[:3], alpha).tolist()
                if len(body_action) >= 3
                else [0.0, 0.0, 0.0]
            )
            body_ctrl.maybe_keepalive()
            body_ctrl.publish(
                mode=mode,
                movement=movement,
                facing_xy=facing_xy,
                speed=speed,
                height=height,
                arm_q=arm_q,
            )

        time.sleep(period)


def main():
    args = parse_args()
    _validate_mode_flags(args)
    _validate_live_confirmation(args)

    episode_json = os.path.abspath(args.episode_json)
    episode = load_episode(episode_json)
    data_items = select_frames(episode.get("data", []), args.start_frame, args.max_frames)
    enabled = _mode_enabled(args.mode)
    sonic_arm_backend = _use_sonic_arm_backend(args, enabled)
    direct_arm_dofs_per_side = _get_direct_arm_dofs_per_side(args.arm)
    sonic_arm_dofs_per_side = _get_sonic_arm_dofs_per_side(args.arm)

    if not data_items:
        raise ValueError("No episode frames selected for replay.")

    logger_mp.info(f"[ReplayPlayer] loaded {episode_json}")
    logger_mp.info(f"[ReplayPlayer] frames_to_replay={len(data_items)} rate={args.rate:.1f}Hz")
    logger_mp.info(f"[ReplayPlayer] mode={args.mode}")
    logger_mp.info(f"[ReplayPlayer] arm={args.arm}")
    logger_mp.info(f"[ReplayPlayer] body_replay={'enabled' if args.include_body else 'disabled'}")
    logger_mp.info(f"[ReplayPlayer] hand_action_space={args.hand_action_space}")
    logger_mp.info(f"[ReplayPlayer] arm_control_backend={args.arm_control_backend}")
    logger_mp.info(
        f"[ReplayPlayer] planner_endpoint=tcp://{args.planner_zmq_host}:{args.planner_zmq_port} "
        f"mode={args.planner_zmq_mode}"
    )

    if args.dry_run:
        for item in data_items[:5]:
            actions = item.get("actions", {}) or {}
            states = item.get("states", {}) or {}
            logger_mp.info(
                f"[ReplayPlayer] idx={item.get('idx')} "
                f"left_arm={actions.get('left_arm', {}).get('qpos', [])} "
                f"right_arm={actions.get('right_arm', {}).get('qpos', [])} "
                f"left_ee={actions.get('left_ee', {}).get('qpos', [])} "
                f"right_ee={actions.get('right_ee', {}).get('qpos', [])} "
                f"body_state={states.get('body', {}).get('qpos', [])} "
                f"body_action={actions.get('body', {}).get('qpos', [])}"
            )
        return

    arm_ctrl = None
    hand_ctrl = None
    body_ctrl = None

    if (enabled["arms"] and not sonic_arm_backend) or enabled["hands"]:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        ChannelFactoryInitialize(0, networkInterface=args.network_interface)

    if enabled["arms"] and not sonic_arm_backend:
        if args.arm == "G1_29":
            from teleop.robot_control.robot_arm import G1_29_ArmController
        else:
            from teleop.robot_control.robot_arm import G1_23_ArmController

    if args.stop_planner_first:
        _send_planner_stop(args.planner_zmq_host, args.planner_zmq_port, args.planner_zmq_mode)
    arm_ctrl = (
        (
            G1_29_ArmController(motion_mode=False, simulation_mode=False)
            if args.arm == "G1_29"
            else G1_23_ArmController(motion_mode=False, simulation_mode=False)
        )
        if enabled["arms"] and not sonic_arm_backend
        else None
    )
    hand_ctrl = InspireHandActionPublisher() if enabled["hands"] else None
    body_ctrl = (
        PlannerReplayPublisher(
            host=args.planner_zmq_host,
            port=args.planner_zmq_port,
            keepalive_sec=args.planner_keepalive_sec,
            zmq_mode=args.planner_zmq_mode,
        )
        if args.include_body or sonic_arm_backend
        else None
    )

    tauff_target = np.zeros(direct_arm_dofs_per_side * 2, dtype=np.float64)
    period = 1.0 / max(args.rate, 1.0)

    try:
        if body_ctrl is not None:
            _planner_command_preroll(
                body_ctrl,
                args.planner_command_preroll_seconds,
                args.planner_command_refresh_hz,
            )
        _run_warmup(
            arm_ctrl,
            hand_ctrl,
            body_ctrl,
            data_items[0],
            enabled,
            args.rate,
            args.warmup_seconds,
            args.hand_action_space,
            sonic_arm_backend,
            direct_arm_dofs_per_side,
            sonic_arm_dofs_per_side,
        )

        for item in data_items:
            actions = item.get("actions", {}) or {}
            states = item.get("states", {}) or {}
            arm_dofs_per_side = sonic_arm_dofs_per_side if sonic_arm_backend else direct_arm_dofs_per_side
            left_arm = _normalize_arm_side(
                actions.get("left_arm", {}).get("qpos", []) or [],
                arm_dofs_per_side,
            )
            right_arm = _normalize_arm_side(
                actions.get("right_arm", {}).get("qpos", []) or [],
                arm_dofs_per_side,
            )
            left_ee = actions.get("left_ee", {}).get("qpos", []) or []
            right_ee = actions.get("right_ee", {}).get("qpos", []) or []
            body = actions.get("body", {}).get("qpos", []) or []
            body_state = states.get("body", {}).get("qpos", []) or []

            arm_q_target = None
            if enabled["arms"]:
                if left_arm.size != arm_dofs_per_side or right_arm.size != arm_dofs_per_side:
                    logger_mp.warning(
                        f"[ReplayPlayer] skip idx={item.get('idx')} because arm action size is invalid "
                        f"(expected {arm_dofs_per_side} per side, got {left_arm.size}/{right_arm.size})"
                    )
                    continue
                arm_q_target = np.concatenate((left_arm, right_arm))
                if not sonic_arm_backend:
                    arm_ctrl.ctrl_dual_arm(arm_q_target, tauff_target)

            if enabled["hands"] and len(left_ee) == 6 and len(right_ee) == 6:
                hand_ctrl.publish(
                    _resolve_hand_command(left_ee, args.hand_action_space),
                    _resolve_hand_command(right_ee, args.hand_action_space),
                )

            if body_ctrl is not None:
                mode, speed, height, facing_xy = _parse_body_state(body_state)
                arm_q_for_body = None
                if sonic_arm_backend:
                    arm_q_for_body = arm_q_target
                elif not enabled["arms"]:
                    arm_q_for_body = (
                        np.concatenate((left_arm, right_arm))
                        if left_arm.size == arm_dofs_per_side and right_arm.size == arm_dofs_per_side
                        else _resolve_body_arm_q(item, actions, sonic_arm_dofs_per_side)
                    )

                if args.include_body and len(body) < 3:
                    logger_mp.warning(f"[ReplayPlayer] skip body at idx={item.get('idx')} because body action size is invalid")
                else:
                    movement = (
                        [float(body[0]), float(body[1]), float(body[2])]
                        if args.include_body and len(body) >= 3
                        else [0.0, 0.0, 0.0]
                    )
                    body_ctrl.maybe_keepalive()
                    body_ctrl.publish(
                        mode=mode if args.include_body else 0,
                        movement=movement,
                        facing_xy=facing_xy if args.include_body else [1.0, 0.0],
                        speed=speed if args.include_body else -1.0,
                        height=height if args.include_body else -1.0,
                        arm_q=arm_q_for_body,
                    )

            logger_mp.info(
                f"[ReplayPlayer] idx={item.get('idx')} "
                f"arms={'on' if enabled['arms'] else 'off'} "
                f"hands={'on' if enabled['hands'] else 'off'} "
                f"body={'on' if args.include_body else 'off'} "
                f"sonic_arm={'on' if sonic_arm_backend else 'off'} "
                f"left_ee={left_ee} right_ee={right_ee} body={body}"
            )
            if args.include_body and len(body) >= 3:
                mode, speed, height, facing_xy = _parse_body_state(body_state)
                print(
                    f"\r[ReplayBody] idx={item.get('idx')} mode={mode} "
                    f"move=({float(body[0]):+.3f},{float(body[1]):+.3f},{float(body[2]):+.3f}) "
                    f"speed={speed:+.2f} height={height:+.2f} "
                    f"facing=({facing_xy[0]:+.2f},{facing_xy[1]:+.2f})",
                    end="",
                    flush=True,
                )
            elif enabled["hands"] and len(right_ee) == 6:
                print(
                    f"\r[ReplayHands] idx={item.get('idx')} "
                    f"L={_resolve_hand_command(left_ee, args.hand_action_space)} "
                    f"R={_resolve_hand_command(right_ee, args.hand_action_space)}",
                    end="",
                    flush=True,
                )
            time.sleep(period)
    except KeyboardInterrupt:
        logger_mp.info("[ReplayPlayer] interrupted by user")
    finally:
        print()
        if body_ctrl is not None:
            try:
                body_ctrl.send_stop()
            except Exception as exc:
                logger_mp.warning(f"[ReplayPlayer] failed to send planner stop: {exc}")
            try:
                body_ctrl.close()
            except Exception as exc:
                logger_mp.warning(f"[ReplayPlayer] failed to close planner publisher: {exc}")
        logger_mp.info("[ReplayPlayer] replay finished")


if __name__ == "__main__":
    main()

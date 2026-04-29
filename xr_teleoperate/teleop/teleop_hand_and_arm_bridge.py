import argparse
import json
import math
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from collections import deque
from multiprocessing import Array, Lock

import logging_mp
import numpy as np
import zmq
from sshkeyboard import listen_keyboard, stop_listening
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState

logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

GR00T_ROOT = "/home/unitree/GR00T-WholeBodyControl"
if GR00T_ROOT not in sys.path:
    sys.path.append(GR00T_ROOT)

from televuer import TeleVuerWrapper
from teleimager.image_client import ImageClient
from teleop.robot_control.robot_arm import (
    G1_23_JointArmIndex,
    G1_23_JointIndex,
    G1_29_JointArmIndex,
    G1_29_JointIndex,
)
from teleop.robot_control.robot_arm_ik import G1_23_ArmIK, G1_29_ArmIK
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.episode_writer_inspire_record import (
    INSPIRE_FTP_JOINT_NAMES,
    INSPIRE_FTP_TACTILE_NAMES,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
)

kTopicLowState = "rt/lowstate"

# Keep tactile metadata local to the bridge so passive recording does not pull in
# the full inspire retargeting dependency chain.
INSPIRE_FTP_TOUCH_FIELDS = [
    ("fingerone_tip_touch", 9),
    ("fingerone_top_touch", 96),
    ("fingerone_palm_touch", 80),
    ("fingertwo_tip_touch", 9),
    ("fingertwo_top_touch", 96),
    ("fingertwo_palm_touch", 80),
    ("fingerthree_tip_touch", 9),
    ("fingerthree_top_touch", 96),
    ("fingerthree_palm_touch", 80),
    ("fingerfour_tip_touch", 9),
    ("fingerfour_top_touch", 96),
    ("fingerfour_palm_touch", 80),
    ("fingerfive_tip_touch", 9),
    ("fingerfive_top_touch", 96),
    ("fingerfive_middle_touch", 9),
    ("fingerfive_palm_touch", 96),
    ("palm_touch", 112),
]
INSPIRE_FTP_TOUCH_NUM_VALUES = sum(length for _, length in INSPIRE_FTP_TOUCH_FIELDS)

G1_ARM_7_NAMES = [
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
]
ARM_EE_QPOS_NAMES = ["x", "y", "z", "roll", "pitch", "yaw"]
BODY_STATE_NAMES = ["mode", "speed", "height_cmd", "facing_x", "facing_y"]
PSI0_STATE_NAMES = (
    [f"left_hand_{i}" for i in range(7)]
    + [f"right_hand_{i}" for i in range(7)]
    + [f"left_arm_{name}" for name in G1_ARM_7_NAMES]
    + [f"right_arm_{name}" for name in G1_ARM_7_NAMES]
    + ["torso_roll", "torso_pitch", "torso_yaw", "height"]
)
PSI0_ACTION_NAMES = PSI0_STATE_NAMES + ["torso_vx", "torso_vy", "torso_vyaw", "target_yaw"]

ROBOT_ARM_CONFIGS = {
    "G1_23": {
        "ik_cls": G1_23_ArmIK,
        "arm_indices": [member.value for member in G1_23_JointArmIndex],
        "left_arm7_indices": [
            G1_23_JointIndex.kLeftShoulderPitch.value,
            G1_23_JointIndex.kLeftShoulderRoll.value,
            G1_23_JointIndex.kLeftShoulderYaw.value,
            G1_23_JointIndex.kLeftElbow.value,
            G1_23_JointIndex.kLeftWristRoll.value,
            G1_23_JointIndex.kLeftWristPitchNotUsed.value,
            G1_23_JointIndex.kLeftWristyawNotUsed.value,
        ],
        "right_arm7_indices": [
            G1_23_JointIndex.kRightShoulderPitch.value,
            G1_23_JointIndex.kRightShoulderRoll.value,
            G1_23_JointIndex.kRightShoulderYaw.value,
            G1_23_JointIndex.kRightElbow.value,
            G1_23_JointIndex.kRightWristRoll.value,
            G1_23_JointIndex.kRightWristPitchNotUsed.value,
            G1_23_JointIndex.kRightWristYawNotUsed.value,
        ],
    },
    "G1_29": {
        "ik_cls": G1_29_ArmIK,
        "arm_indices": [member.value for member in G1_29_JointArmIndex],
        "left_arm7_indices": [
            G1_29_JointIndex.kLeftShoulderPitch.value,
            G1_29_JointIndex.kLeftShoulderRoll.value,
            G1_29_JointIndex.kLeftShoulderYaw.value,
            G1_29_JointIndex.kLeftElbow.value,
            G1_29_JointIndex.kLeftWristRoll.value,
            G1_29_JointIndex.kLeftWristPitch.value,
            G1_29_JointIndex.kLeftWristyaw.value,
        ],
        "right_arm7_indices": [
            G1_29_JointIndex.kRightShoulderPitch.value,
            G1_29_JointIndex.kRightShoulderRoll.value,
            G1_29_JointIndex.kRightShoulderYaw.value,
            G1_29_JointIndex.kRightElbow.value,
            G1_29_JointIndex.kRightWristRoll.value,
            G1_29_JointIndex.kRightWristPitch.value,
            G1_29_JointIndex.kRightWristYaw.value,
        ],
    },
}

START = False
STOP = False
READY = False
RECORD_TOGGLE = False
RECORD_RUNNING = False
KEY_EVENTS = deque()
KEY_LOCK = threading.Lock()

LOCOMOTION_MODES = {
    "idle": 0,
    "slow_walk": 1,
    "walk": 2,
    "run": 3,
    "squat": 4,
    "kneel_two_legs": 5,
    "kneel": 6,
    "lying_face_down": 7,
    "crawling": 8,
    "idle_boxing": 9,
    "walk_boxing": 10,
    "left_punch": 11,
    "right_punch": 12,
    "random_punch": 13,
    "elbow_crawling": 14,
    "left_hook": 15,
    "right_hook": 16,
    "forward_jump": 17,
    "stealth_walk": 18,
    "injured_walk": 19,
    "ledge_walking": 20,
    "object_carrying": 21,
    "stealth_walk_2": 22,
    "happy_dance_walk": 23,
    "zombie_walk": 24,
    "gun_walk": 25,
    "scare_walk": 26,
}

MOTION_SETS = [
    ("Locomotion", ["slow_walk", "walk", "run", "happy_dance_walk", "stealth_walk", "injured_walk"]),
    ("Squat", ["squat", "kneel_two_legs", "kneel", "crawling", "elbow_crawling"]),
    ("Boxing", ["idle_boxing", "walk_boxing", "left_punch", "right_punch", "random_punch", "left_hook", "right_hook"]),
    ("Styled", ["ledge_walking", "object_carrying", "stealth_walk_2", "happy_dance_walk", "zombie_walk", "gun_walk", "scare_walk"]),
]

STATIC_LIKE_MODES = {"idle", "squat", "kneel_two_legs", "kneel", "lying_face_down", "idle_boxing"}

MODE_SPEED_LIMITS = {
    "slow_walk": (0.2, 0.8),
    "run": (1.5, 3.0),
    "crawling": (0.4, 1.0),
    "elbow_crawling": (0.7, 1.0),
    "walk_boxing": (0.7, 1.5),
    "left_punch": (0.7, 1.5),
    "right_punch": (0.7, 1.5),
    "random_punch": (0.7, 1.5),
    "left_hook": (0.7, 1.5),
    "right_hook": (0.7, 1.5),
}


def on_press(key: str) -> None:
    global START, STOP, RECORD_TOGGLE
    if key == "]":
        START = True
    elif key in ("o", "O"):
        STOP = True
    elif key in ("k", "K"):
        RECORD_TOGGLE = True
    else:
        with KEY_LOCK:
            KEY_EVENTS.append(key)


def drain_key_events() -> list[str]:
    with KEY_LOCK:
        keys = list(KEY_EVENTS)
        KEY_EVENTS.clear()
    return keys


class PlannerKeyboardState:
    """Event-driven Sonic planner keyboard state for Terminal 2."""

    def __init__(self, initial_mode: str, speed: float = -1.0, height: float = -1.0) -> None:
        self.motion_set_index = 0
        self.mode_name = initial_mode
        for idx, (_, modes) in enumerate(MOTION_SETS):
            if initial_mode in modes:
                self.motion_set_index = idx
                break
        self.facing_angle = 0.0
        self.movement_momentum = 0.0
        self.momentum_decay_rate = 0.92
        self.momentum_threshold = 0.1
        self.speed = speed
        self.height = height
        self._last_status = ""
        self._last_move_vector = np.zeros(3, dtype=np.float32)
        self._mode_switch_debounce_sec = 0.18
        self._last_mode_switch_at = {"n": 0.0, "p": 0.0}
        self._apply_mode_defaults()

    def _can_switch_mode(self, key_name: str) -> bool:
        now = time.time()
        if now - self._last_mode_switch_at[key_name] < self._mode_switch_debounce_sec:
            return False
        self._last_mode_switch_at[key_name] = now
        return True

    def _apply_mode_defaults(self) -> None:
        limits = MODE_SPEED_LIMITS.get(self.mode_name)
        if limits is not None:
            lo, hi = limits
            if self.speed < 0.0:
                self.speed = lo
            self.speed = min(max(self.speed, lo), hi)
        elif self.mode_name in STATIC_LIKE_MODES:
            self.speed = -1.0

        if self.motion_set_index == 1:
            if self.height < 0.2:
                self.height = 0.8
            self.height = min(max(self.height, 0.2), 0.8)
        else:
            self.height = -1.0

    def _current_facing(self) -> np.ndarray:
        return np.array(
            [np.cos(self.facing_angle), np.sin(self.facing_angle), 0.0],
            dtype=np.float32,
        )

    def _set_move_vector(self, movement: np.ndarray) -> None:
        self._last_move_vector = movement.astype(np.float32)
        self.movement_momentum = 1.0

    def handle_key(self, ch: str) -> None:
        facing = self._current_facing()
        if ch in ("q", "Q"):
            self.facing_angle += np.pi / 6.0
        elif ch in ("e", "E"):
            self.facing_angle -= np.pi / 6.0
        elif ch in ("n", "N") and self._can_switch_mode("n"):
            self.motion_set_index = (self.motion_set_index + 1) % len(MOTION_SETS)
            self.mode_name = MOTION_SETS[self.motion_set_index][1][0]
            self._apply_mode_defaults()
        elif ch in ("p", "P") and self._can_switch_mode("p"):
            self.motion_set_index = (self.motion_set_index - 1) % len(MOTION_SETS)
            self.mode_name = MOTION_SETS[self.motion_set_index][1][0]
            self._apply_mode_defaults()
        elif ch in "12345678":
            mode_idx = int(ch) - 1
            modes = MOTION_SETS[self.motion_set_index][1]
            if mode_idx < len(modes):
                self.mode_name = modes[mode_idx]
                self._apply_mode_defaults()
        elif ch == "9":
            limits = MODE_SPEED_LIMITS.get(self.mode_name)
            if limits is not None:
                self.speed = max(limits[0], self.speed - 0.1)
        elif ch == "0":
            limits = MODE_SPEED_LIMITS.get(self.mode_name)
            if limits is not None:
                self.speed = min(limits[1], self.speed + 0.1)
        elif ch == "-":
            if self.motion_set_index == 1:
                self.height = max(0.2, (0.8 if self.height < 0.0 else self.height) - 0.05)
        elif ch == "=" or ch == "+":
            if self.motion_set_index == 1:
                self.height = min(0.8, (0.8 if self.height < 0.0 else self.height) + 0.05)
        elif ch in ("r", "R", "`", "~"):
            self.movement_momentum = 0.0
            self._last_move_vector = np.zeros(3, dtype=np.float32)
        elif self.mode_name not in STATIC_LIKE_MODES:
            if ch in ("w", "W"):
                self._set_move_vector(facing)
            elif ch in ("s", "S"):
                self._set_move_vector(-facing)
            elif ch == ",":
                self._set_move_vector(
                    np.array([-np.sin(self.facing_angle), np.cos(self.facing_angle), 0.0], dtype=np.float32)
                )
            elif ch == ".":
                self._set_move_vector(
                    np.array([np.sin(self.facing_angle), -np.cos(self.facing_angle), 0.0], dtype=np.float32)
                )
            elif ch in ("a", "A"):
                self.facing_angle += 0.1
                self._set_move_vector(self._current_facing())
            elif ch in ("d", "D"):
                self.facing_angle -= 0.1
                self._set_move_vector(self._current_facing())

    def update(self, key_events: list[str]) -> tuple[int, np.ndarray, np.ndarray, float, float]:
        for ch in key_events:
            self.handle_key(ch)

        facing = self._current_facing()
        if self.movement_momentum > 0.0:
            self.movement_momentum *= self.momentum_decay_rate

        if self.movement_momentum > self.momentum_threshold:
            final_mode = LOCOMOTION_MODES[self.mode_name]
            final_movement = self._last_move_vector * self.movement_momentum
            final_speed = self.speed
            final_height = self.height
        else:
            self._last_move_vector = np.zeros(3, dtype=np.float32)
            if self.motion_set_index == 1:
                final_mode = LOCOMOTION_MODES[self.mode_name]
                final_movement = np.zeros(3, dtype=np.float32)
                final_speed = 0.0 if self.mode_name in {"crawling", "elbow_crawling"} else -1.0
                final_height = self.height
            elif self.motion_set_index == 2:
                final_mode = LOCOMOTION_MODES[self.mode_name]
                final_movement = np.zeros(3, dtype=np.float32)
                final_speed = 0.0 if self.mode_name != "idle_boxing" else -1.0
                if self.mode_name in {"left_punch", "right_punch", "left_hook", "right_hook"}:
                    final_movement = facing.copy()
                final_height = -1.0
            elif self.motion_set_index == 3:
                final_mode = LOCOMOTION_MODES[self.mode_name]
                final_movement = np.zeros(3, dtype=np.float32)
                final_speed = 0.0
                final_height = -1.0
            else:
                final_mode = LOCOMOTION_MODES["idle"]
                final_movement = np.zeros(3, dtype=np.float32)
                final_speed = -1.0
                final_height = -1.0

        status = (
            f"[Planner] set={MOTION_SETS[self.motion_set_index][0]} mode={self.mode_name} "
            f"speed={final_speed:.2f} height={final_height:.2f} "
            f"yaw={self.facing_angle:+.2f} momentum={self.movement_momentum:.2f}"
        )
        if status != self._last_status:
            logger_mp.info(status)
            self._last_status = status

        return final_mode, final_movement, facing, final_speed, final_height


class UDPPlannerCommandServer:
    """Receive planner and bridge control commands over UDP."""

    def __init__(self, host: str, port: int) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.setblocking(False)
        self.message_count = 0
        self.last_packet_time = None
        logger_mp.info(f"[PlannerUDP] Listening on {host}:{port} for remote planner commands")

    @staticmethod
    def _extract_key_events(msg: dict) -> list[str]:
        keys: list[str] = []

        single_key = msg.get("key")
        if isinstance(single_key, str) and single_key:
            keys.append(single_key[0])

        for field_name in ("keys", "planner_keys"):
            field = msg.get(field_name)
            if isinstance(field, list):
                for item in field:
                    if isinstance(item, str) and item:
                        keys.append(item[0])

        planner_text = msg.get("planner_text")
        if isinstance(planner_text, str):
            keys.extend(list(planner_text))

        return keys

    def poll(self) -> tuple[list[str], bool, bool, bool]:
        key_events: list[str] = []
        start_requested = False
        stop_requested = False
        record_toggle_requested = False

        try:
            while True:
                packet, _ = self.sock.recvfrom(4096)
                msg = json.loads(packet.decode("utf-8"))
                if not isinstance(msg, dict):
                    continue

                self.message_count += 1
                self.last_packet_time = time.time()
                key_events.extend(self._extract_key_events(msg))

                if bool(msg.get("start", False)):
                    start_requested = True
                if bool(msg.get("stop", False)) or bool(msg.get("quit", False)):
                    stop_requested = True
                if bool(msg.get("record_toggle", False)):
                    record_toggle_requested = True

                # Convenient aliases for common local shortcuts.
                if msg.get("command") == "start":
                    start_requested = True
                elif msg.get("command") == "stop":
                    stop_requested = True
                elif msg.get("command") == "record_toggle":
                    record_toggle_requested = True
        except BlockingIOError:
            pass

        return key_events, start_requested, stop_requested, record_toggle_requested


class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def get(self):
        with self.lock:
            return self.data

    def set(self, data):
        with self.lock:
            self.data = data


def quaternion_to_rpy(quat) -> np.ndarray:
    """Convert Unitree lowstate quaternion [w, x, y, z] to roll/pitch/yaw."""
    if quat is None or len(quat) < 4:
        return np.zeros(3, dtype=np.float64)
    w, x, y, z = [float(v) for v in quat[:4]]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def extract_lowstate_rpy(msg) -> np.ndarray:
    imu_state = getattr(msg, "imu_state", None)
    if imu_state is None:
        return np.zeros(3, dtype=np.float64)

    rpy = getattr(imu_state, "rpy", None)
    if rpy is not None and len(rpy) >= 3:
        return np.array([float(v) for v in rpy[:3]], dtype=np.float64)

    quat = getattr(imu_state, "quaternion", None)
    return quaternion_to_rpy(quat)


def pad_to_length(values, length: int, fill: float = 0.0) -> list[float]:
    padded = [float(v) for v in values[:length]]
    if len(padded) < length:
        padded.extend([float(fill)] * (length - len(padded)))
    return padded


@dataclass
class ArmState:
    q: np.ndarray
    dq: np.ndarray
    q7: np.ndarray
    dq7: np.ndarray
    torso_rpy: np.ndarray


class ReadOnlyG1ArmState:
    """Read-only lowstate subscriber for warm-starting IK without publishing motor commands."""

    def __init__(self, robot_config: dict):
        self.arm_indices = robot_config["arm_indices"]
        self.arm7_indices = robot_config["left_arm7_indices"] + robot_config["right_arm7_indices"]
        self.subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.subscriber.Init()
        self.buffer = DataBuffer()
        self.thread = threading.Thread(target=self._subscribe, daemon=True)
        self.thread.start()

        while self.buffer.get() is None:
            logger_mp.info("[Bridge] Waiting for rt/lowstate...")
            time.sleep(0.1)

    def _subscribe(self):
        while True:
            msg = self.subscriber.Read()
            if msg is not None:
                q = np.array([msg.motor_state[idx].q for idx in self.arm_indices], dtype=np.float64)
                dq = np.array([msg.motor_state[idx].dq for idx in self.arm_indices], dtype=np.float64)
                q7 = np.array([msg.motor_state[idx].q for idx in self.arm7_indices], dtype=np.float64)
                dq7 = np.array([msg.motor_state[idx].dq for idx in self.arm7_indices], dtype=np.float64)
                self.buffer.set(ArmState(q=q, dq=dq, q7=q7, dq7=dq7, torso_rpy=extract_lowstate_rpy(msg)))
            time.sleep(0.002)

    def get_arm_state(self) -> ArmState:
        state = self.buffer.get()
        if state is None:
            return ArmState(
                q=np.zeros(10),
                dq=np.zeros(10),
                q7=np.zeros(14),
                dq7=np.zeros(14),
                torso_rpy=np.zeros(3),
            )
        return state


class InspireFTPControllerBridge:
    """Controller-mode Inspire FTP publisher using the official DDS hand driver topics."""

    def __init__(self):
        from inspire_sdkpy import inspire_dds
        import inspire_sdkpy.inspire_hand_defaut as inspire_hand_defaut

        self.inspire_hand_defaut = inspire_hand_defaut
        self.left_pub = ChannelPublisher("rt/inspire_hand/ctrl/l", inspire_dds.inspire_hand_ctrl)
        self.left_pub.Init()
        self.right_pub = ChannelPublisher("rt/inspire_hand/ctrl/r", inspire_dds.inspire_hand_ctrl)
        self.right_pub.Init()
        self.debug_count = 0
        logger_mp.info("[Bridge] Inspire FTP DDS publishers ready")

    @staticmethod
    def _normalize_trigger(trigger_value: float) -> float:
        # tv_wrapper reports 10.0=open, 0.0=fully pressed.
        return float(np.clip((10.0 - trigger_value) / 10.0, 0.0, 1.0))

    @staticmethod
    def _normalize_to_scaled(values: list[float]) -> list[int]:
        return [int(np.clip(v * 1000.0, 0.0, 1000.0)) for v in values]

    def publish_from_controller(self, left_trigger: float, right_trigger: float) -> None:
        left_close = self._normalize_trigger(left_trigger)
        right_close = self._normalize_trigger(right_trigger)
        left_open = 1.0 - left_close
        right_open = 1.0 - right_close

        left_msg = self.inspire_hand_defaut.get_inspire_hand_ctrl()
        left_msg.angle_set = self._normalize_to_scaled([left_open] * 6)
        left_msg.mode = 0b0001
        self.left_pub.Write(left_msg)

        right_msg = self.inspire_hand_defaut.get_inspire_hand_ctrl()
        right_msg.angle_set = self._normalize_to_scaled([right_open] * 6)
        right_msg.mode = 0b0001
        self.right_pub.Write(right_msg)

        if self.debug_count < 10:
            logger_mp.info(
                f"[Bridge] Inspire controller cmd L={left_msg.angle_set} R={right_msg.angle_set}"
            )
            self.debug_count += 1


class InspireFTPStateRecorder:
    """Passive Inspire FTP DDS subscriber for recording real hand state and tactile data."""

    def __init__(self):
        from inspire_sdkpy import inspire_dds

        self._touch_count = INSPIRE_FTP_TOUCH_NUM_VALUES
        self.left_state = np.zeros(6, dtype=np.float64)
        self.right_state = np.zeros(6, dtype=np.float64)
        self.left_force = np.zeros(6, dtype=np.float64)
        self.right_force = np.zeros(6, dtype=np.float64)
        self.left_touch = np.zeros(self._touch_count, dtype=np.float64)
        self.right_touch = np.zeros(self._touch_count, dtype=np.float64)
        self._lock = threading.Lock()
        self._last_log_time = time.time()
        self._stats = {
            "left_state": 0,
            "right_state": 0,
            "left_touch": 0,
            "right_touch": 0,
            "left_force_nonzero": 0,
            "right_force_nonzero": 0,
            "left_touch_nonzero": 0,
            "right_touch_nonzero": 0,
            "left_state_time": 0.0,
            "right_state_time": 0.0,
            "left_touch_time": 0.0,
            "right_touch_time": 0.0,
        }

        self.left_state_sub = ChannelSubscriber("rt/inspire_hand/state/l", inspire_dds.inspire_hand_state)
        self.left_state_sub.Init(self._on_left_state, 10)
        self.right_state_sub = ChannelSubscriber("rt/inspire_hand/state/r", inspire_dds.inspire_hand_state)
        self.right_state_sub.Init(self._on_right_state, 10)
        self.left_touch_sub = ChannelSubscriber("rt/inspire_hand/touch/l", inspire_dds.inspire_hand_touch)
        self.left_touch_sub.Init(self._on_left_touch, 10)
        self.right_touch_sub = ChannelSubscriber("rt/inspire_hand/touch/r", inspire_dds.inspire_hand_touch)
        self.right_touch_sub.Init(self._on_right_touch, 10)
        logger_mp.info("[Bridge] Inspire FTP state recorder DDS subscribers ready")

    @staticmethod
    def _copy_state(msg, state_out: np.ndarray, force_out: np.ndarray) -> None:
        if msg is None:
            return
        if hasattr(msg, "angle_act") and len(msg.angle_act) >= 6:
            state_out[:] = np.array(msg.angle_act[:6], dtype=np.float64) / 1000.0
        if hasattr(msg, "force_act") and len(msg.force_act) >= 6:
            force_out[:] = np.array(msg.force_act[:6], dtype=np.float64)

    @staticmethod
    def _flatten_touch_msg(touch_msg) -> np.ndarray | None:
        if touch_msg is None:
            return None
        flat_values = []
        for field_name, _ in INSPIRE_FTP_TOUCH_FIELDS:
            values = getattr(touch_msg, field_name, [])
            flat_values.extend(float(v) for v in values)
        return np.array(flat_values, dtype=np.float64)

    def _handle_state(
        self,
        msg,
        state_out: np.ndarray,
        force_out: np.ndarray,
        count_key: str,
        nonzero_key: str,
        time_key: str,
    ) -> None:
        with self._lock:
            self._copy_state(msg, state_out, force_out)
            self._stats[count_key] += 1
            self._stats[time_key] = time.time()
            if np.any(np.abs(force_out) > 0.0):
                self._stats[nonzero_key] += 1

    def _handle_touch(
        self,
        msg,
        touch_out: np.ndarray,
        count_key: str,
        nonzero_key: str,
        time_key: str,
    ) -> None:
        flat = self._flatten_touch_msg(msg)
        if flat is None or flat.size != touch_out.size:
            return
        with self._lock:
            touch_out[:] = flat
            self._stats[count_key] += 1
            self._stats[time_key] = time.time()
            if np.any(np.abs(touch_out) > 0.0):
                self._stats[nonzero_key] += 1

    def _on_left_state(self, msg) -> None:
        self._handle_state(msg, self.left_state, self.left_force, "left_state", "left_force_nonzero", "left_state_time")

    def _on_right_state(self, msg) -> None:
        self._handle_state(msg, self.right_state, self.right_force, "right_state", "right_force_nonzero", "right_state_time")

    def _on_left_touch(self, msg) -> None:
        self._handle_touch(msg, self.left_touch, "left_touch", "left_touch_nonzero", "left_touch_time")

    def _on_right_touch(self, msg) -> None:
        self._handle_touch(msg, self.right_touch, "right_touch", "right_touch_nonzero", "right_touch_time")

    def _maybe_log_stats(self, stats: dict) -> None:
        now = time.time()
        if now - self._last_log_time < 5.0:
            return
        self._last_log_time = now
        left_state_age = now - stats["left_state_time"] if stats["left_state_time"] > 0.0 else float("inf")
        right_state_age = now - stats["right_state_time"] if stats["right_state_time"] > 0.0 else float("inf")
        left_touch_age = now - stats["left_touch_time"] if stats["left_touch_time"] > 0.0 else float("inf")
        right_touch_age = now - stats["right_touch_time"] if stats["right_touch_time"] > 0.0 else float("inf")
        logger_mp.info(
            "[Bridge] Inspire FTP recorder "
            f"state_msgs L/R={stats['left_state']}/{stats['right_state']} "
            f"touch_msgs L/R={stats['left_touch']}/{stats['right_touch']} "
            f"force_nonzero L/R={stats['left_force_nonzero']}/{stats['right_force_nonzero']} "
            f"touch_nonzero L/R={stats['left_touch_nonzero']}/{stats['right_touch_nonzero']} "
            f"age_s state L/R={left_state_age:.2f}/{right_state_age:.2f} "
            f"touch L/R={left_touch_age:.2f}/{right_touch_age:.2f}"
        )

    def poll(self):
        with self._lock:
            left_state = self.left_state.copy()
            right_state = self.right_state.copy()
            left_force = self.left_force.copy()
            right_force = self.right_force.copy()
            left_touch = self.left_touch.copy()
            right_touch = self.right_touch.copy()
            stats = dict(self._stats)
        self._maybe_log_stats(stats)
        return (
            left_state,
            right_state,
            left_force,
            right_force,
            left_touch,
            right_touch,
        )


class UDPUpperBodyBridge:
    """Receive upper-body wrist poses and Inspire triggers from Windows/SteamVR UDP."""

    G1_HOME_L = np.array([0.30, 0.25, -0.20], dtype=np.float64)
    G1_HOME_R = np.array([0.30, -0.25, -0.20], dtype=np.float64)
    T_VR_TO_G1 = np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    def __init__(self, host: str, port: int, enable_inspire: bool) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.setblocking(False)
        self.left_matrix = np.eye(4, dtype=np.float64)
        self.right_matrix = np.eye(4, dtype=np.float64)
        self.left_matrix[:3, 3] = self.G1_HOME_L
        self.right_matrix[:3, 3] = self.G1_HOME_R
        self.vr_home_l = None
        self.vr_home_r = None
        self.vr_rot_init_l = None
        self.vr_rot_init_r = None
        self.left_trigger = 0.0
        self.right_trigger = 0.0
        self.left_hand_pos = np.zeros((25, 3), dtype=np.float64)
        self.right_hand_pos = np.zeros((25, 3), dtype=np.float64)
        self.message_count = 0
        self.last_packet_time = None
        self._last_status_log_time = 0.0
        self._first_packet_logged = False
        self.enable_inspire = enable_inspire
        self.inspire_pub_l = None
        self.inspire_pub_r = None
        self.inspire_hand_defaut = None

        if enable_inspire:
            try:
                from inspire_sdkpy import inspire_dds
                import inspire_sdkpy.inspire_hand_defaut as inspire_hand_defaut

                self.inspire_hand_defaut = inspire_hand_defaut
                self.inspire_pub_l = ChannelPublisher("rt/inspire_hand/ctrl/l", inspire_dds.inspire_hand_ctrl)
                self.inspire_pub_l.Init()
                self.inspire_pub_r = ChannelPublisher("rt/inspire_hand/ctrl/r", inspire_dds.inspire_hand_ctrl)
                self.inspire_pub_r.Init()
                logger_mp.info("[UDP] Inspire FTP DDS publishers ready")
            except Exception as exc:
                self.enable_inspire = False
                logger_mp.warning(f"[UDP] Failed to init Inspire DDS publishers: {exc}")

        logger_mp.info(f"[UDP] Listening on {host}:{port} for SteamVR packets")

    @staticmethod
    def _normalize_trigger_for_record(trigger_value: float) -> float:
        return float(np.clip(trigger_value, 0.0, 1.0))

    @staticmethod
    def _coerce_close_value(value) -> float | None:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(np.clip(value, 0.0, 1.0))
        return None

    @staticmethod
    def _coerce_hand_points(value) -> np.ndarray | None:
        try:
            arr = np.array(value, dtype=np.float64)
        except Exception:
            return None
        if arr.shape == (25, 3):
            return arr
        if arr.size == 75:
            return arr.reshape(25, 3)
        return None

    def _apply_matrix(self, raw_pose: np.ndarray, side: str) -> np.ndarray:
        if side == "left":
            if self.vr_home_l is None:
                self.vr_home_l = raw_pose[:3, 3].copy()
                self.vr_rot_init_l = raw_pose[:3, :3].copy()
                logger_mp.info("[UDP] Left wrist origin captured")
            home = self.vr_home_l
            rot_init = self.vr_rot_init_l
            g1_home = self.G1_HOME_L
        else:
            if self.vr_home_r is None:
                self.vr_home_r = raw_pose[:3, 3].copy()
                self.vr_rot_init_r = raw_pose[:3, :3].copy()
                logger_mp.info("[UDP] Right wrist origin captured")
            home = self.vr_home_r
            rot_init = self.vr_rot_init_r
            g1_home = self.G1_HOME_R

        delta_p = raw_pose[:3, 3] - home
        out = np.eye(4, dtype=np.float64)
        out[0, 3] = g1_home[0] - delta_p[2]
        out[1, 3] = g1_home[1] - delta_p[0]
        out[2, 3] = g1_home[2] + delta_p[1]

        vr_rot_cur = raw_pose[:3, :3]
        vr_rot_delta = vr_rot_cur @ rot_init.T
        g1_rot_delta = self.T_VR_TO_G1 @ vr_rot_delta @ self.T_VR_TO_G1.T
        out[:3, :3] = g1_rot_delta
        return out

    def _publish_inspire(self, trigger_value: float, is_left_trigger: bool) -> None:
        if not self.enable_inspire or self.inspire_hand_defaut is None:
            return
        angle = int((1.0 - float(np.clip(trigger_value, 0.0, 1.0))) * 1000.0)
        angle = max(0, min(1000, angle))
        msg = self.inspire_hand_defaut.get_inspire_hand_ctrl()
        msg.angle_set = [angle] * 6
        msg.mode = 1
        if is_left_trigger:
            self.inspire_pub_l.Write(msg)
        else:
            self.inspire_pub_r.Write(msg)

    def poll(self) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
        try:
            while True:
                packet, _ = self.sock.recvfrom(4096)
                msg = json.loads(packet.decode("utf-8"))
                self.message_count += 1
                self.last_packet_time = time.time()
                if not self._first_packet_logged:
                    logger_mp.info(
                        f"[UDP] First packet received keys={sorted(list(msg.keys()))[:12]} "
                        f"count={self.message_count}"
                    )
                    self._first_packet_logged = True

                if "left" in msg:
                    raw_l = np.array(msg["left"], dtype=np.float64)
                    self.left_matrix = self._apply_matrix(raw_l, "left")
                if "right" in msg:
                    raw_r = np.array(msg["right"], dtype=np.float64)
                    self.right_matrix = self._apply_matrix(raw_r, "right")

                for field_name in ("left_hand_pos", "left_hand_positions", "left_hand_landmarks"):
                    if field_name in msg:
                        left_hand = self._coerce_hand_points(msg[field_name])
                        if left_hand is not None:
                            self.left_hand_pos = left_hand
                            break
                for field_name in ("right_hand_pos", "right_hand_positions", "right_hand_landmarks"):
                    if field_name in msg:
                        right_hand = self._coerce_hand_points(msg[field_name])
                        if right_hand is not None:
                            self.right_hand_pos = right_hand
                            break

                left_close = None
                for field_name in (
                    "left_trigger",
                    "left_grab",
                    "left_grip",
                    "left_middle",
                    "left_middle_button",
                    "left_grab_button",
                ):
                    if field_name in msg:
                        left_close = self._coerce_close_value(msg[field_name])
                        if left_close is not None:
                            break
                if left_close is not None:
                    self.left_trigger = self._normalize_trigger_for_record(left_close)
                    self._publish_inspire(self.left_trigger, is_left_trigger=True)

                right_close = None
                for field_name in (
                    "right_trigger",
                    "right_grab",
                    "right_grip",
                    "right_middle",
                    "right_middle_button",
                    "right_grab_button",
                ):
                    if field_name in msg:
                        right_close = self._coerce_close_value(msg[field_name])
                        if right_close is not None:
                            break
                if right_close is not None:
                    self.right_trigger = self._normalize_trigger_for_record(right_close)
                    self._publish_inspire(self.right_trigger, is_left_trigger=False)
        except BlockingIOError:
            pass

        now = time.time()
        if now - self._last_status_log_time >= 2.0:
            if self.last_packet_time is None:
                logger_mp.warning("[UDP] No packet received yet on upper-body UDP bridge")
            else:
                age = now - self.last_packet_time
                logger_mp.info(
                    f"[UDP] packets={self.message_count} "
                    f"last_packet_age={age:.2f}s "
                    f"left_trigger={self.left_trigger:.2f} right_trigger={self.right_trigger:.2f}"
                )
            self._last_status_log_time = now

        return (
            self.left_matrix.copy(),
            self.right_matrix.copy(),
            self.left_trigger,
            self.right_trigger,
            self.left_hand_pos.copy(),
            self.right_hand_pos.copy(),
        )


class PlannerBridgePublisher:
    def __init__(self, bind_host: str, port: int, keepalive_sec: float):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://{bind_host}:{port}")
        self.keepalive_sec = keepalive_sec
        self.last_keepalive = 0.0
        self.last_upper_body_position = None
        self.last_upper_body_time = None
        time.sleep(0.1)
        logger_mp.info(f"[Bridge] ZMQ PUB bound to tcp://{bind_host}:{port}")

    def close(self):
        self.socket.close()
        self.context.term()

    def send_start(self):
        self.socket.send(build_command_message(start=True, stop=False, planner=True))
        self.last_keepalive = time.monotonic()

    def send_stop(self):
        self.socket.send(build_command_message(start=False, stop=True, planner=True))

    def maybe_keepalive(self):
        now = time.monotonic()
        if now - self.last_keepalive >= self.keepalive_sec:
            self.socket.send(build_command_message(start=True, stop=False, planner=True))
            self.last_keepalive = now

    @staticmethod
    def arm_q_to_upper_body_position(arm_q: np.ndarray) -> np.ndarray:
        arm_q = np.asarray(arm_q, dtype=np.float32)
        upper = np.zeros(17, dtype=np.float32)
        upper[3] = arm_q[0]   # left_shoulder_pitch
        right_offset = 7 if arm_q.size >= 14 else 5
        upper[4] = arm_q[right_offset]   # right_shoulder_pitch
        upper[5] = arm_q[1]   # left_shoulder_roll
        upper[6] = arm_q[right_offset + 1]   # right_shoulder_roll
        upper[7] = arm_q[2]   # left_shoulder_yaw
        upper[8] = arm_q[right_offset + 2]   # right_shoulder_yaw
        upper[9] = arm_q[3]   # left_elbow
        upper[10] = arm_q[right_offset + 3]  # right_elbow
        upper[11] = arm_q[4]  # left_wrist_roll
        upper[12] = arm_q[right_offset + 4]  # right_wrist_roll
        if arm_q.size >= 14:
            upper[13] = arm_q[5]   # left_wrist_pitch
            upper[14] = arm_q[12]  # right_wrist_pitch
            upper[15] = arm_q[6]   # left_wrist_yaw
            upper[16] = arm_q[13]  # right_wrist_yaw
        return upper

    def _compute_upper_body_velocity(self, upper_pos: np.ndarray) -> np.ndarray:
        now = time.monotonic()
        if self.last_upper_body_position is None or self.last_upper_body_time is None:
            vel = np.zeros_like(upper_pos)
        else:
            dt = max(1e-3, now - self.last_upper_body_time)
            vel = (upper_pos - self.last_upper_body_position) / dt
        self.last_upper_body_position = upper_pos.copy()
        self.last_upper_body_time = now
        return vel.astype(np.float32)

    def publish(
        self,
        mode: int,
        movement: np.ndarray,
        facing: np.ndarray,
        speed: float,
        height: float,
        arm_q: np.ndarray,
    ) -> None:
        upper_pos = self.arm_q_to_upper_body_position(arm_q)
        upper_vel = self._compute_upper_body_velocity(upper_pos)
        payload = build_planner_message(
            mode=mode,
            movement=movement.tolist(),
            facing=facing.tolist(),
            speed=speed,
            height=height,
            upper_body_position=upper_pos.tolist(),
            upper_body_velocity=upper_vel.tolist(),
        )
        self.socket.send(payload)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bridge xr_teleoperate PICO upper body to Sonic planner, while keeping Inspire hand DDS"
    )
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--input-mode", choices=["hand", "controller"], default="controller")
    parser.add_argument(
        "--upper-body-source",
        choices=["udp", "televuer"],
        default="udp",
        help="Upper-body wrist pose source",
    )
    parser.add_argument(
        "--display-mode", choices=["immersive", "ego", "pass-through"], default="immersive"
    )
    parser.add_argument("--arm", choices=sorted(ROBOT_ARM_CONFIGS.keys()), default="G1_29")
    parser.add_argument("--ee", choices=["none", "inspire_ftp"], default="inspire_ftp")
    parser.add_argument("--img-server-ip", type=str, default="192.168.123.164")
    parser.add_argument("--network-interface", type=str, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--bind-host", type=str, default="*")
    parser.add_argument("--zmq-port", type=int, default=5556)
    parser.add_argument("--udp-host", type=str, default="0.0.0.0", help="UDP bind host for Windows bridge")
    parser.add_argument("--udp-port", type=int, default=5005, help="UDP port for Windows bridge")
    parser.add_argument(
        "--planner-udp-host",
        type=str,
        default="0.0.0.0",
        help="UDP bind host for remote planner commands",
    )
    parser.add_argument(
        "--planner-udp-port",
        type=int,
        default=5006,
        help="UDP port for remote planner commands",
    )
    parser.add_argument(
        "--planner-mode",
        choices=sorted(LOCOMOTION_MODES.keys()),
        default="walk",
        help="Initial Sonic planner mode",
    )
    parser.add_argument("--command-keepalive", type=float, default=1.0)
    parser.add_argument("--record", action="store_true", help="Enable episode recording")
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start the Sonic bridge immediately without waiting for ]",
    )
    parser.add_argument("--task-dir", type=str, default="./utils/data/", help="Path to save recorded data")
    parser.add_argument("--task-name", type=str, default="pick cube", help="Task folder name")
    parser.add_argument("--task-goal", type=str, default="pick up cube.", help="Task goal stored in json")
    parser.add_argument("--task-desc", type=str, default="task description", help="Task description")
    parser.add_argument("--task-steps", type=str, default="step1: do this; step2: do that;", help="Task steps")
    return parser.parse_args()


def main():
    global READY, RECORD_RUNNING, RECORD_TOGGLE, START, STOP
    args = parse_args()
    logger_mp.info(f"args: {args}")

    ChannelFactoryInitialize(0, networkInterface=args.network_interface)

    listen_keyboard_thread = threading.Thread(
        target=listen_keyboard,
        kwargs={"on_press": on_press, "until": None, "sequential": False},
        daemon=True,
    )
    listen_keyboard_thread.start()

    img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
    camera_config = img_client.get_cam_config()
    xr_need_local_img = False
    tv_wrapper = None
    if args.upper_body_source == "televuer":
        xr_need_local_img = (not args.headless) and not (
            args.display_mode == "pass-through" or camera_config["head_camera"]["enable_webrtc"]
        )
        tv_wrapper = TeleVuerWrapper(
            use_hand_tracking=args.input_mode == "hand",
            binocular=camera_config["head_camera"]["binocular"],
            img_shape=camera_config["head_camera"]["image_shape"],
            display_mode=args.display_mode,
            zmq=camera_config["head_camera"]["enable_zmq"],
            webrtc=camera_config["head_camera"]["enable_webrtc"],
            webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
        )

    robot_config = ROBOT_ARM_CONFIGS[args.arm]
    arm_ik = robot_config["ik_cls"]()
    arm_state_reader = ReadOnlyG1ArmState(robot_config)
    sonic_pub = PlannerBridgePublisher(
        bind_host=args.bind_host,
        port=args.zmq_port,
        keepalive_sec=args.command_keepalive,
    )
    planner_keyboard = PlannerKeyboardState(initial_mode=args.planner_mode, speed=-1.0, height=-1.0)
    recorder = None
    udp_bridge = None
    planner_udp_server = UDPPlannerCommandServer(
        host=args.planner_udp_host,
        port=args.planner_udp_port,
    )

    if args.record:
        recorder = EpisodeWriter(
            task_dir=os.path.join(args.task_dir, args.task_name),
            task_goal=args.task_goal,
            task_desc=args.task_desc,
            task_steps=args.task_steps,
            frequency=args.frequency,
            rerun_log=not args.headless,
        )
        if args.ee == "inspire_ftp":
            recorder.info["joint_names"]["left_arm"] = G1_ARM_7_NAMES
            recorder.info["joint_names"]["right_arm"] = G1_ARM_7_NAMES
            recorder.info["joint_names"]["left_arm_ee"] = ARM_EE_QPOS_NAMES
            recorder.info["joint_names"]["right_arm_ee"] = ARM_EE_QPOS_NAMES
            recorder.info["joint_names"]["left_ee"] = INSPIRE_FTP_JOINT_NAMES
            recorder.info["joint_names"]["right_ee"] = INSPIRE_FTP_JOINT_NAMES
            recorder.info["joint_names"]["left_hand"] = [f"inspire_ftp_{i}" for i in range(7)]
            recorder.info["joint_names"]["right_hand"] = [f"inspire_ftp_{i}" for i in range(7)]
            recorder.info["joint_names"]["body"] = BODY_STATE_NAMES
            recorder.info["joint_names"]["body_rpy"] = ["roll", "pitch", "yaw"]
            recorder.info["joint_names"]["psi0_state"] = PSI0_STATE_NAMES
            recorder.info["joint_names"]["psi0_action"] = PSI0_ACTION_NAMES
            recorder.info["tactile_names"]["left_ee"] = INSPIRE_FTP_TACTILE_NAMES
            recorder.info["tactile_names"]["right_ee"] = INSPIRE_FTP_TACTILE_NAMES
            recorder.info["tactile_dims"] = {
                "force_act_per_hand": 6,
                "touch_per_hand": INSPIRE_FTP_TOUCH_NUM_VALUES,
                "touch_fields": {name: length for name, length in INSPIRE_FTP_TOUCH_FIELDS},
            }
        recorder.info["camera_names"] = {
            "color_0": "head_camera",
            "color_1": "left_wrist_camera",
            "color_2": "right_wrist_camera",
        }
        recorder.info["camera_config"] = {
            key: {
                "enable_zmq": bool(camera_config.get(key, {}).get("enable_zmq", False)),
                "image_shape": camera_config.get(key, {}).get("image_shape"),
                "fps": camera_config.get(key, {}).get("fps"),
            }
            for key in ("head_camera", "left_wrist_camera", "right_wrist_camera")
            if key in camera_config
        }

    left_hand_pos_array = None
    right_hand_pos_array = None
    dual_hand_data_lock = None
    dual_hand_state_array = None
    dual_hand_action_array = None
    dual_hand_force_array = None
    dual_hand_touch_array = None
    hand_ctrl = None
    hand_ctrl_bridge = None
    inspire_state_recorder = None

    if args.ee == "inspire_ftp":
        inspire_state_recorder = InspireFTPStateRecorder()
        if args.input_mode == "hand":
            from teleop.robot_control.robot_hand_inspire_inspire_record import (
                Inspire_Controller_FTP,
            )

            left_hand_pos_array = Array("d", 75, lock=True)
            right_hand_pos_array = Array("d", 75, lock=True)
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array("d", 12, lock=False)
            dual_hand_action_array = Array("d", 12, lock=False)
            dual_hand_force_array = Array("d", 12, lock=False)
            dual_hand_touch_array = Array("d", INSPIRE_FTP_TOUCH_NUM_VALUES * 2, lock=False)
            hand_ctrl = Inspire_Controller_FTP(
                left_hand_pos_array,
                right_hand_pos_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
                dual_hand_force_array,
                dual_hand_touch_array,
                simulation_mode=False,
            )
        elif args.upper_body_source == "televuer":
            hand_ctrl_bridge = InspireFTPControllerBridge()

    if args.upper_body_source == "udp":
        udp_bridge = UDPUpperBodyBridge(
            host=args.udp_host,
            port=args.udp_port,
            enable_inspire=args.ee == "inspire_ftp",
        )

    logger_mp.info("----------------------------------------------------------------")
    if args.auto_start:
        logger_mp.info("🟢  Auto-start enabled: Sonic planner bridge will start immediately.")
    else:
        logger_mp.info("🟢  Press ] to start Sonic planner bridge.")
    logger_mp.info("🟡  Planner keys live in Terminal 2: W/S A/D Q/E ,/. N/P 1-8 9/0 -/= R ` ~")
    logger_mp.info("🟠  Press [k] to START or SAVE recording.")
    logger_mp.info("🔴  Press [o] to stop and exit.")
    if args.upper_body_source == "udp":
        logger_mp.info(f"🛰️  Upper body source: UDP {args.udp_host}:{args.udp_port} from Windows/SteamVR")
    else:
        logger_mp.info("🕶️  Upper body source: TeleVuer direct XR connection")
    logger_mp.info(
        f"🎮  Planner remote UDP: {args.planner_udp_host}:{args.planner_udp_port} "
        "(JSON: key/keys/planner_text/start/stop/record_toggle)"
    )
    logger_mp.info("⚠️  Keep focus on Terminal 2 while sending planner keyboard commands.")
    READY = True

    if args.auto_start:
        START = True
    else:
        while not START and not STOP:
            time.sleep(0.033)
            _, remote_start, remote_stop, remote_record = planner_udp_server.poll()
            if remote_start:
                START = True
            if remote_stop:
                STOP = True
            if remote_record:
                RECORD_TOGGLE = True
            if tv_wrapper is not None and camera_config["head_camera"]["enable_zmq"] and xr_need_local_img:
                head_img = img_client.get_head_frame()
                tv_wrapper.render_to_xr(head_img)

    if STOP:
        return

    sonic_pub.send_start()
    logger_mp.info("[Bridge] Sonic planner bridge started")
    last_slow_loop_log_time = 0.0
    last_record_path_log_time = 0.0
    record_warn_threshold = 0.05
    loop_index = 0

    try:
        while not STOP:
            loop_index += 1
            start_time = time.time()
            head_img = None
            left_wrist_img = None
            right_wrist_img = None
            head_fetch_time = 0.0
            left_wrist_fetch_time = 0.0
            right_wrist_fetch_time = 0.0
            tactile_poll_time = 0.0
            recorder_add_time = 0.0
            left_trigger = 0.0
            right_trigger = 0.0
            left_udp_hand_pos = None
            right_udp_hand_pos = None
            planner_events = drain_key_events()
            remote_planner_events, remote_start, remote_stop, remote_record = planner_udp_server.poll()
            planner_events.extend(remote_planner_events)
            if remote_start:
                START = True
            if remote_stop:
                STOP = True
            if remote_record:
                RECORD_TOGGLE = True

            udp_done_time = time.time()
            # Drain high-rate UDP teleop packets before any camera/recording work that may stall.
            if args.upper_body_source == "udp":
                (
                    left_wrist_pose,
                    right_wrist_pose,
                    left_trigger,
                    right_trigger,
                    left_udp_hand_pos,
                    right_udp_hand_pos,
                ) = udp_bridge.poll()
                if args.ee == "inspire_ftp" and args.input_mode == "hand" and left_hand_pos_array is not None:
                    with left_hand_pos_array.get_lock():
                        left_hand_pos_array[:] = left_udp_hand_pos.flatten()
                    with right_hand_pos_array.get_lock():
                        right_hand_pos_array[:] = right_udp_hand_pos.flatten()
            else:
                tele_data = tv_wrapper.get_tele_data()
                left_wrist_pose = tele_data.left_wrist_pose
                right_wrist_pose = tele_data.right_wrist_pose

                if args.ee == "inspire_ftp":
                    if args.input_mode == "hand" and left_hand_pos_array is not None:
                        with left_hand_pos_array.get_lock():
                            left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                        with right_hand_pos_array.get_lock():
                            right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
                    elif hand_ctrl_bridge is not None:
                        hand_ctrl_bridge.publish_from_controller(
                            tele_data.left_ctrl_triggerValue,
                            tele_data.right_ctrl_triggerValue,
                        )
                        left_trigger = InspireFTPControllerBridge._normalize_trigger(tele_data.left_ctrl_triggerValue)
                        right_trigger = InspireFTPControllerBridge._normalize_trigger(tele_data.right_ctrl_triggerValue)
            udp_done_time = time.time()

            if tv_wrapper is not None and camera_config["head_camera"]["enable_zmq"] and xr_need_local_img:
                head_fetch_start = time.time()
                head_img = img_client.get_head_frame()
                head_fetch_time = time.time() - head_fetch_start
                tv_wrapper.render_to_xr(head_img)
            elif camera_config["head_camera"]["enable_zmq"] and args.record:
                head_fetch_start = time.time()
                head_img = img_client.get_head_frame()
                head_fetch_time = time.time() - head_fetch_start

            if camera_config["left_wrist_camera"]["enable_zmq"] and args.record:
                left_fetch_start = time.time()
                left_wrist_img = img_client.get_left_wrist_frame()
                left_wrist_fetch_time = time.time() - left_fetch_start
            if camera_config["right_wrist_camera"]["enable_zmq"] and args.record:
                right_fetch_start = time.time()
                right_wrist_img = img_client.get_right_wrist_frame()
                right_wrist_fetch_time = time.time() - right_fetch_start
            image_done_time = time.time()

            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder is not None and recorder.create_episode():
                        RECORD_RUNNING = True
                        logger_mp.info("[Bridge] Recording started")
                    else:
                        logger_mp.error("[Bridge] Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    if recorder is not None:
                        recorder.save_episode()
                    logger_mp.info("[Bridge] Recording save requested")
            record_toggle_done_time = time.time()

            mode, movement, facing, speed, height = planner_keyboard.update(planner_events)
            if loop_index <= 3:
                logger_mp.info(f"[Bridge] Loop {loop_index}: planner update complete")

            if loop_index <= 3:
                logger_mp.info(f"[Bridge] Loop {loop_index}: reading arm state")
            arm_state = arm_state_reader.get_arm_state()
            if loop_index <= 3:
                logger_mp.info(f"[Bridge] Loop {loop_index}: arm state ready")

            if loop_index <= 3:
                logger_mp.info(f"[Bridge] Loop {loop_index}: solving IK")
            sol_q, _ = arm_ik.solve_ik(
                left_wrist_pose,
                right_wrist_pose,
                arm_state.q,
                arm_state.dq,
            )
            ik_done_time = time.time()
            if loop_index <= 3:
                logger_mp.info(f"[Bridge] Loop {loop_index}: IK complete")

            if loop_index <= 3:
                logger_mp.info(f"[Bridge] Loop {loop_index}: publishing planner command")
            sonic_pub.maybe_keepalive()
            sonic_pub.publish(
                mode=mode,
                movement=movement,
                facing=facing,
                speed=speed,
                height=height,
                arm_q=sol_q,
            )
            publish_done_time = time.time()
            if loop_index <= 3:
                logger_mp.info(f"[Bridge] Loop {loop_index}: planner publish complete")

            if args.record and recorder is not None:
                if loop_index <= 3:
                    logger_mp.info(f"[Bridge] Loop {loop_index}: entering record path")
                READY = recorder.is_ready()
                tactile_payload = None
                try:
                    tactile_poll_start = time.time()
                    if args.ee == "inspire_ftp" and args.input_mode == "hand" and dual_hand_state_array is not None:
                        with dual_hand_data_lock:
                            left_ee_state = list(dual_hand_state_array[:6])
                            right_ee_state = list(dual_hand_state_array[-6:])
                            left_hand_action = list(dual_hand_action_array[:6])
                            right_hand_action = list(dual_hand_action_array[-6:])
                            if dual_hand_force_array is not None and dual_hand_touch_array is not None:
                                left_force = list(dual_hand_force_array[:6])
                                right_force = list(dual_hand_force_array[-6:])
                                half_touch = len(dual_hand_touch_array) // 2
                                left_touch = list(dual_hand_touch_array[:half_touch])
                                right_touch = list(dual_hand_touch_array[half_touch:])
                                tactile_payload = {
                                    "left_ee": {
                                        "force_act": left_force,
                                        "touch": left_touch,
                                    },
                                    "right_ee": {
                                        "force_act": right_force,
                                        "touch": right_touch,
                                    },
                                }
                    elif args.ee == "inspire_ftp" and inspire_state_recorder is not None:
                        (
                            left_state_obs,
                            right_state_obs,
                            left_force_obs,
                            right_force_obs,
                            left_touch_obs,
                            right_touch_obs,
                        ) = inspire_state_recorder.poll()
                        left_ee_state = list(left_state_obs)
                        right_ee_state = list(right_state_obs)
                        left_hand_action = [left_trigger] * 6
                        right_hand_action = [right_trigger] * 6
                        tactile_payload = {
                            "left_ee": {
                                "force_act": list(left_force_obs),
                                "touch": list(left_touch_obs),
                            },
                            "right_ee": {
                                "force_act": list(right_force_obs),
                                "touch": list(right_touch_obs),
                            },
                        }
                    elif args.ee == "inspire_ftp":
                        left_ee_state = [left_trigger] * 6
                        right_ee_state = [right_trigger] * 6
                        left_hand_action = [left_trigger] * 6
                        right_hand_action = [right_trigger] * 6
                    else:
                        left_ee_state = []
                        right_ee_state = []
                        left_hand_action = []
                        right_hand_action = []
                    tactile_poll_time = time.time() - tactile_poll_start
                except Exception as exc:
                    tactile_poll_time = time.time() - tactile_poll_start
                    logger_mp.warning(f"[Bridge] Record path tactile polling failed: {exc}")
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    tactile_payload = None

                if RECORD_RUNNING:
                    try:
                        colors = {}
                        depths = {}
                        if head_img is not None:
                            if getattr(head_img, "bgr", None) is None:
                                logger_mp.warning("[Bridge] Record path head image is missing BGR data; skipping head frame.")
                            elif camera_config["head_camera"]["binocular"]:
                                colors["color_0"] = head_img.bgr[:, : camera_config["head_camera"]["image_shape"][1] // 2]
                                colors["color_1"] = head_img.bgr[:, camera_config["head_camera"]["image_shape"][1] // 2 :]
                                if left_wrist_img is not None and getattr(left_wrist_img, "bgr", None) is not None:
                                    colors["color_2"] = left_wrist_img.bgr
                                if right_wrist_img is not None and getattr(right_wrist_img, "bgr", None) is not None:
                                    colors["color_3"] = right_wrist_img.bgr
                            else:
                                colors["color_0"] = head_img.bgr
                                if left_wrist_img is not None and getattr(left_wrist_img, "bgr", None) is not None:
                                    colors["color_1"] = left_wrist_img.bgr
                                if right_wrist_img is not None and getattr(right_wrist_img, "bgr", None) is not None:
                                    colors["color_2"] = right_wrist_img.bgr

                        left_arm_state = arm_state.q7[:7].tolist()
                        right_arm_state = arm_state.q7[7:].tolist()
                        if len(sol_q) >= 14:
                            left_arm_action = sol_q[:7].tolist()
                            right_arm_action = sol_q[7:14].tolist()
                        else:
                            left_arm_action = sol_q[:5].tolist() + arm_state.q7[5:7].tolist()
                            right_arm_action = sol_q[5:].tolist() + arm_state.q7[12:14].tolist()
                        left_arm_ee_state, right_arm_ee_state = arm_ik.compute_ee_qpos(arm_state.q)
                        left_arm_ee_action, right_arm_ee_action = arm_ik.compute_ee_qpos(sol_q)
                        left_hand_state_psi0 = pad_to_length(left_ee_state, 7)
                        right_hand_state_psi0 = pad_to_length(right_ee_state, 7)
                        left_hand_action_psi0 = pad_to_length(left_hand_action, 7)
                        right_hand_action_psi0 = pad_to_length(right_hand_action, 7)
                        torso_rpy = arm_state.torso_rpy.tolist()
                        current_body_state = [float(mode), float(speed), float(height), float(facing[0]), float(facing[1])]
                        current_body_action = movement.tolist()
                        target_yaw = float(math.atan2(float(facing[1]), float(facing[0])))
                        psi0_state = (
                            left_hand_state_psi0
                            + right_hand_state_psi0
                            + left_arm_state
                            + right_arm_state
                            + torso_rpy
                            + [float(height)]
                        )
                        psi0_action = (
                            left_hand_action_psi0
                            + right_hand_action_psi0
                            + left_arm_action
                            + right_arm_action
                            + torso_rpy
                            + [float(height)]
                            + [float(movement[0]), float(movement[1]), float(movement[2]), target_yaw]
                        )

                        states = {
                            "left_arm": {"qpos": left_arm_state, "qvel": [], "torque": []},
                            "right_arm": {"qpos": right_arm_state, "qvel": [], "torque": []},
                            "left_arm_ee": {"qpos": left_arm_ee_state.tolist(), "qvel": [], "torque": []},
                            "right_arm_ee": {"qpos": right_arm_ee_state.tolist(), "qvel": [], "torque": []},
                            "left_hand": {"qpos": left_hand_state_psi0, "qvel": [], "torque": []},
                            "right_hand": {"qpos": right_hand_state_psi0, "qvel": [], "torque": []},
                            "left_ee": {"qpos": left_ee_state, "qvel": [], "torque": []},
                            "right_ee": {"qpos": right_ee_state, "qvel": [], "torque": []},
                            "body": {"qpos": current_body_state, "rpy": torso_rpy, "height": float(height)},
                            "psi0": {"qpos": psi0_state, "names": PSI0_STATE_NAMES},
                        }
                        actions = {
                            "left_arm": {"qpos": left_arm_action, "qvel": [], "torque": []},
                            "right_arm": {"qpos": right_arm_action, "qvel": [], "torque": []},
                            "left_arm_ee": {"qpos": left_arm_ee_action.tolist(), "qvel": [], "torque": []},
                            "right_arm_ee": {"qpos": right_arm_ee_action.tolist(), "qvel": [], "torque": []},
                            "left_hand": {"qpos": left_hand_action_psi0, "qvel": [], "torque": []},
                            "right_hand": {"qpos": right_hand_action_psi0, "qvel": [], "torque": []},
                            "left_ee": {"qpos": left_hand_action, "qvel": [], "torque": []},
                            "right_ee": {"qpos": right_hand_action, "qvel": [], "torque": []},
                            "body": {
                                "qpos": current_body_action,
                                "rpy": torso_rpy,
                                "height": float(height),
                                "target_yaw": target_yaw,
                            },
                            "psi0": {"qpos": psi0_action, "names": PSI0_ACTION_NAMES},
                        }
                        recorder_add_start = time.time()
                        recorder.add_item(
                            colors=colors,
                            depths=depths,
                            states=states,
                            actions=actions,
                            tactiles=tactile_payload,
                        )
                        recorder_add_time = time.time() - recorder_add_start
                    except Exception as exc:
                        recorder_add_time = time.time() - recorder_add_start if "recorder_add_start" in locals() else 0.0
                        logger_mp.warning(f"[Bridge] Record path add_item failed: {exc}")
                if loop_index <= 3:
                    logger_mp.info(f"[Bridge] Loop {loop_index}: record path complete")
            record_done_time = time.time()

            if args.record and (record_done_time - last_record_path_log_time) >= 1.0:
                if (
                    head_fetch_time > record_warn_threshold
                    or left_wrist_fetch_time > record_warn_threshold
                    or right_wrist_fetch_time > record_warn_threshold
                    or tactile_poll_time > record_warn_threshold
                    or recorder_add_time > record_warn_threshold
                ):
                    logger_mp.warning(
                        "[Bridge] Record path timing "
                        f"head={head_fetch_time:.3f}s "
                        f"left_wrist={left_wrist_fetch_time:.3f}s "
                        f"right_wrist={right_wrist_fetch_time:.3f}s "
                        f"tactile={tactile_poll_time:.3f}s "
                        f"add_item={recorder_add_time:.3f}s"
                    )
                    last_record_path_log_time = record_done_time

            loop_elapsed = record_done_time - start_time
            if loop_elapsed > 0.2 and (record_done_time - last_slow_loop_log_time) >= 1.0:
                logger_mp.warning(
                    "[Bridge] Slow loop "
                    f"total={loop_elapsed:.3f}s "
                    f"udp={udp_done_time - start_time:.3f}s "
                    f"image={image_done_time - udp_done_time:.3f}s "
                    f"record_toggle={record_toggle_done_time - image_done_time:.3f}s "
                    f"ik={ik_done_time - record_toggle_done_time:.3f}s "
                    f"publish={publish_done_time - ik_done_time:.3f}s "
                    f"record={record_done_time - publish_done_time:.3f}s"
                )
                last_slow_loop_log_time = record_done_time

            sleep_time = max(0.0, (1.0 / args.frequency) - (time.time() - start_time))
            time.sleep(sleep_time)
    finally:
        try:
            sonic_pub.send_stop()
        except Exception as exc:
            logger_mp.warning(f"[Bridge] Failed to send stop: {exc}")
        try:
            sonic_pub.close()
        except Exception as exc:
            logger_mp.warning(f"[Bridge] Failed to close ZMQ: {exc}")
        try:
            stop_listening()
            listen_keyboard_thread.join(timeout=1.0)
        except Exception as exc:
            logger_mp.warning(f"[Bridge] Failed to stop keyboard listener: {exc}")
        try:
            img_client.close()
        except Exception as exc:
            logger_mp.warning(f"[Bridge] Failed to close image client: {exc}")
        try:
            if tv_wrapper is not None:
                tv_wrapper.close()
        except Exception as exc:
            logger_mp.warning(f"[Bridge] Failed to close TeleVuer: {exc}")
        try:
            if recorder is not None:
                recorder.close()
        except Exception as exc:
            logger_mp.warning(f"[Bridge] Failed to close recorder: {exc}")
        logger_mp.info("[Bridge] Exit complete")


if __name__ == "__main__":
    main()

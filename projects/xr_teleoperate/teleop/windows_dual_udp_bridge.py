import argparse
import json
import socket
import threading
import time
from typing import Any


SPECIAL_KEY_MAP = {
    "space": "r",
    "enter": None,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Windows-side UDP bridge for xr_teleoperate. "
            "Supports OpenVR controllers, external hand-landmark relay, and planner keyboard forwarding."
        )
    )
    parser.add_argument("--robot-ip", required=True, help="Robot IP running teleop_hand_and_arm_bridge.py")
    parser.add_argument("--teleop-port", type=int, default=5005, help="UDP port for wrist/hand teleop data")
    parser.add_argument("--planner-port", type=int, default=5006, help="UDP port for planner keyboard data")
    parser.add_argument(
        "--mode",
        choices=("controller", "hand", "hybrid"),
        default="hybrid",
        help="controller=OpenVR only, hand=external hand relay only, hybrid=both",
    )
    parser.add_argument(
        "--controller-rate",
        type=float,
        default=100.0,
        help="OpenVR controller send rate in Hz",
    )
    parser.add_argument(
        "--hand-source-port",
        type=int,
        default=5100,
        help="Local UDP port to receive hand-tracking JSON on Windows",
    )
    parser.add_argument(
        "--planner-rate",
        type=float,
        default=20.0,
        help="Planner keyboard UDP send rate in Hz",
    )
    parser.add_argument("--start-key", default="f8", help="Global key to send planner start")
    parser.add_argument("--record-key", default="f9", help="Global key to toggle recording")
    parser.add_argument("--stop-key", default="esc", help="Global key to stop keyboard bridge")
    parser.add_argument(
        "--disable-keyboard",
        action="store_true",
        help="Disable planner keyboard forwarding",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print live controller / hand / keyboard status",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=2.0,
        help="Periodic status heartbeat in seconds for teleop UDP sending",
    )
    return parser.parse_args()


def clamp01(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return float(value)
    return None


def normalize_key(key) -> str | None:
    name = getattr(key, "name", None)
    if name is not None:
        lower = name.lower()
        if lower in SPECIAL_KEY_MAP:
            return SPECIAL_KEY_MAP[lower]
        if len(lower) == 1:
            return lower
        return lower

    char = getattr(key, "char", None)
    if char is None:
        return None
    return char.lower()


def coerce_pose_matrix(value: Any) -> list[list[float]] | None:
    try:
        rows = [[float(x) for x in row] for row in value]
    except Exception:
        return None
    if len(rows) != 3 or any(len(row) != 4 for row in rows):
        return None
    return rows


def coerce_hand_points(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list):
        return None

    if len(value) == 25 and all(isinstance(row, list) and len(row) == 3 for row in value):
        try:
            return [[float(x), float(y), float(z)] for x, y, z in value]
        except Exception:
            return None

    if len(value) == 75:
        try:
            flat = [float(x) for x in value]
        except Exception:
            return None
        return [flat[i:i + 3] for i in range(0, 75, 3)]

    return None


def coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return None


def coerce_gesture_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else None
    return str(value)


class PlannerKeyboardBridge:
    def __init__(self, args):
        self.host = args.robot_ip
        self.port = args.planner_port
        self.rate = max(args.planner_rate, 1.0)
        self.start_key = args.start_key.lower()
        self.record_key = args.record_key.lower()
        self.stop_key = args.stop_key.lower()
        self.verbose = args.verbose
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.lock = threading.Lock()
        self.pressed_keys = set()
        self.one_shot_commands = []
        self.listener = None
        self.running = False
        self.thread = None

        self.planner_keys = {
            "w", "a", "s", "d", "q", "e", ",", ".", "n", "p",
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
            "-", "=", "+", "r", "`", "~", "o",
        }

    def _queue_command(self, command_name: str):
        with self.lock:
            self.one_shot_commands.append(command_name)

    def _on_press(self, key):
        name = normalize_key(key)
        if name is None:
            return
        if name == self.start_key:
            self._queue_command("start")
            return
        if name == self.record_key:
            self._queue_command("record_toggle")
            return
        if name == self.stop_key:
            self._queue_command("stop")
            return
        if name in self.planner_keys:
            with self.lock:
                self.pressed_keys.add(name)

    def _on_release(self, key):
        name = normalize_key(key)
        if name is None:
            return
        with self.lock:
            self.pressed_keys.discard(name)

    def start(self):
        from pynput import keyboard

        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(
            f"[KBUDP] planner keyboard -> udp://{self.host}:{self.port} "
            f"(start={self.start_key}, record={self.record_key}, stop={self.stop_key})"
        )

    def _loop(self):
        period = 1.0 / self.rate
        while self.running:
            with self.lock:
                keys = sorted(self.pressed_keys)
                commands = list(self.one_shot_commands)
                self.one_shot_commands.clear()

            if keys:
                payload = {"keys": keys}
                self.sock.sendto(json.dumps(payload).encode("utf-8"), (self.host, self.port))
                if self.verbose:
                    print(f"\r[KBUDP] keys={keys}", end="")

            for command_name in commands:
                payload = {"command": command_name}
                self.sock.sendto(json.dumps(payload).encode("utf-8"), (self.host, self.port))
                if self.verbose:
                    print(f"\n[KBUDP] command={command_name}")

            time.sleep(period)

    def stop(self):
        self.running = False
        if self.listener is not None:
            self.listener.stop()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.sock.close()


class OpenVRControllerSource:
    def __init__(self):
        self.openvr = None
        self.vr_system = None

    def start(self):
        import openvr

        self.openvr = openvr
        openvr.init(openvr.VRApplication_Background)
        self.vr_system = openvr.VRSystem()
        print("✅ OpenVR / SteamVR connected")

    def stop(self):
        if self.openvr is not None:
            self.openvr.shutdown()

    def _button_pressed(self, pressed_mask: int, button_id: int) -> bool:
        try:
            return bool(pressed_mask & self.openvr.ButtonMaskFromId(button_id))
        except Exception:
            return False

    def poll(self) -> dict[str, Any]:
        openvr = self.openvr
        poses = (openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount)()
        self.vr_system.getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0, poses)

        payload = {}
        for device_index in range(openvr.k_unMaxTrackedDeviceCount):
            if not poses[device_index].bPoseIsValid:
                continue
            if self.vr_system.getTrackedDeviceClass(device_index) != openvr.TrackedDeviceClass_Controller:
                continue

            role = self.vr_system.getControllerRoleForTrackedDeviceIndex(device_index)
            matrix = [
                [
                    poses[device_index].mDeviceToAbsoluteTracking[row][0],
                    poses[device_index].mDeviceToAbsoluteTracking[row][1],
                    poses[device_index].mDeviceToAbsoluteTracking[row][2],
                    poses[device_index].mDeviceToAbsoluteTracking[row][3],
                ]
                for row in range(3)
            ]

            result, state = self.vr_system.getControllerState(device_index)
            trigger = state.rAxis[2].x if result else 0.0
            pressed_mask = int(state.ulButtonPressed) if result else 0
            grip_pressed = self._button_pressed(pressed_mask, openvr.k_EButton_Grip)
            menu_pressed = self._button_pressed(pressed_mask, openvr.k_EButton_ApplicationMenu)
            a_pressed = self._button_pressed(pressed_mask, openvr.k_EButton_A)

            close_value = max(float(trigger), 1.0 if grip_pressed else 0.0, 1.0 if menu_pressed else 0.0)

            if role == openvr.TrackedControllerRole_LeftHand:
                payload["left"] = matrix
                payload["left_trigger"] = close_value
                payload["left_trigger_raw"] = float(trigger)
                payload["left_grab_button"] = grip_pressed
                payload["left_middle_button"] = menu_pressed
                payload["left_a_button"] = a_pressed
            elif role == openvr.TrackedControllerRole_RightHand:
                payload["right"] = matrix
                payload["right_trigger"] = close_value
                payload["right_trigger_raw"] = float(trigger)
                payload["right_grab_button"] = grip_pressed
                payload["right_middle_button"] = menu_pressed
                payload["right_a_button"] = a_pressed

        return payload


class UDPHandRelaySource:
    """
    Receives local hand-tracking JSON on Windows and normalizes it to the robot-side bridge format.

    Accepted input examples:
    1. {
         "left_hand_pos": [[x,y,z], ... 25 points ...],
         "right_hand_pos": [[x,y,z], ...],
         "left": [[...],[...],[...]],          # optional wrist 3x4 pose
         "right": [[...],[...],[...]],         # optional wrist 3x4 pose
         "left_grab": 0.8,                     # optional hand-close scalar
         "right_grab": 0.2
       }
    2. Same keys but with aliases:
       left_hand_positions / left_hand_landmarks / right_hand_positions / right_hand_landmarks
       left_trigger / right_trigger / left_grip / right_grip / left_middle / right_middle
    3. Gesture metadata is also accepted:
       left_gesture / right_gesture
       left_pinch / right_pinch
       left_pinch_value / right_pinch_value
       left_grab_strength / right_grab_strength
    """

    def __init__(self, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.setblocking(False)
        self.last_payload: dict[str, Any] = {}
        self.message_count = 0

    def stop(self):
        self.sock.close()

    def _normalize_payload(self, msg: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}

        left_pose = msg.get("left")
        right_pose = msg.get("right")
        left_pose = coerce_pose_matrix(left_pose) if left_pose is not None else None
        right_pose = coerce_pose_matrix(right_pose) if right_pose is not None else None
        if left_pose is not None:
            payload["left"] = left_pose
        if right_pose is not None:
            payload["right"] = right_pose

        for source_key in ("left_hand_pos", "left_hand_positions", "left_hand_landmarks"):
            if source_key in msg:
                points = coerce_hand_points(msg[source_key])
                if points is not None:
                    payload["left_hand_pos"] = points
                    break
        for source_key in ("right_hand_pos", "right_hand_positions", "right_hand_landmarks"):
            if source_key in msg:
                points = coerce_hand_points(msg[source_key])
                if points is not None:
                    payload["right_hand_pos"] = points
                    break

        for source_key, target_key in (
            ("left_trigger", "left_trigger"),
            ("left_grab", "left_trigger"),
            ("left_grip", "left_trigger"),
            ("left_grab_strength", "left_trigger"),
            ("left_pinch_value", "left_trigger"),
            ("left_middle", "left_trigger"),
            ("left_middle_button", "left_trigger"),
            ("left_pinch_strength", "left_trigger"),
            ("right_trigger", "right_trigger"),
            ("right_grab", "right_trigger"),
            ("right_grip", "right_trigger"),
            ("right_grab_strength", "right_trigger"),
            ("right_pinch_value", "right_trigger"),
            ("right_middle", "right_trigger"),
            ("right_middle_button", "right_trigger"),
            ("right_pinch_strength", "right_trigger"),
        ):
            if source_key in msg:
                value = clamp01(msg[source_key])
                if value is not None:
                    payload[target_key] = value

        for source_key, target_key in (
            ("left_pinch", "left_pinch"),
            ("right_pinch", "right_pinch"),
            ("left_grab_button", "left_grab_button"),
            ("right_grab_button", "right_grab_button"),
            ("left_middle_button", "left_middle_button"),
            ("right_middle_button", "right_middle_button"),
        ):
            if source_key in msg:
                value = coerce_bool(msg[source_key])
                if value is not None:
                    payload[target_key] = value

        for source_key, target_key in (
            ("left_gesture", "left_gesture"),
            ("left_gesture_name", "left_gesture"),
            ("right_gesture", "right_gesture"),
            ("right_gesture_name", "right_gesture"),
        ):
            if source_key in msg:
                value = coerce_gesture_name(msg[source_key])
                if value is not None:
                    payload[target_key] = value
                    if target_key.endswith("gesture"):
                        side = target_key.split("_")[0]
                        if value.lower() in {"pinch", "grab", "fist", "close"}:
                            payload.setdefault(f"{side}_trigger", 1.0)
                        elif value.lower() in {"open", "release"}:
                            payload.setdefault(f"{side}_trigger", 0.0)

        return payload

    def poll(self) -> dict[str, Any]:
        updated = False
        try:
            while True:
                packet, _ = self.sock.recvfrom(65535)
                msg = json.loads(packet.decode("utf-8"))
                if not isinstance(msg, dict):
                    continue
                normalized = self._normalize_payload(msg)
                if normalized:
                    self.last_payload = normalized
                    self.message_count += 1
                    updated = True
        except BlockingIOError:
            pass

        if updated:
            return dict(self.last_payload)
        return {}


def merge_payloads(controller_payload: dict[str, Any], hand_payload: dict[str, Any]) -> dict[str, Any]:
    merged = dict(controller_payload)
    merged.update(hand_payload)
    return merged


def format_send_status(
    args,
    send_count: int,
    last_send_time: float | None,
    controller_payload: dict[str, Any],
    hand_payload: dict[str, Any],
) -> str:
    if last_send_time is None:
        age_text = "never"
    else:
        age_text = f"{time.time() - last_send_time:.2f}s"

    controller_has_pose = "left" in controller_payload or "right" in controller_payload
    hand_has_pose = "left" in hand_payload or "right" in hand_payload
    hand_has_points = "left_hand_pos" in hand_payload or "right_hand_pos" in hand_payload

    return (
        f"[STATUS] teleop->udp://{args.robot_ip}:{args.teleop_port} "
        f"sends={send_count} last_send_age={age_text} "
        f"controller_pose={'Y' if controller_has_pose else 'N'} "
        f"hand_pose={'Y' if hand_has_pose else 'N'} "
        f"hand_points={'Y' if hand_has_points else 'N'} "
        f"left={'Y' if 'left' in controller_payload or 'left' in hand_payload else 'N'} "
        f"right={'Y' if 'right' in controller_payload or 'right' in hand_payload else 'N'}"
    )


def main():
    args = parse_args()
    teleop_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    keyboard_bridge = None
    controller_source = None
    hand_source = None

    if not args.disable_keyboard:
        keyboard_bridge = PlannerKeyboardBridge(args)
        keyboard_bridge.start()

    try:
        if args.mode in ("controller", "hybrid"):
            controller_source = OpenVRControllerSource()
            controller_source.start()

        if args.mode in ("hand", "hybrid"):
            hand_source = UDPHandRelaySource(args.hand_source_port)
            print(
                f"✅ Hand relay listening on udp://0.0.0.0:{args.hand_source_port} "
                f"and forwarding to udp://{args.robot_ip}:{args.teleop_port}"
            )

        controller_period = 1.0 / max(args.controller_rate, 1.0)
        status_interval = max(args.status_interval, 0.2)
        send_count = 0
        last_send_time = None
        last_status_time = 0.0
        print(
            f"📡 Teleop UDP bridge started: mode={args.mode} "
            f"teleop=udp://{args.robot_ip}:{args.teleop_port}"
        )

        while True:
            controller_payload = controller_source.poll() if controller_source is not None else {}
            hand_payload = hand_source.poll() if hand_source is not None else {}
            payload = merge_payloads(controller_payload, hand_payload)

            if payload:
                teleop_sock.sendto(json.dumps(payload).encode("utf-8"), (args.robot_ip, args.teleop_port))
                send_count += 1
                last_send_time = time.time()
                if args.verbose:
                    left_trig = payload.get("left_trigger", 0.0)
                    right_trig = payload.get("right_trigger", 0.0)
                    has_left_hand = "left_hand_pos" in payload
                    has_right_hand = "right_hand_pos" in payload
                    left_gesture = payload.get("left_gesture", "-")
                    right_gesture = payload.get("right_gesture", "-")
                    print(
                        "\r"
                        f"📡 send left_trig={left_trig:.2f} right_trig={right_trig:.2f} "
                        f"left_hand={'Y' if has_left_hand else 'N'} "
                        f"right_hand={'Y' if has_right_hand else 'N'} "
                        f"left_gesture={left_gesture} right_gesture={right_gesture}",
                        end="",
                    )

            now = time.time()
            if now - last_status_time >= status_interval:
                print(
                    format_send_status(
                        args=args,
                        send_count=send_count,
                        last_send_time=last_send_time,
                        controller_payload=controller_payload,
                        hand_payload=hand_payload,
                    )
                )
                last_status_time = now

            time.sleep(controller_period)

    except KeyboardInterrupt:
        print("\n🚪 Bridge stopped")
    finally:
        if controller_source is not None:
            controller_source.stop()
        if hand_source is not None:
            hand_source.stop()
        if keyboard_bridge is not None:
            keyboard_bridge.stop()
        teleop_sock.close()


if __name__ == "__main__":
    main()

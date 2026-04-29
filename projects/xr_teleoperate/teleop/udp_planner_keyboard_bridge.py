import argparse
import json
import socket
import threading
import time


SPECIAL_KEY_MAP = {
    "space": "r",
    "enter": None,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Global keyboard capture for Sonic planner UDP control (intended for Windows)"
    )
    parser.add_argument("--host", required=True, help="Robot IP running teleop_hand_and_arm_bridge.py")
    parser.add_argument("--port", type=int, default=5006, help="Planner UDP port")
    parser.add_argument("--rate", type=float, default=20.0, help="Key-state send rate in Hz")
    parser.add_argument("--start-key", default="f8", help="Global key to send planner start")
    parser.add_argument("--record-key", default="f9", help="Global key to toggle recording")
    parser.add_argument("--stop-key", default="esc", help="Global key to stop bridge")
    return parser.parse_args()


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


def main():
    args = parse_args()
    from pynput import keyboard

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pressed_keys = set()
    one_shot_commands = []
    lock = threading.Lock()

    planner_keys = {
        "w", "a", "s", "d", "q", "e", ",", ".", "n", "p",
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
        "-", "=", "+", "r", "`", "~", "o",
    }

    def queue_command(command_name: str):
        with lock:
            one_shot_commands.append(command_name)

    def on_press(key):
        name = normalize_key(key)
        if name is None:
            return
        if name == args.start_key.lower():
            queue_command("start")
            return
        if name == args.record_key.lower():
            queue_command("record_toggle")
            return
        if name == args.stop_key.lower():
            queue_command("stop")
            return
        if name in planner_keys:
            with lock:
                pressed_keys.add(name)

    def on_release(key):
        name = normalize_key(key)
        if name is None:
            return
        with lock:
            pressed_keys.discard(name)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    period = 1.0 / max(args.rate, 1.0)
    print(f"[KBUDP] sending to udp://{args.host}:{args.port} at {args.rate:.1f} Hz")
    print(
        f"[KBUDP] global keys: start={args.start_key} record={args.record_key} stop={args.stop_key}; "
        "planner=WASDQE , . N/P 1-8 9/0 -/="
    )

    try:
        while True:
            with lock:
                keys = sorted(pressed_keys)
                commands = list(one_shot_commands)
                one_shot_commands.clear()

            if keys:
                payload = {"keys": keys}
                sock.sendto(json.dumps(payload).encode("utf-8"), (args.host, args.port))

            for command_name in commands:
                payload = {"command": command_name}
                sock.sendto(json.dumps(payload).encode("utf-8"), (args.host, args.port))

            time.sleep(period)
    except KeyboardInterrupt:
        print("[KBUDP] stopped")
    finally:
        listener.stop()
        sock.close()


if __name__ == "__main__":
    main()

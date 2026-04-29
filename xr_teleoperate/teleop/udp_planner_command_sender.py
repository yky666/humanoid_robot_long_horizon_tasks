import argparse
import json
import socket


def parse_args():
    parser = argparse.ArgumentParser(description="Send UDP planner commands to teleop_hand_and_arm_bridge.py")
    parser.add_argument("--host", default="127.0.0.1", help="Robot IP running the bridge")
    parser.add_argument("--port", type=int, default=5006, help="Planner UDP port")
    parser.add_argument("--key", default=None, help="Single key event, e.g. w or q")
    parser.add_argument("--text", default=None, help="Key text sequence, e.g. wasd")
    parser.add_argument(
        "--command",
        choices=["start", "stop", "record_toggle"],
        default=None,
        help="Bridge control command",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    payload = {}
    if args.key:
        payload["key"] = args.key
    if args.text:
        payload["planner_text"] = args.text
    if args.command:
        payload["command"] = args.command

    if not payload:
        raise SystemExit("Provide at least one of --key, --text, or --command")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(json.dumps(payload).encode("utf-8"), (args.host, args.port))
    print(f"sent {payload} to udp://{args.host}:{args.port}")


if __name__ == "__main__":
    main()

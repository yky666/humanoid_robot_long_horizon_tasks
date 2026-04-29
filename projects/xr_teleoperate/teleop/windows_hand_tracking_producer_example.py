import argparse
import json
import math
import socket
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example Windows-side hand-tracking producer for udp://127.0.0.1:5100"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Local hand bridge host")
    parser.add_argument("--port", type=int, default=5100, help="Local hand bridge port")
    parser.add_argument("--rate", type=float, default=30.0, help="Send rate in Hz")
    return parser.parse_args()


def make_pose(x_sign: float):
    return [
        [1.0, 0.0, 0.0, 0.18 * x_sign],
        [0.0, 1.0, 0.0, 0.06],
        [0.0, 0.0, 1.0, -0.30],
    ]


def make_landmarks(x_sign: float, close_value: float):
    points = []
    for idx in range(25):
        finger_band = idx // 5
        local = idx % 5
        x = x_sign * (0.05 + 0.01 * finger_band)
        y = 0.02 * local
        z = -0.25 + 0.015 * finger_band
        if idx in (4, 8):
            y -= 0.03 * close_value
        points.append([x, y, z])
    return points


def build_packet(t: float):
    left_close = 0.5 + 0.5 * math.sin(t)
    right_close = 0.5 + 0.5 * math.sin(t + math.pi)
    left_gesture = "pinch" if left_close > 0.65 else "open"
    right_gesture = "pinch" if right_close > 0.65 else "open"

    return {
        "timestamp_ms": int(time.time() * 1000.0),
        "source": "example_hand_tracking_producer",
        "left": make_pose(+1.0),
        "right": make_pose(-1.0),
        "left_hand_pos": make_landmarks(+1.0, left_close),
        "right_hand_pos": make_landmarks(-1.0, right_close),
        "left_pinch": left_close > 0.65,
        "right_pinch": right_close > 0.65,
        "left_pinch_value": round(left_close, 4),
        "right_pinch_value": round(right_close, 4),
        "left_gesture": left_gesture,
        "right_gesture": right_gesture,
    }


def main():
    args = parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1.0 / max(args.rate, 1.0)
    start = time.time()
    print(f"[HAND-PRODUCER] sending example packets to udp://{args.host}:{args.port} at {args.rate:.1f} Hz")
    try:
        while True:
            packet = build_packet(time.time() - start)
            sock.sendto(json.dumps(packet).encode("utf-8"), (args.host, args.port))
            print(
                "\r"
                f"[HAND-PRODUCER] left={packet['left_gesture']}({packet['left_pinch_value']:.2f}) "
                f"right={packet['right_gesture']}({packet['right_pinch_value']:.2f})",
                end="",
            )
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n[HAND-PRODUCER] stopped")
    finally:
        sock.close()


if __name__ == "__main__":
    main()

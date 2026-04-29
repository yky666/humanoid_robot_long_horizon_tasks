#!/usr/bin/env python3
"""
Bridge teleimager PUB (head_camera) -> Psi0 REP schema.
- Subscribes to teleimager PUB on tcp://127.0.0.1:55555 (JPEG bytes).
- Serves REP on tcp://0.0.0.0:5556, responding with multipart [rgb_jpeg, ir, depth].
- IR/Depth are left empty to keep wire format compatible; Psi0 client should handle empty payloads.
"""
import zmq
import threading
import time

TELE_HOST = "127.0.0.1"
TELE_PORT = 55555
BRIDGE_BIND = "tcp://0.0.0.0:5556"

ctx = zmq.Context.instance()
sub = ctx.socket(zmq.SUB)
sub.setsockopt(zmq.RCVHWM, 1)
sub.setsockopt_string(zmq.SUBSCRIBE, "")
sub.connect(f"tcp://{TELE_HOST}:{TELE_PORT}")

latest_rgb = None
lock = threading.Lock()

def pull_loop():
    global latest_rgb
    while True:
        try:
            msg = sub.recv()
            with lock:
                latest_rgb = msg
        except Exception as e:
            print(f"[bridge] sub error: {e}")
            time.sleep(0.5)

threading.Thread(target=pull_loop, daemon=True).start()

rep = ctx.socket(zmq.REP)
rep.bind(BRIDGE_BIND)
print(f"[bridge] started. teleimager tcp://{TELE_HOST}:{TELE_PORT} -> REP {BRIDGE_BIND}")

while True:
    try:
        _ = rep.recv()
        with lock:
            rgb = latest_rgb
        if rgb is None:
            rep.send_multipart([b"", b"", b""])
        else:
            rep.send_multipart([rgb, b"", b""])
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"[bridge] rep error: {e}")
        time.sleep(0.2)

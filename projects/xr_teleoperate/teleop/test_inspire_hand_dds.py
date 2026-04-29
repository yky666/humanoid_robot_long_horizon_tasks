import argparse
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal DDS test for Inspire hand command/state topics."
    )
    parser.add_argument(
        "--network-interface",
        type=str,
        default=None,
        help="DDS network interface passed to ChannelFactoryInitialize",
    )
    parser.add_argument(
        "--side",
        choices=("left", "right", "both"),
        default="both",
        help="Which hand command topic to publish",
    )
    parser.add_argument(
        "--value",
        type=float,
        default=1.0,
        help="Open-space hand value in [0,1]. 1=open, 0=closed",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="How long to continuously publish the command",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=30.0,
        help="Command publish rate",
    )
    parser.add_argument(
        "--pulse",
        action="store_true",
        help="Send open -> target -> open sequence for easier visual verification",
    )
    return parser.parse_args()


def _scaled(values):
    return [int(np.clip(float(v) * 1000.0, 0.0, 1000.0)) for v in values]


def _print_state(prefix, msg):
    if msg is None:
        print(f"[{prefix}] state: <none>")
        return
    angle = list(getattr(msg, "angle_act", [])[:6])
    force = list(getattr(msg, "force_act", [])[:6])
    print(f"[{prefix}] angle_act={angle} force_act={force}")


def main():
    args = parse_args()

    from inspire_sdkpy import inspire_dds
    import inspire_sdkpy.inspire_hand_defaut as inspire_hand_default

    ChannelFactoryInitialize(0, networkInterface=args.network_interface)

    left_pub = ChannelPublisher("rt/inspire_hand/ctrl/l", inspire_dds.inspire_hand_ctrl)
    left_pub.Init()
    right_pub = ChannelPublisher("rt/inspire_hand/ctrl/r", inspire_dds.inspire_hand_ctrl)
    right_pub.Init()

    left_state_sub = ChannelSubscriber("rt/inspire_hand/state/l", inspire_dds.inspire_hand_state)
    left_state_sub.Init()
    right_state_sub = ChannelSubscriber("rt/inspire_hand/state/r", inspire_dds.inspire_hand_state)
    right_state_sub.Init()

    value = float(np.clip(args.value, 0.0, 1.0))
    target = _scaled([value] * 6)
    opened = _scaled([1.0] * 6)

    def publish(side: str, angle_set: list[int]):
        msg = inspire_hand_default.get_inspire_hand_ctrl()
        msg.angle_set = angle_set
        msg.mode = 0b0001
        if side in ("left", "both"):
            left_pub.Write(msg)
        if side in ("right", "both"):
            right_pub.Write(msg)

    def run_stage(name: str, angle_set: list[int], duration: float):
        print(f"[DDS-Test] stage={name} side={args.side} angle_set={angle_set} duration={duration:.2f}s")
        period = 1.0 / max(args.hz, 1.0)
        end_time = time.time() + max(duration, 0.0)
        while time.time() < end_time:
            publish(args.side, angle_set)
            time.sleep(period)

        time.sleep(0.2)
        _print_state("left", left_state_sub.Read())
        _print_state("right", right_state_sub.Read())

    if args.pulse:
        run_stage("open", opened, 1.0)
        run_stage("target", target, args.duration)
        run_stage("open", opened, 1.0)
    else:
        run_stage("target", target, args.duration)


if __name__ == "__main__":
    main()

import argparse
import logging
import os
import time

from teleop.utils.rerun_visualizer import RerunEpisodeReader, RerunLogger

try:
    import logging_mp  # type: ignore
except ImportError:
    logging_mp = None


def _build_logger():
    if logging_mp is not None:
        return logging_mp.getLogger(__name__)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return logging.getLogger(__name__)


logger_mp = _build_logger()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a recorded teleoperation episode from utils/data using Rerun."
    )
    parser.add_argument(
        "--task-dir",
        type=str,
        default="./utils/data/pick cube",
        help="Task directory containing episode_xxxx folders",
    )
    parser.add_argument(
        "--episode",
        type=int,
        required=True,
        help="Episode index, e.g. 9 for episode_0009",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=30.0,
        help="Replay rate in Hz",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Visible idx sliding window size in Rerun",
    )
    parser.add_argument(
        "--memory-limit",
        type=str,
        default="300MB",
        help="Rerun viewer memory limit",
    )
    parser.add_argument(
        "--no-sleep",
        action="store_true",
        help="Replay as fast as possible instead of real-time pacing",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print episode statistics only, without launching Rerun",
    )
    parser.add_argument(
        "--save-rrd",
        type=str,
        default=None,
        help="Optional output .rrd path. In headless shells this is used automatically if omitted.",
    )
    parser.add_argument(
        "--serve-web",
        action="store_true",
        help="Serve a web viewer for headless environments instead of exporting only an .rrd file.",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=9876,
        help="Rerun gRPC port used with --serve-web.",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="Rerun web viewer port used with --serve-web.",
    )
    return parser.parse_args()


def summarize_episode(episode_data: list[dict]) -> None:
    num_frames = len(episode_data)
    num_colors = sum(1 for item in episode_data if item.get("colors"))
    num_tactiles = sum(1 for item in episode_data if item.get("tactiles"))

    left_nonzero = 0
    right_nonzero = 0
    for item in episode_data:
        left_vals = item.get("states", {}).get("left_ee", {}).get("qpos", []) or []
        right_vals = item.get("states", {}).get("right_ee", {}).get("qpos", []) or []
        if any(abs(v) > 1e-9 for v in left_vals):
            left_nonzero += 1
        if any(abs(v) > 1e-9 for v in right_vals):
            right_nonzero += 1

    summary_lines = [
        f"[Replay] frames={num_frames}",
        f"[Replay] frames_with_colors={num_colors}",
        f"[Replay] frames_with_tactiles={num_tactiles}",
        f"[Replay] left_ee_nonzero_frames={left_nonzero}",
        f"[Replay] right_ee_nonzero_frames={right_nonzero}",
    ]
    for line in summary_lines:
        print(line)
        logger_mp.info(line)


def _has_graphical_display() -> bool:
    return any(os.environ.get(var) for var in ("DISPLAY", "WAYLAND_DISPLAY", "WAYLAND_SOCKET"))


def main():
    args = parse_args()
    task_dir = os.path.abspath(args.task_dir)
    episode_dir = os.path.join(task_dir, f"episode_{args.episode:04d}")
    json_path = os.path.join(episode_dir, "data.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Episode file not found: {json_path}")

    reader = RerunEpisodeReader(task_dir=task_dir)
    episode_data = reader.return_episode_data(args.episode)

    logger_mp.info(f"[Replay] loaded {json_path}")
    summarize_episode(episode_data)

    if args.summary_only:
        return

    has_display = _has_graphical_display()
    save_rrd = args.save_rrd
    if not has_display and not save_rrd:
        save_rrd = os.path.join(episode_dir, f"episode_{args.episode:04d}_replay.rrd")
        print(f"[Replay] no DISPLAY/WAYLAND found, falling back to file export: {save_rrd}")
        logger_mp.info(f"[Replay] no DISPLAY/WAYLAND found, falling back to file export: {save_rrd}")

    if has_display:
        logger_mp.info(f"[Replay] launching Rerun for episode_{args.episode:04d}")
    elif args.serve_web:
        logger_mp.info(
            f"[Replay] serving headless web viewer on http://<this-host>:{args.web_port} "
            f"with gRPC backend port {args.grpc_port}"
        )
    elif save_rrd:
        logger_mp.info(f"[Replay] exporting replay stream to {save_rrd}")

    replay_logger = RerunLogger(
        prefix="replay/",
        IdxRangeBoundary=args.window,
        memory_limit=args.memory_limit,
        spawn_viewer=has_display,
        save_path=save_rrd,
    )

    if (not has_display) and args.serve_web:
        from teleop.utils.rerun_visualizer import rr

        server_uri = rr.serve_grpc(grpc_port=args.grpc_port, server_memory_limit=args.memory_limit)
        rr.serve_web_viewer(web_port=args.web_port, open_browser=False, connect_to=server_uri)
        print(f"[Replay] web viewer: http://127.0.0.1:{args.web_port}")
        print(f"[Replay] grpc uri: {server_uri}")

    sleep_time = 0.0 if args.no_sleep else (1.0 / max(args.rate, 1.0))
    for item_data in episode_data:
        replay_logger.log_item_data(item_data)
        if sleep_time > 0.0:
            time.sleep(sleep_time)

    logger_mp.info("[Replay] playback completed")
    if has_display:
        input("Press Enter to exit replay...")
    elif args.serve_web:
        print("[Replay] web server is running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("[Replay] stopped web viewer server")
    elif save_rrd:
        print(f"[Replay] saved replay recording to: {save_rrd}")


if __name__ == "__main__":
    main()

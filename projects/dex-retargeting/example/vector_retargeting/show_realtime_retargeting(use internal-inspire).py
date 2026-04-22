import multiprocessing
import socket
import time
from pathlib import Path
from queue import Empty, Full
from typing import Optional, Sequence
from urllib.parse import urlparse

import threading
from flask import Flask, Response

# 建立一个全局变量存放最新渲染的画面
latest_frame = None
_new_frame_event = threading.Event()
app = Flask(__name__)
STREAM_HOST = "0.0.0.0"
STREAM_PORT = 5001
INPUT_RETRY_DELAY_SEC = 1.0
NO_HAND_LOG_INTERVAL_SEC = 1.0
VIDEO_CAPTURE_OPEN_TIMEOUT_MSEC = 5000
VIDEO_CAPTURE_READ_TIMEOUT_MSEC = 5000


def is_network_capture_source(source) -> bool:
    if not isinstance(source, str):
        return False
    return urlparse(source).scheme.lower() in {
        "http",
        "https",
        "rtmp",
        "rtmps",
        "rtsp",
        "rtsps",
        "srt",
        "tcp",
        "udp",
    }

def generate():
    global latest_frame
    last_sent = None
    while True:
        _new_frame_event.wait(timeout=0.5)
        _new_frame_event.clear()
        frame = latest_frame
        if frame is not None and frame is not last_sent:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            last_sent = frame

@app.route('/video')
def video_feed():
    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
            'Pragma': 'no-cache',
            'Expires': '0',
            'X-Accel-Buffering': 'no',
        },
    )


def run_stream_server():
    try:
        app.run(host=STREAM_HOST, port=STREAM_PORT, debug=False, use_reloader=False, threaded=True)
    except Exception:
        logger.exception("Processed video stream server exited unexpectedly.")


def wait_for_stream_server(port: int, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.05)
    return False


def open_video_capture(camera_path: Optional[str]):
    source = 0 if camera_path is None else camera_path
    open_params = []
    api_preferences = [cv2.CAP_ANY]
    if is_network_capture_source(source):
        open_params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            VIDEO_CAPTURE_OPEN_TIMEOUT_MSEC,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            VIDEO_CAPTURE_READ_TIMEOUT_MSEC,
        ]
        api_preferences = [cv2.CAP_FFMPEG, cv2.CAP_ANY]

    for api_preference in api_preferences:
        cap = cv2.VideoCapture()
        try:
            opened = (
                cap.open(source, api_preference, open_params)
                if open_params
                else cap.open(source, api_preference)
            )
        except (TypeError, cv2.error):
            if not open_params:
                cap.release()
                continue
            try:
                opened = cap.open(source, api_preference)
            except cv2.error:
                cap.release()
                continue

        if opened:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        cap.release()

    return cv2.VideoCapture()

import cv2
import numpy as np
import sapien
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector

INSPIRE_SDK_COMMAND_JOINT_NAMES = {
    RobotName.inspire: {
        HandType.right: (
            "pinky_proximal_joint",
            "ring_proximal_joint",
            "middle_proximal_joint",
            "index_proximal_joint",
            "thumb_proximal_pitch_joint",
            "thumb_proximal_yaw_joint",
        ),
        HandType.left: (
            "pinky_proximal_joint",
            "ring_proximal_joint",
            "middle_proximal_joint",
            "index_proximal_joint",
            "thumb_proximal_pitch_joint",
            "thumb_proximal_yaw_joint",
        ),
    },
}


class InspireHandDDSPublisher:
    def __init__(
        self,
        robot_name: RobotName,
        hand_type: HandType,
        min_publish_period: float,
        network: Optional[str],
        min_command_value: int,
        max_command_value: int,
        topic_side: Optional[str] = None,
    ):
        try:
            from inspire_sdkpy import inspire_hand_defaut, inspire_dds
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelPublisher,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Inspire SDK publishing requested, but inspire_sdkpy or unitree_sdk2py is unavailable."
            ) from exc

        if min_publish_period < 0:
            raise ValueError("inspire_min_publish_period must be non-negative.")
        if min_command_value >= max_command_value:
            raise ValueError("inspire_min_command must be smaller than inspire_max_command.")

        if network is None:
            ChannelFactoryInitialize(0)
        else:
            ChannelFactoryInitialize(0, network)

        try:
            command_joint_names = INSPIRE_SDK_COMMAND_JOINT_NAMES[robot_name][hand_type]
        except KeyError as exc:
            raise ValueError(
                "Inspire SDK publishing is currently only supported for inspire hands."
            ) from exc

        topic_suffix = topic_side if topic_side else ("r" if hand_type is HandType.right else "l")
        self.publisher = ChannelPublisher(
            f"rt/inspire_hand/ctrl/{topic_suffix}", inspire_dds.inspire_hand_ctrl
        )
        self.publisher.Init()

        self.get_command = inspire_hand_defaut.get_inspire_hand_ctrl
        self.command_joint_names = list(command_joint_names)
        self.min_publish_period = float(min_publish_period)
        self.min_command_value = int(min_command_value)
        self.max_command_value = int(max_command_value)
        self.source_indices: Optional[np.ndarray] = None
        self.lower_limits: Optional[np.ndarray] = None
        self.upper_limits: Optional[np.ndarray] = None
        self.last_publish_time = 0.0
        self.warned_no_subscriber = False

    def configure_source_joints(
        self,
        source_joint_names: Sequence[str],
        joint_limits_by_name: dict[str, np.ndarray],
    ):
        missing = [
            joint_name
            for joint_name in self.command_joint_names
            if joint_name not in source_joint_names
        ]
        if missing:
            raise ValueError(
                f"Missing joints in retargeting output required for Inspire SDK publishing: {missing}"
            )

        self.source_indices = np.array(
            [source_joint_names.index(joint_name) for joint_name in self.command_joint_names],
            dtype=int,
        )
        ordered_limits = np.asarray(
            [joint_limits_by_name[joint_name] for joint_name in self.command_joint_names],
            dtype=np.float32,
        )
        self.lower_limits = ordered_limits[:, 0]
        self.upper_limits = ordered_limits[:, 1]

    def publish(self, qpos: np.ndarray):
        if self.source_indices is None:
            raise RuntimeError("Inspire SDK source joints are not configured.")

        now = time.monotonic()
        if now - self.last_publish_time < self.min_publish_period:
            return

        positions = np.asarray(qpos[self.source_indices], dtype=np.float64)
        if self.lower_limits is not None and self.upper_limits is not None:
            positions = np.clip(positions, self.lower_limits, self.upper_limits)
            limits_span = self.upper_limits - self.lower_limits
            normalized = np.divide(
                positions - self.lower_limits,
                limits_span,
                out=np.zeros_like(positions),
                where=limits_span > 1e-6,
            )
            normalized = 1.0 - normalized  # invert: joint-max (closed fist) → 0, joint-min (open) → 1000
        else:
            normalized = np.zeros_like(positions)

        command_span = self.max_command_value - self.min_command_value
        scaled = np.rint(self.min_command_value + normalized * command_span).astype(int)
        scaled = np.clip(scaled, self.min_command_value, self.max_command_value)

        command = self.get_command()
        command.angle_set = scaled.tolist()
        command.mode = 0b0001

        published = self.publisher.Write(command)
        if not published and not self.warned_no_subscriber:
            logger.warning(
                "Inspire DDS command was dropped because no SDK driver subscriber is ready yet."
            )
            self.warned_no_subscriber = True
        elif published:
            self.warned_no_subscriber = False

        self.last_publish_time = now


def build_inspire_sdk_publisher(
    robot_name: RobotName,
    hand_type: HandType,
    min_publish_period: float,
    network: Optional[str],
    min_command_value: int,
    max_command_value: int,
    topic_side: Optional[str] = None,
) -> InspireHandDDSPublisher:
    return InspireHandDDSPublisher(
        robot_name=robot_name,
        hand_type=hand_type,
        min_publish_period=min_publish_period,
        network=network,
        min_command_value=min_command_value,
        max_command_value=max_command_value,
        topic_side=topic_side,
    )


def start_retargeting(
    queue: multiprocessing.Queue,
    robot_dir: str,
    config_path: str,
    robot_name: RobotName,
    hand_type: HandType,
    selfie: bool,
    publish_to_inspire_sdk: bool,
    inspire_network: Optional[str],
    inspire_min_publish_period: float,
    inspire_min_command: int,
    inspire_max_command: int,
    inspire_topic_side: Optional[str] = None,
):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    config = RetargetingConfig.load_from_file(config_path)
    # Disable the low-pass filter to eliminate smoothing latency
    config.low_pass_alpha = 1.0
    retargeting = config.build()

    detector_hand_type = "Right" if hand_type is HandType.right else "Left"

    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    # Setup
    logger.info("Creating SAPIEN scene")
    scene = sapien.Scene()
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )
    scene.add_area_light_for_ray_tracing(
        sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
    )
    logger.info("SAPIEN scene configured")

    # Camera
    logger.info("Creating render camera")
    cam = scene.add_camera(
        name="Cheese!", width=320, height=320, fovy=1, near=0.1, far=10
    )
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))
    logger.info("Render camera ready")

    # Initializing the hand detector before the SAPIEN scene can trigger a segfault
    # when MediaPipe/TFLite and EGL/OpenGL are both active in the same process.
    logger.info("Initializing hand detector")
    detector = SingleHandDetector(hand_type=detector_hand_type, selfie=selfie)
    logger.info("Hand detector initialized")

    # 在后台启动处理后视频流服务，并确认端口已真正开始监听。
    logger.info("Starting processed video stream server")
    threading.Thread(target=run_stream_server, daemon=True).start()
    if wait_for_stream_server(STREAM_PORT):
        logger.info(f"Linux 3D 渲染推流已启动！请在浏览器访问 {STREAM_PORT} 端口。")
    else:
        logger.error(f"Linux 3D 渲染推流启动失败：{STREAM_PORT} 端口未进入监听状态。")

    # 🚫 把下面这几行全部注释掉，防止 SSH 报错！
    # viewer = Viewer()
    # viewer.set_scene(scene)
    # viewer.control_window.show_origin_frame = False
    # viewer.control_window.move_speed = 0.01
    # viewer.control_window.toggle_camera_lines(False)
    # viewer.set_camera_pose(cam.get_local_pose())

    # Load robot and set it to a good pose to take picture
    logger.info("Loading robot model")
    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_model_name = filepath.stem
    loader.load_multiple_collisions_from_file = True
    if "ability" in robot_model_name:
        loader.scale = 1.5
    elif "dclaw" in robot_model_name:
        loader.scale = 1.25
    elif "allegro" in robot_model_name:
        loader.scale = 1.4
    elif "shadow" in robot_model_name:
        loader.scale = 0.9
    elif "bhand" in robot_model_name:
        loader.scale = 1.5
    elif "leap" in robot_model_name:
        loader.scale = 1.4
    elif "svh" in robot_model_name:
        loader.scale = 1.5

    if "glb" not in robot_model_name:
        glb_path = filepath.with_stem(filepath.stem + "_glb")
        filepath = str(glb_path if glb_path.exists() else filepath)
    else:
        filepath = str(filepath)

    robot = loader.load(filepath)

    if "ability" in robot_model_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "shadow" in robot_model_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "dclaw" in robot_model_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "allegro" in robot_model_name:
        robot.set_pose(sapien.Pose([0, 0, -0.05]))
    elif "bhand" in robot_model_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "leap" in robot_model_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "svh" in robot_model_name:
        robot.set_pose(sapien.Pose([0, 0, -0.13]))

    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)

    inspire_sdk_publisher = None
    if publish_to_inspire_sdk:
        if config.target_joint_names is None:
            raise ValueError(
                "Inspire SDK publishing requires target_joint_names in the retargeting config."
            )
        inspire_sdk_publisher = build_inspire_sdk_publisher(
            robot_name=robot_name,
            hand_type=hand_type,
            min_publish_period=inspire_min_publish_period,
            network=inspire_network,
            min_command_value=inspire_min_command,
            max_command_value=inspire_max_command,
            topic_side=inspire_topic_side,
        )
        inspire_sdk_publisher.configure_source_joints(
            retargeting_joint_names,
            {
                joint_name: joint_limit
                for joint_name, joint_limit in zip(
                    config.target_joint_names, retargeting.joint_limits, strict=True
                )
            },
        )
        _effective_side = inspire_topic_side if inspire_topic_side else ("r" if hand_type is HandType.right else "l")
        logger.info(
            "Inspire SDK DDS publishing enabled on topic {}".format(
                f"rt/inspire_hand/ctrl/{_effective_side}"
            )
        )

    # ── Async 3D render thread ──────────────────────────────────────
    # Render is the main bottleneck; moving it off the critical path keeps
    # detection + DDS publish running at full speed (~30-50 Hz).
    _render_lock = threading.Lock()
    _render_qpos = None        # latest qpos for the render thread
    _render_bgr_human = None   # latest camera frame with skeleton overlay
    _render_stop = threading.Event()

    def _render_loop():
        global latest_frame
        while not _render_stop.is_set():
            with _render_lock:
                qpos_snap = _render_qpos
                bgr_snap = _render_bgr_human
            if qpos_snap is None or bgr_snap is None:
                time.sleep(0.01)
                continue
            robot.set_qpos(qpos_snap[retargeting_to_sapien])
            scene.update_render()
            cam.take_picture()
            rgb_render = cam.get_picture("Color")[..., :3]
            render_bgr = (np.clip(rgb_render, 0, 1) * 255).astype(np.uint8)[..., ::-1]
            human_bgr_resized = cv2.resize(bgr_snap, (320, 320))
            combined_frame = cv2.hconcat([human_bgr_resized, render_bgr])
            _, buffer = cv2.imencode('.jpg', combined_frame, [cv2.IMWRITE_JPEG_QUALITY, 35])
            latest_frame = buffer.tobytes()
            _new_frame_event.set()

    render_thread = threading.Thread(target=_render_loop, daemon=True)
    render_thread.start()

    last_no_hand_log_time = 0.0
    last_empty_queue_log_time = 0.0

    while True:
        try:
            bgr = queue.get(timeout=5)
            while True:
                try:
                    bgr = queue.get_nowait()
                except Empty:
                    break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Empty:
            now = time.monotonic()
            if now - last_empty_queue_log_time >= INPUT_RETRY_DELAY_SEC:
                logger.warning(
                    "No input frame received in 5 secs. Waiting for camera stream to recover."
                )
                last_empty_queue_log_time = now
            continue

        _, joint_pos, keypoint_2d, _ = detector.detect(rgb)
        bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")

        if joint_pos is None:
            now = time.monotonic()
            if now - last_no_hand_log_time >= NO_HAND_LOG_INTERVAL_SEC:
                logger.warning(f"{detector_hand_type} hand is not detected.")
                last_no_hand_log_time = now
            with _render_lock:
                _render_bgr_human = bgr.copy()
        else:
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                indices = indices
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            qpos = retargeting.retarget(ref_value)

            # Publish to real hand ASAP (no rendering in the way)
            if inspire_sdk_publisher is not None:
                inspire_sdk_publisher.publish(qpos)

            # Hand off to render thread (non-blocking)
            with _render_lock:
                _render_qpos = qpos.copy()
                _render_bgr_human = bgr.copy()


def produce_frame(queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    cap: Optional[cv2.VideoCapture] = None
    stream_was_ready = False

    while True:
        if cap is None or not cap.isOpened():
            cap = open_video_capture(camera_path)
            if not cap.isOpened():
                if stream_was_ready:
                    logger.warning("Input camera stream is unavailable. Retrying connection.")
                else:
                    logger.warning("Unable to open input camera stream. Retrying connection.")
                stream_was_ready = False
                time.sleep(INPUT_RETRY_DELAY_SEC)
                continue

            logger.info("Input camera stream connected.")
            stream_was_ready = True

        success, image = cap.read()
        if not success:
            if stream_was_ready:
                logger.warning("Input camera stream read failed. Reconnecting.")
            stream_was_ready = False
            cap.release()
            cap = None
            time.sleep(INPUT_RETRY_DELAY_SEC)
            continue

        # Drop stale frames instead of building up latency in the IPC queue.
        try:
            queue.put_nowait(image)
        except Full:
            try:
                queue.get_nowait()
            except Empty:
                pass
            try:
                queue.put_nowait(image)
            except Full:
                pass


def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    camera_path: Optional[str] = None,
    selfie: bool = False,
    publish_to_inspire_sdk: bool = False,
    inspire_network: Optional[str] = None,
    inspire_min_publish_period: float = 0.02,
    inspire_min_command: int = 0,
    inspire_max_command: int = 1000,
    inspire_topic_side: Optional[str] = None,
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
        camera_path: the device path to feed to opencv to open the web camera. It will use 0 by default.
        selfie: whether the input stream is mirrored like a selfie preview; set this to true if left/right hand labels appear reversed.
        publish_to_inspire_sdk: whether to publish retargeted commands to the Inspire SDK DDS topic.
        inspire_network: optional DDS network interface passed to ChannelFactoryInitialize.
        inspire_min_publish_period: minimum spacing between published Inspire DDS commands.
        inspire_min_command: lower bound of the integer angle command sent to the Inspire SDK.
        inspire_max_command: upper bound of the integer angle command sent to the Inspire SDK.
        inspire_topic_side: override DDS topic suffix ('r' or 'l'). Defaults to hand_type if not set.
    """
    if publish_to_inspire_sdk and robot_name not in INSPIRE_SDK_COMMAND_JOINT_NAMES:
        raise ValueError(
            "publish_to_inspire_sdk is currently only supported with --robot-name inspire."
        )

    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )

    queue = multiprocessing.Queue(maxsize=2)
    producer_process = multiprocessing.Process(
        target=produce_frame, args=(queue, camera_path)
    )
    consumer_process = multiprocessing.Process(
        target=start_retargeting,
        args=(
            queue,
            str(robot_dir),
            str(config_path),
            robot_name,
            hand_type,
            selfie,
            publish_to_inspire_sdk,
            inspire_network,
            inspire_min_publish_period,
            inspire_min_command,
            inspire_max_command,
            inspire_topic_side,
        ),
    )

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()

    print("done")


if __name__ == "__main__":
    tyro.cli(main)

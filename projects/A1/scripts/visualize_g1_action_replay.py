#!/usr/bin/env python3

"""Render G1 GT vs predicted action chunks from an A1 evaluation JSON.

This is an offline visual replay utility. It does not run a closed-loop GR00T
policy; it visualizes already-saved ``pred_action`` and ``gt_action`` arrays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pandas as pd


ARM_JOINTS = [
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "shoulder_yaw_joint",
    "elbow_joint",
    "wrist_roll_joint",
    "wrist_pitch_joint",
    "wrist_yaw_joint",
]

# The A1 26D action files have 6 hand channels per side while this MJCF hand has
# 7 joints per side. This mapping preserves the available thumb/index/middle
# motion and leaves the extra distal joints tied to their proximal channels.
LEFT_HAND_MAP = {
    "left_hand_middle_0_joint": 14,
    "left_hand_middle_1_joint": 14,
    "left_hand_index_0_joint": 17,
    "left_hand_index_1_joint": 17,
    "left_hand_thumb_0_joint": 19,
    "left_hand_thumb_1_joint": 18,
    "left_hand_thumb_2_joint": 18,
}
RIGHT_HAND_MAP = {
    "right_hand_middle_0_joint": 20,
    "right_hand_middle_1_joint": 20,
    "right_hand_index_0_joint": 23,
    "right_hand_index_1_joint": 23,
    "right_hand_thumb_0_joint": 25,
    "right_hand_thumb_1_joint": 24,
    "right_hand_thumb_2_joint": 24,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--full-episode",
        action="store_true",
        help=(
            "Aggregate all saved chunks onto the dataset timeline and render the full episode. "
            "Requires --dataset-root."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="LeRobot dataset root used to read full-episode GT actions when --full-episode is set.",
    )
    parser.add_argument(
        "--raw-episode-dir",
        type=Path,
        default=None,
        help="Raw episode directory with data.json. Used to read full-episode 26D A1 GT actions.",
    )
    parser.add_argument(
        "--aggregation",
        choices=("average", "latest", "earliest"),
        default="average",
        help="How to combine overlapping saved prediction chunks in --full-episode mode.",
    )
    parser.add_argument(
        "--allow-missing-pred",
        action="store_true",
        help=(
            "Render full GT episode even when saved prediction chunks do not cover every "
            "frame. Missing prediction frames are rendered with the GT pose and labeled."
        ),
    )
    parser.add_argument(
        "--g1-xml",
        type=Path,
        default=Path(
            "/home/sys01/yangky/test/humanoid_robot_long_horizon_tasks/"
            "projects/xr_teleoperate/assets/g1/g1_body29_hand14.xml"
        ),
    )
    parser.add_argument("--output-mp4", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def joint_qpos_addr(model: mujoco.MjModel, joint_name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise KeyError(f"Missing joint in model: {joint_name}")
    return int(model.jnt_qposadr[jid])


def set_pose(model: mujoco.MjModel, data: mujoco.MjData, action: np.ndarray) -> None:
    data.qpos[:] = 0.0
    # Free base: keep G1 upright and slightly above ground.
    data.qpos[2] = 0.78
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

    for side, offset in [("left", 0), ("right", 7)]:
        for i, suffix in enumerate(ARM_JOINTS):
            addr = joint_qpos_addr(model, f"{side}_{suffix}")
            data.qpos[addr] = action[offset + i]

    for joint_name, action_idx in {**LEFT_HAND_MAP, **RIGHT_HAND_MAP}.items():
        addr = joint_qpos_addr(model, joint_name)
        low, high = model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)]
        value = float(action[action_idx])
        if "left_hand_middle" in joint_name or "left_hand_index" in joint_name:
            value = -abs(value) * abs(low)
        elif "right_hand_middle" in joint_name or "right_hand_index" in joint_name:
            value = abs(value) * high
        elif "thumb_2" in joint_name:
            value = np.clip(value, low, high)
        else:
            value = np.clip(value, low, high)
        data.qpos[addr] = value

    mujoco.mj_forward(model, data)


def load_full_episode_actions(dataset_root: Path) -> tuple[np.ndarray, list[int]]:
    parquet = dataset_root / "data" / "chunk-000" / "episode_000000.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"Missing episode parquet: {parquet}")
    df = pd.read_parquet(parquet)
    if "actions" in df.columns:
        action_key = "actions"
    elif "action" in df.columns:
        action_key = "action"
    else:
        raise ValueError(f"No action/actions column found in {parquet}; columns={list(df.columns)}")
    actions = np.stack(df[action_key].to_numpy()).astype(np.float32)
    frame_indices = [int(v) for v in df["frame_index"].to_numpy()]
    if actions.shape[1] != 26:
        raise ValueError(f"This replay utility expects 26D A1 G1 actions, got {actions.shape[1]}D")
    return actions, frame_indices


def load_raw_episode_actions(raw_episode_dir: Path) -> tuple[np.ndarray, list[int]]:
    data_path = raw_episode_dir / "data.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing raw episode data.json: {data_path}")
    payload = json.loads(data_path.read_text())
    frames = payload.get("data", [])
    if not frames:
        raise ValueError(f"No frames found in {data_path}")

    actions = []
    frame_indices = []
    for frame_index, frame in enumerate(frames):
        raw_actions = frame["actions"]
        action = np.concatenate(
            [
                np.asarray(raw_actions["left_arm"]["qpos"], dtype=np.float32),
                np.asarray(raw_actions["right_arm"]["qpos"], dtype=np.float32),
                np.asarray(raw_actions["left_ee"]["qpos"], dtype=np.float32),
                np.asarray(raw_actions["right_ee"]["qpos"], dtype=np.float32),
            ]
        )
        if action.shape[0] != 26:
            raise ValueError(f"Expected 26D raw A1 action at frame {frame_index}, got {action.shape[0]}D")
        actions.append(action)
        frame_indices.append(frame_index)
    return np.stack(actions).astype(np.float32), frame_indices


def aggregate_saved_chunks(
    samples: list[dict],
    episode_len: int,
    action_dim: int,
    aggregation: str,
    allow_missing_pred: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.zeros((episode_len, action_dim), dtype=np.float32)
    counts = np.zeros((episode_len,), dtype=np.int32)
    written_order = np.full((episode_len,), -1, dtype=np.int32)

    sorted_samples = sorted(samples, key=lambda s: int(s["dataset_index"]))
    for order, sample in enumerate(sorted_samples):
        start = int(sample["dataset_index"])
        chunk = np.asarray(sample["pred_action"], dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != action_dim:
            raise ValueError(f"Bad pred_action shape for dataset_index={start}: {chunk.shape}")
        for j, value in enumerate(chunk):
            t = start + j
            if t >= episode_len:
                break
            if aggregation == "average":
                pred[t] += value
                counts[t] += 1
            elif aggregation == "earliest":
                if counts[t] == 0:
                    pred[t] = value
                    counts[t] = 1
                    written_order[t] = order
            elif aggregation == "latest":
                pred[t] = value
                counts[t] = 1
                written_order[t] = order

    if aggregation == "average":
        covered = counts > 0
        pred[covered] /= counts[covered, None]
    if not np.all(counts > 0):
        missing = np.where(counts == 0)[0].tolist()
        if allow_missing_pred:
            return pred, counts
        raise ValueError(
            f"Saved chunks do not cover the full episode. Missing {len(missing)} frames, "
            f"first missing frames: {missing[:20]}. Re-run inference on every frame."
        )
    return pred, counts


def render(model: mujoco.MjModel, renderer: mujoco.Renderer, data: mujoco.MjData) -> np.ndarray:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.array([0.0, 0.0, 0.65])
    cam.distance = 2.0
    cam.azimuth = 145
    cam.elevation = -15
    renderer.update_scene(data, camera=cam)
    return renderer.render()


def label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input_json.read_text())
    samples = payload["samples"]
    if args.full_episode:
        dataset_root = args.dataset_root or Path(payload.get("dataset_path", ""))
        if args.raw_episode_dir is not None:
            gt, frame_indices = load_raw_episode_actions(args.raw_episode_dir)
        elif dataset_root:
            gt, frame_indices = load_full_episode_actions(dataset_root)
        else:
            raise ValueError("--dataset-root is required when input JSON lacks dataset_path")
        pred, coverage = aggregate_saved_chunks(
            samples=samples,
            episode_len=len(gt),
            action_dim=gt.shape[1],
            aggregation=args.aggregation,
            allow_missing_pred=args.allow_missing_pred,
        )
        covered = coverage > 0
        pred_for_metrics = pred[covered]
        gt_for_metrics = gt[covered]
        pred = pred.copy()
        pred[~covered] = gt[~covered]
        sample_label = (
            f"full_episode frames={len(gt)} chunks={len(samples)} "
            f"covered={int(covered.sum())}/{len(gt)} "
            f"coverage={int(coverage.min())}-{int(coverage.max())}"
        )
        mean_l1 = float(np.mean(np.abs(pred_for_metrics - gt_for_metrics)))
        mean_mse = float(np.mean(np.square(pred_for_metrics - gt_for_metrics)))
    else:
        sample = samples[args.sample_index]
        pred = np.asarray(sample["pred_action"], dtype=np.float32)
        gt = np.asarray(sample["gt_action"], dtype=np.float32)
        sample_label = f"sample={args.sample_index}"
        mean_l1 = float(sample["l1"])
        mean_mse = float(sample["mse"])
        coverage = np.ones((len(gt),), dtype=np.int32)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")
    if pred.shape[1] != 26:
        raise ValueError(f"This replay utility expects 26D A1 G1 actions, got {pred.shape[1]}D")

    model = mujoco.MjModel.from_xml_path(str(args.g1_xml))
    gt_data = mujoco.MjData(model)
    pred_data = mujoco.MjData(model)
    gt_renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    pred_renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    args.output_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width * 2, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {args.output_mp4}")

    for t in range(pred.shape[0]):
        set_pose(model, gt_data, gt[t])
        set_pose(model, pred_data, pred[t])
        gt_frame = label(render(model, gt_renderer, gt_data), "GT")
        pred_status = "Pred" if coverage[t] > 0 else "Pred missing; showing GT pose"
        pred_frame = label(
            render(model, pred_renderer, pred_data),
            f"{pred_status}  {sample_label} frame={t} L1={mean_l1:.4f} MSE={mean_mse:.4f}",
        )
        frame = np.concatenate([gt_frame, pred_frame], axis=1)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    print(args.output_mp4)


if __name__ == "__main__":
    main()

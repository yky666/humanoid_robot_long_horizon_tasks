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
    sample = samples[args.sample_index]
    pred = np.asarray(sample["pred_action"], dtype=np.float32)
    gt = np.asarray(sample["gt_action"], dtype=np.float32)
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
        pred_frame = label(
            render(model, pred_renderer, pred_data),
            f"Pred  sample={args.sample_index} frame={t} L1={sample['l1']:.4f} MSE={sample['mse']:.4f}",
        )
        frame = np.concatenate([gt_frame, pred_frame], axis=1)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    print(args.output_mp4)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""Export GR00T open-loop action chunks as JSON for metric analysis and replay.

The native output keeps the REAL_G1 53D action layout.  The optional A1 26D
projection is only for visual replay with the existing MuJoCo utility: it maps
arm joints directly and truncates the 7D hand commands to 6D per side.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import logging
from pathlib import Path
import re
from typing import Any

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.open_loop_eval import parse_action_gr00t, parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import pandas as pd
import torch
import tyro


REAL_G1_ACTION_KEYS = [
    "left_wrist_eef_9d",
    "right_wrist_eef_9d",
    "left_hand",
    "right_hand",
    "left_arm",
    "right_arm",
    "waist",
    "base_height_command",
    "navigate_command",
]


def to_serializable_array(value: np.ndarray) -> list:
    return np.asarray(value, dtype=np.float32).tolist()


def extract_columns(traj: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = {}
    for column in columns:
        values[column] = np.vstack([arr for arr in traj[column]])
    return np.concatenate([values[column] for column in columns], axis=-1)


def split_action_by_key(action: np.ndarray, action_dims: dict[str, int]) -> dict[str, np.ndarray]:
    offset = 0
    result = {}
    for key in REAL_G1_ACTION_KEYS:
        dim = action_dims[key]
        result[key] = action[..., offset : offset + dim]
        offset += dim
    return result


def real_g1_to_a1_26d(action: np.ndarray, action_dims: dict[str, int]) -> np.ndarray:
    grouped = split_action_by_key(action, action_dims)
    left_arm = grouped["left_arm"]
    right_arm = grouped["right_arm"]
    left_hand = grouped["left_hand"][..., :6]
    right_hand = grouped["right_hand"][..., :6]
    return np.concatenate([left_arm, right_arm, left_hand, right_hand], axis=-1)


def checkpoint_step(model_path: str) -> int | None:
    match = re.search(r"checkpoint-(\d+)", model_path)
    return int(match.group(1)) if match else None


def evaluate_and_export(
    policy: Gr00tPolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    embodiment_tag: EmbodimentTag,
    steps: int | None,
    action_horizon: int,
    output_json: Path,
    output_a1_json: Path | None,
    model_path: str,
) -> dict[str, Any]:
    traj = loader[traj_id]
    traj_length = len(traj)
    obs_min_delta = min(
        min(config.delta_indices)
        for name, config in loader.modality_configs.items()
        if name != "action"
    )
    start_step = max(0, -obs_min_delta)
    max_steps = traj_length - start_step
    actual_steps = max_steps if steps is None else min(steps, max_steps)
    if actual_steps <= 0:
        raise ValueError(f"No evaluable frames for trajectory {traj_id}: length={traj_length}")

    state_keys = loader.modality_configs["state"].modality_keys
    action_keys = loader.modality_configs["action"].modality_keys
    action_dims = {
        key: int(np.asarray(traj[f"action.{key}"].iloc[0]).shape[-1]) for key in action_keys
    }
    if action_keys != REAL_G1_ACTION_KEYS:
        raise ValueError(f"This exporter currently expects REAL_G1 action keys, got {action_keys}")

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")

    samples = []
    pred_across_time = []
    gt_across_time = extract_columns(traj, [f"action.{key}" for key in action_keys])[
        start_step : start_step + actual_steps
    ]
    state_across_time = extract_columns(traj, [f"state.{key}" for key in state_keys])[
        start_step : start_step + actual_steps
    ]

    for step_count in range(start_step, start_step + actual_steps, action_horizon):
        data_point = extract_step_data(traj, step_count, modality_configs, embodiment_tag)
        obs = {}
        for key, value in data_point.states.items():
            obs[f"state.{key}"] = value
        for key, value in data_point.images.items():
            obs[f"video.{key}"] = np.asarray(value)
        for language_key in loader.modality_configs["language"].modality_keys:
            obs[language_key] = data_point.text

        parsed_obs = parse_observation_gr00t(obs, loader.modality_configs)
        with torch.inference_mode():
            raw_action, _ = policy.get_action(parsed_obs)
        action_chunk = parse_action_gr00t(raw_action)
        concat_pred = np.concatenate(
            [
                np.asarray(action_chunk[f"action.{key}"], dtype=np.float32)
                for key in action_keys
            ],
            axis=-1,
        )
        local_start = step_count - start_step
        remaining = actual_steps - local_start
        clipped_pred = concat_pred[:remaining]
        clipped_gt = gt_across_time[local_start : local_start + len(clipped_pred)]
        pred_across_time.append(clipped_pred)
        samples.append(
            {
                "sample_index": len(samples),
                "dataset_index": int(local_start),
                "source_frame_index": int(step_count),
                "pred_action": to_serializable_array(clipped_pred),
                "gt_action": to_serializable_array(clipped_gt),
                "l1": float(np.mean(np.abs(clipped_pred - clipped_gt))),
                "mse": float(np.mean(np.square(clipped_pred - clipped_gt))),
                "metadata": {
                    "frame_index": int(step_count),
                    "traj_id": int(traj_id),
                    "action_horizon": int(action_horizon),
                    "action_layout": "real_g1_53d",
                    "action_keys": list(action_keys),
                    "action_dims": action_dims,
                },
            }
        )

    pred_across_time_arr = np.concatenate(pred_across_time, axis=0)[:actual_steps]
    mse = float(np.mean(np.square(pred_across_time_arr - gt_across_time)))
    l1 = float(np.mean(np.abs(pred_across_time_arr - gt_across_time)))
    payload = {
        "schema": "gr00t_open_loop_actions_v1",
        "model": "GR00T",
        "model_path": str(model_path),
        "checkpoint_step": checkpoint_step(str(model_path)),
        "dataset_path": str(loader.dataset_path),
        "traj_id": int(traj_id),
        "embodiment_tag": embodiment_tag.name,
        "action_layout": "real_g1_53d",
        "action_keys": list(action_keys),
        "action_dims": action_dims,
        "start_step": int(start_step),
        "num_steps": int(actual_steps),
        "action_horizon": int(action_horizon),
        "avg_l1_loss": l1,
        "avg_mse_loss": mse,
        "state_action_l1": l1,
        "state_action_mse": mse,
        "state_shape": list(state_across_time.shape),
        "gt_action_shape": list(gt_across_time.shape),
        "pred_action_shape": list(pred_across_time_arr.shape),
        "samples": samples,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if output_a1_json is not None:
        a1_samples = []
        for sample in samples:
            pred = np.asarray(sample["pred_action"], dtype=np.float32)
            gt = np.asarray(sample["gt_action"], dtype=np.float32)
            pred_a1 = real_g1_to_a1_26d(pred, action_dims)
            gt_a1 = real_g1_to_a1_26d(gt, action_dims)
            a1_samples.append(
                {
                    **{k: v for k, v in sample.items() if k not in {"pred_action", "gt_action"}},
                    "pred_action": to_serializable_array(pred_a1),
                    "gt_action": to_serializable_array(gt_a1),
                    "l1": float(np.mean(np.abs(pred_a1 - gt_a1))),
                    "mse": float(np.mean(np.square(pred_a1 - gt_a1))),
                    "metadata": {
                        **sample["metadata"],
                        "action_layout": "a1_26d_approx_from_real_g1",
                        "projection_note": (
                            "Approximate replay projection: left/right arm are direct; "
                            "left/right hand use the first 6 of 7 REAL_G1 hand commands. "
                            "REAL_G1 EEF xyz+rot6d commands are not visualized here."
                        ),
                    },
                }
            )
        pred_a1_all = real_g1_to_a1_26d(pred_across_time_arr, action_dims)
        gt_a1_all = real_g1_to_a1_26d(gt_across_time, action_dims)
        a1_payload = {
            **{k: v for k, v in payload.items() if k not in {"samples", "action_keys", "action_dims"}},
            "schema": "gr00t_open_loop_a1_26d_approx_v1",
            "action_layout": "a1_26d_approx_from_real_g1",
            "action_keys": ["left_arm", "right_arm", "left_hand_first6", "right_hand_first6"],
            "action_dims": {"left_arm": 7, "right_arm": 7, "left_hand_first6": 6, "right_hand_first6": 6},
            "avg_l1_loss": float(np.mean(np.abs(pred_a1_all - gt_a1_all))),
            "avg_mse_loss": float(np.mean(np.square(pred_a1_all - gt_a1_all))),
            "projection_note": (
                "This JSON is for A1-style MuJoCo visual replay only. It does not include "
                "the REAL_G1 EEF xyz+rot6d commands and should not be used as a physical "
                "closed-loop command representation."
            ),
            "samples": a1_samples,
        }
        output_a1_json.parent.mkdir(parents=True, exist_ok=True)
        output_a1_json.write_text(json.dumps(a1_payload, indent=2), encoding="utf-8")

    return payload


def main(
    model_path: str,
    dataset_path: str,
    output_json: Path,
    embodiment_tag: str = "REAL_G1",
    traj_id: int = 0,
    steps: int | None = None,
    action_horizon: int = 40,
    output_a1_json: Path | None = None,
) -> None:
    logging.basicConfig(level=logging.INFO)
    resolved_tag = EmbodimentTag.resolve(embodiment_tag)
    policy = Gr00tPolicy(
        embodiment_tag=resolved_tag,
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    modality = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(
        dataset_path=dataset_path,
        modality_configs=modality,
        video_backend="torchcodec",
        video_backend_kwargs=None,
    )
    payload = evaluate_and_export(
        policy=policy,
        loader=loader,
        traj_id=traj_id,
        embodiment_tag=resolved_tag,
        steps=steps,
        action_horizon=action_horizon,
        output_json=output_json,
        output_a1_json=output_a1_json,
        model_path=model_path,
    )
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_a1_json": str(output_a1_json) if output_a1_json else None,
                "num_steps": payload["num_steps"],
                "avg_l1_loss": payload["avg_l1_loss"],
                "avg_mse_loss": payload["avg_mse_loss"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    tyro.cli(main)

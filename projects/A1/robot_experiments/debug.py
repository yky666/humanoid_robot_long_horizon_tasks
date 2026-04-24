import collections
import dataclasses
import logging
import math
import pathlib
import json
import imageio
import numpy as np
from typing import Optional, Union
from pathlib import Path
import draccus
from dataclasses import dataclass,replace
from a1.data.vla.lerobot_datasets import LeRobotDatasetWrapper

import requests
import base64
import cv2

from robot_experiments.robot_utils import (
    DATE_TIME,
    # get_action,
    get_image_resize_size,
    # get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


from robot_experiments.vla_utils import (
    resize_image_for_policy,
    prepare_images_for_vla,
    _unnormalize_actions,
    _load_dataset_stats,
    normalize_proprio,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

@dataclasses.dataclass
class GenerateConfig:
    dataset_path: str = "data/vlabench"
    url: str = "http://localhost:8000"
    n_episode: int = 50
    fixed_action_dim: int = 32
    chunk_size: int = 50
    use_proprio: bool = True
    use_wrist_image: bool = True
    seed: int = 42
    output_json: Optional[str] = None


class DummyPolicy:
    """
    Example policy class.
    Users should implement the __init__ and run_policy methods according to their own logic.
    """

    def __init__(self, base_url: str = "http://localhost:7777"):
        """
        Initialize the policy.
        Args:
            port (str): Port to the model checkpoint file.
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def run_policy(self, input_data):
        """
        Run inference using the policy/model.
        Args:
            input_data: Input data for inference.
        Returns:
            list: Inference results.
        """
        def encode_image_to_base64(image_data):
            image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
            image_data = cv2.imencode('.png', image_data)[-1].tobytes()
            return base64.b64encode(image_data).decode('utf-8')

        payload = {
            "instruction": input_data['question'],
            "images": [
                encode_image_to_base64(image_data) for image_data in input_data['images']
            ],
            "proprio_data": input_data['proprio'].tolist(),
        }


        response = self.session.post(
            f"{self.base_url}/inference", 
            json=payload, 
            timeout=60  # Longer timeout for inference
        )
        response.raise_for_status()
        result = response.json()
        actions = result['predicted_actions']
        return actions

# def normalize_action(action: np.ndarray, metadata, normalization_type: NormalizationType):  
#     """Normalizes the action and proprio fields of a trajectory using the given metadata."""  
#     # keys_to_normalize = {"action": "actions", "state": "state"}  
      
#     normalized_action = action.copy()  
  
#     if normalization_type == NormalizationType.NORMAL:  
#         normalized_action = (action - metadata['mean']) / (metadata['std'] + 1e-8)
#         return normalized_action  
#     elif normalization_type == NormalizationType.BOUNDS:
#         normalized_action = np.clip(2 * (action - metadata['min']) / (metadata['max'] - metadata['min'] + 1e-8) - 1, -1, 1)
#         return normalized_action  
#     elif normalization_type == NormalizationType.BOUNDS_Q99:
#         normalized_action = np.clip(2 * (action - metadata['q01']) / (metadata['q99'] - metadata['q01'] + 1e-8) - 1, -1, 1)
#         return normalized_action  
#     raise ValueError(f"Unknown Normalization Type {normalization_type}")

@draccus.wrap()
def main(cfg: GenerateConfig) -> None:
    rng = np.random.RandomState(cfg.seed)
    dataset = LeRobotDatasetWrapper(
        dataset_path=cfg.dataset_path,
        fixed_action_dim=cfg.fixed_action_dim,
        chunk_size=cfg.chunk_size,
        use_proprio=cfg.use_proprio,
        use_wrist_image=cfg.use_wrist_image,
    )
    policy = DummyPolicy(base_url=cfg.url)

    num_samples = min(len(dataset), cfg.n_episode)
    logger.info(f"Debug inference started: num_samples={num_samples}, dataset_path={cfg.dataset_path}")
    avg_l1_loss = 0
    avg_mse_loss = 0
    replace = num_samples > len(dataset)
    sampled_indices = rng.choice(len(dataset), size=num_samples, replace=replace)
    per_sample_metrics = []
    for i, idx in enumerate(sampled_indices):
        idx = int(idx)
        item = dataset.get(idx, rng)
        # # 设置打印位数
        # np.set_printoptions(precision=16)
        # print(item['proprio'])
        logger.info(f"sample={i} idx={idx} question={item['question']}")
        gt_action = np.asarray(item['action'], dtype=np.float32)
        actions = np.asarray(policy.run_policy(item), dtype=np.float32)
        if actions.shape != gt_action.shape:
            raise ValueError(
                f"Predicted action shape {actions.shape} does not match ground truth {gt_action.shape}"
            )
        mse = np.mean(np.square(actions - gt_action))
        l1 = np.mean(np.abs(actions - gt_action))
        avg_l1_loss += l1
        avg_mse_loss += mse
        per_sample_metrics.append(
            {
                "sample_index": i,
                "dataset_index": idx,
                "question": item["question"],
                "mse": float(mse),
                "l1": float(l1),
            }
        )
        logger.info(f"sample={i} mse={mse} l1={l1}")
    summary = {
        "dataset_path": cfg.dataset_path,
        "url": cfg.url,
        "num_samples": num_samples,
        "fixed_action_dim": cfg.fixed_action_dim,
        "chunk_size": cfg.chunk_size,
        "use_proprio": cfg.use_proprio,
        "use_wrist_image": cfg.use_wrist_image,
        "seed": cfg.seed,
        "avg_l1_loss": float(avg_l1_loss / num_samples),
        "avg_mse_loss": float(avg_mse_loss / num_samples),
        "samples": per_sample_metrics,
    }
    if cfg.output_json:
        output_path = Path(cfg.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info(f"Saved debug evaluation summary to {output_path}")
    print('num_samples', num_samples)
    print('avg_l1_loss', summary['avg_l1_loss'])
    print('avg_mse_loss', summary['avg_mse_loss'])
if __name__ == "__main__":
    main()

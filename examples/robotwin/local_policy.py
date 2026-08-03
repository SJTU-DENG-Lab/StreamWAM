"""In-process RoboTwin 2.0 policy adapter for StarWAM.

This plugs a StarWAM checkpoint directly into the official RoboTwin evaluation
harness (``RoboTwin/script/eval_policy.py``). Use it when SAPIEN and the
Torch/StarWAM stack live in the same Python environment.

RoboTwin harness entry points: get_model / eval / reset_model.

All benchmark-neutral logic (recipe/checkpoint loading, text conditioning,
normalization, flow-matching inference, replan queue) lives in
:class:`starwam.eval.policy.StarwamPolicy`. This file only handles the
RoboTwin-specific pieces:

  * 3-camera observation -> the exact 384x320 grid used at training time
    (head 256x320 on top, [left|right] 128x160 each on the bottom). The resize
    reuses ``starwam.data.lerobot._resize_frames`` so eval pixels match training
    pixels bit-for-bit.
  * 14-D dual-arm proprio state from ``observation["joint_action"]["vector"]``.
  * 14-D action executed via ``task_env.take_action(action, action_type="qpos")``
    in native qpos order (no reindexing).

Camera order MUST match the recipe's ``data.video_keys``:
[head, left_wrist, right_wrist].
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# Make the StarWAM package importable when launched from the RoboTwin repo.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from starwam.data.lerobot import _resize_frames  # noqa: E402
from starwam.eval.policy import StarwamPolicy  # noqa: E402


def _to_chw_uint8(rgb: np.ndarray) -> torch.Tensor:
    """RoboTwin camera RGB ``[H, W, 3]`` uint8 -> ``[1, 3, H, W]`` uint8 tensor."""
    arr = np.ascontiguousarray(rgb)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(torch.uint8)


def build_robotwin_image(observation: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Compose the RoboTwin 3-camera grid identical to the training layout.

    Returns a ``[1, 3, 384, 320]`` tensor in ``[-1, 1]``.
    """
    obs = observation["observation"]
    head = _to_chw_uint8(obs["head_camera"]["rgb"])
    left = _to_chw_uint8(obs["left_camera"]["rgb"])
    right = _to_chw_uint8(obs["right_camera"]["rgb"])

    top = _resize_frames(head, (256, 320)).float() / 255.0
    left_r = _resize_frames(left, (128, 160)).float() / 255.0
    right_r = _resize_frames(right, (128, 160)).float() / 255.0
    bottom = torch.cat([left_r, right_r], dim=-1)
    frame = torch.cat([top, bottom], dim=-2)
    frame = frame * 2.0 - 1.0
    return frame.to(device=device, dtype=dtype)


class RoboTwinStarwamModel:
    """Wraps :class:`StarwamPolicy` with RoboTwin obs/action conventions."""

    def __init__(self, policy: StarwamPolicy) -> None:
        self.policy = policy

    def step(self, task_env: Any, observation: Optional[Dict[str, Any]]) -> None:
        if self.policy.needs_observation():
            if observation is None:
                raise ValueError("Observation required on a replan step but got None.")
            instruction = str(task_env.get_instruction())
            image = build_robotwin_image(observation, self.policy.device, self.policy.dtype)
            state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
            self.policy.fill_queue(image, state, instruction)

        if self.policy.needs_observation():
            return
        action = self.policy.pop_action()
        task_env.take_action(action, action_type="qpos")

    def reset(self) -> None:
        self.policy.reset()


def _get(usr_args: Dict[str, Any], key: str, default: Any = None) -> Any:
    value = usr_args.get(key, default)
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return default
    return value


def get_model(usr_args: Dict[str, Any]) -> RoboTwinStarwamModel:
    config_path = _get(usr_args, "config_path") or _get(usr_args, "config")
    checkpoint = _get(usr_args, "checkpoint") or _get(usr_args, "ckpt_setting")
    if config_path is None or checkpoint is None:
        raise ValueError("`config_path` and `checkpoint` are required in deploy_policy args.")

    config_path = Path(str(config_path))
    if not config_path.is_absolute() and not config_path.is_file():
        config_path = _REPO_ROOT / config_path

    overrides = _get(usr_args, "overrides")
    if isinstance(overrides, str):
        overrides = overrides.split()

    device = str(_get(usr_args, "device", "cuda:0"))
    num_inference_steps = _get(usr_args, "num_inference_steps")
    action_num_inference_steps = _get(usr_args, "action_num_inference_steps")
    replan_steps = int(_get(usr_args, "replan_steps", 8))
    seed = _get(usr_args, "seed")

    policy = StarwamPolicy(
        config_path=str(config_path),
        checkpoint=str(checkpoint),
        overrides=list(overrides) if overrides else None,
        device=device,
        num_inference_steps=int(num_inference_steps) if num_inference_steps is not None else None,
        action_num_inference_steps=(
            int(action_num_inference_steps) if action_num_inference_steps is not None else None
        ),
        replan_steps=replan_steps,
        seed=int(seed) if seed is not None else None,
    )
    return RoboTwinStarwamModel(policy)


def eval(TASK_ENV: Any, model: RoboTwinStarwamModel, observation: Optional[Dict[str, Any]]) -> None:
    model.step(TASK_ENV, observation)


def reset_model(model: RoboTwinStarwamModel) -> None:
    model.reset()

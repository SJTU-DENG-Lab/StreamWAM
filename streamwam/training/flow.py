"""Flow-matching helpers for StreamWAM."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from streamwam.modules.scheduler import FlowMatchScheduler


def add_flow_noise(
    scheduler: FlowMatchScheduler,
    sample: Tensor,
    pin_first_latent_step: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    noise = torch.randn_like(sample)
    timesteps = scheduler.sample_timesteps(sample.shape[0], sample.device, sample.dtype)
    noisy = scheduler.add_noise(sample, noise, timesteps)
    if pin_first_latent_step and sample.shape[2] > 1:
        noisy[:, :, 0:1] = sample[:, :, 0:1]
    target = scheduler.training_target(sample, noise, timesteps)
    return noisy, target, timesteps


def build_inference_schedule(scheduler_config, num_inference_steps: int, device, dtype) -> tuple[Tensor, Tensor]:
    scheduler = FlowMatchScheduler(
        num_train_timesteps=scheduler_config.num_train_timesteps,
        shift=scheduler_config.infer_shift,
    )
    return scheduler.build_inference_schedule(num_inference_steps, device, dtype)


def video_latent_pad_mask(
    image_is_pad: Optional[Tensor],
    latent_steps: int,
    include_initial_video_step: bool,
    temporal_factor: int = 4,
) -> Optional[Tensor]:
    if image_is_pad is None:
        return None
    if image_is_pad.shape[1] < 1:
        raise ValueError("image_is_pad must contain at least one frame")
    if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
        raise ValueError(
            "Cannot align image_is_pad with video latent steps: "
            f"num_frames={image_is_pad.shape[1]}, temporal_factor={temporal_factor}"
        )
    tail_is_pad = image_is_pad[:, 1:]
    latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
    if include_initial_video_step:
        video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
    else:
        video_is_pad = latent_tail_is_pad
    if video_is_pad.shape[1] != latent_steps:
        raise ValueError(
            "Video-loss mask shape mismatch: "
            f"mask steps={video_is_pad.shape[1]}, latent steps={latent_steps}"
        )
    return video_is_pad

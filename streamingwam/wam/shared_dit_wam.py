"""Shared-DiT WAM taxonomy entry."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from streamingwam.backbone.base import BaseBackbone
from streamingwam.config import FrameworkConfig
from streamingwam.modules.scheduler import FlowMatchScheduler
from streamingwam.training.flow import add_flow_noise, build_inference_schedule, video_latent_pad_mask
from streamingwam.training.loss import flow_matching_loss
from streamingwam.training.metrics import action_monitor_metrics
from streamingwam.wam.base import WAMModel


class SharedDiTWAM(WAMModel):
    """Shared-DiT WAM with action/state register tokens."""

    taxonomy_model_family = "shared_dit_wam"

    def __init__(
        self,
        backbone: BaseBackbone,
        config: FrameworkConfig,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone = backbone
        self._device = device
        self._dtype = dtype
        self.action_dim = int(config.action_dim)
        self.chunk_size = int(config.chunk_size)
        self.state_dim = self._resolve_state_dim(config)
        self.action_tokens_per_state = self._resolve_action_tokens_per_state(config)

        self.shared_dit = backbone.build_shared_dit_core(
            config,
            state_dim=self.state_dim,
            action_tokens_per_state=self.action_tokens_per_state,
            device=device,
            dtype=dtype,
        )
        self.video_scheduler = FlowMatchScheduler(
            num_train_timesteps=config.video_scheduler.num_train_timesteps,
            shift=config.video_scheduler.train_shift,
        )
        self.action_scheduler = FlowMatchScheduler(
            num_train_timesteps=config.action_scheduler.num_train_timesteps,
            shift=config.action_scheduler.train_shift,
        )
        self.loss_lambda_video = float(config.loss_lambda_video)
        self.loss_lambda_action = float(config.loss_lambda_action)
        self.pin_first_latent_step = bool(getattr(config, "shared_dit_pin_first_latent_step", True))

        self.shared_dit.to(device=device, dtype=dtype)

    @staticmethod
    def _resolve_state_dim(config: FrameworkConfig) -> int:
        state_dim = getattr(config, "max_state_dim", None) or getattr(config, "proprio_dim", None)
        if not state_dim or int(state_dim) <= 0:
            raise ValueError("shared_dit_wam requires framework.max_state_dim or framework.proprio_dim")
        return int(state_dim)

    @staticmethod
    def _resolve_action_tokens_per_state(config: FrameworkConfig) -> int:
        num_action_per_block = getattr(config, "num_action_per_block", None)
        num_state_per_block = max(1, int(getattr(config, "num_state_per_block", 1)))
        if num_action_per_block is None:
            return 4
        return max(1, int(num_action_per_block) // num_state_per_block)

    def training_step(self, sample: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        video = sample["video"]
        action = sample["action"]
        context = sample["context"]
        context_mask = sample.get("context_mask")
        action_is_pad = sample.get("action_is_pad")
        image_is_pad = sample.get("image_is_pad")

        state, state_is_pad = self._state_from_sample(sample, action)

        with torch.no_grad():
            video_latents = self.backbone.encode_video(video)

        noisy_video, target_video, t_video = add_flow_noise(
            self.video_scheduler,
            video_latents,
            pin_first_latent_step=self.pin_first_latent_step,
        )
        noisy_action, target_action, t_action = add_flow_noise(self.action_scheduler, action)
        clean_video = video_latents if getattr(self.config, "shared_dit_clean_context", "full_video") == "full_video" else None

        pred_video, pred_action = self.shared_dit(
            noisy_video=noisy_video,
            video_timestep=t_video,
            context=context,
            noisy_action=noisy_action,
            action_timestep=t_action,
            state=state,
            context_mask=context_mask,
            clean_video=clean_video,
            state_is_pad=state_is_pad,
        )

        include_initial_video_step = not (video_latents.shape[2] > 1)
        if video_latents.shape[2] > 1:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]
        video_is_pad = video_latent_pad_mask(
            image_is_pad,
            latent_steps=pred_video.shape[2],
            include_initial_video_step=include_initial_video_step,
        )

        loss_video = flow_matching_loss(
            pred_video,
            target_video,
            t_video,
            self.video_scheduler,
            is_pad_mask=video_is_pad,
        )
        loss_action = flow_matching_loss(
            pred_action,
            target_action,
            t_action,
            self.action_scheduler,
            is_pad_mask=action_is_pad,
        )
        loss = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action

        loss_dict = {
            "loss_video": loss_video.item(),
            "loss_action": loss_action.item(),
            "loss_total": loss.item(),
        }
        loss_dict.update(action_monitor_metrics(
            pred_action.detach(),
            target_action.detach(),
            action.detach(),
            is_pad=action_is_pad,
            gripper_dim=getattr(self.config, "action_gripper_dim", -1),
        ))
        return loss, loss_dict

    def _state_from_sample(self, sample: dict[str, Any], action: Tensor) -> tuple[Tensor, Optional[Tensor]]:
        proprio = sample.get("proprio")
        if proprio is None:
            raise ValueError("shared_dit_wam requires sample['proprio']; set framework.proprio_dim and data.state_key")
        if proprio.ndim != 3:
            raise ValueError(f"sample['proprio'] must be [B, T, D], got {tuple(proprio.shape)}")
        if proprio.shape[-1] < self.state_dim:
            raise ValueError(f"sample['proprio'] dim {proprio.shape[-1]} is smaller than state_dim={self.state_dim}")
        state = proprio[..., :self.state_dim]
        state_is_pad = sample.get("proprio_is_pad")
        if state_is_pad is None:
            state_is_pad = sample.get("action_is_pad")
        target_len = max(1, (action.shape[1] + self.action_tokens_per_state - 1) // self.action_tokens_per_state)
        state = self._resample_sequence(state, target_len)
        if state_is_pad is not None:
            state_is_pad = self._resample_sequence(state_is_pad.unsqueeze(-1).to(state.dtype), target_len).squeeze(-1) > 0.5
        return state, state_is_pad

    @staticmethod
    def _resample_sequence(sequence: Tensor, target_len: int) -> Tensor:
        if sequence.shape[1] == target_len:
            return sequence
        if sequence.shape[1] <= 0:
            raise ValueError("Cannot resample an empty sequence")
        indices = torch.linspace(0, sequence.shape[1] - 1, target_len, device=sequence.device)
        indices = indices.round().long().clamp(0, sequence.shape[1] - 1)
        return sequence.index_select(1, indices)

    @torch.no_grad()
    def infer_action(
        self,
        input_image: Tensor,
        context: Tensor,
        context_mask: Optional[Tensor],
        action_horizon: int,
        num_inference_steps: int = 20,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Tensor:
        action_steps = int(kwargs.get("action_num_inference_steps") or num_inference_steps)
        out = self.infer_joint(
            input_image=input_image,
            context=context,
            context_mask=context_mask,
            num_video_frames=int(kwargs["num_video_frames"]),
            action_horizon=action_horizon,
            num_inference_steps=num_inference_steps,
            action_num_inference_steps=action_steps,
            seed=seed,
            proprio=kwargs.get("proprio"),
        )
        return out["action"]

    @torch.no_grad()
    def infer_joint(
        self,
        input_image: Tensor,
        context: Tensor,
        context_mask: Optional[Tensor],
        num_video_frames: int,
        action_horizon: int,
        num_inference_steps: int = 20,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        device = input_image.device
        dtype = input_image.dtype
        seed = kwargs.get("seed")
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

        if input_image.ndim != 4:
            raise ValueError(f"input_image must be [B, C, H, W], got {tuple(input_image.shape)}")
        if int(action_horizon) != self.chunk_size:
            raise ValueError(f"SharedDiTWAM was built for action_horizon={self.chunk_size}, got {action_horizon}")

        video = input_image.unsqueeze(2).expand(-1, -1, int(num_video_frames), -1, -1).contiguous()
        clean_video = self.backbone.encode_video(video)
        batch, channels, latent_steps, height, width = clean_video.shape
        if latent_steps <= 1:
            raise ValueError(f"SharedDiT inference needs future latent steps, got latent_steps={latent_steps}")

        video_latents = torch.randn(
            batch,
            channels,
            latent_steps,
            height,
            width,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        video_latents[:, :, 0:1] = clean_video[:, :, 0:1]
        action_latents = torch.randn(
            batch,
            int(action_horizon),
            self.action_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        state, state_is_pad = self._state_from_proprio(kwargs.get("proprio"), batch, latent_steps, device, dtype)
        video_timesteps, video_deltas = build_inference_schedule(
            self.config.video_scheduler,
            int(num_inference_steps),
            device,
            dtype,
        )
        action_steps = int(kwargs.get("action_num_inference_steps") or num_inference_steps)
        action_timesteps, action_deltas = build_inference_schedule(
            self.config.action_scheduler,
            action_steps,
            device,
            dtype,
        )
        total_steps = max(int(num_inference_steps), action_steps)

        for i in range(total_steps):
            if i < int(num_inference_steps):
                t_video = video_timesteps[i].expand(batch)
                video_delta = video_deltas[i]
            else:
                t_video = torch.zeros(batch, device=device, dtype=dtype)
                video_delta = torch.zeros((), device=device, dtype=dtype)
            if i < action_steps:
                t_action = action_timesteps[i].expand(batch)
                action_delta = action_deltas[i]
            else:
                t_action = torch.zeros(batch, device=device, dtype=dtype)
                action_delta = torch.zeros((), device=device, dtype=dtype)

            pred_video, pred_action = self.shared_dit(
                noisy_video=video_latents,
                video_timestep=t_video,
                context=context,
                noisy_action=action_latents,
                action_timestep=t_action,
                state=state,
                context_mask=context_mask,
                clean_video=video_latents,
                state_is_pad=state_is_pad,
            )
            if i < int(num_inference_steps):
                video_latents = FlowMatchScheduler.step(pred_video, video_delta, video_latents)
                video_latents[:, :, 0:1] = clean_video[:, :, 0:1]
            if i < action_steps:
                action_latents = FlowMatchScheduler.step(pred_action, action_delta, action_latents)

        return {"video": self.backbone.decode_latents(video_latents), "action": action_latents}

    def _state_from_proprio(
        self,
        proprio: Optional[Tensor],
        batch: int,
        latent_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Optional[Tensor]]:
        num_blocks = (int(latent_steps) - 1) // max(1, int(getattr(self.config, "num_frame_per_block", 1)))
        if num_blocks <= 0:
            raise ValueError(
                f"Cannot align proprio with Shared-DiT blocks: latent_steps={latent_steps}, "
                f"num_frame_per_block={getattr(self.config, 'num_frame_per_block', 1)}"
            )
        target_len = num_blocks * max(1, int(getattr(self.config, "num_state_per_block", 1)))
        if proprio is None:
            raise ValueError("shared_dit_wam inference requires proprio")
        proprio = proprio.to(device=device, dtype=dtype)
        if proprio.ndim == 2:
            proprio = proprio.unsqueeze(1).expand(batch, target_len, -1)
        elif proprio.ndim == 3:
            proprio = self._resample_sequence(proprio, target_len)
        else:
            raise ValueError(f"proprio must be [B, D] or [B, T, D], got {tuple(proprio.shape)}")
        if proprio.shape[0] != batch:
            raise ValueError(f"proprio batch {proprio.shape[0]} does not match image batch {batch}")
        if proprio.shape[-1] < self.state_dim:
            raise ValueError(f"proprio dim {proprio.shape[-1]} is smaller than state_dim={self.state_dim}")
        return proprio[..., :self.state_dim], None

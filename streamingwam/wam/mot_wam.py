"""MoT WAM implementation.

MoT WAM uses separate video and action experts, with per-layer joint attention
between their Q/K/V streams. The implementation keeps Fast-WAM-aligned details:
first-frame clean pinning, configurable first-frame/full-video action attention,
velocity target ``noise - sample``, weighted flow matching, and action inference.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from streamingwam.action_model import build_action_expert, load_action_dit_init
from streamingwam.training.flow import add_flow_noise, build_inference_schedule, video_latent_pad_mask
from streamingwam.wam.base import WAMModel
from streamingwam.backbone.base import BaseBackbone
from streamingwam.config import FrameworkConfig
from streamingwam.inference.consistency import (
    action_consistency_boundary,
    normalize_sampling_method,
    sample_joint_consistency_noise,
    video_consistency_boundary,
)
from streamingwam.modules.mot import MoT
from streamingwam.modules.scheduler import FlowMatchScheduler
from streamingwam.training.loss import flow_matching_loss
from streamingwam.training.metrics import action_monitor_metrics


def _validate_consistency_inference(
    *,
    num_inference_steps: int,
    action_video_conditioning: str,
    action_horizon: int,
    configured_horizon: int,
    num_video_frames: int,
    temporal_compress: int,
) -> None:
    if int(num_inference_steps) != 1:
        raise ValueError("Joint CD consistency sampling requires exactly one inference step")
    if action_video_conditioning != "full_video":
        raise ValueError("Joint CD consistency sampling requires full-video joint inference")
    if int(configured_horizon) != 32:
        raise ValueError(
            f"Joint CD checkpoint requires configured chunk_size=32, got {configured_horizon}"
        )
    if int(action_horizon) != int(configured_horizon):
        raise ValueError(
            f"Joint CD model requires action_horizon={configured_horizon}, got {action_horizon}"
        )
    if int(num_video_frames) != 9:
        raise ValueError(
            f"Joint CD checkpoint requires num_video_frames=9, got {num_video_frames}"
        )
    if int(temporal_compress) != 4:
        raise ValueError(
            f"Joint CD checkpoint requires temporal_compress=4, got {temporal_compress}"
        )
    if temporal_compress <= 0 or (int(num_video_frames) - 1) % temporal_compress != 0:
        raise ValueError(
            "Joint CD video geometry requires (num_video_frames - 1) to be "
            f"divisible by temporal_compress, got {num_video_frames} and {temporal_compress}"
        )
    latent_chunks = (int(num_video_frames) - 1) // temporal_compress
    if latent_chunks != 2:
        raise ValueError(
            f"Joint CD checkpoint requires 2 future latent chunks, got {latent_chunks}"
        )
    if latent_chunks <= 0 or int(action_horizon) % latent_chunks != 0:
        raise ValueError(
            "Joint CD action horizon must divide evenly over future latent chunks; "
            f"got action_horizon={action_horizon}, latent_chunks={latent_chunks}"
        )


def _validate_first_frame_consistency_inference(*, num_inference_steps: int) -> None:
    if int(num_inference_steps) != 1:
        raise ValueError("First-frame CD consistency sampling requires exactly one inference step")


def _shifted_consistency_anchors(
    num_steps: int,
    shift: float,
    device: torch.device,
) -> Tensor:
    if int(num_steps) < 1:
        raise ValueError(f"consistency teacher steps must be positive, got {num_steps}")
    unit = torch.linspace(1.0, 0.0, int(num_steps) + 1, device=device, dtype=torch.float32)
    return float(shift) * unit / (1.0 + (float(shift) - 1.0) * unit)


def _sample_euler_noise(
    shape: tuple[int, ...],
    *,
    seed: int | None,
    rand_device: str | torch.device | None,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Sample native or FastWAM-reference Euler noise without mixing contracts."""

    if rand_device is None:
        generator = (
            None if seed is None else torch.Generator(device=device).manual_seed(seed)
        )
        return torch.randn(
            shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    generator = (
        None
        if seed is None
        else torch.Generator(device=rand_device).manual_seed(seed)
    )
    return torch.randn(
        shape,
        generator=generator,
        device=rand_device,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)


def _consistency_timesteps(
    timestep_video: Tensor,
    timestep_action: Tensor,
    *,
    latent_steps: int,
    action_horizon: int,
) -> tuple[Tensor, Tensor]:
    video = timestep_video.reshape(1, 1).expand(1, latent_steps).clone()
    video[:, 0] = 0
    action = timestep_action.reshape(1, 1).expand(1, action_horizon).clone()
    return video, action


class MoTWAM(WAMModel):
    """Fast-WAM/Motus-style multi-stream WAM."""

    taxonomy_model_family = "mot_wam"

    def __init__(
        self,
        backbone: BaseBackbone,
        config: FrameworkConfig,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.config = config
        self._device = device
        self._dtype = dtype
        self.backbone = backbone

        self.action_expert = build_action_expert(backbone.info, config)
        load_action_dit_init(
            self.action_expert,
            getattr(config, "action_expert_init_from", None),
            head_init=getattr(config, "action_expert_head_init", "random"),
        )

        self.mot = MoT(
            experts={"video": backbone.get_dit(), "action": self.action_expert},
            checkpoint_mixed_attn=config.mot_checkpoint_mixed_attn,
        )

        self.video_scheduler = FlowMatchScheduler(
            num_train_timesteps=config.video_scheduler.num_train_timesteps,
            shift=config.video_scheduler.train_shift,
        )
        self.action_scheduler = FlowMatchScheduler(
            num_train_timesteps=config.action_scheduler.num_train_timesteps,
            shift=config.action_scheduler.train_shift,
        )
        self.loss_lambda_video = config.loss_lambda_video
        self.loss_lambda_action = config.loss_lambda_action
        self.action_video_conditioning = getattr(config, "action_video_conditioning", "first_frame")
        if self.action_video_conditioning not in {"first_frame", "full_video"}:
            raise ValueError(
                "framework.action_video_conditioning must be 'first_frame' or 'full_video', "
                f"got {self.action_video_conditioning!r}"
            )
        self.proprio_dim = None
        self.proprio_encoder = None
        if getattr(config, "proprio_dim", None):
            self.proprio_dim = int(config.proprio_dim)
            if self.proprio_dim > 0:
                self.proprio_encoder = nn.Linear(self.proprio_dim, backbone.info.text_dim).to(device=device, dtype=dtype)
            else:
                self.proprio_dim = None

        self.backbone.get_dit().to(device=device, dtype=dtype)
        self.action_expert.to(device=device, dtype=dtype)
        self.mot.to(device=device, dtype=dtype)

    def training_step(self, sample: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        video = sample["video"]
        action = sample["action"]
        context = sample["context"]
        context_mask = sample.get("context_mask")
        action_is_pad = sample.get("action_is_pad")
        image_is_pad = sample.get("image_is_pad")
        device = video.device
        context, context_mask = self._append_proprio_from_sample(sample, context, context_mask)

        with torch.no_grad():
            video_latents = self.backbone.encode_video(video)

        noisy_video, target_video, t_video = add_flow_noise(
            self.video_scheduler,
            video_latents,
            pin_first_latent_step=True,
        )
        noisy_action, target_action, t_action = add_flow_noise(self.action_scheduler, action)

        video_dit = self.backbone.get_dit()
        video_state = video_dit.pre_dit(noisy_video, t_video, context, context_mask)
        action_state = self.action_expert.pre_dit(noisy_action, t_action, context, context_mask)

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_state["tokens"].shape[1],
            action_seq_len=action_state["tokens"].shape[1],
            video_tokens_per_frame=self._compute_video_tokens_per_frame(video_state),
            device=device,
            action_video_conditioning=self.action_video_conditioning,
        )
        output_tokens = self.mot({"video": video_state, "action": action_state}, attention_mask)

        pred_video = video_dit.post_dit(output_tokens["video"], video_state["meta"], video_state["t"])
        pred_action = self.action_expert.post_dit(output_tokens["action"])

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

    @torch.no_grad()
    def infer_action(
        self,
        input_image: Tensor,
        context: Tensor,
        context_mask: Tensor,
        action_horizon: int,
        num_inference_steps: int = 10,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Tensor:
        proprio = kwargs.get("proprio")
        num_video_frames = kwargs.get("num_video_frames")
        rand_device = kwargs.get("rand_device")
        sampling_method = normalize_sampling_method(kwargs.get("sampling_method", "euler"))
        if sampling_method == "consistency":
            if self.action_video_conditioning == "first_frame":
                _validate_first_frame_consistency_inference(
                    num_inference_steps=num_inference_steps,
                )
            else:
                if num_video_frames is None:
                    raise ValueError("num_video_frames is required for Joint CD inference")
                _validate_consistency_inference(
                    num_inference_steps=num_inference_steps,
                    action_video_conditioning=self.action_video_conditioning,
                    action_horizon=action_horizon,
                    configured_horizon=int(self.config.chunk_size),
                    num_video_frames=int(num_video_frames),
                    temporal_compress=int(self.backbone.get_vae().temporal_compress),
                )
        if self.action_video_conditioning == "full_video":
            if num_video_frames is None:
                raise ValueError("num_video_frames is required for full-video action inference")
            out = self.infer_joint(
                input_image=input_image,
                context=context,
                context_mask=context_mask,
                num_video_frames=int(num_video_frames),
                action_horizon=action_horizon,
                num_inference_steps=num_inference_steps,
                seed=seed,
                proprio=proprio,
                rand_device=rand_device,
                sampling_method=sampling_method,
                decode_video=False,
            )
            return out["action"]

        device = input_image.device
        dtype = input_image.dtype
        context, context_mask = self._append_proprio_to_context(context, context_mask, proprio)

        video_single = input_image.unsqueeze(2)
        video_latents = self.backbone.encode_video(video_single)
        video_dit = self.backbone.get_dit()
        t_zero = torch.zeros(1, device=device, dtype=torch.long)
        video_state = video_dit.pre_dit(video_latents, t_zero, context, context_mask)
        video_kv_cache = self.mot.prefill_video_cache(video_state)

        action_latents = _sample_euler_noise(
            (1, action_horizon, self.config.action_dim),
            seed=seed,
            rand_device=rand_device,
            device=device,
            dtype=dtype,
        )
        if sampling_method == "consistency":
            teacher_steps = int(kwargs.get("consistency_teacher_steps", 4))
            sigma = _shifted_consistency_anchors(
                teacher_steps,
                float(self.config.action_scheduler.infer_shift),
                device,
            )[0]
            timestep = (sigma * 1000.0).reshape(1).to(dtype=dtype)
            action_state = self.action_expert.pre_dit(
                action_latents,
                timestep,
                context,
                context_mask,
            )
            action_out = self.mot.forward_action_with_video_cache(
                action_state,
                video_kv_cache,
            )
            velocity = self.action_expert.post_dit(action_out)
            return action_latents.float() - sigma.float() * velocity.float()

        timesteps, deltas = build_inference_schedule(
            self.config.action_scheduler,
            num_inference_steps,
            device,
            dtype,
        )
        for i in range(num_inference_steps):
            t_action = timesteps[i].reshape(1)
            action_state = self.action_expert.pre_dit(action_latents, t_action, context, context_mask)
            action_out = self.mot.forward_action_with_video_cache(action_state, video_kv_cache)
            velocity = self.action_expert.post_dit(action_out)
            action_latents = FlowMatchScheduler.step(velocity, deltas[i], action_latents)
        return action_latents

    @torch.no_grad()
    def infer_joint(
        self,
        input_image: Tensor,
        context: Tensor,
        context_mask: Tensor,
        num_video_frames: int,
        action_horizon: int,
        num_inference_steps: int = 20,
        **kwargs,
    ) -> dict[str, Any]:
        device = input_image.device
        dtype = input_image.dtype
        seed = kwargs.get("seed")
        rand_device = kwargs.get("rand_device")
        sampling_method = normalize_sampling_method(kwargs.get("sampling_method", "euler"))
        if sampling_method == "consistency":
            _validate_consistency_inference(
                num_inference_steps=num_inference_steps,
                action_video_conditioning=self.action_video_conditioning,
                action_horizon=action_horizon,
                configured_horizon=int(self.config.chunk_size),
                num_video_frames=int(num_video_frames),
                temporal_compress=int(self.backbone.get_vae().temporal_compress),
            )
        context, context_mask = self._append_proprio_to_context(context, context_mask, kwargs.get("proprio"))

        video_single = input_image.unsqueeze(2)
        first_frame_latents = self.backbone.encode_video(video_single)
        _, channels, _, height_latent, width_latent = first_frame_latents.shape
        vae = self.backbone.get_vae()
        latent_steps = max(1, (int(num_video_frames) - 1) // vae.temporal_compress + 1)

        video_shape = (1, channels, latent_steps, height_latent, width_latent)
        action_shape = (1, action_horizon, self.config.action_dim)
        if sampling_method == "consistency":
            video_latents, action_latents = sample_joint_consistency_noise(
                video_shape,
                action_shape,
                seed=seed,
                device=device,
                dtype=dtype,
            )
        elif rand_device is not None:
            video_latents = _sample_euler_noise(
                video_shape,
                seed=seed,
                rand_device=rand_device,
                device=device,
                dtype=dtype,
            )
            action_latents = _sample_euler_noise(
                action_shape,
                seed=seed,
                rand_device=rand_device,
                device=device,
                dtype=dtype,
            )
        else:
            generator = (
                torch.Generator(device=device).manual_seed(seed)
                if seed is not None
                else None
            )
            video_latents = torch.randn(
                video_shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            action_latents = torch.randn(
                action_shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        video_latents[:, :, 0:1] = first_frame_latents

        video_timesteps, video_deltas = build_inference_schedule(
            self.config.video_scheduler,
            num_inference_steps,
            device,
            dtype,
        )
        action_timesteps, action_deltas = build_inference_schedule(
            self.config.action_scheduler,
            num_inference_steps,
            device,
            dtype,
        )
        video_dit = self.backbone.get_dit()

        for i in range(num_inference_steps):
            t_video = video_timesteps[i].reshape(1)
            video_token_timesteps = None
            if sampling_method == "consistency":
                video_token_timesteps, t_action = _consistency_timesteps(
                    video_timesteps[i],
                    action_timesteps[i],
                    latent_steps=latent_steps,
                    action_horizon=action_horizon,
                )
            else:
                t_action = action_timesteps[i].reshape(1)
            video_state = video_dit.pre_dit(video_latents, t_video, context, context_mask)
            action_state = self.action_expert.pre_dit(action_latents, t_action, context, context_mask)
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_state["tokens"].shape[1],
                action_seq_len=action_state["tokens"].shape[1],
                video_tokens_per_frame=self._compute_video_tokens_per_frame(video_state),
                device=device,
                action_video_conditioning=self.action_video_conditioning,
            )
            output_tokens = self.mot({"video": video_state, "action": action_state}, attention_mask)
            video_velocity = video_dit.post_dit(output_tokens["video"], video_state["meta"], video_state["t"])
            action_velocity = self.action_expert.post_dit(output_tokens["action"])
            if sampling_method == "euler":
                video_latents = FlowMatchScheduler.step(video_velocity, video_deltas[i], video_latents)
                action_latents = FlowMatchScheduler.step(action_velocity, action_deltas[i], action_latents)
            else:
                assert video_token_timesteps is not None
                sigma_video = video_token_timesteps / float(self.video_scheduler.num_train_timesteps)
                sigma_action = t_action / float(self.action_scheduler.num_train_timesteps)
                video_latents = video_consistency_boundary(
                    video_latents, video_velocity, sigma_video
                )
                action_latents = action_consistency_boundary(
                    action_latents, action_velocity, sigma_action
                )
            video_latents[:, :, 0:1] = first_frame_latents

        out: dict[str, Any] = {"action": action_latents}
        if bool(kwargs.get("decode_video", True)):
            out["video"] = self.backbone.decode_latents(video_latents)
        return out

    def _append_proprio_from_sample(
        self,
        sample: dict[str, Any],
        context: Tensor,
        context_mask: Optional[Tensor],
    ) -> tuple[Tensor, Optional[Tensor]]:
        if self.proprio_encoder is None:
            return context, context_mask
        proprio = sample.get("proprio")
        if proprio is None:
            raise ValueError("sample['proprio'] is required when framework.proprio_dim is enabled")
        if proprio.ndim != 3:
            raise ValueError(f"sample['proprio'] must be [B, T, D], got {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"sample['proprio'] last dim must be {self.proprio_dim}, got {proprio.shape[-1]}"
            )
        return self._append_proprio_to_context(context, context_mask, proprio[:, 0, :])

    def _append_proprio_to_context(
        self,
        context: Tensor,
        context_mask: Optional[Tensor],
        proprio: Optional[Tensor],
    ) -> tuple[Tensor, Optional[Tensor]]:
        if self.proprio_encoder is None:
            return context, context_mask
        if proprio is None:
            raise ValueError("proprio is required when framework.proprio_dim is enabled")
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 2:
            raise ValueError(f"proprio must be [B, D] or [D], got {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(f"proprio last dim must be {self.proprio_dim}, got {proprio.shape[-1]}")
        if context_mask is None:
            context_mask = torch.ones(context.shape[:2], dtype=torch.bool, device=context.device)
        encoder_param = next(self.proprio_encoder.parameters())
        proprio_token = self.proprio_encoder(
            proprio.to(device=encoder_param.device, dtype=encoder_param.dtype).unsqueeze(1)
        ).to(device=context.device, dtype=context.dtype)
        proprio_mask = torch.ones((context.shape[0], 1), dtype=torch.bool, device=context.device)
        return torch.cat([context, proprio_token], dim=1), torch.cat([context_mask, proprio_mask], dim=1)

    def _compute_video_tokens_per_frame(self, video_state: dict) -> int:
        meta = video_state.get("meta", {})
        height = int(meta.get("H", 0))
        width = int(meta.get("W", 0))
        _, patch_h, patch_w = self.backbone.info.patch_size
        return max(1, height // patch_h) * max(1, width // patch_w)

    @staticmethod
    def _build_mot_attention_mask(
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        action_video_conditioning: str = "first_frame",
    ) -> Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        first_frame = min(max(video_tokens_per_frame, 1), video_seq_len)

        video_block = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
        video_block[:first_frame, first_frame:] = False
        mask[:video_seq_len, :video_seq_len] = video_block

        mask[video_seq_len:, video_seq_len:] = True
        if action_video_conditioning == "first_frame":
            mask[video_seq_len:, :first_frame] = True
        elif action_video_conditioning == "full_video":
            mask[video_seq_len:, :video_seq_len] = True
        else:
            raise ValueError(
                "action_video_conditioning must be 'first_frame' or 'full_video', "
                f"got {action_video_conditioning!r}"
            )
        return mask

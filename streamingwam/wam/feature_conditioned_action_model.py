"""Feature-conditioned action model."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from streamingwam.action_model import build_action_expert, load_action_dit_init
from streamingwam.backbone.base import BaseBackbone
from streamingwam.config import FrameworkConfig
from streamingwam.modules.scheduler import FlowMatchScheduler
from streamingwam.modules.wan_block import sinusoidal_embedding_1d
from streamingwam.training.flow import add_flow_noise, build_inference_schedule
from streamingwam.training.loss import flow_matching_loss
from streamingwam.training.metrics import action_monitor_metrics
from streamingwam.wam.base import WAMModel


class FeatureConditionedActionModel(WAMModel):
    """Action flow model conditioned on intermediate video-DiT features."""

    taxonomy_model_family = "feature_conditioned_action_model"

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

        self.feature_input = getattr(config, "feature_condition_input", "observation")
        self.feature_noise = getattr(config, "feature_condition_noise", "none")
        self.feature_layer = int(getattr(config, "feature_condition_layer", -1))
        self.feature_num_tokens = getattr(config, "feature_condition_num_tokens", None)
        self.feature_include_text = bool(getattr(config, "feature_condition_include_text", True))
        self.feature_include_timestep = bool(getattr(config, "feature_condition_include_timestep", True))
        self.feature_train_backbone = bool(getattr(config, "feature_condition_train_backbone", False))
        self.feature_pin_first_latent_step = bool(getattr(config, "feature_condition_pin_first_latent_step", True))
        self.feature_inference_video_steps = getattr(config, "feature_condition_inference_video_steps", None)

        if self.feature_input not in {"observation", "ground_truth_video", "generated_video"}:
            raise ValueError(
                "framework.feature_condition_input must be 'observation', 'ground_truth_video', "
                f"or 'generated_video', got {self.feature_input!r}"
            )
        if self.feature_noise not in {"none", "random_flow_noise", "scheduler"}:
            raise ValueError(
                "framework.feature_condition_noise must be 'none', 'random_flow_noise', or 'scheduler', "
                f"got {self.feature_noise!r}"
            )
        if self.feature_input == "observation" and self.feature_noise != "none":
            raise ValueError("feature_condition_input='observation' expects feature_condition_noise='none'")
        if self.feature_input == "ground_truth_video" and self.feature_noise == "scheduler":
            raise ValueError("feature_condition_noise='scheduler' is only valid for generated-video inference")

        self.action_expert = build_action_expert(backbone.info, config)
        load_action_dit_init(
            self.action_expert,
            getattr(config, "action_expert_init_from", None),
            head_init=getattr(config, "action_expert_head_init", "random"),
        )

        self.video_scheduler = FlowMatchScheduler(
            num_train_timesteps=config.video_scheduler.num_train_timesteps,
            shift=config.video_scheduler.train_shift,
        )
        self.action_scheduler = FlowMatchScheduler(
            num_train_timesteps=config.action_scheduler.num_train_timesteps,
            shift=config.action_scheduler.train_shift,
        )
        self.loss_lambda_action = float(config.loss_lambda_action)

        self.feature_projector = nn.Linear(backbone.info.hidden_dim, backbone.info.text_dim)
        self.feature_timestep_embedding = nn.Sequential(
            nn.Linear(backbone.info.freq_dim, backbone.info.text_dim),
            nn.SiLU(),
            nn.Linear(backbone.info.text_dim, backbone.info.text_dim),
        )

        self.proprio_dim = None
        self.proprio_encoder = None
        if getattr(config, "proprio_dim", None):
            self.proprio_dim = int(config.proprio_dim)
            if self.proprio_dim > 0:
                self.proprio_encoder = nn.Linear(self.proprio_dim, backbone.info.text_dim)
            else:
                self.proprio_dim = None

        self.backbone.get_dit().to(device=device, dtype=dtype)
        self.action_expert.to(device=device, dtype=dtype)
        self.feature_projector.to(device=device, dtype=dtype)
        self.feature_timestep_embedding.to(device=device, dtype=dtype)
        if self.proprio_encoder is not None:
            self.proprio_encoder.to(device=device, dtype=dtype)
        if not self.feature_train_backbone:
            self.backbone.get_dit().requires_grad_(False)

    def training_step(self, sample: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        video = sample["video"]
        action = sample["action"]
        context = sample["context"]
        context_mask = sample.get("context_mask")
        action_is_pad = sample.get("action_is_pad")
        context, context_mask = self._append_proprio_from_sample(sample, context, context_mask)

        action_context, action_context_mask, feature_t = self._build_training_context(video, context, context_mask)
        noisy_action, target_action, t_action = add_flow_noise(self.action_scheduler, action)
        pred_action = self.action_expert(noisy_action, t_action, action_context, action_context_mask)

        loss_action = flow_matching_loss(
            pred_action,
            target_action,
            t_action,
            self.action_scheduler,
            is_pad_mask=action_is_pad,
        )
        loss = self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": 0.0,
            "loss_action": loss_action.item(),
            "loss_total": loss.item(),
            "feature_timestep_mean": float(feature_t.detach().float().mean().item()),
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
        device = input_image.device
        dtype = input_image.dtype
        context, context_mask = self._append_proprio_to_context(context, context_mask, kwargs.get("proprio"))
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

        if self.feature_input == "generated_video" or self.feature_noise == "scheduler":
            num_video_frames = kwargs.get("num_video_frames")
            if num_video_frames is None:
                raise ValueError("num_video_frames is required for generated-video feature inference")
            features, feature_t = self._generate_video_features(
                input_image,
                context,
                context_mask,
                int(num_video_frames),
                generator,
            )
        else:
            video = input_image.unsqueeze(2)
            video_latents = self.backbone.encode_video(video)
            feature_t = torch.zeros(video_latents.shape[0], device=device, dtype=dtype)
            features = self._extract_video_features(video_latents, feature_t, context, context_mask)

        action_context, action_context_mask = self._compose_action_context(features, feature_t, context, context_mask)
        action_latents = torch.randn(
            input_image.shape[0],
            action_horizon,
            self.config.action_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        timesteps, deltas = build_inference_schedule(
            self.config.action_scheduler,
            num_inference_steps,
            device,
            dtype,
        )
        for i in range(num_inference_steps):
            t_action = timesteps[i].expand(input_image.shape[0])
            velocity = self.action_expert(action_latents, t_action, action_context, action_context_mask)
            action_latents = FlowMatchScheduler.step(velocity, deltas[i], action_latents)
        return action_latents

    def _build_training_context(
        self,
        video: Tensor,
        context: Tensor,
        context_mask: Optional[Tensor],
    ) -> tuple[Tensor, Optional[Tensor], Tensor]:
        with torch.no_grad():
            if self.feature_input == "observation":
                feature_video = video[:, :, :1]
            elif self.feature_input == "ground_truth_video":
                feature_video = video
            elif self.feature_input == "generated_video":
                raise ValueError("feature_condition_input='generated_video' is only supported for inference")
            else:
                raise AssertionError(f"Unhandled feature input: {self.feature_input}")
            video_latents = self.backbone.encode_video(feature_video)

        if self.feature_noise == "none":
            feature_latents = video_latents
            feature_t = torch.zeros(video_latents.shape[0], device=video_latents.device, dtype=video_latents.dtype)
        elif self.feature_noise == "random_flow_noise":
            feature_latents, _, feature_t = add_flow_noise(
                self.video_scheduler,
                video_latents,
                pin_first_latent_step=self.feature_pin_first_latent_step,
            )
        else:
            raise AssertionError(f"Unhandled feature noise policy: {self.feature_noise}")

        if self.feature_train_backbone:
            features = self._extract_video_features(feature_latents, feature_t, context, context_mask)
        else:
            with torch.no_grad():
                features = self._extract_video_features(feature_latents, feature_t, context, context_mask)
        return (*self._compose_action_context(features, feature_t, context, context_mask), feature_t)

    def _extract_video_features(
        self,
        video_latents: Tensor,
        timestep: Tensor,
        context: Tensor,
        context_mask: Optional[Tensor],
    ) -> Tensor:
        video_dit = self.backbone.get_dit()
        state = video_dit.pre_dit(video_latents, timestep, context, context_mask)
        features, _ = self._run_video_blocks(state, run_to_end=False)
        return features

    def _run_video_blocks(self, state: dict[str, Tensor], run_to_end: bool) -> tuple[Tensor, Tensor]:
        video_dit = self.backbone.get_dit()
        tokens = state["tokens"]
        freqs = state["freqs"]
        t_mod = state["t_mod"]
        ctx = state["context"]
        ctx_mask = state["context_mask"]
        num_blocks = len(video_dit.blocks)
        layer_idx = self.feature_layer if self.feature_layer >= 0 else num_blocks + self.feature_layer
        if layer_idx < 0 or layer_idx >= num_blocks:
            raise ValueError(
                f"feature_condition_layer={self.feature_layer} resolves to {layer_idx}, "
                f"but the video DiT has {num_blocks} blocks"
            )
        features = None
        for idx, block in enumerate(video_dit.blocks):
            tokens = block(tokens, ctx, t_mod, freqs, ctx_mask)
            if idx == layer_idx:
                features = tokens
                if not run_to_end:
                    break
        if features is None:
            raise RuntimeError("Failed to capture video feature tokens")
        return features, tokens

    def _compose_action_context(
        self,
        features: Tensor,
        feature_t: Tensor,
        context: Tensor,
        context_mask: Optional[Tensor],
    ) -> tuple[Tensor, Optional[Tensor]]:
        features = self._pool_features(features)
        projector_param = self.feature_projector.weight
        feature_context = self.feature_projector(features.to(device=projector_param.device, dtype=projector_param.dtype))
        feature_context = feature_context.to(device=context.device, dtype=context.dtype)
        parts = []
        masks = []
        if self.feature_include_text:
            parts.append(context)
            if context_mask is not None:
                masks.append(context_mask)
        if self.feature_include_timestep:
            t_token = self._feature_timestep_token(feature_t, context)
            parts.append(t_token)
            if context_mask is not None:
                masks.append(torch.ones((context.shape[0], 1), dtype=torch.bool, device=context.device))
        parts.append(feature_context)
        if context_mask is not None:
            masks.append(torch.ones(feature_context.shape[:2], dtype=torch.bool, device=context.device))
        action_context = torch.cat(parts, dim=1)
        action_context_mask = torch.cat(masks, dim=1) if context_mask is not None else None
        return action_context, action_context_mask

    def _pool_features(self, features: Tensor) -> Tensor:
        if self.feature_num_tokens is None:
            return features
        num_tokens = int(self.feature_num_tokens)
        if num_tokens <= 0 or features.shape[1] <= num_tokens:
            return features
        return F.adaptive_avg_pool1d(features.transpose(1, 2), num_tokens).transpose(1, 2)

    def _feature_timestep_token(self, feature_t: Tensor, context: Tensor) -> Tensor:
        t = feature_t.reshape(-1).to(device=context.device, dtype=torch.float32)
        emb = sinusoidal_embedding_1d(self.backbone.info.freq_dim, t)
        token = self.feature_timestep_embedding(emb.to(dtype=next(self.feature_timestep_embedding.parameters()).dtype))
        return token.to(device=context.device, dtype=context.dtype).unsqueeze(1)

    def _generate_video_features(
        self,
        input_image: Tensor,
        context: Tensor,
        context_mask: Optional[Tensor],
        num_video_frames: int,
        generator: Optional[torch.Generator],
    ) -> tuple[Tensor, Tensor]:
        device = input_image.device
        dtype = input_image.dtype
        video_single = input_image.unsqueeze(2)
        first_frame_latents = self.backbone.encode_video(video_single)
        _, channels, _, height_latent, width_latent = first_frame_latents.shape
        vae = self.backbone.get_vae()
        if vae is None:
            raise RuntimeError("Generated-video feature inference requires a loaded VAE")
        latent_steps = max(1, (num_video_frames - 1) // vae.temporal_compress + 1)
        video_latents = torch.randn(
            input_image.shape[0],
            channels,
            latent_steps,
            height_latent,
            width_latent,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        video_latents[:, :, 0:1] = first_frame_latents
        num_steps = int(self.feature_inference_video_steps or 8)
        timesteps, deltas = build_inference_schedule(self.config.video_scheduler, num_steps, device, dtype)
        video_dit = self.backbone.get_dit()
        features = None
        feature_t = None
        for i in range(num_steps):
            feature_t = timesteps[i].expand(input_image.shape[0])
            state = video_dit.pre_dit(video_latents, feature_t, context, context_mask)
            features, tokens = self._run_video_blocks(state, run_to_end=True)
            velocity = video_dit.post_dit(tokens, state["meta"], state["t"])
            video_latents = FlowMatchScheduler.step(velocity, deltas[i], video_latents)
            video_latents[:, :, 0:1] = first_frame_latents
        if features is None or feature_t is None:
            raise RuntimeError("Generated-video feature extraction produced no features")
        return features, feature_t

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
            raise ValueError(f"sample['proprio'] last dim must be {self.proprio_dim}, got {proprio.shape[-1]}")
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

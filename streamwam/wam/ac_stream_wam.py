"""FastWAM AC-Stream checkpoint-specific MoT WAM variant."""

from __future__ import annotations

import torch
from torch import Tensor

from streamwam.inference.consistency import (
    action_consistency_boundary,
    normalize_sampling_method,
    sample_joint_consistency_noise,
    video_consistency_boundary,
)
from streamwam.inference.ac_stream import (
    AC_STREAM_ACTION_HORIZON,
    AC_STREAM_CONDITION_SLOTS,
    AC_STREAM_DELAY,
    AC_STREAM_LAUNCH_AFTER_STEPS,
    AC_STREAM_STRIDE,
    ACStreamAccelerationRuntime,
    apply_ac_stream_hard_prefix_,
    validate_ac_stream_geometry,
)
from streamwam.modules.ac_stream import ACStreamMoT, ACStreamSlotEncoder
from streamwam.modules.ac_stream import build_ac_stream_condition_mask, build_ac_stream_policy_mask
from streamwam.modules.ac_stream import (
    build_starwam_rtc_condition_mask,
    build_starwam_rtc_policy_mask,
)
from streamwam.training.flow import build_inference_schedule
from streamwam.wam.mot_wam import MoTWAM


AC_STREAM_SLOT_ENCODER_NAME = "rtc_1step_selfatt_z1_slot_encoder"
STARWAM_RTC_ARCHITECTURE = "starwam_rtc_h32_s16_d8_z1_method3_v2"
FASTWAM_AC_STREAM_ARCHITECTURE = "fastwam_stage2_selfatt_z1"


def validate_ac_stream_accelerated_contract(
    *,
    input_image: Tensor,
    video_expert: torch.nn.Module,
    action_expert: torch.nn.Module,
    expected_image_shape: tuple[int, int, int] = (3, 224, 448),
) -> None:
    """Reject inputs that cannot reuse the fixed wyx Stage-2 z1 graph."""

    from streamwam.backbone.wan22 import Wan22Dit

    if input_image.dtype != torch.bfloat16:
        raise ValueError(
            "AC-Stream accelerated inference requires BF16 input images, "
            f"got {input_image.dtype}"
        )
    if int(input_image.shape[0]) != 1:
        raise ValueError(
            "AC-Stream accelerated inference requires batch size 1, "
            f"got {input_image.shape[0]}"
        )
    if input_image.ndim != 4 or tuple(input_image.shape[1:]) != tuple(expected_image_shape):
        expected_label = "x".join(str(value) for value in expected_image_shape)
        raise ValueError(
            f"AC-Stream accelerated inference requires {expected_label} RGB "
            f"input, got {tuple(input_image.shape)}"
        )
    if not isinstance(video_expert, Wan22Dit) or (
        int(getattr(video_expert, "hidden_dim", 0)) != 3072
        or int(getattr(video_expert, "num_layers", 0)) != 30
        or int(getattr(video_expert, "num_heads", 0)) != 24
    ):
        raise ValueError(
            "AC-Stream accelerated inference requires the Wan2.2 5B video expert"
        )
    parameter = next(video_expert.parameters(), None)
    if parameter is None or parameter.dtype != torch.bfloat16:
        actual = None if parameter is None else parameter.dtype
        raise ValueError(
            "AC-Stream accelerated inference requires BF16 parameters, "
            f"got {actual}"
        )
    action_parameter = next(action_expert.parameters(), None)
    if action_parameter is None or action_parameter.dtype != torch.bfloat16:
        actual = None if action_parameter is None else action_parameter.dtype
        raise ValueError(
            "AC-Stream accelerated inference requires ActionDiT BF16 parameters, "
            f"got {actual}"
        )


class ACStreamWAM(MoTWAM):
    """MoT WAM carrying the Stage-2 z1 slot parameters used by AC-Stream."""

    inference_variant = "ac-stream"

    def __init__(self, backbone, config, device: str = "cpu", dtype=torch.bfloat16) -> None:
        super().__init__(backbone, config, device=device, dtype=dtype)
        self.ac_stream_architecture = str(
            getattr(config, "ac_stream_architecture", FASTWAM_AC_STREAM_ARCHITECTURE)
        ).strip().lower()
        if self.ac_stream_architecture not in {
            FASTWAM_AC_STREAM_ARCHITECTURE,
            STARWAM_RTC_ARCHITECTURE,
        }:
            raise ValueError(
                f"Unsupported AC-Stream architecture {self.ac_stream_architecture!r}"
            )
        video_expert = self.backbone.get_dit()
        architecture = getattr(
            self,
            "ac_stream_architecture",
            FASTWAM_AC_STREAM_ARCHITECTURE,
        )
        if architecture == STARWAM_RTC_ARCHITECTURE:
            self.rtc_slot_state_embedding = torch.nn.Embedding(
                2,
                self.action_expert.hidden_dim,
                device=device,
                dtype=dtype,
            )
            torch.nn.init.normal_(
                self.rtc_slot_state_embedding.weight,
                std=self.action_expert.hidden_dim**-0.5,
            )
        else:
            slot_encoder = ACStreamSlotEncoder(self.action_expert.hidden_dim).to(
                device=device,
                dtype=dtype,
            )
            setattr(video_expert, AC_STREAM_SLOT_ENCODER_NAME, slot_encoder)
        self.mot = ACStreamMoT(
            experts={"video": video_expert, "action": self.action_expert},
            checkpoint_mixed_attn=config.mot_checkpoint_mixed_attn,
        ).to(device=device, dtype=dtype)
        self._ac_stream_acceleration: ACStreamAccelerationRuntime | None = None

    def enable_ac_stream_acceleration(self) -> None:
        if self._ac_stream_acceleration is not None:
            return
        parameter = next(self.parameters(), None)
        if parameter is None or parameter.device.type != "cuda":
            raise RuntimeError("AC-Stream acceleration requires a CUDA model")
        self._ac_stream_acceleration = ACStreamAccelerationRuntime()

    def mark_ac_stream_prewarmed(self, delay: int) -> None:
        runtime = self._ac_stream_acceleration
        if runtime is None:
            raise RuntimeError("AC-Stream acceleration is not enabled")
        runtime.mark_prewarmed(delay)

    @property
    def ac_stream_prewarm_complete(self) -> bool:
        runtime = getattr(self, "_ac_stream_acceleration", None)
        return bool(runtime is not None and runtime.prewarm_complete)

    def ac_stream_acceleration_status(self) -> dict[str, object]:
        runtime = getattr(self, "_ac_stream_acceleration", None)
        if runtime is None:
            return {"backend": "eager", "compile_active": False}
        return runtime.status()

    def _build_ac_stream_condition_state(
        self,
        clean_action_prefix: Tensor,
        *,
        known_prefix_length: int,
        context: Tensor,
        context_mask: Tensor | None,
        projected_context: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch = clean_action_prefix.shape[0]
        condition_actions = clean_action_prefix.new_zeros(
            batch,
            AC_STREAM_CONDITION_SLOTS,
            int(self.config.action_dim),
        )
        if known_prefix_length:
            condition_actions[:, :AC_STREAM_DELAY] = clean_action_prefix
        condition_timestep = clean_action_prefix.new_zeros(
            batch,
            AC_STREAM_CONDITION_SLOTS,
        )
        condition_kwargs = {}
        if projected_context is not None:
            condition_kwargs["projected_context"] = projected_context
        state = self.action_expert.pre_dit(
            condition_actions,
            condition_timestep,
            context,
            context_mask,
            **condition_kwargs,
        )
        known_mask = torch.arange(
            AC_STREAM_CONDITION_SLOTS,
            device=state["tokens"].device,
        ).unsqueeze(0) < int(known_prefix_length)
        known_mask = known_mask.expand(batch, -1)
        architecture = getattr(
            self,
            "ac_stream_architecture",
            FASTWAM_AC_STREAM_ARCHITECTURE,
        )
        if architecture == STARWAM_RTC_ARCHITECTURE:
            content = torch.where(
                known_mask.unsqueeze(-1),
                state["tokens"],
                torch.zeros_like(state["tokens"]),
            )
            state["tokens"] = content + self.rtc_slot_state_embedding(
                known_mask.to(torch.long)
            ).to(content.dtype)
        else:
            slot_encoder = getattr(
                self.backbone.get_dit(),
                AC_STREAM_SLOT_ENCODER_NAME,
                None,
            )
            if not isinstance(slot_encoder, ACStreamSlotEncoder):
                raise RuntimeError("AC-Stream slot encoder is not installed on the video expert")
            state["tokens"] = slot_encoder(state["tokens"], known_mask)
        return state

    @torch.inference_mode()
    def infer_action(
        self,
        input_image: Tensor,
        context: Tensor,
        context_mask: Tensor,
        action_horizon: int,
        num_inference_steps: int = 1,
        seed: int | None = None,
        **kwargs,
    ) -> Tensor:
        method = normalize_sampling_method(kwargs.get("sampling_method", "ac-stream"))
        if method != "ac-stream":
            raise ValueError(
                f"ACStreamWAM requires sampling_method='ac-stream', got {method!r}"
            )
        architecture = getattr(
            self,
            "ac_stream_architecture",
            FASTWAM_AC_STREAM_ARCHITECTURE,
        )
        num_video_frames = int(kwargs.get("num_video_frames", 0))
        validate_ac_stream_geometry(
            action_horizon=action_horizon,
            stride=int(kwargs.get("ac_stream_stride", AC_STREAM_STRIDE)),
            delay=int(kwargs.get("ac_stream_delay", AC_STREAM_DELAY)),
            launch_after_steps=int(
                kwargs.get("ac_stream_launch_after_steps", AC_STREAM_LAUNCH_AFTER_STEPS)
            ),
            num_video_frames=num_video_frames,
            temporal_compress=int(self.backbone.get_vae().temporal_compress),
            num_inference_steps=num_inference_steps,
        )
        if int(self.config.chunk_size) != AC_STREAM_ACTION_HORIZON:
            raise ValueError(
                f"AC-Stream config requires chunk_size={AC_STREAM_ACTION_HORIZON}, "
                f"got {self.config.chunk_size}"
            )

        previous_action_chunk = kwargs.get("ac_stream_prev_action_chunk")
        phase_delay = int(kwargs.get("ac_stream_inference_delay", 0))
        if previous_action_chunk is None:
            if phase_delay != 0:
                raise ValueError("AC-Stream D0 requires ac_stream_inference_delay=0")
            known_prefix_length = 0
        else:
            if phase_delay != AC_STREAM_DELAY:
                raise ValueError(
                    f"AC-Stream D8 requires ac_stream_inference_delay={AC_STREAM_DELAY}, "
                    f"got {phase_delay}"
                )
            known_prefix_length = AC_STREAM_DELAY

        device = input_image.device
        dtype = input_image.dtype
        acceleration = getattr(self, "_ac_stream_acceleration", None)
        if acceleration is not None:
            expected_image_shape = (
                (3, 384, 320)
                if architecture == STARWAM_RTC_ARCHITECTURE
                else (3, 224, 448)
            )
            validate_ac_stream_accelerated_contract(
                input_image=input_image,
                video_expert=self.backbone.get_dit(),
                action_expert=self.action_expert,
                expected_image_shape=expected_image_shape,
            )
        static_context = context
        context, context_mask = self._append_proprio_to_context(
            context,
            context_mask,
            kwargs.get("proprio"),
        )
        projected_contexts = None
        if acceleration is not None:
            context_key = kwargs.get("ac_stream_context_key")
            if not isinstance(context_key, str) or not context_key:
                raise ValueError(
                    "AC-Stream accelerated inference requires ac_stream_context_key"
                )
            dynamic_context = context[:, static_context.shape[1] :]
            if dynamic_context.shape[1] != 1:
                raise ValueError(
                    "AC-Stream acceleration requires exactly one dynamic proprio token, "
                    f"got {dynamic_context.shape[1]}"
                )
            projected_contexts = acceleration.prepare_contexts(
                context_key=context_key,
                static_context=static_context,
                dynamic_context=dynamic_context,
                video_expert=self.backbone.get_dit(),
                action_expert=self.action_expert,
            )
        first_frame_latents = self.backbone.encode_video(input_image.unsqueeze(2))
        batch, channels, _, height, width = first_frame_latents.shape
        latent_steps = (num_video_frames - 1) // int(
            self.backbone.get_vae().temporal_compress
        ) + 1
        if architecture == STARWAM_RTC_ARCHITECTURE:
            generator = (
                None
                if seed is None
                else torch.Generator(device=device).manual_seed(seed)
            )
            video_latents = torch.randn(
                (batch, channels, latent_steps, height, width),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            action_latents = torch.randn(
                (batch, AC_STREAM_ACTION_HORIZON, int(self.config.action_dim)),
                generator=generator,
                device=device,
                dtype=dtype,
            )
        else:
            video_latents, action_latents = sample_joint_consistency_noise(
                (batch, channels, latent_steps, height, width),
                (batch, AC_STREAM_ACTION_HORIZON, int(self.config.action_dim)),
                seed=seed,
                device=device,
                dtype=dtype,
            )
        video_latents[:, :, :1].copy_(first_frame_latents)

        def video_schedule():
            return build_inference_schedule(
                self.config.video_scheduler,
                num_inference_steps,
                device,
                dtype,
            )

        def action_schedule():
            return build_inference_schedule(
                self.config.action_scheduler,
                num_inference_steps,
                device,
                dtype,
            )

        if acceleration is None:
            video_timesteps, _ = video_schedule()
            action_timesteps, _ = action_schedule()
        else:
            video_timesteps, _ = acceleration.get_schedule(
                (
                    "video",
                    num_inference_steps,
                    str(device),
                    str(dtype),
                    float(self.config.video_scheduler.infer_shift),
                ),
                video_schedule,
            )
            action_timesteps, _ = acceleration.get_schedule(
                (
                    "action",
                    num_inference_steps,
                    str(device),
                    str(dtype),
                    float(self.config.action_scheduler.infer_shift),
                ),
                action_schedule,
            )
        timestep_video = video_timesteps[0].reshape(1).expand(batch)
        timestep_action = action_timesteps[0].reshape(1, 1).expand(
            batch,
            AC_STREAM_ACTION_HORIZON,
        ).clone()
        if previous_action_chunk is None:
            clean_action_prefix = action_latents.new_zeros(
                batch,
                AC_STREAM_DELAY,
                int(self.config.action_dim),
            )
        else:
            clean_action_prefix = apply_ac_stream_hard_prefix_(
                action_latents,
                timestep_action,
                previous_action_chunk,
            )

        video_expert = self.backbone.get_dit()
        video_pre_kwargs = {}
        action_pre_kwargs = {}
        if projected_contexts is not None:
            video_pre_kwargs["projected_context"] = projected_contexts["video"]
            action_pre_kwargs["projected_context"] = projected_contexts["action"]
        video_state = video_expert.pre_dit(
            video_latents,
            timestep_video,
            context,
            context_mask,
            **video_pre_kwargs,
        )
        action_state = self.action_expert.pre_dit(
            action_latents,
            timestep_action,
            context,
            context_mask,
            **action_pre_kwargs,
        )
        condition_state = self._build_ac_stream_condition_state(
            clean_action_prefix,
            known_prefix_length=known_prefix_length,
            context=context,
            context_mask=context_mask,
            projected_context=(
                projected_contexts["action"]
                if projected_contexts is not None
                else None
            ),
        )
        tokens_per_frame = int(video_state["meta"]["tokens_per_frame"])
        video_seq_len = int(video_state["tokens"].shape[1])
        action_seq_len = int(action_state["tokens"].shape[1])
        condition_seq_len = int(condition_state["tokens"].shape[1])

        def policy_mask_builder():
            if architecture == STARWAM_RTC_ARCHITECTURE:
                return build_starwam_rtc_policy_mask(
                    batch_size=batch,
                    video_seq_len=video_seq_len,
                    policy_seq_len=action_seq_len,
                    video_tokens_per_frame=tokens_per_frame,
                    known_prefix_length=known_prefix_length,
                    device=device,
                )
            return build_ac_stream_policy_mask(
                video_seq_len=video_seq_len,
                action_seq_len=action_seq_len,
                video_tokens_per_frame=tokens_per_frame,
                known_prefix_length=known_prefix_length,
                device=device,
            )

        def condition_mask_builder():
            if architecture == STARWAM_RTC_ARCHITECTURE:
                return build_starwam_rtc_condition_mask(
                    batch_size=batch,
                    condition_seq_len=condition_seq_len,
                    video_tokens_per_frame=tokens_per_frame,
                    known_prefix_length=known_prefix_length,
                    device=device,
                )
            return build_ac_stream_condition_mask(
                video_seq_len=video_seq_len,
                condition_seq_len=condition_seq_len,
                video_tokens_per_frame=tokens_per_frame,
                device=device,
            )

        if acceleration is None:
            policy_mask = policy_mask_builder()
            condition_mask = condition_mask_builder()
        else:
            policy_mask = acceleration.get_attention_mask(
                (
                    "policy",
                    architecture,
                    str(device),
                    video_seq_len,
                    action_seq_len,
                    tokens_per_frame,
                    known_prefix_length,
                ),
                policy_mask_builder,
            )
            condition_mask = acceleration.get_attention_mask(
                (
                    "condition",
                    architecture,
                    str(device),
                    video_seq_len,
                    condition_seq_len,
                    tokens_per_frame,
                    known_prefix_length,
                ),
                condition_mask_builder,
            )
        expert_states = {
            "video": video_state,
            "action": action_state,
            "condition": condition_state,
        }
        action_condition_active = torch.full(
            (batch,),
            bool(known_prefix_length),
            dtype=torch.bool,
            device=device,
        )
        video_mask = None
        if architecture == STARWAM_RTC_ARCHITECTURE:
            video_mask = self._build_mot_attention_mask(
                video_seq_len,
                0,
                tokens_per_frame,
                device,
                "first_frame",
            )[:video_seq_len, :video_seq_len]
        if acceleration is None:
            if architecture == STARWAM_RTC_ARCHITECTURE:
                assert video_mask is not None
                output = self.mot.forward_starwam_rtc(
                    expert_states,
                    video_attention_mask=video_mask,
                    policy_attention_mask=policy_mask,
                    condition_attention_mask=condition_mask,
                    video_tokens_per_frame=tokens_per_frame,
                    action_condition_active=action_condition_active,
                )
            else:
                output = self.mot.forward_ac_stream(
                    expert_states,
                    policy_attention_mask=policy_mask,
                    condition_attention_mask=condition_mask,
                    video_tokens_per_frame=tokens_per_frame,
                    action_condition_active=action_condition_active,
                )
        else:
            static_kv = acceleration.static_cross_attention_kv
            static_context_length = acceleration.static_context_length
            compiled_kwargs = {
                "tokens_all": {
                    name: state["tokens"] for name, state in expert_states.items()
                },
                "freqs_all": {
                    name: state["freqs"] for name, state in expert_states.items()
                },
                "t_mod_all": {
                    name: state["t_mod"] for name, state in expert_states.items()
                },
                "context_all": {
                    name: {
                        "context": state["context"],
                        "mask": state["context_mask"],
                        "static_cross_attention_kv": static_kv[
                            "video" if name == "video" else "action"
                        ],
                        "static_context_length": static_context_length,
                    }
                    for name, state in expert_states.items()
                },
                "policy_attention_mask": policy_mask,
                "condition_attention_mask": condition_mask,
                "video_tokens_per_frame": tokens_per_frame,
                "action_condition_active": action_condition_active,
            }
            if video_mask is not None:
                compiled_kwargs["video_attention_mask"] = video_mask
            output = acceleration.run_mot(
                self.mot,
                architecture=architecture,
                **compiled_kwargs,
            )
        video_velocity = video_expert.post_dit(
            output["video"],
            video_state["meta"],
            video_state["t"],
        )
        action_velocity = self.action_expert.post_dit(output["action"])
        if architecture == STARWAM_RTC_ARCHITECTURE:
            # yzy's RoboTwin RTC checkpoint is trained against the direct
            # sigma=1 consistency boundary in float32.  It returns immediately
            # from the action path and clamps the original (not bf16-rounded)
            # D8 prefix after the boundary.
            predicted_action = action_latents.float() - action_velocity.float()
            if previous_action_chunk is not None:
                prefix = previous_action_chunk.to(device=device, dtype=torch.float32)
                if prefix.ndim == 2:
                    prefix = prefix.unsqueeze(0)
                predicted_action[:, :AC_STREAM_DELAY].copy_(
                    prefix[:, :AC_STREAM_DELAY]
                )
            return predicted_action
        video_token_timestep = timestep_video.reshape(batch, 1).expand(
            batch,
            latent_steps,
        ).clone()
        video_token_timestep[:, 0] = 0
        video_latents = video_consistency_boundary(
            video_latents,
            video_velocity,
            video_token_timestep / float(self.video_scheduler.num_train_timesteps),
        )
        action_latents = action_consistency_boundary(
            action_latents,
            action_velocity,
            timestep_action / float(self.action_scheduler.num_train_timesteps),
        )
        video_latents[:, :, :1].copy_(first_frame_latents)
        if previous_action_chunk is not None:
            action_latents[:, :AC_STREAM_DELAY].copy_(clean_action_prefix)
        return action_latents

"""FastWAM RTC-AC checkpoint-specific MoT WAM variant."""

from __future__ import annotations

import torch
from torch import Tensor

from starwam.inference.consistency import (
    action_consistency_boundary,
    normalize_sampling_method,
    sample_joint_consistency_noise,
    video_consistency_boundary,
)
from starwam.inference.rtc_ac import (
    RTC_AC_ACTION_DIM,
    RTC_AC_ACTION_HORIZON,
    RTC_AC_CONDITION_SLOTS,
    RTC_AC_DELAY,
    RTC_AC_LAUNCH_AFTER_STEPS,
    RTC_AC_STRIDE,
    RTCACAccelerationRuntime,
    apply_rtc_ac_hard_prefix_,
    validate_rtc_ac_geometry,
)
from starwam.modules.rtc_ac import RTCACMoT, RTCACSlotEncoder
from starwam.modules.rtc_ac import build_rtc_ac_condition_mask, build_rtc_ac_policy_mask
from starwam.training.flow import build_inference_schedule
from starwam.wam.mot_wam import MoTWAM


RTC_AC_SLOT_ENCODER_NAME = "rtc_1step_selfatt_z1_slot_encoder"


def validate_rtc_ac_accelerated_contract(
    *,
    input_image: Tensor,
    video_expert: torch.nn.Module,
    action_expert: torch.nn.Module,
) -> None:
    """Reject inputs that cannot reuse the fixed wyx Stage-2 z1 graph."""

    from starwam.backbone.wan22 import Wan22Dit

    if input_image.dtype != torch.bfloat16:
        raise ValueError(
            "RTC-AC accelerated inference requires BF16 input images, "
            f"got {input_image.dtype}"
        )
    if int(input_image.shape[0]) != 1:
        raise ValueError(
            "RTC-AC accelerated inference requires batch size 1, "
            f"got {input_image.shape[0]}"
        )
    if input_image.ndim != 4 or tuple(input_image.shape[1:]) != (3, 224, 448):
        raise ValueError(
            "RTC-AC accelerated inference requires concatenated 224x448 RGB "
            f"input, got {tuple(input_image.shape)}"
        )
    if not isinstance(video_expert, Wan22Dit) or (
        int(getattr(video_expert, "hidden_dim", 0)) != 3072
        or int(getattr(video_expert, "num_layers", 0)) != 30
        or int(getattr(video_expert, "num_heads", 0)) != 24
    ):
        raise ValueError(
            "RTC-AC accelerated inference requires the Wan2.2 5B video expert"
        )
    parameter = next(video_expert.parameters(), None)
    if parameter is None or parameter.dtype != torch.bfloat16:
        actual = None if parameter is None else parameter.dtype
        raise ValueError(
            "RTC-AC accelerated inference requires BF16 parameters, "
            f"got {actual}"
        )
    action_parameter = next(action_expert.parameters(), None)
    if action_parameter is None or action_parameter.dtype != torch.bfloat16:
        actual = None if action_parameter is None else action_parameter.dtype
        raise ValueError(
            "RTC-AC accelerated inference requires ActionDiT BF16 parameters, "
            f"got {actual}"
        )


class RTCACWAM(MoTWAM):
    """MoT WAM carrying the Stage-2 z1 slot parameters used by RTC-AC."""

    inference_variant = "rtc_ac"

    def __init__(self, backbone, config, device: str = "cpu", dtype=torch.bfloat16) -> None:
        super().__init__(backbone, config, device=device, dtype=dtype)
        video_expert = self.backbone.get_dit()
        slot_encoder = RTCACSlotEncoder(self.action_expert.hidden_dim).to(
            device=device,
            dtype=dtype,
        )
        setattr(video_expert, RTC_AC_SLOT_ENCODER_NAME, slot_encoder)
        self.mot = RTCACMoT(
            experts={"video": video_expert, "action": self.action_expert},
            checkpoint_mixed_attn=config.mot_checkpoint_mixed_attn,
        ).to(device=device, dtype=dtype)
        self._rtc_ac_acceleration: RTCACAccelerationRuntime | None = None

    def enable_rtc_ac_acceleration(self) -> None:
        if self._rtc_ac_acceleration is not None:
            return
        parameter = next(self.parameters(), None)
        if parameter is None or parameter.device.type != "cuda":
            raise RuntimeError("RTC-AC acceleration requires a CUDA model")
        self._rtc_ac_acceleration = RTCACAccelerationRuntime()

    def mark_rtc_ac_prewarmed(self, delay: int) -> None:
        runtime = self._rtc_ac_acceleration
        if runtime is None:
            raise RuntimeError("RTC-AC acceleration is not enabled")
        runtime.mark_prewarmed(delay)

    @property
    def rtc_ac_prewarm_complete(self) -> bool:
        runtime = getattr(self, "_rtc_ac_acceleration", None)
        return bool(runtime is not None and runtime.prewarm_complete)

    def rtc_ac_acceleration_status(self) -> dict[str, object]:
        runtime = getattr(self, "_rtc_ac_acceleration", None)
        if runtime is None:
            return {"backend": "eager", "compile_active": False}
        return runtime.status()

    def _build_rtc_ac_condition_state(
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
            RTC_AC_CONDITION_SLOTS,
            RTC_AC_ACTION_DIM,
        )
        if known_prefix_length:
            condition_actions[:, :RTC_AC_DELAY] = clean_action_prefix
        condition_timestep = clean_action_prefix.new_zeros(
            batch,
            RTC_AC_CONDITION_SLOTS,
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
            RTC_AC_CONDITION_SLOTS,
            device=state["tokens"].device,
        ).unsqueeze(0) < int(known_prefix_length)
        known_mask = known_mask.expand(batch, -1)
        slot_encoder = getattr(
            self.backbone.get_dit(),
            RTC_AC_SLOT_ENCODER_NAME,
            None,
        )
        if not isinstance(slot_encoder, RTCACSlotEncoder):
            raise RuntimeError("RTC-AC slot encoder is not installed on the video expert")
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
        method = normalize_sampling_method(kwargs.get("sampling_method", "rtc_ac"))
        if method != "rtc_ac":
            raise ValueError(
                f"RTCACWAM requires sampling_method='rtc_ac', got {method!r}"
            )
        num_video_frames = int(kwargs.get("num_video_frames", 0))
        validate_rtc_ac_geometry(
            action_horizon=action_horizon,
            stride=int(kwargs.get("rtc_ac_stride", RTC_AC_STRIDE)),
            delay=int(kwargs.get("rtc_ac_delay", RTC_AC_DELAY)),
            launch_after_steps=int(
                kwargs.get("rtc_ac_launch_after_steps", RTC_AC_LAUNCH_AFTER_STEPS)
            ),
            num_video_frames=num_video_frames,
            temporal_compress=int(self.backbone.get_vae().temporal_compress),
            num_inference_steps=num_inference_steps,
        )
        if int(self.config.chunk_size) != RTC_AC_ACTION_HORIZON:
            raise ValueError(
                f"RTC-AC config requires chunk_size={RTC_AC_ACTION_HORIZON}, "
                f"got {self.config.chunk_size}"
            )

        previous_action_chunk = kwargs.get("rtc_prev_action_chunk")
        phase_delay = int(kwargs.get("rtc_inference_delay", 0))
        if previous_action_chunk is None:
            if phase_delay != 0:
                raise ValueError("RTC-AC D0 requires rtc_inference_delay=0")
            known_prefix_length = 0
        else:
            if phase_delay != RTC_AC_DELAY:
                raise ValueError(
                    f"RTC-AC D8 requires rtc_inference_delay={RTC_AC_DELAY}, "
                    f"got {phase_delay}"
                )
            known_prefix_length = RTC_AC_DELAY

        device = input_image.device
        dtype = input_image.dtype
        acceleration = getattr(self, "_rtc_ac_acceleration", None)
        if acceleration is not None:
            validate_rtc_ac_accelerated_contract(
                input_image=input_image,
                video_expert=self.backbone.get_dit(),
                action_expert=self.action_expert,
            )
        static_context = context
        context, context_mask = self._append_proprio_to_context(
            context,
            context_mask,
            kwargs.get("proprio"),
        )
        projected_contexts = None
        if acceleration is not None:
            context_key = kwargs.get("rtc_ac_context_key")
            if not isinstance(context_key, str) or not context_key:
                raise ValueError(
                    "RTC-AC accelerated inference requires rtc_ac_context_key"
                )
            dynamic_context = context[:, static_context.shape[1] :]
            if dynamic_context.shape[1] != 1:
                raise ValueError(
                    "RTC-AC acceleration requires exactly one dynamic proprio token, "
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
        video_latents, action_latents = sample_joint_consistency_noise(
            (batch, channels, latent_steps, height, width),
            (batch, RTC_AC_ACTION_HORIZON, RTC_AC_ACTION_DIM),
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
            RTC_AC_ACTION_HORIZON,
        ).clone()
        if previous_action_chunk is None:
            clean_action_prefix = action_latents.new_zeros(
                batch,
                RTC_AC_DELAY,
                RTC_AC_ACTION_DIM,
            )
        else:
            clean_action_prefix = apply_rtc_ac_hard_prefix_(
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
        condition_state = self._build_rtc_ac_condition_state(
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
            return build_rtc_ac_policy_mask(
                video_seq_len=video_seq_len,
                action_seq_len=action_seq_len,
                video_tokens_per_frame=tokens_per_frame,
                known_prefix_length=known_prefix_length,
                device=device,
            )

        def condition_mask_builder():
            return build_rtc_ac_condition_mask(
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
                    str(device),
                    video_seq_len,
                    condition_seq_len,
                    tokens_per_frame,
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
        if acceleration is None:
            output = self.mot.forward_rtc_ac(
                expert_states,
                policy_attention_mask=policy_mask,
                condition_attention_mask=condition_mask,
                video_tokens_per_frame=tokens_per_frame,
                action_condition_active=action_condition_active,
            )
        else:
            static_kv = acceleration.static_cross_attention_kv
            static_context_length = acceleration.static_context_length
            output = acceleration.run_mot(
                self.mot,
                tokens_all={
                    name: state["tokens"] for name, state in expert_states.items()
                },
                freqs_all={
                    name: state["freqs"] for name, state in expert_states.items()
                },
                t_mod_all={
                    name: state["t_mod"] for name, state in expert_states.items()
                },
                context_all={
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
                policy_attention_mask=policy_mask,
                condition_attention_mask=condition_mask,
                video_tokens_per_frame=tokens_per_frame,
                action_condition_active=action_condition_active,
            )
        video_velocity = video_expert.post_dit(
            output["video"],
            video_state["meta"],
            video_state["t"],
        )
        action_velocity = self.action_expert.post_dit(output["action"])
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
            action_latents[:, :RTC_AC_DELAY].copy_(clean_action_prefix)
        return action_latents

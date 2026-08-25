from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
import torch

from streamwam.inference import normalize_sampling_method
from streamwam.inference.ac_stream import (
    ACStreamController,
    ACStreamOverlapRecord,
    ACStreamPrediction,
    apply_ac_stream_hard_prefix_,
    build_ac_stream_overlap_record,
    build_ac_stream_prev_action_target,
    validate_ac_stream_geometry,
)
from streamwam.modules.ac_stream import ACStreamSlotEncoder
from streamwam.modules.ac_stream import (
    ACStreamMoT,
    build_ac_stream_condition_mask,
    build_ac_stream_policy_mask,
    build_starwam_rtc_condition_mask,
    build_starwam_rtc_policy_mask,
)


def test_ac_stream_is_a_public_sampling_method() -> None:
    assert normalize_sampling_method("ac-stream") == "ac-stream"
    assert normalize_sampling_method("AC-Stream") == "ac-stream"


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"action_horizon": 16}, "action_horizon=32"),
        ({"stride": 8}, "stride=16"),
        ({"delay": 4}, "delay=8"),
        ({"launch_after_steps": 7}, "launch_after_steps=8"),
        ({"num_video_frames": 17}, "num_video_frames=9"),
        ({"temporal_compress": 2}, "temporal_compress=4"),
        ({"num_inference_steps": 2}, "num_inference_steps=1"),
    ],
)
def test_ac_stream_rejects_checkpoint_incompatible_geometry(
    override: dict[str, int],
    match: str,
) -> None:
    values = {
        "action_horizon": 32,
        "stride": 16,
        "delay": 8,
        "launch_after_steps": 8,
        "num_video_frames": 9,
        "temporal_compress": 4,
        "num_inference_steps": 1,
    }
    values.update(override)
    with pytest.raises(ValueError, match=match):
        validate_ac_stream_geometry(**values)


def test_ac_stream_aligns_d8_prefix_to_launch_cursor() -> None:
    current = torch.arange(32 * 7, dtype=torch.float32).reshape(32, 7)
    target = build_ac_stream_prev_action_target(
        current,
        cursor=8,
        action_horizon=32,
        execute_horizon=16,
    )
    torch.testing.assert_close(target[:16], current[8:24])
    torch.testing.assert_close(target[16:], torch.zeros(16, 7))
    assert target.data_ptr() != current.data_ptr()


def test_ac_stream_hard_prefix_clamps_first_eight_tokens_and_sigma() -> None:
    latents = torch.full((1, 32, 7), -3.0)
    timesteps = torch.full((1, 32), 1000.0)
    prefix_source = torch.arange(32 * 7, dtype=torch.float32).reshape(1, 32, 7)

    clean = apply_ac_stream_hard_prefix_(latents, timesteps, prefix_source)

    torch.testing.assert_close(latents[:, :8], prefix_source[:, :8])
    torch.testing.assert_close(latents[:, 8:], torch.full((1, 24, 7), -3.0))
    torch.testing.assert_close(timesteps[:, :8], torch.zeros((1, 8)))
    torch.testing.assert_close(timesteps[:, 8:], torch.full((1, 24), 1000.0))
    torch.testing.assert_close(clean, prefix_source[:, :8])
    assert clean.data_ptr() != prefix_source.data_ptr()


def test_ac_stream_hard_prefix_supports_robotwin_action_dimension() -> None:
    latents = torch.randn(1, 32, 14)
    timesteps = torch.full((1, 32), 1000.0)
    previous = torch.randn(32, 14)

    clean = apply_ac_stream_hard_prefix_(latents, timesteps, previous)

    assert clean.shape == (1, 8, 14)
    torch.testing.assert_close(latents[:, :8], previous[:8].unsqueeze(0))
    assert bool((timesteps[:, :8] == 0).all())


def test_ac_stream_prediction_supports_robotwin_action_dimension() -> None:
    model = torch.zeros((32, 14))

    prediction = ACStreamPrediction(
        env_actions=model.numpy(),
        model_actions=model,
        communication_ms=1.0,
        inference_ms=2.0,
    )

    assert prediction.model_actions.shape == (32, 14)


def test_ac_stream_slot_encoder_keeps_known_content_and_zeros_unknown_content() -> None:
    encoder = ACStreamSlotEncoder(hidden_dim=2)
    with torch.no_grad():
        encoder.state_embedding.weight.copy_(
            torch.tensor([[10.0, 20.0], [100.0, 200.0]])
        )
    tokens = torch.arange(32, dtype=torch.float32).reshape(1, 16, 2)
    known = torch.zeros((1, 16), dtype=torch.bool)
    known[:, :8] = True

    encoded = encoder(tokens, known)

    torch.testing.assert_close(encoded[:, :8], tokens[:, :8] + torch.tensor([100.0, 200.0]))
    torch.testing.assert_close(
        encoded[:, 8:],
        torch.tensor([10.0, 20.0]).reshape(1, 1, 2).expand(1, 8, 2),
    )


def test_ac_stream_d0_policy_mask_has_no_clean_prefix_restriction() -> None:
    mask = build_ac_stream_policy_mask(
        video_seq_len=6,
        action_seq_len=32,
        video_tokens_per_frame=2,
        known_prefix_length=0,
        device=torch.device("cpu"),
    )
    assert tuple(mask.shape) == (38, 38)
    assert not bool(mask[:6, 6:].any())
    assert bool(mask[6:, :].all())
    assert not bool(mask[:2, 2:6].any())
    assert bool(mask[2:6, :6].all())


def test_starwam_rtc_d0_policy_reads_only_z0_and_policy_stream() -> None:
    mask = build_starwam_rtc_policy_mask(
        batch_size=1,
        video_seq_len=6,
        policy_seq_len=32,
        video_tokens_per_frame=2,
        known_prefix_length=0,
        device=torch.device("cpu"),
    )

    assert mask.shape == (1, 1, 32, 38)
    assert bool(mask[..., :2].all())
    assert not bool(mask[..., 2:6].any())
    assert bool(mask[..., 6:].all())


def test_starwam_rtc_d8_policy_separates_clean_and_noisy_queries() -> None:
    mask = build_starwam_rtc_policy_mask(
        batch_size=1,
        video_seq_len=6,
        policy_seq_len=32,
        video_tokens_per_frame=2,
        known_prefix_length=8,
        device=torch.device("cpu"),
    )

    clean = mask[0, 0, :8]
    noisy = mask[0, 0, 8:]
    assert bool(clean[:, :2].all())
    assert not bool(clean[:, 2:6].any())
    assert bool(clean[:, 6:14].all())
    assert not bool(clean[:, 14:].any())
    assert bool(noisy.all())


def test_starwam_rtc_condition_mask_matches_known_slot_routing() -> None:
    mask = build_starwam_rtc_condition_mask(
        batch_size=1,
        condition_seq_len=16,
        video_tokens_per_frame=2,
        known_prefix_length=8,
        device=torch.device("cpu"),
    )

    assert mask.shape == (1, 1, 16, 18)
    known = mask[0, 0, :8]
    unknown = mask[0, 0, 8:]
    assert bool(known[:, :10].all())
    assert not bool(known[:, 10:].any())
    assert bool(unknown.all())


def test_ac_stream_d8_policy_mask_isolates_clean_prefix() -> None:
    mask = build_ac_stream_policy_mask(
        video_seq_len=6,
        action_seq_len=32,
        video_tokens_per_frame=2,
        known_prefix_length=8,
        device=torch.device("cpu"),
    )
    clean = mask[6:14]
    assert bool(clean[:, :2].all())
    assert not bool(clean[:, 2:6].any())
    assert bool(clean[:, 6:14].all())
    assert not bool(clean[:, 14:].any())
    assert bool(mask[14:, :].all())


def test_ac_stream_condition_mask_injects_slots_only_into_z1() -> None:
    mask = build_ac_stream_condition_mask(
        video_seq_len=6,
        condition_seq_len=16,
        video_tokens_per_frame=2,
        device=torch.device("cpu"),
    )
    assert tuple(mask.shape) == (22, 22)
    assert not bool(mask[:2, 6:].any())
    assert bool(mask[2:4, 6:].all())
    assert not bool(mask[4:6, 6:].any())
    assert bool(mask[6:14, :2].all())
    assert bool(mask[6:14, 6:14].all())
    assert not bool(mask[6:14, 14:].any())
    assert bool(mask[14:, :2].all())
    assert bool(mask[14:, 6:].all())


def test_builder_selects_ac_stream_wam_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    import streamwam.backbone
    import streamwam.wam
    from streamwam.builder import build_framework
    from streamwam.config import StreamWAMConfig

    marker = object()

    class FakeACStreamWAM:
        def __new__(cls, *args, **kwargs):
            return marker

    class FakeStandardWAM:
        def __new__(cls, *args, **kwargs):
            return object()

    monkeypatch.setattr(streamwam.backbone, "build_backbone", lambda *args, **kwargs: object())
    monkeypatch.setattr(streamwam.wam, "ACStreamWAM", FakeACStreamWAM, raising=False)
    monkeypatch.setattr(streamwam.wam, "MoTWAM", FakeStandardWAM)
    config = StreamWAMConfig()
    config.framework.variant = "ac-stream"

    assert build_framework(config) is marker


def test_framework_config_exposes_starwam_rtc_architecture() -> None:
    from streamwam.config import FrameworkConfig

    config = FrameworkConfig(
        variant="ac-stream",
        ac_stream_architecture="starwam_rtc_h32_s16_d8_z1_method3_v2",
    )

    assert config.ac_stream_architecture == "starwam_rtc_h32_s16_d8_z1_method3_v2"


def test_starwam_rtc_model_uses_original_top_level_slot_key(monkeypatch) -> None:
    from streamwam.config import FrameworkConfig
    from streamwam.wam.ac_stream_wam import ACStreamWAM, AC_STREAM_SLOT_ENCODER_NAME
    from streamwam.wam.mot_wam import MoTWAM

    class FakeExpert(torch.nn.Module):
        hidden_dim = 4

        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList()

    class FakeBackbone:
        def __init__(self) -> None:
            self.video = FakeExpert()

        def get_dit(self):
            return self.video

    def fake_base_init(self, backbone, config, device="cpu", dtype=torch.float32):
        del device, dtype
        torch.nn.Module.__init__(self)
        self.backbone = backbone
        self.config = config
        self.action_expert = FakeExpert()

    monkeypatch.setattr(MoTWAM, "__init__", fake_base_init)
    config = FrameworkConfig(
        variant="ac-stream",
        ac_stream_architecture="starwam_rtc_h32_s16_d8_z1_method3_v2",
    )
    model = ACStreamWAM(FakeBackbone(), config, dtype=torch.float32)

    assert "rtc_slot_state_embedding.weight" in model.state_dict()
    assert not hasattr(model.backbone.get_dit(), AC_STREAM_SLOT_ENCODER_NAME)


def test_ac_stream_three_stream_residual_changes_only_z1_video_tokens() -> None:
    class FakeBlock(torch.nn.Module):
        num_heads = 1
        attn_head_dim = 1

        def get_qkv(self, tokens, t_mod, freqs):
            del t_mod, freqs
            return torch.zeros_like(tokens), torch.zeros_like(tokens), tokens

        def post_attention(self, tokens, attn_out, t_mod, context, context_mask):
            del t_mod, context, context_mask
            return tokens + attn_out

    class FakeExpert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([FakeBlock()])

    mot = ACStreamMoT(
        experts={"video": FakeExpert(), "action": FakeExpert()},
        checkpoint_mixed_attn=False,
    )
    video = torch.zeros((1, 6, 1))
    action = torch.zeros((1, 32, 1))
    condition = torch.ones((1, 16, 1))
    common_state = {
        "freqs": torch.zeros((1,)),
        "t_mod": torch.zeros((1,)),
        "context": torch.zeros((1, 1, 1)),
        "context_mask": torch.ones((1, 1), dtype=torch.bool),
    }
    states = {
        "video": {"tokens": video, **common_state},
        "action": {"tokens": action, **common_state},
        "condition": {"tokens": condition, **common_state},
    }
    policy_mask = build_ac_stream_policy_mask(
        video_seq_len=6,
        action_seq_len=32,
        video_tokens_per_frame=2,
        known_prefix_length=8,
        device=torch.device("cpu"),
    )
    condition_mask = build_ac_stream_condition_mask(
        video_seq_len=6,
        condition_seq_len=16,
        video_tokens_per_frame=2,
        device=torch.device("cpu"),
    )

    inactive = mot.forward_ac_stream(
        states,
        policy_attention_mask=policy_mask,
        condition_attention_mask=condition_mask,
        video_tokens_per_frame=2,
        action_condition_active=torch.tensor([False]),
    )["video"]
    active = mot.forward_ac_stream(
        states,
        policy_attention_mask=policy_mask,
        condition_attention_mask=condition_mask,
        video_tokens_per_frame=2,
        action_condition_active=torch.tensor([True]),
    )["video"]

    torch.testing.assert_close(active[:, :2], inactive[:, :2])
    assert bool((active[:, 2:4] > inactive[:, 2:4]).all())
    torch.testing.assert_close(active[:, 4:], inactive[:, 4:])


def test_starwam_rtc_three_stream_residual_changes_only_z1_video_tokens() -> None:
    class FakeBlock(torch.nn.Module):
        num_heads = 1
        attn_head_dim = 1

        def get_qkv(self, tokens, t_mod, freqs):
            del t_mod, freqs
            return torch.zeros_like(tokens), torch.zeros_like(tokens), tokens

        def post_attention(self, tokens, attn_out, t_mod, context, context_mask):
            del t_mod, context, context_mask
            return tokens + attn_out

    class FakeExpert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([FakeBlock()])

    mot = ACStreamMoT(
        experts={"video": FakeExpert(), "action": FakeExpert()},
        checkpoint_mixed_attn=False,
    )
    common_state = {
        "freqs": torch.zeros((1,)),
        "t_mod": torch.zeros((1,)),
        "context": torch.zeros((1, 1, 1)),
        "context_mask": torch.ones((1, 1), dtype=torch.bool),
    }
    states = {
        "video": {"tokens": torch.zeros((1, 6, 1)), **common_state},
        "action": {"tokens": torch.zeros((1, 32, 1)), **common_state},
        "condition": {"tokens": torch.ones((1, 16, 1)), **common_state},
    }
    video_mask = torch.ones((6, 6), dtype=torch.bool)
    policy_mask = build_starwam_rtc_policy_mask(
        batch_size=1,
        video_seq_len=6,
        policy_seq_len=32,
        video_tokens_per_frame=2,
        known_prefix_length=8,
        device=torch.device("cpu"),
    )
    condition_mask = build_starwam_rtc_condition_mask(
        batch_size=1,
        condition_seq_len=16,
        video_tokens_per_frame=2,
        known_prefix_length=8,
        device=torch.device("cpu"),
    )

    inactive = mot.forward_starwam_rtc(
        states,
        video_attention_mask=video_mask,
        policy_attention_mask=policy_mask,
        condition_attention_mask=condition_mask,
        video_tokens_per_frame=2,
        action_condition_active=torch.tensor([False]),
    )["video"]
    active = mot.forward_starwam_rtc(
        states,
        video_attention_mask=video_mask,
        policy_attention_mask=policy_mask,
        condition_attention_mask=condition_mask,
        video_tokens_per_frame=2,
        action_condition_active=torch.tensor([True]),
    )["video"]

    torch.testing.assert_close(active[:, :2], inactive[:, :2])
    assert bool((active[:, 2:4] > inactive[:, 2:4]).all())
    torch.testing.assert_close(active[:, 4:], inactive[:, 4:])


def test_ac_stream_wam_d8_returns_exact_clean_prefix() -> None:
    from streamwam.config import SchedulerConfig
    from streamwam.wam.ac_stream_wam import ACStreamWAM, AC_STREAM_SLOT_ENCODER_NAME

    class FakeVAE:
        temporal_compress = 4

    class FakeVideoExpert(torch.nn.Module):
        def pre_dit(self, latents, timestep, context, context_mask):
            del timestep
            batch = latents.shape[0]
            return {
                "tokens": torch.zeros((batch, 3, 1), device=latents.device, dtype=latents.dtype),
                "freqs": torch.zeros((3, 1, 1), device=latents.device),
                "t_mod": torch.zeros((batch, 3, 6, 1), device=latents.device),
                "context": context,
                "context_mask": context_mask,
                "meta": {"tokens_per_frame": 1, "latent_shape": tuple(latents.shape)},
                "t": torch.zeros((batch, 3, 1), device=latents.device),
                "latent_shape": tuple(latents.shape),
            }

        def post_dit(self, tokens, meta, timestep):
            del tokens, timestep
            return torch.zeros(meta["latent_shape"])

    class FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.video = FakeVideoExpert()
            setattr(self.video, AC_STREAM_SLOT_ENCODER_NAME, ACStreamSlotEncoder(1))
            self.vae = FakeVAE()
            self.info = SimpleNamespace(patch_size=(1, 1, 1))

        def get_dit(self):
            return self.video

        def get_vae(self):
            return self.vae

        def encode_video(self, video):
            return torch.zeros((video.shape[0], 1, 1, 1, 1), dtype=video.dtype)

    class FakeActionExpert(torch.nn.Module):
        hidden_dim = 1

        def pre_dit(self, action_tokens, timestep, context, context_mask):
            del timestep
            batch, steps = action_tokens.shape[:2]
            return {
                "tokens": torch.zeros((batch, steps, 1), dtype=action_tokens.dtype),
                "freqs": torch.zeros((steps, 1, 1)),
                "t_mod": torch.zeros((batch, steps, 6, 1)),
                "context": context,
                "context_mask": context_mask,
            }

        def post_dit(self, tokens):
            return torch.zeros((tokens.shape[0], tokens.shape[1], 7), dtype=tokens.dtype)

    class FakeMoT(torch.nn.Module):
        def forward_ac_stream(self, states, **kwargs):
            del kwargs
            assert torch.is_inference_mode_enabled()
            return {"video": states["video"]["tokens"], "action": states["action"]["tokens"]}

    model = ACStreamWAM.__new__(ACStreamWAM)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        chunk_size=32,
        action_dim=7,
        video_scheduler=SchedulerConfig(),
        action_scheduler=SchedulerConfig(),
    )
    model.backbone = FakeBackbone()
    model.action_expert = FakeActionExpert()
    model.mot = FakeMoT()
    model.proprio_encoder = None
    model.proprio_dim = None
    model.action_video_conditioning = "full_video"
    model.video_scheduler = SimpleNamespace(num_train_timesteps=1000)
    model.action_scheduler = SimpleNamespace(num_train_timesteps=1000)
    previous = torch.arange(32 * 7, dtype=torch.float32).reshape(32, 7) / 100.0

    action = model.infer_action(
        input_image=torch.zeros((1, 3, 2, 2)),
        context=torch.zeros((1, 2, 1)),
        context_mask=torch.ones((1, 2), dtype=torch.bool),
        action_horizon=32,
        num_inference_steps=1,
        num_video_frames=9,
        sampling_method="ac-stream",
        ac_stream_prev_action_chunk=previous,
        ac_stream_inference_delay=8,
        seed=42,
    )

    torch.testing.assert_close(action[0, :8], previous[:8])


def test_starwam_rtc_wam_d8_returns_exact_14d_clean_prefix() -> None:
    from streamwam.config import SchedulerConfig
    from streamwam.wam.ac_stream_wam import ACStreamWAM, STARWAM_RTC_ARCHITECTURE

    class FakeVAE:
        temporal_compress = 4

    class FakeVideoExpert(torch.nn.Module):
        def pre_dit(self, latents, timestep, context, context_mask, **kwargs):
            del timestep, kwargs
            batch = latents.shape[0]
            return {
                "tokens": torch.zeros((batch, 3, 1), dtype=latents.dtype),
                "freqs": torch.zeros((3, 1, 1)),
                "t_mod": torch.zeros((batch, 3, 6, 1)),
                "context": context,
                "context_mask": context_mask,
                "meta": {"tokens_per_frame": 1, "latent_shape": tuple(latents.shape)},
                "t": torch.zeros((batch, 3, 1)),
            }

        def post_dit(self, tokens, meta, timestep):
            del tokens, timestep
            return torch.zeros(meta["latent_shape"])

    class FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.video = FakeVideoExpert()
            self.vae = FakeVAE()

        def get_dit(self):
            return self.video

        def get_vae(self):
            return self.vae

        def encode_video(self, video):
            return torch.zeros((video.shape[0], 1, 1, 1, 1), dtype=video.dtype)

    class FakeActionExpert(torch.nn.Module):
        hidden_dim = 1

        def pre_dit(self, action, timestep, context, context_mask, **kwargs):
            del timestep, kwargs
            batch, length = action.shape[:2]
            return {
                "tokens": torch.zeros((batch, length, 1), dtype=action.dtype),
                "freqs": torch.zeros((length, 1, 1)),
                "t_mod": torch.zeros((batch, length, 6, 1)),
                "context": context,
                "context_mask": context_mask,
            }

        def post_dit(self, tokens):
            return torch.zeros((tokens.shape[0], tokens.shape[1], 14), dtype=tokens.dtype)

    class FakeMoT(torch.nn.Module):
        def forward_starwam_rtc(self, states, **kwargs):
            assert kwargs["policy_attention_mask"].shape == (1, 1, 32, 35)
            assert kwargs["condition_attention_mask"].shape == (1, 1, 16, 17)
            return {"video": states["video"]["tokens"], "action": states["action"]["tokens"]}

    model = ACStreamWAM.__new__(ACStreamWAM)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        chunk_size=32,
        action_dim=14,
        video_scheduler=SchedulerConfig(),
        action_scheduler=SchedulerConfig(),
    )
    model.ac_stream_architecture = STARWAM_RTC_ARCHITECTURE
    model.backbone = FakeBackbone()
    model.action_expert = FakeActionExpert()
    model.rtc_slot_state_embedding = torch.nn.Embedding(2, 1)
    model.mot = FakeMoT()
    model.proprio_encoder = None
    model.proprio_dim = None
    model.action_video_conditioning = "first_frame"
    model.video_scheduler = SimpleNamespace(num_train_timesteps=1000)
    model.action_scheduler = SimpleNamespace(num_train_timesteps=1000)
    previous = torch.arange(32 * 14, dtype=torch.float32).reshape(32, 14) / 100.0 + 0.001

    action = model.infer_action(
        input_image=torch.zeros((1, 3, 384, 320), dtype=torch.bfloat16),
        context=torch.zeros((1, 2, 1), dtype=torch.bfloat16),
        context_mask=torch.ones((1, 2), dtype=torch.bool),
        action_horizon=32,
        num_inference_steps=1,
        num_video_frames=9,
        sampling_method="ac-stream",
        ac_stream_prev_action_chunk=previous,
        ac_stream_inference_delay=8,
        seed=42,
    )

    assert action.shape == (1, 32, 14)
    assert action.dtype == torch.float32
    torch.testing.assert_close(action[0, :8], previous[:8])


def test_ac_stream_controller_launches_d8_at_eight_and_swaps_to_cursor_eight() -> None:
    calls: list[tuple[int, torch.Tensor | None]] = []

    def predict(observation, previous_target, delay):
        calls.append((int(delay), None if previous_target is None else previous_target.clone()))
        base = 0.0 if previous_target is None else 100.0
        model = torch.arange(32, dtype=torch.float32).unsqueeze(1).expand(32, 7) + base
        return ACStreamPrediction(
            env_actions=model.numpy().copy(),
            model_actions=model,
            communication_ms=1.0,
            inference_ms=2.0,
        )

    controller = ACStreamController(predict, block_on_miss=True)
    controller.start_episode({"step": 0})
    executed = []
    for step in range(16):
        executed.append(float(controller.next_action({"step": step})[0]))
        controller.mark_action_executed()
    first_from_new_chunk = float(controller.next_action({"step": 16})[0])
    controller.close()

    assert executed == [float(index) for index in range(16)]
    assert first_from_new_chunk == 108.0
    assert [delay for delay, _ in calls] == [0, 8]
    assert calls[0][1] is None
    assert calls[1][1] is not None
    torch.testing.assert_close(
        calls[1][1][:16],
        torch.arange(8, 24, dtype=torch.float32).unsqueeze(1).expand(16, 7),
    )
    torch.testing.assert_close(calls[1][1][16:], torch.zeros((16, 7)))


def test_ac_stream_close_returns_generated_but_uninstalled_prediction() -> None:
    def predict(observation, previous_target, delay):
        del observation, previous_target
        model = torch.full((32, 7), float(delay))
        return ACStreamPrediction(
            env_actions=model.numpy().copy(),
            model_actions=model,
            communication_ms=3.0,
            inference_ms=4.0,
        )

    controller = ACStreamController(predict)
    controller.start_episode({})
    controller.pop_installed_predictions()
    for _ in range(8):
        controller.next_action({})
        controller.mark_action_executed()
    controller.next_action({})  # launches D8 at cursor 8

    pending = controller.close()

    assert pending is not None
    assert pending.communication_ms == 3.0
    assert pending.inference_ms == 4.0


def test_ac_stream_rejects_nonblocking_miss_policy() -> None:
    with pytest.raises(ValueError, match="block_on_miss=True"):
        ACStreamController(lambda *_: None, block_on_miss=False)


def test_ac_stream_predictor_error_is_not_rethrown_during_close() -> None:
    calls = 0

    def predict(observation, previous_target, delay):
        nonlocal calls
        del observation, previous_target, delay
        calls += 1
        if calls == 2:
            raise RuntimeError("predict failed")
        model = torch.zeros((32, 7))
        return ACStreamPrediction(model.numpy().copy(), model, 0.0, 0.0)

    controller = ACStreamController(predict)
    controller.start_episode({})
    for _ in range(16):
        controller.next_action({})
        controller.mark_action_executed()

    with pytest.raises(RuntimeError, match="predict failed"):
        controller.next_action({})

    assert controller.close() is None
    assert controller.close() is None


def test_ac_stream_overlap_record_for_prediction_ready_before_boundary() -> None:
    record = build_ac_stream_overlap_record(
        inference_started_ns=100_000_000,
        inference_completed_ns=300_000_000,
        prediction_completed_ns=320_000_000,
        overlap_window_started_ns=0,
        boundary_ns=1_000_000_000,
        swap_ns=1_000_000_000,
    )

    assert record == ACStreamOverlapRecord(
        inference_wall_ms=200.0,
        action_overlap_ms=200.0,
        boundary_wait_ms=0.0,
        ready_before_boundary=True,
        episode_end_before_boundary=False,
    )


def test_ac_stream_overlap_counts_only_measured_action_execution_intervals() -> None:
    record = build_ac_stream_overlap_record(
        inference_started_ns=100_000_000,
        inference_completed_ns=500_000_000,
        prediction_completed_ns=510_000_000,
        overlap_window_started_ns=0,
        action_execution_intervals_ns=[
            (50_000_000, 150_000_000),
            (200_000_000, 300_000_000),
            (600_000_000, 700_000_000),
        ],
        boundary_ns=1_000_000_000,
        swap_ns=1_000_000_000,
    )

    assert record.inference_wall_ms == 400.0
    assert record.action_overlap_ms == 150.0


def test_ac_stream_overlap_record_for_boundary_miss() -> None:
    record = build_ac_stream_overlap_record(
        inference_started_ns=100_000_000,
        inference_completed_ns=1_200_000_000,
        prediction_completed_ns=1_230_000_000,
        overlap_window_started_ns=0,
        boundary_ns=1_000_000_000,
        swap_ns=1_250_000_000,
    )

    assert record.inference_wall_ms == 1100.0
    assert record.action_overlap_ms == 900.0
    assert record.boundary_wait_ms == 230.0
    assert record.ready_before_boundary is False
    assert record.hidden_inference_ratio == pytest.approx(9 / 11)


def test_ac_stream_overlap_record_for_episode_end_drain() -> None:
    record = build_ac_stream_overlap_record(
        inference_started_ns=100_000_000,
        inference_completed_ns=800_000_000,
        prediction_completed_ns=850_000_000,
        overlap_window_started_ns=0,
        episode_end_ns=500_000_000,
    )

    assert record.inference_wall_ms == 700.0
    assert record.action_overlap_ms == 400.0
    assert record.boundary_wait_ms == 0.0
    assert record.ready_before_boundary is None
    assert record.episode_end_before_boundary is True


def test_ac_stream_controller_latches_boundary_at_last_action_completion() -> None:
    release_d8 = threading.Event()
    d8_completed = threading.Event()

    def predict(observation, previous_target, delay):
        del observation, previous_target
        started_ns = time.perf_counter_ns()
        if delay:
            assert release_d8.wait(timeout=1.0)
        completed_ns = time.perf_counter_ns()
        model = torch.zeros((32, 7))
        prediction = ACStreamPrediction(
            model.numpy().copy(),
            model,
            0.0,
            (completed_ns - started_ns) / 1e6,
            inference_started_ns=started_ns,
            inference_completed_ns=completed_ns,
        )
        if delay:
            d8_completed.set()
        return prediction

    controller = ACStreamController(predict)
    controller.start_episode({})
    controller.pop_installed_predictions()
    for _ in range(16):
        controller.next_action({})
        action_completed_ns = time.perf_counter_ns()
        controller.mark_action_executed(
            started_ns=action_completed_ns - 1_000,
            completed_ns=action_completed_ns,
        )

    release_d8.set()
    assert d8_completed.wait(timeout=1.0)
    controller.next_action({})
    records = controller.pop_overlap_records()
    controller.close()

    assert len(records) == 1
    assert records[0].ready_before_boundary is False
    assert records[0].boundary_wait_ms > 0.0

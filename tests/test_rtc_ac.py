from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
import torch

from streamwam.inference import normalize_sampling_method
from streamwam.inference.rtc_ac import (
    RTCACController,
    RTCACOverlapRecord,
    RTCACPrediction,
    apply_rtc_ac_hard_prefix_,
    build_rtc_ac_overlap_record,
    build_rtc_ac_prev_action_target,
    validate_rtc_ac_geometry,
)
from streamwam.modules.rtc_ac import RTCACSlotEncoder
from streamwam.modules.rtc_ac import (
    RTCACMoT,
    build_rtc_ac_condition_mask,
    build_rtc_ac_policy_mask,
)


def test_rtc_ac_is_a_public_sampling_method() -> None:
    assert normalize_sampling_method("rtc_ac") == "rtc_ac"
    assert normalize_sampling_method("RTC-AC") == "rtc_ac"


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
def test_rtc_ac_rejects_checkpoint_incompatible_geometry(
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
        validate_rtc_ac_geometry(**values)


def test_rtc_ac_aligns_d8_prefix_to_launch_cursor() -> None:
    current = torch.arange(32 * 7, dtype=torch.float32).reshape(32, 7)
    target = build_rtc_ac_prev_action_target(
        current,
        cursor=8,
        action_horizon=32,
        execute_horizon=16,
    )
    torch.testing.assert_close(target[:16], current[8:24])
    torch.testing.assert_close(target[16:], torch.zeros(16, 7))
    assert target.data_ptr() != current.data_ptr()


def test_rtc_ac_hard_prefix_clamps_first_eight_tokens_and_sigma() -> None:
    latents = torch.full((1, 32, 7), -3.0)
    timesteps = torch.full((1, 32), 1000.0)
    prefix_source = torch.arange(32 * 7, dtype=torch.float32).reshape(1, 32, 7)

    clean = apply_rtc_ac_hard_prefix_(latents, timesteps, prefix_source)

    torch.testing.assert_close(latents[:, :8], prefix_source[:, :8])
    torch.testing.assert_close(latents[:, 8:], torch.full((1, 24, 7), -3.0))
    torch.testing.assert_close(timesteps[:, :8], torch.zeros((1, 8)))
    torch.testing.assert_close(timesteps[:, 8:], torch.full((1, 24), 1000.0))
    torch.testing.assert_close(clean, prefix_source[:, :8])
    assert clean.data_ptr() != prefix_source.data_ptr()


def test_rtc_ac_slot_encoder_keeps_known_content_and_zeros_unknown_content() -> None:
    encoder = RTCACSlotEncoder(hidden_dim=2)
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


def test_rtc_ac_d0_policy_mask_has_no_clean_prefix_restriction() -> None:
    mask = build_rtc_ac_policy_mask(
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


def test_rtc_ac_d8_policy_mask_isolates_clean_prefix() -> None:
    mask = build_rtc_ac_policy_mask(
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


def test_rtc_ac_condition_mask_injects_slots_only_into_z1() -> None:
    mask = build_rtc_ac_condition_mask(
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


def test_builder_selects_rtc_ac_wam_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    import streamwam.backbone
    import streamwam.wam
    from streamwam.builder import build_framework
    from streamwam.config import StreamWAMConfig

    marker = object()

    class FakeRTCACWAM:
        def __new__(cls, *args, **kwargs):
            return marker

    class FakeStandardWAM:
        def __new__(cls, *args, **kwargs):
            return object()

    monkeypatch.setattr(streamwam.backbone, "build_backbone", lambda *args, **kwargs: object())
    monkeypatch.setattr(streamwam.wam, "RTCACWAM", FakeRTCACWAM, raising=False)
    monkeypatch.setattr(streamwam.wam, "MoTWAM", FakeStandardWAM)
    config = StreamWAMConfig()
    config.framework.variant = "rtc_ac"

    assert build_framework(config) is marker


def test_rtc_ac_three_stream_residual_changes_only_z1_video_tokens() -> None:
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

    mot = RTCACMoT(
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
    policy_mask = build_rtc_ac_policy_mask(
        video_seq_len=6,
        action_seq_len=32,
        video_tokens_per_frame=2,
        known_prefix_length=8,
        device=torch.device("cpu"),
    )
    condition_mask = build_rtc_ac_condition_mask(
        video_seq_len=6,
        condition_seq_len=16,
        video_tokens_per_frame=2,
        device=torch.device("cpu"),
    )

    inactive = mot.forward_rtc_ac(
        states,
        policy_attention_mask=policy_mask,
        condition_attention_mask=condition_mask,
        video_tokens_per_frame=2,
        action_condition_active=torch.tensor([False]),
    )["video"]
    active = mot.forward_rtc_ac(
        states,
        policy_attention_mask=policy_mask,
        condition_attention_mask=condition_mask,
        video_tokens_per_frame=2,
        action_condition_active=torch.tensor([True]),
    )["video"]

    torch.testing.assert_close(active[:, :2], inactive[:, :2])
    assert bool((active[:, 2:4] > inactive[:, 2:4]).all())
    torch.testing.assert_close(active[:, 4:], inactive[:, 4:])


def test_rtc_ac_wam_d8_returns_exact_clean_prefix() -> None:
    from streamwam.config import SchedulerConfig
    from streamwam.wam.rtc_ac_wam import RTCACWAM, RTC_AC_SLOT_ENCODER_NAME

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
            setattr(self.video, RTC_AC_SLOT_ENCODER_NAME, RTCACSlotEncoder(1))
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
        def forward_rtc_ac(self, states, **kwargs):
            del kwargs
            assert torch.is_inference_mode_enabled()
            return {"video": states["video"]["tokens"], "action": states["action"]["tokens"]}

    model = RTCACWAM.__new__(RTCACWAM)
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
        sampling_method="rtc_ac",
        rtc_prev_action_chunk=previous,
        rtc_inference_delay=8,
        seed=42,
    )

    torch.testing.assert_close(action[0, :8], previous[:8])


def test_rtc_ac_controller_launches_d8_at_eight_and_swaps_to_cursor_eight() -> None:
    calls: list[tuple[int, torch.Tensor | None]] = []

    def predict(observation, previous_target, delay):
        calls.append((int(delay), None if previous_target is None else previous_target.clone()))
        base = 0.0 if previous_target is None else 100.0
        model = torch.arange(32, dtype=torch.float32).unsqueeze(1).expand(32, 7) + base
        return RTCACPrediction(
            env_actions=model.numpy().copy(),
            model_actions=model,
            communication_ms=1.0,
            inference_ms=2.0,
        )

    controller = RTCACController(predict, block_on_miss=True)
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


def test_rtc_ac_close_returns_generated_but_uninstalled_prediction() -> None:
    def predict(observation, previous_target, delay):
        del observation, previous_target
        model = torch.full((32, 7), float(delay))
        return RTCACPrediction(
            env_actions=model.numpy().copy(),
            model_actions=model,
            communication_ms=3.0,
            inference_ms=4.0,
        )

    controller = RTCACController(predict)
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


def test_rtc_ac_rejects_nonblocking_miss_policy() -> None:
    with pytest.raises(ValueError, match="block_on_miss=True"):
        RTCACController(lambda *_: None, block_on_miss=False)


def test_rtc_ac_predictor_error_is_not_rethrown_during_close() -> None:
    calls = 0

    def predict(observation, previous_target, delay):
        nonlocal calls
        del observation, previous_target, delay
        calls += 1
        if calls == 2:
            raise RuntimeError("predict failed")
        model = torch.zeros((32, 7))
        return RTCACPrediction(model.numpy().copy(), model, 0.0, 0.0)

    controller = RTCACController(predict)
    controller.start_episode({})
    for _ in range(16):
        controller.next_action({})
        controller.mark_action_executed()

    with pytest.raises(RuntimeError, match="predict failed"):
        controller.next_action({})

    assert controller.close() is None
    assert controller.close() is None


def test_rtc_ac_overlap_record_for_prediction_ready_before_boundary() -> None:
    record = build_rtc_ac_overlap_record(
        inference_started_ns=100_000_000,
        inference_completed_ns=300_000_000,
        prediction_completed_ns=320_000_000,
        overlap_window_started_ns=0,
        boundary_ns=1_000_000_000,
        swap_ns=1_000_000_000,
    )

    assert record == RTCACOverlapRecord(
        inference_wall_ms=200.0,
        action_overlap_ms=200.0,
        boundary_wait_ms=0.0,
        ready_before_boundary=True,
        episode_end_before_boundary=False,
    )


def test_rtc_ac_overlap_counts_only_measured_action_execution_intervals() -> None:
    record = build_rtc_ac_overlap_record(
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


def test_rtc_ac_overlap_record_for_boundary_miss() -> None:
    record = build_rtc_ac_overlap_record(
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


def test_rtc_ac_overlap_record_for_episode_end_drain() -> None:
    record = build_rtc_ac_overlap_record(
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


def test_rtc_ac_controller_latches_boundary_at_last_action_completion() -> None:
    release_d8 = threading.Event()
    d8_completed = threading.Event()

    def predict(observation, previous_target, delay):
        del observation, previous_target
        started_ns = time.perf_counter_ns()
        if delay:
            assert release_d8.wait(timeout=1.0)
        completed_ns = time.perf_counter_ns()
        model = torch.zeros((32, 7))
        prediction = RTCACPrediction(
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

    controller = RTCACController(predict)
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

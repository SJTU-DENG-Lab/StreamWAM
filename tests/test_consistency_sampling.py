import math
from types import SimpleNamespace

import pytest
import torch

from streamwam.inference.consistency import (
    sample_joint_consistency_noise,
    action_consistency_boundary,
    normalize_sampling_method,
    video_consistency_boundary,
)
from streamwam.wam.mot_wam import (
    MoTWAM,
    _sample_euler_noise,
    _validate_consistency_inference,
)


class _TinyVideoDiT:
    def pre_dit(self, latents, timestep, context, context_mask):
        return {
            "tokens": latents,
            "timestep": timestep,
            "context": context,
            "context_mask": context_mask,
        }


class _TinyBackbone:
    def __init__(self) -> None:
        self.dit = _TinyVideoDiT()

    def encode_video(self, video):
        return video

    def get_dit(self):
        return self.dit


class _TinyActionExpert:
    def __init__(self, velocity: torch.Tensor) -> None:
        self.velocity = velocity
        self.last_timestep = None

    def pre_dit(self, action, timestep, context, context_mask):
        self.last_timestep = timestep.clone()
        return {"action": action}

    def post_dit(self, output):
        return self.velocity.to(output)


class _TinyMoT:
    def prefill_video_cache(self, state):
        return state

    def forward_action_with_video_cache(self, action_state, video_cache):
        del video_cache
        return action_state["action"]


class _FirstFrameOwner:
    action_video_conditioning = "first_frame"

    def __init__(self, velocity: torch.Tensor) -> None:
        self.config = SimpleNamespace(
            action_dim=2,
            chunk_size=4,
            action_scheduler=SimpleNamespace(infer_shift=5.0),
        )
        self.backbone = _TinyBackbone()
        self.action_expert = _TinyActionExpert(velocity)
        self.mot = _TinyMoT()

    def _append_proprio_to_context(self, context, context_mask, proprio):
        del proprio
        return context, context_mask

    def infer_joint(self, **kwargs):
        raise AssertionError(f"first-frame CD must not call infer_joint: {kwargs}")


def test_action_consistency_boundary_supports_tokenwise_sigma() -> None:
    sample = torch.tensor([[[2.0], [4.0]]], dtype=torch.float32)
    velocity = torch.tensor([[[0.5], [1.0]]], dtype=torch.float32)
    sigma = torch.tensor([[1.0, 0.25]], dtype=torch.float32)

    actual = action_consistency_boundary(sample, velocity, sigma)

    torch.testing.assert_close(actual, torch.tensor([[[1.5], [3.75]]]))


def test_video_consistency_boundary_uses_karras_scalings() -> None:
    sample = torch.ones((1, 1, 1, 1, 1), dtype=torch.float32) * 2.0
    velocity = torch.ones_like(sample) * 0.5
    sigma = torch.tensor([1.0], dtype=torch.float32)

    actual = video_consistency_boundary(sample, velocity, sigma, sigma_data=0.5)

    c_skip = 0.2
    c_out = 0.5 / math.sqrt(1.25)
    expected = torch.full_like(sample, c_skip * 2.0 + c_out * 1.5)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == sample.dtype


def test_sampling_method_aliases_are_explicit() -> None:
    assert normalize_sampling_method("flow") == "euler"
    assert normalize_sampling_method("cd") == "consistency"
    with pytest.raises(ValueError, match="Unsupported sampling_method"):
        normalize_sampling_method("unknown")


def test_joint_consistency_noise_matches_fastwam_cpu_generators() -> None:
    video_shape = (1, 2, 3, 2, 2)
    action_shape = (1, 4, 2)
    seed = 42

    expected_video = torch.randn(
        video_shape,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        device="cpu",
        dtype=torch.float32,
    ).to(dtype=torch.bfloat16)
    expected_action = torch.randn(
        action_shape,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        device="cpu",
        dtype=torch.float32,
    ).to(dtype=torch.bfloat16)

    actual_video, actual_action = sample_joint_consistency_noise(
        video_shape,
        action_shape,
        seed=seed,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    torch.testing.assert_close(actual_video, expected_video, rtol=0, atol=0)
    torch.testing.assert_close(actual_action, expected_action, rtol=0, atol=0)


def test_fastwam_euler_noise_uses_cpu_float32_before_target_cast() -> None:
    shape = (1, 4, 2)
    seed = 42
    expected = torch.randn(
        shape,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        device="cpu",
        dtype=torch.float32,
    ).to(dtype=torch.bfloat16)

    actual = _sample_euler_noise(
        shape,
        seed=seed,
        rand_device="cpu",
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_native_euler_noise_preserves_target_dtype_sampling() -> None:
    shape = (1, 4, 2)
    seed = 17
    expected = torch.randn(
        shape,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        device="cpu",
        dtype=torch.bfloat16,
    )

    actual = _sample_euler_noise(
        shape,
        seed=seed,
        rand_device=None,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("steps", "conditioning", "horizon", "expected"),
    [
        (2, "full_video", 32, "one inference step"),
        (1, "full_video", 16, "action_horizon=32"),
    ],
)
def test_joint_cd_rejects_incompatible_geometry(
    steps: int,
    conditioning: str,
    horizon: int,
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=expected):
        _validate_consistency_inference(
            num_inference_steps=steps,
            action_video_conditioning=conditioning,
            action_horizon=horizon,
            configured_horizon=32,
            num_video_frames=9,
            temporal_compress=4,
        )


def test_first_frame_cd_matches_yzy_action_boundary() -> None:
    velocity = torch.full((1, 4, 2), 0.25)
    owner = _FirstFrameOwner(velocity)
    seed = 19
    expected_noise = torch.randn(
        (1, 4, 2),
        generator=torch.Generator(device="cpu").manual_seed(seed),
        dtype=torch.float32,
    )

    actual = MoTWAM.infer_action(
        owner,
        input_image=torch.zeros((1, 3, 2, 2)),
        context=torch.zeros((1, 2, 3)),
        context_mask=torch.ones((1, 2), dtype=torch.bool),
        action_horizon=4,
        num_inference_steps=1,
        seed=seed,
        sampling_method="consistency",
    )

    torch.testing.assert_close(actual, expected_noise - velocity)
    torch.testing.assert_close(owner.action_expert.last_timestep, torch.tensor([1000.0]))


def test_first_frame_cd_requires_one_inference_step() -> None:
    owner = _FirstFrameOwner(torch.zeros((1, 4, 2)))

    with pytest.raises(ValueError, match="exactly one inference step"):
        MoTWAM.infer_action(
            owner,
            input_image=torch.zeros((1, 3, 2, 2)),
            context=torch.zeros((1, 2, 3)),
            context_mask=torch.ones((1, 2), dtype=torch.bool),
            action_horizon=4,
            num_inference_steps=2,
            sampling_method="consistency",
        )


def test_joint_cd_rejects_non_32_configured_horizon() -> None:
    with pytest.raises(ValueError, match="configured chunk_size=32"):
        _validate_consistency_inference(
            num_inference_steps=1,
            action_video_conditioning="full_video",
            action_horizon=16,
            configured_horizon=16,
            num_video_frames=9,
            temporal_compress=4,
        )


def test_joint_cd_rejects_video_geometry_other_than_nine_frames() -> None:
    with pytest.raises(ValueError, match="num_video_frames=9"):
        _validate_consistency_inference(
            num_inference_steps=1,
            action_video_conditioning="full_video",
            action_horizon=32,
            configured_horizon=32,
            num_video_frames=13,
            temporal_compress=4,
        )


@pytest.mark.parametrize("temporal_compress", [2, 8])
def test_joint_cd_requires_temporal_compress_four(temporal_compress: int) -> None:
    with pytest.raises(ValueError, match="temporal_compress=4"):
        _validate_consistency_inference(
            num_inference_steps=1,
            action_video_conditioning="full_video",
            action_horizon=32,
            configured_horizon=32,
            num_video_frames=9,
            temporal_compress=temporal_compress,
        )

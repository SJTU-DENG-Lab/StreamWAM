from types import SimpleNamespace

import numpy as np
import pytest
import torch

from streamwam.eval.policy import StreamWAMPolicy
from examples.robotwin.runtime import resolve_inference_runtime


class _RecordingModel:
    def __init__(self) -> None:
        self.kwargs = None

    def infer_action(self, **kwargs):
        self.kwargs = kwargs
        return torch.zeros((1, kwargs["action_horizon"], 14), dtype=torch.float32)


def _tiny_policy(inference_mode: str) -> StreamWAMPolicy:
    policy = object.__new__(StreamWAMPolicy)
    policy.device = torch.device("cpu")
    policy.dtype = torch.float32
    policy.model = _RecordingModel()
    policy.config = SimpleNamespace(
        framework=SimpleNamespace(proprio_dim=None),
        data=SimpleNamespace(num_frames=9, action_freq_ratio=1, action_norm_mode="zscore"),
        inference=SimpleNamespace(consistency_teacher_steps=4),
    )
    policy.inference_mode = inference_mode
    policy.action_horizon = 32
    policy.num_inference_steps = 4 if inference_mode == "baseline" else 1
    policy.action_num_inference_steps = policy.num_inference_steps
    policy.seed = 42
    policy._action_stats = None
    policy._state_stats = None
    policy._encode_context = lambda instruction: (
        torch.zeros((1, 2, 4)),
        torch.ones((1, 2), dtype=torch.bool),
    )
    return policy


@pytest.mark.parametrize(
    ("inference_mode", "sampling_method"),
    [("baseline", "euler"), ("cd", "consistency")],
)
def test_robotwin_policy_maps_mode_to_sampling_method(
    inference_mode: str,
    sampling_method: str,
) -> None:
    policy = _tiny_policy(inference_mode)

    action = policy.predict_chunk(
        torch.zeros((1, 3, 384, 320)),
        np.zeros(14, dtype=np.float32),
        "pick up the object",
    )

    assert action.shape == (32, 14)
    assert policy.model.kwargs["sampling_method"] == sampling_method
    assert policy.model.kwargs["consistency_teacher_steps"] == 4


def test_robotwin_policy_rejects_unknown_inference_mode() -> None:
    policy = _tiny_policy("unknown")

    with pytest.raises(ValueError, match="Unsupported inference mode"):
        policy.predict_chunk(
            torch.zeros((1, 3, 384, 320)),
            np.zeros(14, dtype=np.float32),
            "pick up the object",
        )


def test_robotwin_ac_stream_policy_returns_model_and_environment_chunks() -> None:
    policy = _tiny_policy("ac-stream")
    previous = torch.ones((32, 14))

    prediction = policy.predict_ac_stream_chunk(
        torch.zeros((1, 3, 384, 320)),
        np.zeros(14, dtype=np.float32),
        "pick up the object",
        previous_action_chunk=previous,
        inference_delay=8,
    )

    assert prediction.env_actions.shape == (32, 14)
    assert prediction.model_actions.shape == (32, 14)
    assert prediction.inference_ms >= 0.0
    torch.testing.assert_close(policy.model.kwargs["ac_stream_prev_action_chunk"], previous)
    assert policy.model.kwargs["ac_stream_inference_delay"] == 8
    assert policy.model.kwargs["ac_stream_context_key"] == "pick up the object"


def test_robotwin_runtime_resolves_three_modes() -> None:
    assert resolve_inference_runtime("baseline").backend == "eager"
    assert resolve_inference_runtime("cd").backend == "eager"
    assert resolve_inference_runtime("ac-stream", accelerated=True).backend == "accelerated"
    assert resolve_inference_runtime("ac-stream", eager=True).backend == "eager"


@pytest.mark.parametrize("mode", ["baseline", "cd"])
def test_robotwin_runtime_rejects_ac_stream_backend_for_sync_modes(mode: str) -> None:
    with pytest.raises(ValueError, match="only valid for inference_mode='ac-stream'"):
        resolve_inference_runtime(mode, accelerated=True)


def test_robotwin_runtime_rejects_conflicting_ac_stream_backends() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_inference_runtime("ac-stream", accelerated=True, eager=True)

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from streamingwam.eval.policy import StreamingWAMPolicy
from streamingwam.inference.ac_stream import ACStreamController, ACStreamPrediction
from examples.robotwin.runtime import resolve_inference_runtime
from examples.robotwin.multigpu_rollout import resolve_sampling_steps
from examples.robotwin.timing import aggregate_evaluation


class _RecordingModel:
    def __init__(self) -> None:
        self.kwargs = None

    def infer_action(self, **kwargs):
        self.kwargs = kwargs
        return torch.zeros((1, kwargs["action_horizon"], 14), dtype=torch.float32)


def _tiny_policy(inference_mode: str) -> StreamingWAMPolicy:
    policy = object.__new__(StreamingWAMPolicy)
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


@pytest.mark.parametrize(
    ("checkpoint_format", "mode", "expected"),
    [
        ("starwam", "baseline", (4, 4)),
        ("starwam", "cd", (1, 1)),
        ("starwam", "ac-stream", (1, 1)),
        ("fastwam", "baseline", (10, 10)),
        ("fastwam", "cd", (1, 2)),
        ("streamingwam", "ac-stream", (1, 2)),
    ],
)
def test_robotwin_sampling_steps_follow_checkpoint_family(
    checkpoint_format: str,
    mode: str,
    expected: tuple[int, int],
) -> None:
    assert resolve_sampling_steps(checkpoint_format, mode, None, None) == expected


def test_robotwin_sampling_step_overrides_are_preserved() -> None:
    assert resolve_sampling_steps("fastwam", "baseline", 3, 5) == (3, 5)


def test_ac_stream_requests_observation_only_at_d0_and_d8_launch() -> None:
    calls = []

    def predict(observation, previous, delay):
        calls.append((observation, previous, delay))
        actions = torch.zeros((32, 7), dtype=torch.float32)
        return ACStreamPrediction(
            env_actions=actions.numpy(),
            model_actions=actions,
            communication_ms=0.0,
            inference_ms=1.0,
        )

    controller = ACStreamController(predict)
    controller.start_episode("d0")
    assert controller.observation_required is False
    for _ in range(8):
        controller.next_action(None)
        controller.mark_action_executed()
    assert controller.observation_required is True
    controller.next_action("d8")
    controller.mark_action_executed()
    assert controller.observation_required is False
    controller.close()
    assert [call[2] for call in calls] == [0, 8]


def test_robotwin_summary_uses_success_only_macro_total_and_d8_chunk() -> None:
    records = [
        {"record_type": "inference", "regime": "d0", "model_inference_ms": 100.0},
        {"record_type": "inference", "regime": "d8", "model_inference_ms": 40.0},
        {"record_type": "inference", "regime": "d8", "model_inference_ms": 50.0},
        {"record_type": "episode", "task": "task", "config": "clean", "success": True, "total_time_s": 10.0},
        {"record_type": "episode", "task": "task", "config": "clean", "success": True, "total_time_s": 30.0},
        {"record_type": "episode", "task": "task", "config": "clean", "success": False, "total_time_s": 999.0},
        {"record_type": "episode", "task": "task", "config": "random", "success": True, "total_time_s": 100.0},
    ]

    summary = aggregate_evaluation(records)

    assert summary["chunk_time_ms"] == pytest.approx(45.0)
    assert summary["total_time_per_episode_s"] == pytest.approx(60.0)
    assert summary["successes"] == 3
    assert summary["episodes"] == 4
    assert summary["by_setting"]["clean/task"]["total_time_success_s"] == pytest.approx(20.0)

from types import SimpleNamespace

import numpy as np
import torch

from examples.robotwin import policy_server


class _FakePolicy:
    def __init__(self, inference_mode: str) -> None:
        self.inference_mode = inference_mode
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.calls = []
        self.reset_count = 0
        self.model = SimpleNamespace(
            ac_stream_acceleration_status=lambda: {
                "backend": "accelerated", "compile_active": True,
                "prewarmed_d0": True, "prewarmed_d8": True,
                "runtime": {"torch_version": "2.7.1"},
            }
        )

    def reset(self) -> None:
        self.reset_count += 1

    def predict_chunk(self, image, state, instruction):
        self.calls.append(("sync", image, state, instruction))
        return np.ones((32, 14), dtype=np.float32)

    def predict_chunk_prediction(self, image, state, instruction):
        self.calls.append(("sync", image, state, instruction))
        action = np.ones((32, 14), dtype=np.float32)
        return SimpleNamespace(env_actions=action, inference_ms=8.25)

    def predict_ac_stream_chunk(
        self,
        image,
        state,
        instruction,
        *,
        previous_action_chunk,
        inference_delay,
    ):
        self.calls.append(
            (
                "ac-stream",
                image,
                state,
                instruction,
                previous_action_chunk,
                inference_delay,
            )
        )
        model = torch.full((32, 14), 2.0)
        return SimpleNamespace(
            env_actions=model.numpy(),
            model_actions=model,
            inference_ms=12.5,
        )


def _request(**extra):
    return {
        "cmd": "infer",
        "request_id": 7,
        "head": np.zeros((2, 2, 3), dtype=np.uint8),
        "left": np.zeros((2, 2, 3), dtype=np.uint8),
        "right": np.zeros((2, 2, 3), dtype=np.uint8),
        "state": np.zeros(14, dtype=np.float32),
        "instruction": "move the object",
        **extra,
    }


def test_policy_server_handles_sync_mode_request(monkeypatch) -> None:
    policy = _FakePolicy("baseline")
    image = torch.zeros((1, 3, 384, 320))
    monkeypatch.setattr(policy_server, "_build_robotwin_image", lambda *args: image)

    response = policy_server.handle_request(policy, _request())

    assert response["request_id"] == 7
    assert response["action"].shape == (32, 14)
    assert "model_action" not in response
    assert response["model_inference_ms"] == 8.25
    assert response["server_total_ms"] >= response["model_inference_ms"]
    assert policy.calls[0][0] == "sync"


def test_policy_server_handles_ac_stream_d8_request(monkeypatch) -> None:
    policy = _FakePolicy("ac-stream")
    image = torch.zeros((1, 3, 384, 320))
    monkeypatch.setattr(policy_server, "_build_robotwin_image", lambda *args: image)
    previous = np.ones((32, 14), dtype=np.float32)

    response = policy_server.handle_request(
        policy,
        _request(previous_action_chunk=previous, inference_delay=8),
    )

    assert response["request_id"] == 7
    assert response["action"].shape == (32, 14)
    assert response["model_action"].shape == (32, 14)
    assert response["model_inference_ms"] == 12.5
    assert response["server_total_ms"] >= 0.0
    assert response["backend"] == "accelerated"
    assert response["runtime"]["torch_version"] == "2.7.1"
    assert policy.calls[0][-1] == 8
    torch.testing.assert_close(policy.calls[0][-2], torch.from_numpy(previous))


def test_policy_server_reset_is_structured() -> None:
    policy = _FakePolicy("baseline")

    response = policy_server.handle_request(
        policy,
        {"cmd": "reset", "request_id": 9},
    )

    assert response == {"ok": True, "request_id": 9}
    assert policy.reset_count == 1


def test_eager_ac_stream_metadata_includes_runtime() -> None:
    policy = _FakePolicy("ac-stream")
    policy.model = SimpleNamespace(
        ac_stream_acceleration_status=lambda: {
            "backend": "eager", "compile_active": False,
        }
    )

    metadata = policy_server._runtime_metadata(policy)

    assert metadata["backend"] == "eager"
    assert metadata["runtime"]["python_executable"]
    assert metadata["runtime"]["torch_version"] == torch.__version__

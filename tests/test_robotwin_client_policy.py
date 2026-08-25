import time

import numpy as np
import torch

from examples.robotwin.client_policy import RemoteStreamWAMModel
from streamwam.inference.ac_stream import ACStreamPrediction


class _TaskEnv:
    def __init__(self) -> None:
        self.actions = []
        self.eval_success = False
        self.take_action_cnt = 0
        self.step_lim = 100

    def get_instruction(self):
        return "move the object"

    def take_action(self, action, action_type):
        assert action_type == "qpos"
        time.sleep(0.004)
        self.actions.append(np.asarray(action).copy())
        self.take_action_cnt += 1


def _observation():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    return {
        "observation": {
            "head_camera": {"rgb": rgb},
            "left_camera": {"rgb": rgb},
            "right_camera": {"rgb": rgb},
        },
        "joint_action": {"vector": np.zeros(14, dtype=np.float32)},
    }


def test_ac_stream_client_launches_d8_while_actions_continue(monkeypatch) -> None:
    monkeypatch.setattr(RemoteStreamWAMModel, "_connect", lambda self, timeout: object())
    model = RemoteStreamWAMModel(
        host="127.0.0.1",
        port=8765,
        replan_steps=16,
        inference_mode="ac-stream",
        prewarm=False,
    )
    calls = []

    def predict(snapshot, previous, delay):
        calls.append((delay, previous))
        if delay == 8:
            time.sleep(0.015)
        base = 0.0 if delay == 0 else 100.0
        actions = torch.arange(32, dtype=torch.float32).unsqueeze(1).expand(32, 14) + base
        return ACStreamPrediction(
            env_actions=actions.numpy().copy(),
            model_actions=actions,
            communication_ms=1.0,
            inference_ms=2.0,
        )

    model._predict_remote = predict
    env = _TaskEnv()
    for _ in range(17):
        model.step(env, _observation())

    assert [delay for delay, _ in calls] == [0, 8]
    assert [float(action[0]) for action in env.actions[:16]] == list(map(float, range(16)))
    assert float(env.actions[16][0]) == 108.0
    model.close()


def test_client_excludes_prewarm_and_records_terminal_episode(monkeypatch) -> None:
    monkeypatch.setattr(RemoteStreamWAMModel, "_connect", lambda self, timeout: object())
    model = RemoteStreamWAMModel(
        host="127.0.0.1",
        port=8765,
        replan_steps=24,
        inference_mode="baseline",
        prewarm=True,
    )
    calls = []

    def infer(*args, warmup=False, **kwargs):
        calls.append(warmup)
        model._timing_records.append({
            "record_type": "inference",
            "warmup": warmup,
            "model_inference_ms": 10.0,
            "server_total_ms": 12.0,
        })
        return np.zeros((32, 14), dtype=np.float32)

    model._infer = infer
    env = _TaskEnv()
    model.step(env, _observation())
    env.eval_success = True
    model.step(env, None)

    assert calls == [True, False]
    assert model._episode_records[0]["success"] is True
    assert model._episode_records[0]["total_time_s"] >= 0.0
    assert [record["warmup"] for record in model._timing_records] == [True, False]


def test_finish_episode_drains_inflight_ac_stream_before_worker_collects_records(monkeypatch) -> None:
    monkeypatch.setattr(RemoteStreamWAMModel, "_connect", lambda self, timeout: object())
    model = RemoteStreamWAMModel(
        host="127.0.0.1", port=8765, replan_steps=16,
        inference_mode="ac-stream", prewarm=False,
    )
    model._episode_started_ns = time.perf_counter_ns()

    class PendingController:
        def close(self):
            model._timing_records.append({
                "record_type": "inference", "warmup": False,
                "episode": model._episode_index,
            })

    model._ac_stream_controller = PendingController()
    model.finish_episode(success=True)

    assert model._ac_stream_controller is None
    assert len(model._timing_records) == 1
    assert model._timing_records[0]["episode"] == 0
    assert model._episode_records[0]["success"] is True

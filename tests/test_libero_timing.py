import numpy as np
import pytest
import torch

from examples.libero.rollout import (
    _prewarm_ac_stream_if_needed,
    _prewarm_sync_if_needed,
    _save_video,
)
from examples.libero.timing import GlobalTimingSummary
from streamwam.inference.ac_stream import ACStreamOverlapRecord
from streamwam.inference.ac_stream import ACStreamPrediction


def test_global_timing_averages_all_chunks() -> None:
    timing = GlobalTimingSummary()
    timing.task_count = 2
    timing.trial_count = 3
    first = timing.add_chunk(communication_ms=2.0, inference_ms=10.0)
    first.add_action_execution(30.0)
    second = timing.add_chunk(communication_ms=4.0, inference_ms=20.0)
    second.add_action_execution(50.0)
    timing.add_episode_wall(100.0)
    timing.add_episode_wall(300.0)

    summary = timing.as_dict(command_wall_ms=200.0)

    assert summary == {
        "tasks_executed": 2,
        "trials_executed": 3,
        "chunks_executed": 2,
        "average_inference_ms_per_chunk": 15.0,
        "average_communication_ms_per_chunk": 3.0,
        "average_action_execution_ms_per_chunk": 40.0,
        "average_total_ms_per_chunk": 58.0,
        "average_episode_wall_ms": 200.0,
        "evaluation_workload_wall_ms": 400.0,
        "command_wall_ms": 200.0,
    }


def test_global_timing_empty_averages_are_zero() -> None:
    summary = GlobalTimingSummary().as_dict(command_wall_ms=5.0)

    assert summary["chunks_executed"] == 0
    assert summary["average_total_ms_per_chunk"] == 0.0
    assert summary["average_episode_wall_ms"] == 0.0
    assert summary["evaluation_workload_wall_ms"] == 0.0


def test_global_timing_formats_one_summary_block() -> None:
    timing = GlobalTimingSummary()
    timing.task_count = 1
    timing.trial_count = 1
    chunk = timing.add_chunk(communication_ms=2.0, inference_ms=10.0)
    chunk.add_action_execution(30.0)

    rendered = timing.format_summary(command_wall_ms=1000.0)

    assert rendered.count("LIBERO Timing Summary") == 1
    assert "average inference/chunk" in rendered
    assert "average communication/chunk" in rendered
    assert "average action execution/chunk" in rendered
    assert "AC-Stream Async Overlap" not in rendered


def test_global_timing_aggregates_ac_stream_overlap_once() -> None:
    timing = GlobalTimingSummary()
    timing.enable_ac_stream()
    first = timing.add_chunk(communication_ms=10.0, inference_ms=100.0)
    first.add_action_execution(890.0)
    second = timing.add_chunk(communication_ms=20.0, inference_ms=200.0)
    second.add_action_execution(780.0)
    timing.add_ac_stream_overlap(
        ACStreamOverlapRecord(
            inference_wall_ms=200.0,
            action_overlap_ms=150.0,
            boundary_wait_ms=50.0,
            ready_before_boundary=False,
            episode_end_before_boundary=False,
        )
    )
    timing.add_ac_stream_overlap(
        ACStreamOverlapRecord(
            inference_wall_ms=100.0,
            action_overlap_ms=100.0,
            boundary_wait_ms=0.0,
            ready_before_boundary=None,
            episode_end_before_boundary=True,
        )
    )

    summary = timing.as_dict(command_wall_ms=2500.0)
    overlap = summary["ac_stream_overlap"]

    assert overlap == {
        "async_d8_inferences": 2,
        "boundary_evaluated_inferences": 1,
        "ready_before_boundary": 0,
        "ready_before_boundary_rate": 0.0,
        "deadline_misses": 1,
        "deadline_miss_rate": 1.0,
        "average_inference_wall_ms": 150.0,
        "first_background_d8_inference_ms": 200.0,
        "first_background_d8_inference_count": 1,
        "steady_state_d8_count": 1,
        "steady_state_d8_mean_ms": 100.0,
        "steady_state_d8_p50_ms": 100.0,
        "steady_state_d8_p90_ms": 100.0,
        "steady_state_d8_inference_ms_values": [100.0],
        "average_action_overlap_ms": 125.0,
        "average_boundary_wait_ms": 50.0,
        "average_hidden_inference_ratio": 0.875,
        "average_effective_total_ms_per_chunk": 875.0,
        "inference_hidden_ms_per_chunk": 125.0,
    }
    rendered = timing.format_summary(command_wall_ms=2500.0)
    assert rendered.count("AC-Stream Async Overlap") == 1
    assert "ready before chunk boundary     : 0/1 (0.00%)" in rendered
    assert "average hidden inference ratio  : 87.50%" in rendered
    assert "average effective time/chunk    : 875.00 ms" in rendered
    assert "first background D8 inference  : 200.00 ms" in rendered
    assert "steady-state D8 inference       : mean=100.00 ms p50=100.00 ms" in rendered


def test_ac_stream_timing_excludes_exactly_first_background_d8_from_steady_state() -> None:
    timing = GlobalTimingSummary()
    timing.enable_ac_stream()
    samples = [114.14, 45.38, 40.52, 41.86, 34.63]
    for sample in samples:
        timing.add_ac_stream_overlap(
            ACStreamOverlapRecord(
                inference_wall_ms=sample,
                action_overlap_ms=0.0,
                boundary_wait_ms=0.0,
                ready_before_boundary=True,
                episode_end_before_boundary=False,
            )
        )

    overlap = timing.as_dict(command_wall_ms=0.0)["ac_stream_overlap"]

    assert overlap["average_inference_wall_ms"] == pytest.approx(sum(samples) / 5)
    assert overlap["first_background_d8_inference_ms"] == pytest.approx(114.14)
    assert overlap["steady_state_d8_count"] == 4
    assert overlap["steady_state_d8_mean_ms"] == pytest.approx(40.5975)
    assert overlap["steady_state_d8_p50_ms"] == pytest.approx(41.19)
    assert overlap["steady_state_d8_p90_ms"] == pytest.approx(44.324)


def test_empty_ac_stream_summary_is_safe() -> None:
    timing = GlobalTimingSummary()
    timing.enable_ac_stream()

    overlap = timing.as_dict(command_wall_ms=0.0)["ac_stream_overlap"]

    assert overlap["async_d8_inferences"] == 0
    assert overlap["average_hidden_inference_ratio"] == 0.0
    assert overlap["steady_state_d8_count"] == 0


def test_save_video_defaults_to_30_fps(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_mimwrite(path, frames, fps):
        captured.update(path=path, frame_count=len(frames), fps=fps)

    monkeypatch.setattr("imageio.mimwrite", fake_mimwrite)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    _save_video(tmp_path / "rollout.mp4", [frame])

    assert captured["fps"] == 30
    assert captured["frame_count"] == 1


def test_sync_policy_prewarms_once_per_task() -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.reset_count = 0
            self.steps = 0

        def reset(self) -> None:
            self.reset_count += 1

        def set_init_state(self, initial_state):
            return {"state": initial_state}

        def step(self, action):
            del action
            self.steps += 1
            return {"state": self.steps}, 0.0, False, {}

    calls = []
    env = FakeEnv()
    prewarmed_tasks = set()

    for task_key in ("suite/task0", "suite/task0", "suite/task1"):
        _prewarm_sync_if_needed(
            task_key=task_key,
            prewarmed_tasks=prewarmed_tasks,
            env=env,
            initial_state="initial",
            num_steps_wait=3,
            predict=lambda observation: calls.append(observation),
        )

    assert env.reset_count == 2
    assert env.steps == 6
    assert len(calls) == 2
    assert prewarmed_tasks == {"suite/task0", "suite/task1"}


def test_ac_stream_acceleration_prewarms_d0_and_d8_once_per_task() -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.reset_count = 0
            self.steps = 0

        def reset(self) -> None:
            self.reset_count += 1

        def set_init_state(self, initial_state):
            return {"state": initial_state}

        def step(self, action):
            del action
            self.steps += 1
            return {"state": self.steps}, 0.0, False, {}

    class FakeModel:
        def __init__(self) -> None:
            self.ac_stream_prewarm_complete = False
            self.delays = []

        def mark_ac_stream_prewarmed(self, delay: int) -> None:
            self.delays.append(delay)
            self.ac_stream_prewarm_complete = set(self.delays) == {0, 8}

    calls = []

    def predict(observation, previous_target, delay):
        calls.append((observation, previous_target, delay))
        model_actions = torch.full((32, 7), float(delay))
        return ACStreamPrediction(
            env_actions=model_actions.numpy(),
            model_actions=model_actions,
            communication_ms=1000.0,
            inference_ms=2000.0,
        )

    env = FakeEnv()
    model = FakeModel()
    prewarmed_tasks = set()

    _prewarm_ac_stream_if_needed(
        task_key="suite/task0",
        prewarmed_tasks=prewarmed_tasks,
        env=env,
        initial_state="initial",
        num_steps_wait=3,
        model=model,
        predict=predict,
        accelerated=True,
    )
    _prewarm_ac_stream_if_needed(
        task_key="suite/task0",
        prewarmed_tasks=prewarmed_tasks,
        env=env,
        initial_state="initial",
        num_steps_wait=3,
        model=model,
        predict=predict,
        accelerated=True,
    )
    _prewarm_ac_stream_if_needed(
        task_key="suite/task1",
        prewarmed_tasks=prewarmed_tasks,
        env=env,
        initial_state="initial",
        num_steps_wait=3,
        model=model,
        predict=predict,
        accelerated=True,
    )

    assert env.reset_count == 2
    assert env.steps == 6
    assert [call[2] for call in calls] == [0, 8, 0, 8]
    assert calls[0][1] is None
    torch.testing.assert_close(calls[1][1], torch.zeros(32, 7))
    assert model.delays == [0, 8]
    assert prewarmed_tasks == {"suite/task0", "suite/task1"}

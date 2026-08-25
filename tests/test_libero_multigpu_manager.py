from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import examples.libero.multigpu_rollout as multigpu_rollout
from examples.libero.multigpu_rollout import (
    _build_arg_parser,
    _format_summary,
    _prepare_output_dir,
    build_worker_command,
    merge_worker_results,
)
from streamwam.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_acceleration_reports_merge_per_worker_diagnostics() -> None:
    common = {
        "backend": "accelerated",
        "compile_active": True,
        "compile_mode": "reduce-overhead",
        "prewarmed_d0": True,
        "prewarmed_d8": True,
    }
    reports = [
        {
            **common,
            "dynamo_unique_graphs": 1,
            "dynamo_recompiles": 0,
            "inductor_cudagraph_skips": 0,
            "runtime": {"torch_version": "2.7.1", "gpu_name": "H100"},
        },
        {
            **common,
            "dynamo_unique_graphs": 2,
            "dynamo_recompiles": 1,
            "inductor_cudagraph_skips": 3,
            "runtime": {"torch_version": "2.7.1", "gpu_name": "A100"},
        },
    ]

    merged = multigpu_rollout._merge_acceleration_reports(reports)

    assert merged["dynamo_unique_graphs"] == 3
    assert merged["dynamo_recompiles"] == 1
    assert merged["inductor_cudagraph_skips"] == 3
    assert merged["runtime"]["torch_version"] == "2.7.1"
    assert merged["runtime"]["gpu_names"] == ["H100", "A100"]
    assert len(merged["worker_diagnostics"]) == 2


def test_fastwam_official_launcher_emits_baseline_protocol(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "GPU_IDS": "2,4,6,7",
            "BACKBONE_PATH": "/models/test-wan22",
            "LIBERO_HOME_PATH": "/datasets/test-LIBERO",
        }
    )

    completed = subprocess.run(
        [
            "bash",
            str(
                REPO_ROOT
                / "examples/libero/scripts/launch_streamwam_libero_fastwam_4gpu.sh"
            ),
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    arguments = completed.stdout.splitlines()

    expected_options = {
        "--gpus": "2,4,6,7",
        "--suites": "libero_spatial,libero_object,libero_goal,libero_10",
        "--num-trials": "1",
        "--config": "examples/libero/configs/recipes/streamwam_libero_mot_wan22_5b.yaml",
        "--checkpoint-format": "fastwam",
        "--checkpoint": "checkpoints/fastwam_release/libero_uncond_2cam224.pt",
        "--backbone-path": "/models/test-wan22",
        "--stats-path": (
            "checkpoints/fastwam_release/"
            "libero_uncond_2cam224_dataset_stats.json"
        ),
        "--libero-home": "/datasets/test-LIBERO",
        "--num-steps-wait": "30",
        "--replan-steps": "10",
        "--num-inference-steps": "10",
        "--sampling-method": "euler",
        "--mujoco-gl": "egl",
    }
    assert arguments[0] == "examples/libero/multigpu_rollout.py"
    for option, expected_value in expected_options.items():
        assert arguments[arguments.index(option) + 1] == expected_value
    assert "--fixed-seed" in arguments
    assert "--save-video" in arguments


def test_fastwam_joint_launcher_emits_synchronous_joint_protocol(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "GPU_IDS": "1,3,5,7",
            "BACKBONE_PATH": "/models/test-wan22",
            "LIBERO_HOME_PATH": "/datasets/test-LIBERO",
        }
    )

    completed = subprocess.run(
        [
            "bash",
            str(
                REPO_ROOT
                / "examples/libero/scripts/launch_streamwam_libero_fastwam_joint_4gpu.sh"
            ),
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    arguments = completed.stdout.splitlines()
    parsed = _build_arg_parser().parse_args(arguments[1:])

    assert arguments[0] == "examples/libero/multigpu_rollout.py"
    assert parsed.gpus == "1,3,5,7"
    assert parsed.suites == "libero_spatial,libero_object,libero_goal,libero_10"
    assert parsed.num_trials == 1
    assert parsed.config == (
        "examples/libero/configs/recipes/streamwam_libero_fastwam_joint_wan22_5b.yaml"
    )
    assert parsed.checkpoint_format == "fastwam"
    assert parsed.checkpoint == "checkpoints/fastwam_joint_step_040000.pt"
    assert parsed.stats_path == "checkpoints/fastwam_joint_dataset_stats.json"
    assert parsed.replan_steps == 16
    assert parsed.num_inference_steps == 10
    assert parsed.sampling_method == "euler"
    assert parsed.fixed_seed is True


def test_fastwam_joint_launcher_allows_fifty_trial_override(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "BACKBONE_PATH": "/models/test-wan22",
            "LIBERO_HOME_PATH": "/datasets/test-LIBERO",
        }
    )

    completed = subprocess.run(
        [
            "bash",
            str(
                REPO_ROOT
                / "examples/libero/scripts/launch_streamwam_libero_fastwam_joint_4gpu.sh"
            ),
            "--num-trials",
            "50",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    parsed = _build_arg_parser().parse_args(completed.stdout.splitlines()[1:])

    assert parsed.num_trials == 50


def test_fastwam_joint_recipe_selects_full_video_euler_inference() -> None:
    config = load_config(
        REPO_ROOT
        / "examples/libero/configs/recipes/streamwam_libero_fastwam_joint_wan22_5b.yaml"
    )

    assert config.framework.action_video_conditioning == "full_video"
    assert config.framework.chunk_size == 32
    assert config.data.num_frames == 33
    assert config.inference.sampling_method == "euler"
    assert config.inference.num_inference_steps == 10
    assert config.inference.replan_steps == 16


def test_worker_command_isolates_visible_gpu_and_uses_local_cuda_zero(tmp_path) -> None:
    args = _build_arg_parser().parse_args(
        [
            "--gpus",
            "2,5",
            "--config",
            "recipe.yaml",
            "--checkpoint",
            "model.pt",
            "--backbone-path",
            "/models/wan",
            "--stats-path",
            "stats.json",
            "--libero-home",
            "/src/LIBERO",
            "--save-video",
        ]
    )

    command, environment = build_worker_command(
        args=args,
        gpu="5",
        worker_id=1,
        manifest_path=tmp_path / "worker_1.json",
        output_dir=tmp_path / "worker_gpu5",
    )

    assert environment["CUDA_VISIBLE_DEVICES"] == "5"
    assert environment["LIBERO_CONFIG_PATH"] == str(
        tmp_path / "worker_gpu5" / "libero_config"
    )
    assert command[command.index("--device") + 1] == "cuda:0"
    assert command[command.index("--work-manifest") + 1] == str(tmp_path / "worker_1.json")
    assert command[command.index("--worker-id") + 1] == "worker_1"
    assert "--suppress-final-summary" in command
    assert "--save-video" in command


def test_manager_accepts_ac_stream_sampling_method() -> None:
    args = _build_arg_parser().parse_args(
        [
            "--gpus",
            "0,1,2,3",
            "--config",
            "ac-stream.yaml",
            "--checkpoint",
            "ac-stream.pt",
            "--sampling-method",
            "ac-stream",
        ]
    )

    assert args.sampling_method == "ac-stream"


def test_manager_forwards_ac_stream_acceleration_to_worker(tmp_path) -> None:
    args = _build_arg_parser().parse_args(
        [
            "--gpus",
            "0",
            "--config",
            "ac-stream.yaml",
            "--checkpoint",
            "ac-stream.pt",
            "--sampling-method",
            "ac-stream",
            "--ac-stream-accelerated",
        ]
    )

    command, _ = build_worker_command(
        args=args,
        gpu="0",
        worker_id=0,
        manifest_path=tmp_path / "manifest.json",
        output_dir=tmp_path / "worker",
    )

    assert "--ac-stream-accelerated" in command


def test_ac_stream_launcher_forwards_acceleration_flag(tmp_path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment.update(
        {
            "BACKBONE_PATH": str(tmp_path / "backbone"),
            "CHECKPOINT_PATH": str(tmp_path / "checkpoint.pt"),
            "LIBERO_HOME_PATH": str(tmp_path / "LIBERO"),
            "STATS_PATH": str(tmp_path / "stats.json"),
        }
    )

    completed = subprocess.run(
        [
            "bash",
            str(
                REPO_ROOT
                / "examples/libero/scripts/launch_streamwam_libero_ac_stream_4gpu.sh"
            ),
            "--ac-stream-accelerated",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--ac-stream-accelerated" in completed.stdout.splitlines()


def test_ac_stream_launcher_uses_explicit_python_bin(tmp_path) -> None:
    invocation_log = tmp_path / "python-invocations.txt"
    selected_python = tmp_path / "selected-python"
    selected_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'CALL\\n' >> \"$PYTHON_INVOCATION_LOG\"\n"
        "printf '%s\\n' \"$@\" >> \"$PYTHON_INVOCATION_LOG\"\n"
        "exec \"$REAL_PYTHON\" \"$@\"\n",
        encoding="utf-8",
    )
    selected_python.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "BACKBONE_PATH": str(tmp_path / "backbone"),
            "CHECKPOINT_PATH": str(tmp_path / "checkpoint.pt"),
            "LIBERO_HOME_PATH": str(tmp_path / "LIBERO"),
            "PYTHON_BIN": str(selected_python),
            "PYTHON_INVOCATION_LOG": str(invocation_log),
            "REAL_PYTHON": sys.executable,
            "STATS_PATH": str(tmp_path / "stats.json"),
        }
    )

    subprocess.run(
        [
            "bash",
            str(
                REPO_ROOT
                / "examples/libero/scripts/launch_streamwam_libero_ac_stream_4gpu.sh"
            ),
            "--help",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    invocations = invocation_log.read_text(encoding="utf-8")
    assert "examples/libero/multigpu_rollout.py" in invocations


def test_terminal_summary_keeps_results_and_only_two_timing_metrics() -> None:
    result = {
        "total_trials": 8,
        "total_successes": 8,
        "success_rate": 1.0,
        "timing_summary": {
            "tasks_executed": 8,
            "chunks_executed": 93,
            "average_inference_ms_per_chunk": 42.73,
            "average_communication_ms_per_chunk": 6.29,
            "average_action_execution_ms_per_chunk": 298.21,
            "average_total_ms_per_chunk": 347.23,
            "average_episode_wall_ms": 4530.0,
            "evaluation_workload_wall_ms": 36280.0,
            "command_wall_ms": 324640.0,
            "readme_aligned": {
                "chunk_time_ms_mean": 42.73,
                "long_successful_episode_s_mean": 5.74,
                "long_successful_episode_count": 4,
                "short_successful_episode_s_mean": 3.33,
                "short_successful_episode_count": 4,
            },
        },
        "ac_stream_acceleration": {
            "backend": "accelerated",
            "compile_active": True,
        },
    }

    assert _format_summary(
        result,
        ["0", "1", "2", "3"],
        results_path="/outputs/results.json",
    ).splitlines() == [
        "=== Multi-GPU LIBERO Evaluation Summary ===",
        "GPUs: 0,1,2,3",
        "Tasks: 8",
        "Trials: 8",
        "Success: 8/8 (1.0000)",
        "Chunks: 93",
        "Chunk Time: 42.73 ms",
        "Total Time / Episode: 4.53 s",
        "Results: /outputs/results.json",
    ]


def test_merge_worker_results_weights_timing_by_chunk_and_merges_split_task(tmp_path) -> None:
    first = {
        "checkpoint": "model.pt",
        "task_results": {
            "libero_spatial/0": {
                "task_suite_name": "libero_spatial",
                "task_id": 0,
                "task_description": "pick object",
                "successes": 1,
                "trials": 1,
                "success_rate": 1.0,
                "episodes": [{"trial": 0, "success": True}],
            }
        },
        "total_successes": 1,
        "total_trials": 1,
        "timing_summary": {
            "chunks_executed": 2,
            "average_inference_ms_per_chunk": 10.0,
            "average_communication_ms_per_chunk": 2.0,
            "average_action_execution_ms_per_chunk": 4.0,
            "average_total_ms_per_chunk": 16.0,
            "average_episode_wall_ms": 100.0,
            "evaluation_workload_wall_ms": 100.0,
        },
    }
    second = {
        "checkpoint": "model.pt",
        "task_results": {
            "libero_spatial/0": {
                "task_suite_name": "libero_spatial",
                "task_id": 0,
                "task_description": "pick object",
                "successes": 0,
                "trials": 1,
                "success_rate": 0.0,
                "episodes": [{"trial": 1, "success": False}],
            }
        },
        "total_successes": 0,
        "total_trials": 1,
        "timing_summary": {
            "chunks_executed": 1,
            "average_inference_ms_per_chunk": 40.0,
            "average_communication_ms_per_chunk": 8.0,
            "average_action_execution_ms_per_chunk": 10.0,
            "average_total_ms_per_chunk": 58.0,
            "average_episode_wall_ms": 300.0,
            "evaluation_workload_wall_ms": 300.0,
        },
    }
    paths = []
    for index, payload in enumerate((first, second)):
        path = tmp_path / f"worker_{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    merged = merge_worker_results(
        paths,
        command_wall_ms=1234.0,
        expected_trials=[
            ("libero_spatial", 0, 0),
            ("libero_spatial", 0, 1),
        ],
    )

    assert merged["total_successes"] == 1
    assert merged["total_trials"] == 2
    assert merged["success_rate"] == 0.5
    task = merged["task_results"]["libero_spatial/0"]
    assert task["trials"] == 2
    assert [episode["trial"] for episode in task["episodes"]] == [0, 1]
    timing = merged["timing_summary"]
    assert timing["tasks_executed"] == 1
    assert timing["trials_executed"] == 2
    assert timing["chunks_executed"] == 3
    assert timing["average_inference_ms_per_chunk"] == 20.0
    assert timing["average_communication_ms_per_chunk"] == 4.0
    assert timing["average_action_execution_ms_per_chunk"] == 6.0
    assert timing["average_total_ms_per_chunk"] == 30.0
    assert timing["average_episode_wall_ms"] == 200.0
    assert timing["evaluation_workload_wall_ms"] == 400.0
    assert timing["command_wall_ms"] == 1234.0


def test_merge_reports_readme_aligned_successful_long_and_short_timing(tmp_path) -> None:
    tasks = {
        "libero_10/0": {
            "task_suite_name": "libero_10",
            "task_id": 0,
            "task_description": "long task",
            "successes": 1,
            "trials": 2,
            "success_rate": 0.5,
            "episodes": [
                {"trial": 0, "success": True, "episode_wall_ms": 5000.0},
                {"trial": 1, "success": False, "episode_wall_ms": 9000.0},
            ],
        },
        "libero_goal/0": {
            "task_suite_name": "libero_goal",
            "task_id": 0,
            "task_description": "short task",
            "successes": 2,
            "trials": 2,
            "success_rate": 1.0,
            "episodes": [
                {"trial": 0, "success": True, "episode_wall_ms": 3000.0},
                {"trial": 1, "success": True, "episode_wall_ms": 4000.0},
            ],
        },
    }
    payload = {
        "checkpoint": "model.pt",
        "task_results": tasks,
        "total_successes": 3,
        "total_trials": 4,
        "timing_summary": {
            "chunks_executed": 4,
            "average_inference_ms_per_chunk": 41.0,
            "average_communication_ms_per_chunk": 2.0,
            "average_action_execution_ms_per_chunk": 100.0,
            "average_total_ms_per_chunk": 143.0,
            "average_episode_wall_ms": 5250.0,
            "evaluation_workload_wall_ms": 21000.0,
        },
    }
    path = tmp_path / "worker.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    merged = merge_worker_results(
        [path],
        command_wall_ms=22000.0,
        expected_trials=[
            (suite, 0, trial)
            for suite in ("libero_10", "libero_goal")
            for trial in (0, 1)
        ],
    )

    reference = merged["timing_summary"]["readme_aligned"]
    assert reference["chunk_time_ms_mean"] == 41.0
    assert reference["long_successful_episode_s_mean"] == 5.0
    assert reference["long_successful_episode_count"] == 1
    assert reference["short_successful_episode_s_mean"] == 3.5
    assert reference["short_successful_episode_count"] == 2


def test_merge_worker_results_reconstructs_ac_stream_overlap_totals(tmp_path) -> None:
    common_task = {
        "task_suite_name": "libero_goal",
        "task_id": 0,
        "task_description": "open drawer",
        "successes": 1,
        "trials": 1,
        "success_rate": 1.0,
    }
    payloads = []
    acceleration_status = {
        "backend": "accelerated",
        "compile_active": True,
        "prewarmed_d0": True,
        "prewarmed_d8": True,
    }
    for trial, chunks, overlap in (
        (
            0,
            2,
            {
                "async_d8_inferences": 2,
                "boundary_evaluated_inferences": 2,
                "ready_before_boundary": 2,
                "ready_before_boundary_rate": 1.0,
                "deadline_misses": 0,
                "deadline_miss_rate": 0.0,
                "average_inference_wall_ms": 100.0,
                "first_background_d8_inference_ms": 150.0,
                "first_background_d8_inference_count": 1,
                "steady_state_d8_count": 1,
                "steady_state_d8_mean_ms": 50.0,
                "steady_state_d8_p50_ms": 50.0,
                "steady_state_d8_p90_ms": 50.0,
                "steady_state_d8_inference_ms_values": [50.0],
                "average_action_overlap_ms": 80.0,
                "average_boundary_wait_ms": 0.0,
                "average_hidden_inference_ratio": 0.8,
                "average_effective_total_ms_per_chunk": 120.0,
                "inference_hidden_ms_per_chunk": 80.0,
            },
        ),
        (
            1,
            1,
            {
                "async_d8_inferences": 1,
                "boundary_evaluated_inferences": 1,
                "ready_before_boundary": 0,
                "ready_before_boundary_rate": 0.0,
                "deadline_misses": 1,
                "deadline_miss_rate": 1.0,
                "average_inference_wall_ms": 400.0,
                "first_background_d8_inference_ms": 400.0,
                "first_background_d8_inference_count": 1,
                "steady_state_d8_count": 0,
                "steady_state_d8_mean_ms": 0.0,
                "steady_state_d8_p50_ms": 0.0,
                "steady_state_d8_p90_ms": 0.0,
                "steady_state_d8_inference_ms_values": [],
                "average_action_overlap_ms": 100.0,
                "average_boundary_wait_ms": 50.0,
                "average_hidden_inference_ratio": 0.25,
                "average_effective_total_ms_per_chunk": 100.0,
                "inference_hidden_ms_per_chunk": 100.0,
            },
        ),
    ):
        payloads.append(
            {
                "checkpoint": "ac-stream.pt",
                "task_results": {
                    "libero_goal/0": {
                        **common_task,
                        "episodes": [{"trial": trial, "success": True}],
                    }
                },
                "total_successes": 1,
                "total_trials": 1,
                "ac_stream_acceleration": acceleration_status,
                "timing_summary": {
                    "chunks_executed": chunks,
                    "average_inference_ms_per_chunk": 100.0,
                    "average_communication_ms_per_chunk": 10.0,
                    "average_action_execution_ms_per_chunk": 90.0,
                    "average_total_ms_per_chunk": 200.0,
                    "average_episode_wall_ms": 1000.0,
                    "evaluation_workload_wall_ms": 1000.0,
                    "ac_stream_overlap": overlap,
                },
            }
        )
    paths = []
    for index, payload in enumerate(payloads):
        path = tmp_path / f"ac_stream_worker_{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    merged = merge_worker_results(
        paths,
        command_wall_ms=1500.0,
        expected_trials=[("libero_goal", 0, 0), ("libero_goal", 0, 1)],
    )
    overlap = merged["timing_summary"]["ac_stream_overlap"]

    assert merged["ac_stream_acceleration"] == acceleration_status
    assert overlap == {
        "async_d8_inferences": 3,
        "boundary_evaluated_inferences": 3,
        "ready_before_boundary": 2,
        "ready_before_boundary_rate": 2 / 3,
        "deadline_misses": 1,
        "deadline_miss_rate": 1 / 3,
        "average_inference_wall_ms": 200.0,
        "first_background_d8_inference_ms": 275.0,
        "first_background_d8_inference_count": 2,
        "steady_state_d8_count": 1,
        "steady_state_d8_mean_ms": 50.0,
        "steady_state_d8_p50_ms": 50.0,
        "steady_state_d8_p90_ms": 50.0,
        "steady_state_d8_inference_ms_values": [50.0],
        "average_action_overlap_ms": 260.0 / 3,
        "average_boundary_wait_ms": 50.0 / 3,
        "average_hidden_inference_ratio": 1.85 / 3,
        "average_effective_total_ms_per_chunk": 340.0 / 3,
        "inference_hidden_ms_per_chunk": 260.0 / 3,
    }


def test_merge_rejects_missing_manifest_trial(tmp_path) -> None:
    result_path = tmp_path / "worker.json"
    result_path.write_text(
        json.dumps(
            {
                "checkpoint": "model.pt",
                "task_results": {},
                "total_successes": 0,
                "total_trials": 0,
                "timing_summary": {
                    "chunks_executed": 0,
                    "average_inference_ms_per_chunk": 0.0,
                    "average_communication_ms_per_chunk": 0.0,
                    "average_action_execution_ms_per_chunk": 0.0,
                    "average_total_ms_per_chunk": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        merge_worker_results(
            [result_path],
            command_wall_ms=1.0,
            expected_trials=[("libero_goal", 2, 0)],
        )
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("missing manifest trial was accepted")


def test_merge_rejects_unexpected_worker_trial(tmp_path) -> None:
    result_path = tmp_path / "worker.json"
    result_path.write_text(
        json.dumps(
            {
                "checkpoint": "model.pt",
                "task_results": {
                    "libero_goal/2": {
                        "task_suite_name": "libero_goal",
                        "task_id": 2,
                        "task_description": "task",
                        "successes": 0,
                        "trials": 1,
                        "success_rate": 0.0,
                        "episodes": [{"trial": 9, "success": False}],
                    }
                },
                "total_successes": 0,
                "total_trials": 1,
                "timing_summary": {
                    "chunks_executed": 0,
                    "average_inference_ms_per_chunk": 0.0,
                    "average_communication_ms_per_chunk": 0.0,
                    "average_action_execution_ms_per_chunk": 0.0,
                    "average_total_ms_per_chunk": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        merge_worker_results(
            [result_path],
            command_wall_ms=1.0,
            expected_trials=[("libero_goal", 2, 0)],
        )
    except ValueError as exc:
        assert "unexpected" in str(exc)
    else:
        raise AssertionError("unexpected worker trial was accepted")


def test_output_directory_must_be_new(tmp_path) -> None:
    output_dir = tmp_path / "run"
    _prepare_output_dir(output_dir)

    try:
        _prepare_output_dir(output_dir)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing output directory was reused")


def test_manager_script_can_run_directly_from_outside_repo(tmp_path) -> None:
    script = Path(__file__).resolve().parents[1] / "examples" / "libero" / "multigpu_rollout.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--gpus" in result.stdout

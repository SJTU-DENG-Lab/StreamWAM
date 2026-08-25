import json
from pathlib import Path

import pytest

from examples.robotwin.multigpu_rollout import (
    _build_arg_parser,
    _completed_timing_records,
    _job_result_path,
    _run_job_queue,
    _select_pending_jobs,
    _terminate_process_group,
    _timeout_result,
    _start_server_group,
    _terminate_processes,
    _validate_result_identities,
    build_server_command,
    format_summary,
)
from examples.robotwin.workload import RoboTwinJob


def test_server_command_uses_explicit_inference_python_and_mode() -> None:
    args = _build_arg_parser().parse_args([
        "--config", "recipe.yaml", "--checkpoint", "model.pt",
        "--stats-path", "stats.json", "--backbone-path", "wan",
        "--robotwin-home", "RoboTwin", "--inference-python", "/wyx/python",
        "--simulator-python", "/motus/python", "--inference-mode", "cd",
    ])

    command = build_server_command(args, port=9000)

    assert command[0] == "/wyx/python"
    assert command[1:3] == ["-m", "examples.robotwin.policy_server"]
    assert command[command.index("--inference-mode") + 1] == "cd"
    assert command[command.index("--seed") + 1] == "42"
    assert "--ac-stream-accelerated" not in command


def test_ac_stream_defaults_accelerated_and_rejects_backend_on_cd() -> None:
    parser = _build_arg_parser()
    common = [
        "--config", "recipe.yaml", "--checkpoint", "model.pt",
        "--stats-path", "stats.json", "--backbone-path", "wan",
        "--robotwin-home", "RoboTwin", "--inference-python", "/wyx/python",
        "--simulator-python", "/motus/python",
    ]
    args = parser.parse_args(common + ["--inference-mode", "ac-stream"])
    assert "--ac-stream-accelerated" in build_server_command(args, port=9000)

    bad = parser.parse_args(common + ["--inference-mode", "cd", "--ac-stream-eager"])
    with pytest.raises(ValueError, match="only valid"):
        build_server_command(bad, port=9000)


def test_final_summary_is_concise_and_contains_results_path(tmp_path: Path) -> None:
    text = format_summary(
        mode="baseline", gpu_ids=["0", "3"], jobs=100,
        summary={
            "successes": 90, "episodes": 100,
            "chunk_time_ms": 80.0, "total_time_per_episode_s": 8.0,
        },
        results_path=tmp_path / "results.json",
    )

    assert "Success: 90/100 (0.9000)" in text
    assert "Chunk Time: 80.00 ms" in text
    assert "Total Time / Episode: 8.00 s" in text
    assert str(tmp_path / "results.json") in text
    assert len(text.splitlines()) <= 8


def test_incomplete_summary_handles_no_completed_episode(tmp_path: Path) -> None:
    text = format_summary(
        mode="baseline", gpu_ids=["0"], jobs=1,
        summary={
            "successes": 0, "episodes": 0,
            "chunk_time_ms": None, "total_time_per_episode_s": None,
        },
        results_path=tmp_path / "results.json",
        completed=0, skipped=1, status="INCOMPLETE",
    )

    assert "Success: N/A" in text
    assert "Chunk Time: N/A" in text
    assert "Total Time / Episode: N/A" in text


def test_result_validation_compares_exact_expected_identities() -> None:
    jobs = [RoboTwinJob("adjust_bottle", "demo_clean", 0)]
    with pytest.raises(RuntimeError, match="missing=.*adjust_bottle"):
        _validate_result_identities(
            jobs,
            [{"task": "click_bell", "config": "demo_clean", "trial": 0}],
        )


def test_all_servers_start_before_manager_waits_for_readiness(tmp_path: Path) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args([
        "--config", "recipe.yaml", "--checkpoint", "model.pt",
        "--stats-path", "stats.json", "--backbone-path", "wan",
        "--robotwin-home", "RoboTwin", "--inference-python", "/wyx/python",
        "--simulator-python", "/motus/python", "--inference-mode", "baseline",
    ])
    events = []

    class Process:
        def __init__(self, command, **kwargs):
            del kwargs
            self.command = command
            self.returncode = None
            events.append(("start", command[command.index("--port") + 1]))

        def poll(self):
            return None

    def ready(port, process):
        assert process.poll() is None
        events.append(("ready", str(port)))

    servers = _start_server_group(
        args, ["0", "1", "2", "3"], tmp_path,
        popen=Process, wait_for_server=ready,
    )

    assert events[:4] == [
        ("start", "18765"), ("start", "18766"),
        ("start", "18767"), ("start", "18768"),
    ]
    assert len(servers) == 4


def test_cleanup_terminates_workers_and_servers() -> None:
    events = []

    class Process:
        returncode = None

        def __init__(self, name):
            self.name = name

        def poll(self):
            return self.returncode

        def terminate(self):
            events.append(("terminate", self.name))
            self.returncode = -15

        def wait(self, timeout):
            events.append(("wait", self.name, timeout))

    _terminate_processes([Process("worker"), Process("server")])

    assert events == [
        ("terminate", "worker"), ("terminate", "server"),
        ("wait", "worker", 10), ("wait", "server", 10),
    ]


def test_timeout_is_skipped_without_becoming_policy_failure() -> None:
    job = RoboTwinJob("adjust_bottle", "demo_clean", 0)

    result = _timeout_result(job, elapsed_seconds=1200.5, phase="expert_check")

    assert result["status"] == "skipped_timeout"
    assert result["success"] is None
    assert result["episodes"] == 0
    assert result["timeout_seconds"] == 1200.5
    assert result["phase"] == "expert_check"


def test_resume_skips_atomic_completed_and_timeout_sidecars(tmp_path: Path) -> None:
    jobs = [
        RoboTwinJob("adjust_bottle", "demo_clean", 0),
        RoboTwinJob("click_bell", "demo_clean", 0),
        RoboTwinJob("turn_switch", "demo_clean", 0),
    ]
    for job, status in zip(jobs[:2], ("completed", "skipped_timeout")):
        path = _job_result_path(tmp_path, job)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**job.__dict__, "status": status}), encoding="utf-8")

    pending, existing = _select_pending_jobs(jobs, tmp_path)

    assert pending == [jobs[2]]
    assert [result["status"] for result in existing] == ["completed", "skipped_timeout"]


def test_dynamic_queue_times_out_once_then_refills_gpu(tmp_path: Path) -> None:
    jobs = [
        RoboTwinJob("adjust_bottle", "demo_clean", 0),
        RoboTwinJob("click_bell", "demo_clean", 0),
    ]
    parser = _build_arg_parser()
    args = parser.parse_args([
        "--config", "recipe.yaml", "--checkpoint", "model.pt",
        "--stats-path", "stats.json", "--backbone-path", "wan",
        "--robotwin-home", "RoboTwin", "--inference-python", "/wyx/python",
        "--simulator-python", "/motus/python", "--inference-mode", "baseline",
        "--job-timeout-seconds", "5",
    ])
    launches = []
    killed = []
    cleaned = []
    dumped = []
    now = [0.0]

    class Process:
        _next_pid = 100

        def __init__(self, command, **kwargs):
            assert kwargs["start_new_session"] is True
            self.command = command
            self.pid = Process._next_pid
            Process._next_pid += 1
            self.task = json.loads(Path(command[command.index("--job-file") + 1]).read_text())["task"]
            self.returncode = None
            launches.append(self.task)
            if self.task == "click_bell":
                output = Path(command[command.index("--output") + 1])
                output.write_text(json.dumps({
                    "gpu_id": "0",
                    "result": {**jobs[1].__dict__, "status": "completed", "success": 1, "episodes": 1},
                    "timing": [{"record_type": "episode", "success": True, "total_time_s": 2.0}],
                }), encoding="utf-8")
                self.returncode = 0

        def poll(self):
            return self.returncode

    def sleep(seconds):
        now[0] += max(seconds, 1.0)

    results = _run_job_queue(
        args, jobs,
        [{"gpu": "0", "port": 18765, "worker_dir": tmp_path / "worker_gpu0", "environment": {}}],
        tmp_path,
        popen=Process,
        monotonic=lambda: now[0],
        sleep=sleep,
        terminate_group=lambda process: killed.append(process.task),
        cleanup_finished_group=lambda process: cleaned.append(process.task),
        dump_stack=lambda process: dumped.append(process.task),
    )

    assert launches == ["adjust_bottle", "click_bell"]
    assert killed == ["adjust_bottle"]
    assert dumped == ["adjust_bottle"]
    assert cleaned == ["click_bell"]
    assert [result["status"] for result in results] == ["skipped_timeout", "completed"]


def test_dynamic_queue_rejects_jobs_without_any_gpu_server(tmp_path: Path) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args([
        "--config", "recipe.yaml", "--checkpoint", "model.pt",
        "--stats-path", "stats.json", "--backbone-path", "wan",
        "--robotwin-home", "RoboTwin", "--inference-python", "/wyx/python",
        "--simulator-python", "/motus/python", "--inference-mode", "baseline",
    ])

    with pytest.raises(ValueError, match="GPU server"):
        _run_job_queue(
            args,
            [RoboTwinJob("adjust_bottle", "demo_clean", 0)],
            [],
            tmp_path,
        )


def test_infrastructure_error_timing_is_excluded_from_final_aggregation() -> None:
    records = _completed_timing_records([
        {
            "status": "completed",
            "timing": [{"record_type": "episode", "total_time_s": 2.0}],
        },
        {
            "status": "infrastructure_error",
            "timing": [
                {"record_type": "inference", "model_inference_ms": 999.0},
                {"record_type": "episode", "total_time_s": 999.0},
            ],
        },
    ])

    assert records == [{"record_type": "episode", "total_time_s": 2.0}]


def test_terminate_process_group_cleans_descendants_after_leader_exits(monkeypatch) -> None:
    signals = []

    class Process:
        pid = 4321

        @staticmethod
        def poll():
            return 0

    monkeypatch.setattr(
        "examples.robotwin.multigpu_rollout.os.killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )

    _terminate_process_group(Process())

    assert signals == [(4321, __import__("signal").SIGKILL)]


def test_terminate_process_group_force_cleans_children_after_leader_stops(monkeypatch) -> None:
    signals = []

    class Process:
        pid = 4321
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            assert timeout == 10
            self.returncode = -15

    monkeypatch.setattr(
        "examples.robotwin.multigpu_rollout.os.killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )

    _terminate_process_group(Process())

    signal_module = __import__("signal")
    assert signals == [
        (4321, signal_module.SIGTERM),
        (4321, signal_module.SIGKILL),
    ]


def test_simulator_launch_failure_is_recorded_once_and_queue_continues(tmp_path: Path) -> None:
    jobs = [
        RoboTwinJob("adjust_bottle", "demo_clean", 0),
        RoboTwinJob("click_bell", "demo_clean", 0),
    ]
    args = _build_arg_parser().parse_args([
        "--config", "recipe.yaml", "--checkpoint", "model.pt",
        "--stats-path", "stats.json", "--backbone-path", "wan",
        "--robotwin-home", "RoboTwin", "--inference-python", "/wyx/python",
        "--simulator-python", "/motus/python", "--inference-mode", "baseline",
    ])
    launches = []

    class Process:
        pid = 1234
        returncode = 0

        def __init__(self, command, **kwargs):
            del kwargs
            task = json.loads(
                Path(command[command.index("--job-file") + 1]).read_text()
            )["task"]
            launches.append(task)
            if task == "adjust_bottle":
                raise OSError("cannot spawn simulator")
            output = Path(command[command.index("--output") + 1])
            output.write_text(json.dumps({
                "result": {
                    **jobs[1].__dict__, "status": "completed",
                    "success": 1, "episodes": 1,
                },
                "timing": [],
            }))

        @staticmethod
        def poll():
            return 0

    results = _run_job_queue(
        args, jobs,
        [{"gpu": "0", "port": 18765, "environment": {}}],
        tmp_path,
        popen=Process,
        cleanup_finished_group=lambda process: None,
        sleep=lambda seconds: None,
    )

    assert launches == ["adjust_bottle", "click_bell"]
    assert [result["status"] for result in results] == [
        "infrastructure_error", "completed",
    ]
    persisted = json.loads(_job_result_path(tmp_path, jobs[0]).read_text())
    assert persisted["status"] == "infrastructure_error"

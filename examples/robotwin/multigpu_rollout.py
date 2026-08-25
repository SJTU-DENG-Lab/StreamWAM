"""Four-GPU (or arbitrary-GPU) RoboTwin evaluation manager."""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from examples.robotwin.runtime import resolve_inference_runtime
from examples.robotwin.timing import aggregate_evaluation
from examples.robotwin.workload import build_workload


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StreamWAM RoboTwin multi-GPU evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-format", default="starwam", choices=("streamwam", "fastwam", "starwam"))
    parser.add_argument("--stats-path", required=True)
    parser.add_argument("--backbone-path", required=True)
    parser.add_argument("--text-cache-path", default="/inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/yzy/starwam/cache/text_embeds_cache")
    parser.add_argument("--robotwin-home", required=True)
    parser.add_argument("--inference-python", required=True)
    parser.add_argument("--simulator-python", required=True)
    parser.add_argument("--inference-mode", choices=("baseline", "cd", "ac-stream"), required=True)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--ac-stream-accelerated", action="store_true")
    backend.add_argument("--ac-stream-eager", action="store_true")
    parser.add_argument("--gpu-ids", default=os.environ.get("GPU_IDS", "0,1,2,3"))
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--configs", default="demo_clean,demo_randomized")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port-base", type=int, default=18765)
    parser.add_argument("--job-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--output-dir", default=None)
    return parser


def build_server_command(args: argparse.Namespace, *, port: int) -> list[str]:
    accelerated = bool(
        args.ac_stream_accelerated
        or (args.inference_mode == "ac-stream" and not args.ac_stream_eager)
    )
    runtime = resolve_inference_runtime(
        args.inference_mode,
        accelerated=accelerated,
        eager=args.ac_stream_eager,
    )
    command = [
        args.inference_python, "-m", "examples.robotwin.policy_server",
        "--config", args.config, "--checkpoint", args.checkpoint,
        "--checkpoint-format", args.checkpoint_format,
        "--inference-mode", args.inference_mode,
        "--num-inference-steps", "4" if args.inference_mode == "baseline" else "1",
        "--action-num-inference-steps", "4" if args.inference_mode == "baseline" else "1",
        "--seed", str(args.seed),
        "--device", "cuda:0", "--host", "127.0.0.1", "--port", str(port),
        "--override",
        f"backbone.pretrained_model_id={args.backbone_path}",
        f"data.action_stats_path={args.stats_path}",
        f"data.state_stats_path={args.stats_path}",
        f"data.text_embedding_cache_dir={args.text_cache_path}",
    ]
    if runtime.accelerated:
        command.append("--ac-stream-accelerated")
    elif runtime.inference_mode == "ac-stream":
        command.append("--ac-stream-eager")
    return command


def format_summary(
    *, mode: str, gpu_ids: list[str], jobs: int, summary: dict,
    results_path: Path, completed: int | None = None,
    skipped: int = 0, status: str = "COMPLETE",
) -> str:
    successes, episodes = int(summary["successes"]), int(summary["episodes"])
    rate = successes / episodes if episodes else 0.0
    chunk_time_ms = summary.get("chunk_time_ms")
    episode_time_s = summary.get("total_time_per_episode_s")
    chunk_time = "N/A" if chunk_time_ms is None else f"{chunk_time_ms:.2f} ms"
    episode_time = "N/A" if episode_time_s is None else f"{episode_time_s:.2f} s"
    lines = [
        "=== RoboTwin Evaluation Summary ===",
        f"Mode: {mode}",
        f"GPUs: {','.join(gpu_ids)} | Jobs: {jobs}",
    ]
    if completed is not None:
        lines.append(
            f"Planned: {jobs} | Completed: {completed} | Skipped: {skipped}"
        )
    lines.extend([
        f"Success: {successes}/{episodes} ({rate:.4f})" if episodes else "Success: N/A",
        f"Chunk Time: {chunk_time}",
        f"Total Time / Episode: {episode_time}",
        f"Status: {status} | Results: {results_path}",
    ])
    return "\n".join(lines)


def _validate_result_identities(jobs, task_results: list[dict]) -> None:
    expected = {job.identity for job in jobs}
    actual_list = [
        (str(item["task"]), str(item["config"]), int(item["trial"]))
        for item in task_results
    ]
    actual = set(actual_list)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    duplicates = len(actual_list) - len(actual)
    if missing or unexpected or duplicates:
        raise RuntimeError(
            "RoboTwin result identities do not match the workload: "
            f"missing={missing}, unexpected={unexpected}, duplicates={duplicates}"
        )


def _job_result_path(output_dir: Path, job) -> Path:
    return (
        output_dir / "jobs" / f"trial_{int(job.trial):04d}"
        / str(job.config) / str(job.task) / "result.json"
    )


def _select_pending_jobs(jobs, output_dir: Path) -> tuple[list, list[dict]]:
    pending, existing = [], []
    for job in jobs:
        path = _job_result_path(output_dir, job)
        if not path.is_file():
            pending.append(job)
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pending.append(job)
            continue
        if result.get("status") in {"completed", "skipped_timeout", "infrastructure_error"}:
            existing.append(result)
        else:
            pending.append(job)
    return pending, existing


def _timeout_result(job, *, elapsed_seconds: float, phase: str | None) -> dict:
    return {
        **job.__dict__,
        "status": "skipped_timeout",
        "success": None,
        "episodes": 0,
        "error": "job watchdog timeout",
        "timeout_seconds": float(elapsed_seconds),
        "phase": phase or "unknown",
    }


def _infrastructure_error_result(job, error: object) -> dict:
    return {
        **job.__dict__,
        "status": "infrastructure_error",
        "success": None,
        "episodes": 0,
        "error": str(error),
    }


def _completed_timing_records(results: list[dict]) -> list[dict]:
    """Return benchmark records only from fully completed jobs."""

    return [
        record
        for result in results
        if result.get("status") == "completed"
        for record in result.get("timing", [])
    ]


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_phase(path: Path) -> str:
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("phase", "unknown"))
    except (OSError, json.JSONDecodeError):
        return "unknown"


def _terminate_process_group(process) -> None:
    """Terminate a job and every renderer/helper process in its POSIX group."""

    if process.poll() is not None:
        _cleanup_finished_process_group(process)
        return
    pid = getattr(process, "pid", None)
    if pid is None:
        process.terminate()
    else:
        try:
            # Jobs are launched with start_new_session=True, so pid == pgid.
            # Addressing the group directly still works if the leader races
            # with us and exits while renderer children remain alive.
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if pid is None:
            process.kill()
        else:
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
        process.wait(timeout=10)
    # The leader exiting does not prove that renderer/helper children honored
    # SIGTERM. Force-remove anything still attached to this one-job session.
    _cleanup_finished_process_group(process)


def _cleanup_finished_process_group(process) -> None:
    """Remove renderer/helper descendants after the one-job leader exits."""

    pid = getattr(process, "pid", None)
    if pid is None:
        return
    try:
        # Every job starts a fresh session, so its PID is also its process-group
        # ID. The leader has already exited; any remaining members are leaks.
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _dump_process_stack(process) -> None:
    pid = getattr(process, "pid", None)
    if pid is None or process.poll() is not None:
        return
    try:
        os.kill(pid, signal.SIGUSR1)
    except (ProcessLookupError, PermissionError):
        pass


def _run_job_queue(
    args: argparse.Namespace,
    jobs,
    server_group: list[dict],
    output_dir: Path,
    *,
    popen=subprocess.Popen,
    monotonic=time.monotonic,
    sleep=time.sleep,
    terminate_group=_terminate_process_group,
    cleanup_finished_group=_cleanup_finished_process_group,
    dump_stack=_dump_process_stack,
) -> list[dict]:
    """Dynamically supervise one isolated simulator process per job."""

    if jobs and not server_group:
        raise ValueError("At least one GPU server is required for RoboTwin jobs")
    timeout = float(args.job_timeout_seconds)
    if timeout <= 0:
        raise ValueError("--job-timeout-seconds must be greater than zero")
    pending = deque(jobs)
    active: dict[str, dict] = {}
    results: list[dict] = []
    warmed = {str(item["gpu"]): False for item in server_group}

    def launch(item: dict, job) -> None:
        gpu = str(item["gpu"])
        result_path = _job_result_path(output_dir, job)
        job_dir = result_path.parent
        job_dir.mkdir(parents=True, exist_ok=True)
        job_file = job_dir / "job.json"
        worker_output = job_dir / "worker_result.json"
        status_output = job_dir / "status.json"
        worker_output.unlink(missing_ok=True)
        status_output.unlink(missing_ok=True)
        _atomic_json(job_file, job.__dict__)
        log = open(job_dir / "worker.log", "w", encoding="utf-8")
        replan_steps = 16 if args.inference_mode == "ac-stream" else 24
        command = [
            args.simulator_python, "-m", "examples.robotwin.robotwin_worker",
            "--gpu-id", gpu, "--robotwin-home", args.robotwin_home,
            "--policy-dir", str(_ROOT / "examples" / "robotwin"),
            "--server-port", str(item["port"]),
            "--inference-mode", args.inference_mode,
            "--replan-steps", str(replan_steps),
            "--job-file", str(job_file),
            "--output", str(worker_output),
            "--status-output", str(status_output),
            "--seed", str(args.seed),
            "--prewarm" if not warmed[gpu] else "--no-prewarm",
        ]
        try:
            process = popen(
                command, cwd=_ROOT, env=item["environment"],
                stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            log.close()
            raise
        active[gpu] = {
            "job": job, "process": process, "started": monotonic(),
            "result_path": result_path, "worker_output": worker_output,
            "status_output": status_output, "log": log,
        }

    try:
        while pending or active:
            for item in server_group:
                gpu = str(item["gpu"])
                if gpu not in active and pending:
                    job = pending.popleft()
                    try:
                        launch(item, job)
                    except Exception as error:  # spawn failure is terminal; never retry
                        canonical = {
                            **_infrastructure_error_result(job, error),
                            "gpu_id": gpu,
                            "timing": [],
                        }
                        _atomic_json(_job_result_path(output_dir, job), canonical)
                        results.append(canonical)
                        print(
                            f"\rProgress: {len(results)} finished this run | "
                            f"success={sum(r.get('success') == 1 for r in results)} | "
                            f"skipped={sum(r.get('status') != 'completed' for r in results)}",
                            end="", flush=True,
                        )

            completed_gpus = []
            for gpu, state in list(active.items()):
                process = state["process"]
                elapsed = monotonic() - state["started"]
                if process.poll() is None and elapsed < timeout:
                    continue
                job = state["job"]
                if process.poll() is None:
                    phase = _read_phase(state["status_output"])
                    dump_stack(process)
                    sleep(0.2)
                    terminate_group(process)
                    result = _timeout_result(
                        job, elapsed_seconds=elapsed, phase=phase,
                    )
                    canonical = {**result, "gpu_id": gpu, "timing": []}
                else:
                    cleanup_finished_group(process)
                    try:
                        payload = json.loads(
                            state["worker_output"].read_text(encoding="utf-8")
                        )
                        result = dict(payload["result"])
                        canonical = {
                            **result, "gpu_id": gpu,
                            "timing": list(payload.get("timing", [])),
                        }
                        if result.get("status") == "completed":
                            warmed[gpu] = True
                    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
                        result = _infrastructure_error_result(
                            job, f"worker produced no valid result: {error}",
                        )
                        canonical = {**result, "gpu_id": gpu, "timing": []}
                _atomic_json(state["result_path"], canonical)
                state["log"].close()
                results.append(canonical)
                completed_gpus.append(gpu)
                completed = len(results)
                successes = sum(item.get("success") == 1 for item in results)
                skipped = sum(item.get("status") != "completed" for item in results)
                print(
                    f"\rProgress: {completed} finished this run | "
                    f"success={successes} | skipped={skipped}",
                    end="", flush=True,
                )
            for gpu in completed_gpus:
                del active[gpu]
            if pending or active:
                sleep(0.2)
        if results:
            print(flush=True)
        return results
    finally:
        for state in active.values():
            terminate_group(state["process"])
            if not state["log"].closed:
                state["log"].close()


def _wait_for_server(port: int, process: subprocess.Popen, timeout: float = 900.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"policy server exited early with code {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return
        except OSError:
            time.sleep(1.0)
    raise TimeoutError(f"policy server on port {port} did not become ready")


def _terminate_processes(processes: list) -> None:
    """Best-effort cleanup for inference servers and all their descendants."""

    for process in processes:
        if process.poll() is None:
            pid = getattr(process, "pid", None)
            if pid is None:
                process.terminate()
            else:
                try:
                    os.killpg(pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pid = getattr(process, "pid", None)
            if pid is None:
                process.kill()
            else:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()
            process.wait(timeout=10)
        _cleanup_finished_process_group(process)


def _start_server_group(
    args: argparse.Namespace,
    gpu_ids: list[str],
    output_dir: Path,
    *,
    popen=subprocess.Popen,
    wait_for_server=_wait_for_server,
) -> list[dict]:
    """Launch every GPU server first, then wait until the whole group is ready."""

    group: list[dict] = []
    try:
        for worker_index, gpu in enumerate(gpu_ids):
            worker_dir = output_dir / f"worker_gpu{gpu}"
            worker_dir.mkdir(parents=True, exist_ok=True)
            port = args.port_base + worker_index
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["PYTHONPATH"] = (
                str(_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
            )
            log = open(worker_dir / "server.log", "w", encoding="utf-8")
            try:
                process = popen(
                    build_server_command(args, port=port),
                    cwd=_ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except BaseException:
                log.close()
                raise
            group.append({
                "gpu": gpu,
                "port": port,
                "worker_dir": worker_dir,
                "environment": environment,
                "process": process,
                "log": log,
            })
        for item in group:
            wait_for_server(item["port"], item["process"])
        return group
    except BaseException:
        _terminate_processes([item["process"] for item in group])
        for item in group:
            item["log"].close()
        raise


def main() -> None:
    args = _build_arg_parser().parse_args()
    gpu_ids = _csv(args.gpu_ids)
    jobs = build_workload(
        num_trials=args.num_trials,
        tasks=_csv(args.tasks) if args.tasks else None,
        configs=_csv(args.configs),
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(args.output_dir or (_ROOT / "outputs" / f"robotwin_multigpu_{stamp}")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    server_group: list[dict] = []
    try:
        pending, existing = _select_pending_jobs(jobs, output_dir)
        if pending:
            print(
                f"Starting {len(gpu_ids)} inference servers on GPUs "
                f"{','.join(gpu_ids)} ...",
                flush=True,
            )
            server_group = _start_server_group(args, gpu_ids, output_dir)
            print(
                f"All {len(server_group)} inference servers are ready; "
                f"dynamically scheduling {len(pending)} RoboTwin jobs ...",
                flush=True,
            )
            new_results = _run_job_queue(
                args, pending, server_group, output_dir,
            )
        else:
            new_results = []
        task_results = existing + new_results
        _validate_result_identities(jobs, task_results)
        records = _completed_timing_records(task_results)
        summary = aggregate_evaluation(records)
        backend_values = sorted({
            str(record.get("backend", "eager"))
            for record in records
            if record.get("record_type") == "inference"
        })
        runtime_values = [
            record.get("runtime", {})
            for record in records
            if record.get("record_type") == "inference" and record.get("runtime")
        ]
        requested_acceleration = bool(
            args.ac_stream_accelerated
            or (args.inference_mode == "ac-stream" and not args.ac_stream_eager)
        )
        mode_label = (
            f"ac-stream-{'accelerated' if requested_acceleration else 'eager'}"
            if args.inference_mode == "ac-stream"
            else args.inference_mode
        )
        completed = sum(item.get("status") == "completed" for item in task_results)
        skipped = sum(item.get("status") == "skipped_timeout" for item in task_results)
        infrastructure_errors = sum(
            item.get("status") == "infrastructure_error" for item in task_results
        )
        evaluation_status = (
            "COMPLETE"
            if completed == len(jobs)
            else "INCOMPLETE"
        )
        summary.update({
            "planned": len(jobs), "completed": completed,
            "skipped_timeout": skipped,
            "infrastructure_errors": infrastructure_errors,
            "status": evaluation_status,
        })
        results_path = output_dir / "results.json"
        _atomic_json(results_path, {
            "mode": mode_label, "gpu_ids": gpu_ids,
            "backend": backend_values,
            "runtime": runtime_values[0] if runtime_values else {},
            "summary": summary, "jobs": task_results, "timing": records,
        })
        print(format_summary(
            mode=mode_label, gpu_ids=gpu_ids, jobs=len(jobs),
            summary=summary, results_path=results_path,
            completed=completed,
            skipped=skipped + infrastructure_errors,
            status=evaluation_status,
        ))
        if evaluation_status != "COMPLETE":
            raise SystemExit(2)
    finally:
        _terminate_processes([item["process"] for item in server_group])
        for item in server_group:
            item["log"].close()


if __name__ == "__main__":
    main()

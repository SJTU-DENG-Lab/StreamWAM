"""Run balanced LIBERO evaluation with one persistent worker per GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples.libero.workload import build_worker_manifests, iter_manifest_trials  # noqa: E402


DEFAULT_SUITES = "libero_spatial,libero_object,libero_goal,libero_10"
TIMING_AVERAGES = (
    "average_inference_ms_per_chunk",
    "average_communication_ms_per_chunk",
    "average_action_execution_ms_per_chunk",
    "average_total_ms_per_chunk",
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate LIBERO suites across a configurable balanced GPU pool"
    )
    parser.add_argument("--gpus", required=True, help="Comma-separated physical GPU IDs, e.g. 0,1,2,3")
    parser.add_argument("--suites", default=DEFAULT_SUITES, help="Comma-separated LIBERO suites")
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Optional comma-separated task IDs applied to every selected suite",
    )
    parser.add_argument("--num-trials", type=int, default=1, help="Trials per task")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-format", choices=("streamwam", "fastwam"), default="streamwam")
    parser.add_argument("--backbone-path", default=None)
    parser.add_argument("--stats-path", default=None)
    parser.add_argument("--libero-home", default=os.environ.get("LIBERO_HOME"))
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--action-num-inference-steps", type=int, default=None)
    parser.add_argument(
        "--sampling-method",
        choices=("euler", "consistency", "ac-stream"),
        default=None,
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-seed", action="store_true")
    parser.add_argument("--ac-stream-accelerated", action="store_true")
    parser.add_argument("--mujoco-gl", choices=("osmesa", "egl", "glfw"), default=None)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--output-dir", default=None)
    return parser


def _split_csv(value: str, name: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one value")
    return items


def _append_option(command: list[str], name: str, value: Any) -> None:
    if value is not None:
        command.extend([name, str(value)])


def build_worker_command(
    *,
    args: argparse.Namespace,
    gpu: str,
    worker_id: int,
    manifest_path: Path,
    output_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    """Build an isolated rollout command for one physical GPU."""

    command = [
        sys.executable,
        str(REPO_ROOT / "examples" / "libero" / "rollout.py"),
        "--config",
        args.config,
        "--checkpoint",
        args.checkpoint,
        "--checkpoint-format",
        args.checkpoint_format,
        "--device",
        "cuda:0",
        "--seed",
        str(args.seed),
        "--num-steps-wait",
        str(args.num_steps_wait),
        "--output-dir",
        str(output_dir),
        "--work-manifest",
        str(manifest_path),
        "--worker-id",
        f"worker_{worker_id}",
        "--suppress-final-summary",
    ]
    for option, value in (
        ("--backbone-path", args.backbone_path),
        ("--stats-path", args.stats_path),
        ("--libero-home", args.libero_home),
        ("--replan-steps", args.replan_steps),
        ("--num-inference-steps", args.num_inference_steps),
        ("--action-num-inference-steps", args.action_num_inference_steps),
        ("--sampling-method", args.sampling_method),
        ("--max-steps", args.max_steps),
        ("--mujoco-gl", args.mujoco_gl),
    ):
        _append_option(command, option, value)
    if args.fixed_seed:
        command.append("--fixed-seed")
    if args.ac_stream_accelerated:
        command.append("--ac-stream-accelerated")
    if args.save_video:
        command.append("--save-video")
    if args.override:
        command.extend(["--override", *args.override])

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["LIBERO_CONFIG_PATH"] = str(output_dir / "libero_config")
    return command, environment


def _read_result(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(path)


def _prepare_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(percentile) / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _merge_acceleration_reports(
    reports: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Require one acceleration contract while retaining per-worker evidence."""

    if not reports:
        raise ValueError("No AC-Stream acceleration reports were provided")
    if all(report == reports[0] for report in reports[1:]):
        return reports[0]

    diagnostic_keys = {
        "dynamo_unique_graphs",
        "dynamo_recompiles",
        "inductor_cudagraph_skips",
        "runtime",
    }
    expected_contract = {
        key: value
        for key, value in reports[0].items()
        if key not in diagnostic_keys
    }
    for report in reports[1:]:
        contract = {
            key: value for key, value in report.items() if key not in diagnostic_keys
        }
        if contract != expected_contract:
            raise ValueError("Workers reported different AC-Stream acceleration contracts")

    runtime_variable_keys = {"gpu_name", "gpu_compute_capability"}
    runtimes = [dict(report.get("runtime") or {}) for report in reports]
    expected_runtime = {
        key: value
        for key, value in runtimes[0].items()
        if key not in runtime_variable_keys
    }
    for runtime in runtimes[1:]:
        comparable = {
            key: value
            for key, value in runtime.items()
            if key not in runtime_variable_keys
        }
        if comparable != expected_runtime:
            raise ValueError("Workers reported different AC-Stream software runtimes")

    merged = dict(expected_contract)
    for key in (
        "dynamo_unique_graphs",
        "dynamo_recompiles",
        "inductor_cudagraph_skips",
    ):
        merged[key] = sum(int(report.get(key, 0) or 0) for report in reports)
    merged["runtime"] = {
        **expected_runtime,
        "gpu_names": [runtime.get("gpu_name") for runtime in runtimes],
        "gpu_compute_capabilities": [
            runtime.get("gpu_compute_capability") for runtime in runtimes
        ],
    }
    merged["worker_diagnostics"] = [
        {
            "dynamo_unique_graphs": int(report.get("dynamo_unique_graphs", 0) or 0),
            "dynamo_recompiles": int(report.get("dynamo_recompiles", 0) or 0),
            "inductor_cudagraph_skips": int(
                report.get("inductor_cudagraph_skips", 0) or 0
            ),
            "gpu_name": runtime.get("gpu_name"),
            "gpu_compute_capability": runtime.get("gpu_compute_capability"),
        }
        for report, runtime in zip(reports, runtimes)
    ]
    return merged


def merge_worker_results(
    result_paths: Sequence[Path],
    *,
    command_wall_ms: float,
    expected_trials: Sequence[tuple[str, int, int]],
) -> dict[str, Any]:
    """Merge worker JSON files and weight timing averages by chunk count."""

    if not result_paths:
        raise ValueError("No worker results were provided")
    worker_results = [_read_result(Path(path)) for path in result_paths]
    expected_trial_set = set(expected_trials)
    if len(expected_trial_set) != len(expected_trials):
        raise ValueError("Expected manifests contain duplicate trial units")
    checkpoints = {result["checkpoint"] for result in worker_results}
    if len(checkpoints) != 1:
        raise ValueError(f"Workers reported different checkpoints: {sorted(checkpoints)}")
    acceleration_reports = [result.get("ac_stream_acceleration") for result in worker_results]
    acceleration_report = None
    if any(report is not None for report in acceleration_reports):
        if not all(report is not None for report in acceleration_reports):
            raise ValueError("Workers mixed reported and unreported AC-Stream acceleration")
        acceleration_report = _merge_acceleration_reports(
            [report for report in acceleration_reports if report is not None]
        )

    merged_tasks: dict[str, dict[str, Any]] = {}
    total_successes = 0
    total_trials = 0
    total_chunks = 0
    weighted_timing = {name: 0.0 for name in TIMING_AVERAGES}
    evaluation_workload_wall_ms = 0.0
    ac_stream_presence: list[bool] = []
    ac_stream_async_count = 0
    ac_stream_boundary_count = 0
    ac_stream_ready_count = 0
    ac_stream_miss_count = 0
    ac_stream_inference_wall_sum_ms = 0.0
    ac_stream_action_overlap_sum_ms = 0.0
    ac_stream_boundary_wait_sum_ms = 0.0
    ac_stream_hidden_ratio_sum = 0.0
    ac_stream_hidden_per_chunk_sum_ms = 0.0
    ac_stream_first_background_d8_values: list[float] = []
    ac_stream_steady_state_d8_values: list[float] = []
    actual_trial_set: set[tuple[str, int, int]] = set()

    for result in worker_results:
        worker_successes = 0
        worker_trials = 0
        total_successes += int(result["total_successes"])
        total_trials += int(result["total_trials"])
        worker_timing = result["timing_summary"]
        evaluation_workload_wall_ms += float(
            worker_timing.get("evaluation_workload_wall_ms", 0.0)
        )
        chunks = int(worker_timing["chunks_executed"])
        total_chunks += chunks
        for name in TIMING_AVERAGES:
            weighted_timing[name] += float(worker_timing[name]) * chunks
        worker_overlap = worker_timing.get("ac_stream_overlap")
        ac_stream_presence.append(worker_overlap is not None)
        if worker_overlap is not None:
            async_count = int(worker_overlap["async_d8_inferences"])
            boundary_count = int(worker_overlap["boundary_evaluated_inferences"])
            ac_stream_async_count += async_count
            ac_stream_boundary_count += boundary_count
            ac_stream_ready_count += int(worker_overlap["ready_before_boundary"])
            ac_stream_miss_count += int(worker_overlap["deadline_misses"])
            ac_stream_inference_wall_sum_ms += (
                float(worker_overlap["average_inference_wall_ms"]) * async_count
            )
            ac_stream_action_overlap_sum_ms += (
                float(worker_overlap["average_action_overlap_ms"]) * async_count
            )
            ac_stream_boundary_wait_sum_ms += (
                float(worker_overlap["average_boundary_wait_ms"]) * boundary_count
            )
            ac_stream_hidden_ratio_sum += (
                float(worker_overlap["average_hidden_inference_ratio"]) * async_count
            )
            ac_stream_hidden_per_chunk_sum_ms += (
                float(worker_overlap["inference_hidden_ms_per_chunk"]) * chunks
            )
            if int(worker_overlap.get("first_background_d8_inference_count", 0)):
                ac_stream_first_background_d8_values.append(
                    float(worker_overlap["first_background_d8_inference_ms"])
                )
            ac_stream_steady_state_d8_values.extend(
                float(value)
                for value in worker_overlap.get(
                    "steady_state_d8_inference_ms_values", []
                )
            )

        for task_key, task in result["task_results"].items():
            episodes = task["episodes"]
            episode_successes = sum(bool(episode["success"]) for episode in episodes)
            if int(task["trials"]) != len(episodes):
                raise ValueError(f"Worker task {task_key} trials does not match its episodes")
            if int(task["successes"]) != episode_successes:
                raise ValueError(f"Worker task {task_key} successes does not match its episodes")
            worker_trials += len(episodes)
            worker_successes += episode_successes
            suite_name = str(task["task_suite_name"])
            task_id = int(task["task_id"])
            for episode in episodes:
                unit = (suite_name, task_id, int(episode["trial"]))
                if unit in actual_trial_set:
                    raise ValueError(f"Duplicate worker trial unit: {unit}")
                actual_trial_set.add(unit)
            if task_key not in merged_tasks:
                merged_tasks[task_key] = {
                    **task,
                    "successes": 0,
                    "trials": 0,
                    "success_rate": 0.0,
                    "episodes": [],
                }
            merged = merged_tasks[task_key]
            merged["successes"] += int(task["successes"])
            merged["trials"] += int(task["trials"])
            merged["episodes"].extend(task["episodes"])
        if worker_trials != int(result["total_trials"]):
            raise ValueError("Worker total_trials does not match its task episodes")
        if worker_successes != int(result["total_successes"]):
            raise ValueError("Worker total_successes does not match its task episodes")

    missing = expected_trial_set - actual_trial_set
    unexpected = actual_trial_set - expected_trial_set
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)[:5]}")
        if unexpected:
            details.append(f"unexpected={sorted(unexpected)[:5]}")
        raise ValueError("Worker results do not match manifests: " + " ".join(details))
    if any(ac_stream_presence) and not all(ac_stream_presence):
        raise ValueError("Workers mixed AC-Stream and non-AC-Stream timing summaries")

    for task_key, task in merged_tasks.items():
        trial_ids = [int(episode["trial"]) for episode in task["episodes"]]
        if len(trial_ids) != len(set(trial_ids)):
            raise ValueError(f"Duplicate trial IDs found while merging {task_key}")
        task["episodes"].sort(key=lambda episode: int(episode["trial"]))
        task["success_rate"] = task["successes"] / max(task["trials"], 1)

    timing_summary: dict[str, Any] = {
        "tasks_executed": len(merged_tasks),
        "trials_executed": total_trials,
        "chunks_executed": total_chunks,
    }
    for name in TIMING_AVERAGES:
        timing_summary[name] = weighted_timing[name] / total_chunks if total_chunks else 0.0
    timing_summary["average_episode_wall_ms"] = (
        evaluation_workload_wall_ms / total_trials if total_trials else 0.0
    )
    timing_summary["evaluation_workload_wall_ms"] = evaluation_workload_wall_ms
    timing_summary["command_wall_ms"] = float(command_wall_ms)
    successful_episode_ms: dict[str, list[float]] = {
        "libero_10": [],
        "libero_goal": [],
    }
    for task in merged_tasks.values():
        suite_name = str(task["task_suite_name"])
        if suite_name not in successful_episode_ms:
            continue
        successful_episode_ms[suite_name].extend(
            float(episode["episode_wall_ms"])
            for episode in task["episodes"]
            if bool(episode["success"]) and "episode_wall_ms" in episode
        )
    if any(successful_episode_ms.values()):
        long_values = successful_episode_ms["libero_10"]
        short_values = successful_episode_ms["libero_goal"]
        timing_summary["readme_aligned"] = {
            "chunk_time_ms_mean": timing_summary["average_inference_ms_per_chunk"],
            "long_successful_episode_s_mean": (
                sum(long_values) / len(long_values) / 1000.0 if long_values else 0.0
            ),
            "long_successful_episode_count": len(long_values),
            "short_successful_episode_s_mean": (
                sum(short_values) / len(short_values) / 1000.0 if short_values else 0.0
            ),
            "short_successful_episode_count": len(short_values),
        }
    if ac_stream_presence and all(ac_stream_presence):
        arithmetic_total_ms = weighted_timing["average_total_ms_per_chunk"]
        timing_summary["ac_stream_overlap"] = {
            "async_d8_inferences": ac_stream_async_count,
            "boundary_evaluated_inferences": ac_stream_boundary_count,
            "ready_before_boundary": ac_stream_ready_count,
            "ready_before_boundary_rate": (
                ac_stream_ready_count / ac_stream_boundary_count if ac_stream_boundary_count else 0.0
            ),
            "deadline_misses": ac_stream_miss_count,
            "deadline_miss_rate": (
                ac_stream_miss_count / ac_stream_boundary_count if ac_stream_boundary_count else 0.0
            ),
            "average_inference_wall_ms": (
                ac_stream_inference_wall_sum_ms / ac_stream_async_count if ac_stream_async_count else 0.0
            ),
            "first_background_d8_inference_ms": (
                sum(ac_stream_first_background_d8_values)
                / len(ac_stream_first_background_d8_values)
                if ac_stream_first_background_d8_values
                else 0.0
            ),
            "first_background_d8_inference_count": len(
                ac_stream_first_background_d8_values
            ),
            "steady_state_d8_count": len(ac_stream_steady_state_d8_values),
            "steady_state_d8_mean_ms": (
                sum(ac_stream_steady_state_d8_values) / len(ac_stream_steady_state_d8_values)
                if ac_stream_steady_state_d8_values
                else 0.0
            ),
            "steady_state_d8_p50_ms": _percentile(
                ac_stream_steady_state_d8_values, 50
            ),
            "steady_state_d8_p90_ms": _percentile(
                ac_stream_steady_state_d8_values, 90
            ),
            "steady_state_d8_inference_ms_values": ac_stream_steady_state_d8_values,
            "average_action_overlap_ms": (
                ac_stream_action_overlap_sum_ms / ac_stream_async_count if ac_stream_async_count else 0.0
            ),
            "average_boundary_wait_ms": (
                ac_stream_boundary_wait_sum_ms / ac_stream_boundary_count
                if ac_stream_boundary_count
                else 0.0
            ),
            "average_hidden_inference_ratio": (
                ac_stream_hidden_ratio_sum / ac_stream_async_count if ac_stream_async_count else 0.0
            ),
            "average_effective_total_ms_per_chunk": (
                max(0.0, arithmetic_total_ms - ac_stream_hidden_per_chunk_sum_ms)
                / total_chunks
                if total_chunks
                else 0.0
            ),
            "inference_hidden_ms_per_chunk": (
                ac_stream_hidden_per_chunk_sum_ms / total_chunks if total_chunks else 0.0
            ),
        }

    merged_result = {
        "checkpoint": next(iter(checkpoints)),
        "task_suites": sorted(
            {str(task["task_suite_name"]) for task in merged_tasks.values()}
        ),
        "task_results": merged_tasks,
        "total_successes": total_successes,
        "total_trials": total_trials,
        "success_rate": total_successes / max(total_trials, 1),
        "timing_summary": timing_summary,
    }
    if acceleration_report is not None:
        merged_result["ac_stream_acceleration"] = acceleration_report
    return merged_result


def _format_summary(
    result: dict[str, Any],
    gpus: Sequence[str],
    *,
    results_path: str | Path | None = None,
) -> str:
    timing = result["timing_summary"]
    lines = [
        "=== Multi-GPU LIBERO Evaluation Summary ===",
        f"GPUs: {','.join(gpus)}",
        f"Tasks: {timing['tasks_executed']}",
        f"Trials: {result['total_trials']}",
        "Success: "
        f"{result['total_successes']}/{result['total_trials']} "
        f"({result['success_rate']:.4f})",
        f"Chunks: {timing['chunks_executed']}",
        f"Chunk Time: {timing['average_inference_ms_per_chunk']:.2f} ms",
        f"Total Time / Episode: {timing['average_episode_wall_ms'] / 1000.0:.2f} s",
    ]
    if results_path is not None:
        lines.append(f"Results: {results_path}")
    return "\n".join(lines)


def _terminate_running(processes: Sequence[subprocess.Popen[Any]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def main() -> None:
    args = _build_arg_parser().parse_args()
    gpus = _split_csv(args.gpus, "--gpus")
    suites = _split_csv(args.suites, "--suites")
    task_ids = (
        [int(task_id) for task_id in _split_csv(args.task_ids, "--task-ids")]
        if args.task_ids is not None
        else None
    )
    manifests = build_worker_manifests(
        suites=suites,
        num_trials=args.num_trials,
        gpus=gpus,
        task_ids=task_ids,
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
    output_dir = Path(args.output_dir or f"outputs/libero_multigpu_{timestamp}").resolve()
    _prepare_output_dir(output_dir)
    assignment_payload = {
        "gpus": gpus,
        "suites": suites,
        "task_ids": task_ids,
        "num_trials_per_task": args.num_trials,
        "total_workload_size": sum(item["workload_size"] for item in manifests),
        "workers": manifests,
    }
    _write_json_atomic(output_dir / "assignments.json", assignment_payload)

    processes: list[subprocess.Popen[Any]] = []
    log_handles: list[Any] = []
    worker_result_paths: list[Path] = []
    command_start = time.perf_counter_ns()
    try:
        for manifest in manifests:
            worker_id = int(manifest["worker_id"])
            gpu = str(manifest["gpu"])
            manifest_path = output_dir / f"worker_{worker_id}_manifest.json"
            worker_dir = output_dir / f"worker_gpu{gpu}"
            worker_dir.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(manifest_path, manifest)
            command, environment = build_worker_command(
                args=args,
                gpu=gpu,
                worker_id=worker_id,
                manifest_path=manifest_path,
                output_dir=worker_dir,
            )
            log_handle = open(worker_dir / "worker.log", "w", encoding="utf-8")
            log_handles.append(log_handle)
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
            )
            worker_result_paths.append(worker_dir / "results.json")

        failed: tuple[int, int] | None = None
        while any(process.poll() is None for process in processes):
            for index, process in enumerate(processes):
                return_code = process.poll()
                if return_code not in (None, 0):
                    failed = index, return_code
                    break
            if failed is not None:
                break
            time.sleep(0.2)
        if failed is None:
            for index, process in enumerate(processes):
                return_code = process.wait()
                if return_code != 0:
                    failed = index, return_code
                    break
        if failed is not None:
            _terminate_running(processes)
            index, return_code = failed
            gpu = manifests[index]["gpu"]
            raise RuntimeError(
                f"Worker {index} on GPU {gpu} failed with exit code {return_code}; "
                f"see {output_dir / f'worker_gpu{gpu}' / 'worker.log'}"
            )
    except BaseException:
        _terminate_running(processes)
        raise
    finally:
        for handle in log_handles:
            handle.close()

    command_wall_ms = (time.perf_counter_ns() - command_start) / 1e6
    expected_trials = [
        trial
        for manifest in manifests
        for trial in iter_manifest_trials(manifest)
    ]
    result = merge_worker_results(
        worker_result_paths,
        command_wall_ms=command_wall_ms,
        expected_trials=expected_trials,
    )
    result["gpus"] = gpus
    result["num_workers"] = len(gpus)
    result["assignments_file"] = str(output_dir / "assignments.json")
    _write_json_atomic(output_dir / "results.json", result)
    print(
        _format_summary(
            result,
            gpus,
            results_path=output_dir / "results.json",
        )
    )


if __name__ == "__main__":
    main()

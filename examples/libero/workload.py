"""Deterministic LIBERO workload manifests for multi-GPU evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_TASK_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}


def _validate_inputs(
    suites: Sequence[str],
    num_trials: int,
    gpus: Sequence[str],
    task_counts: Mapping[str, int],
) -> None:
    if not suites:
        raise ValueError("At least one LIBERO suite is required")
    if len(set(suites)) != len(suites):
        raise ValueError("LIBERO suite names must be unique")
    if num_trials <= 0:
        raise ValueError("num_trials must be positive")
    if not gpus:
        raise ValueError("At least one GPU is required")
    if any(not gpu.strip() for gpu in gpus):
        raise ValueError("GPU IDs must not be empty")
    if len(set(gpus)) != len(gpus):
        raise ValueError("GPU IDs must be unique")
    unknown = [suite for suite in suites if suite not in task_counts]
    if unknown:
        raise ValueError(f"Unknown LIBERO suites: {', '.join(unknown)}")
    invalid = [suite for suite in suites if task_counts[suite] <= 0]
    if invalid:
        raise ValueError(f"Task counts must be positive: {', '.join(invalid)}")


def _group_assignments(
    trials: Sequence[tuple[str, int, int]],
) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    for suite, task_id, trial_id in trials:
        if (
            not assignments
            or assignments[-1]["task_suite_name"] != suite
            or assignments[-1]["task_id"] != task_id
        ):
            assignments.append(
                {
                    "task_suite_name": suite,
                    "task_id": task_id,
                    "trial_ids": [],
                }
            )
        assignments[-1]["trial_ids"].append(trial_id)
    return assignments


def build_worker_manifests(
    *,
    suites: Sequence[str],
    num_trials: int,
    gpus: Sequence[str],
    task_counts: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Split ordered ``(suite, task, trial)`` units across GPU workers."""

    counts = DEFAULT_TASK_COUNTS if task_counts is None else task_counts
    normalized_suites = [str(suite).strip() for suite in suites]
    normalized_gpus = [str(gpu).strip() for gpu in gpus]
    _validate_inputs(normalized_suites, num_trials, normalized_gpus, counts)

    trials = [
        (suite, task_id, trial_id)
        for suite in normalized_suites
        for task_id in range(counts[suite])
        for trial_id in range(num_trials)
    ]
    if len(normalized_gpus) > len(trials):
        raise ValueError("GPU count cannot exceed the number of evaluation trials")

    manifests: list[dict[str, Any]] = []
    for worker_id, gpu in enumerate(normalized_gpus):
        # Striding keeps counts within one while mixing suites/tasks across
        # workers, avoiding a slow suite dominating a single GPU's queue.
        worker_trials = trials[worker_id :: len(normalized_gpus)]
        manifests.append(
            {
                "worker_id": worker_id,
                "gpu": gpu,
                "workload_size": len(worker_trials),
                "assignments": _group_assignments(worker_trials),
            }
        )
    return manifests


def iter_manifest_trials(manifest: Mapping[str, Any]) -> Iterator[tuple[str, int, int]]:
    """Yield the explicit trial units stored in a worker manifest."""

    for assignment in manifest["assignments"]:
        suite = assignment["task_suite_name"]
        task_id = assignment["task_id"]
        for trial_id in assignment["trial_ids"]:
            yield suite, task_id, trial_id


def load_worker_manifest(path: str | Path) -> dict[str, Any]:
    """Load a worker manifest and validate its declared trial count."""

    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("assignments"), list):
        raise ValueError("Worker manifest must contain an assignments list")
    try:
        actual_size = sum(1 for _ in iter_manifest_trials(manifest))
    except (KeyError, TypeError) as exc:
        raise ValueError("Worker manifest contains an invalid assignment") from exc
    if manifest.get("workload_size") != actual_size:
        raise ValueError(
            "Worker manifest workload_size does not match its explicit trial IDs: "
            f"declared={manifest.get('workload_size')} actual={actual_size}"
        )
    return manifest

"""Benchmark-facing RoboTwin success, Chunk Time, and Total Time aggregation."""

from __future__ import annotations

from collections import defaultdict
import statistics
from typing import Iterable


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _chunk_records(records: list[dict]) -> list[dict]:
    inference = [
        record
        for record in records
        if record.get("record_type") == "inference"
        and not bool(record.get("warmup"))
        and "model_inference_ms" in record
    ]
    d8 = [record for record in inference if record.get("regime") == "d8"]
    if d8 or any(record.get("regime") == "d0" for record in inference):
        return d8
    return inference


def aggregate_evaluation(records: Iterable[dict]) -> dict:
    """Aggregate fixed-episode results without hiding task-level variation.

    Chunk Time is the CUDA-synchronized model call (D8 for AC-Stream, sync for
    baseline/CD). Total Time directly averages all completed episodes,
    including failures. Successful-only timing remains a diagnostic.
    """

    records = list(records)
    selected_chunks = _chunk_records(records)
    episodes = [
        record
        for record in records
        if record.get("record_type") == "episode" and "total_time_s" in record
    ]
    successful = [record for record in episodes if record.get("success") is True]

    setting_episodes: dict[tuple[str, str], list[dict]] = defaultdict(list)
    episode_times: dict[tuple[str, str], list[float]] = defaultdict(list)
    setting_chunks: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in episodes:
        key = (str(record.get("config")), str(record.get("task")))
        setting_episodes[key].append(record)
        episode_times[key].append(float(record["total_time_s"]))
    for record in selected_chunks:
        key = (str(record.get("config")), str(record.get("task")))
        setting_chunks[key].append(float(record["model_inference_ms"]))

    by_setting = {}
    for key, selected in sorted(setting_episodes.items()):
        config, task = key
        successes = sum(record.get("success") is True for record in selected)
        attempts = len(selected)
        times = episode_times.get(key, [])
        by_setting[f"{config}/{task}"] = {
            "task": task,
            "config": config,
            "successes": successes,
            "episodes": attempts,
            "success_rate": successes / attempts if attempts else None,
            "total_time_s": _mean(times),
            "timed_episodes": len(times),
            "chunk_time_ms": _mean(setting_chunks.get(key, [])),
        }

    task_settings: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in setting_episodes:
        task_settings[key[1]].append(key)
    by_task = {}
    for task, keys in sorted(task_settings.items()):
        selected = [record for key in keys for record in setting_episodes[key]]
        successes = sum(record.get("success") is True for record in selected)
        attempts = len(selected)
        times = [float(record["total_time_s"]) for record in selected]
        by_task[task] = {
            "successes": successes,
            "episodes": attempts,
            "success_rate": successes / attempts if attempts else None,
            "total_time_s": _mean(times),
            "covered_configs": len(keys),
        }

    config_settings: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in setting_episodes:
        config_settings[key[0]].append(key)
    by_config = {}
    for config, keys in sorted(config_settings.items()):
        selected = [record for key in keys for record in setting_episodes[key]]
        successes = sum(record.get("success") is True for record in selected)
        attempts = len(selected)
        times = [float(record["total_time_s"]) for record in selected]
        config_chunk_values = [
            value for key in keys for value in setting_chunks.get(key, [])
        ]
        by_config[config] = {
            "successes": successes,
            "episodes": attempts,
            "success_rate": successes / attempts if attempts else None,
            "total_time_s": _mean(times),
            "covered_settings": len(keys),
            "chunk_time_ms": _mean(config_chunk_values),
        }

    all_episode_times = [float(record["total_time_s"]) for record in episodes]
    chunks = [float(record["model_inference_ms"]) for record in selected_chunks]
    all_inference = [
        record
        for record in records
        if record.get("record_type") == "inference"
        and not bool(record.get("warmup"))
        and "model_inference_ms" in record
    ]
    by_regime = {
        regime: _mean(
            [
                float(record["model_inference_ms"])
                for record in all_inference
                if record.get("regime") == regime
            ]
        )
        for regime in ("sync", "d0", "d8")
        if any(record.get("regime") == regime for record in all_inference)
    }

    return {
        "chunk_time_ms": _mean(chunks),
        "chunk_time_by_regime_ms": by_regime,
        "total_time_per_episode_s": _mean(all_episode_times),
        "successful_episode_time_s": _mean(
            [float(record["total_time_s"]) for record in successful]
        ),
        "chunks": len(chunks),
        "successes": len(successful),
        "episodes": len(episodes),
        "success_rate": len(successful) / len(episodes) if episodes else None,
        "timed_settings": len(episode_times),
        "by_config": by_config,
        "by_task": by_task,
        "by_setting": by_setting,
    }

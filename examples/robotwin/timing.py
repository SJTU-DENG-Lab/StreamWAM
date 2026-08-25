"""Small, benchmark-facing RoboTwin timing aggregation helpers."""

from __future__ import annotations

import statistics
from typing import Iterable


def aggregate_evaluation(records: Iterable[dict]) -> dict:
    records = list(records)
    chunks = [
        float(record["model_inference_ms"])
        for record in records
        if record.get("record_type") == "inference"
        and not bool(record.get("warmup"))
        and "model_inference_ms" in record
    ]
    episodes = [
        record
        for record in records
        if record.get("record_type") == "episode" and "total_time_s" in record
    ]
    episode_times = [float(record["total_time_s"]) for record in episodes]
    return {
        "chunk_time_ms": statistics.fmean(chunks) if chunks else None,
        "total_time_per_episode_s": statistics.fmean(episode_times) if episode_times else None,
        "chunks": len(chunks),
        "successes": sum(record.get("success") is True for record in episodes),
        "episodes": len(episodes),
    }

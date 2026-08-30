"""Coarse LIBERO chunk timing and global aggregation.

Only three material components are measured:

* communication: observation/context/proprio preparation, CPU-to-device input
  transfer, device-to-CPU action transfer, denormalization, and NumPy conversion;
* inference: the synchronized ``model.infer_action`` call;
* action execution: wall time spent in ``env.step(action)`` for actions from a
  generated chunk.

Small Python, rendering, reset, warm-up, logging, and video-writing overheads
are intentionally not itemized. No per-chunk/task timing is printed or saved;
this module produces one aggregate for the complete evaluation command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from streamingwam.inference.ac_stream import ACStreamOverlapRecord


@dataclass
class ChunkTiming:
    communication_ms: float
    inference_ms: float
    action_execution_ms: float = 0.0

    def add_action_execution(self, elapsed_ms: float) -> None:
        self.action_execution_ms += float(elapsed_ms)

    @property
    def total_ms(self) -> float:
        return self.communication_ms + self.inference_ms + self.action_execution_ms


@dataclass
class GlobalTimingSummary:
    task_count: int = 0
    trial_count: int = 0
    chunks: list[ChunkTiming] = field(default_factory=list)
    ac_stream_enabled: bool = False
    ac_stream_overlap_records: list[ACStreamOverlapRecord] = field(default_factory=list)
    episode_wall_ms: list[float] = field(default_factory=list)

    def enable_ac_stream(self) -> None:
        self.ac_stream_enabled = True

    def add_ac_stream_overlap(self, record: ACStreamOverlapRecord) -> None:
        if not self.ac_stream_enabled:
            raise RuntimeError("Enable AC-Stream timing before adding overlap records")
        self.ac_stream_overlap_records.append(record)

    def add_episode_wall(self, elapsed_ms: float) -> None:
        self.episode_wall_ms.append(float(elapsed_ms))

    def add_chunk(self, *, communication_ms: float, inference_ms: float) -> ChunkTiming:
        chunk = ChunkTiming(
            communication_ms=float(communication_ms),
            inference_ms=float(inference_ms),
        )
        self.chunks.append(chunk)
        return chunk

    def _ac_stream_overlap_dict(self) -> dict[str, int | float]:
        records = self.ac_stream_overlap_records
        boundary_records = [
            record for record in records if record.ready_before_boundary is not None
        ]

        def average(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        ready = sum(record.ready_before_boundary is True for record in boundary_records)
        misses = sum(record.ready_before_boundary is False for record in boundary_records)
        boundary_count = len(boundary_records)
        overlap_total_ms = sum(record.action_overlap_ms for record in records)
        inference_samples = [record.inference_wall_ms for record in records]
        steady_state_samples = inference_samples[1:]
        chunk_count = len(self.chunks)
        arithmetic_total_ms = sum(chunk.total_ms for chunk in self.chunks)
        return {
            "async_d8_inferences": len(records),
            "boundary_evaluated_inferences": boundary_count,
            "ready_before_boundary": ready,
            "ready_before_boundary_rate": ready / boundary_count if boundary_count else 0.0,
            "deadline_misses": misses,
            "deadline_miss_rate": misses / boundary_count if boundary_count else 0.0,
            "average_inference_wall_ms": average(
                inference_samples
            ),
            "first_background_d8_inference_ms": (
                inference_samples[0] if inference_samples else 0.0
            ),
            "first_background_d8_inference_count": 1 if inference_samples else 0,
            "steady_state_d8_count": len(steady_state_samples),
            "steady_state_d8_mean_ms": average(steady_state_samples),
            "steady_state_d8_p50_ms": (
                float(np.percentile(steady_state_samples, 50))
                if steady_state_samples
                else 0.0
            ),
            "steady_state_d8_p90_ms": (
                float(np.percentile(steady_state_samples, 90))
                if steady_state_samples
                else 0.0
            ),
            "steady_state_d8_inference_ms_values": steady_state_samples,
            "average_action_overlap_ms": average(
                [record.action_overlap_ms for record in records]
            ),
            "average_boundary_wait_ms": average(
                [record.boundary_wait_ms for record in boundary_records]
            ),
            "average_hidden_inference_ratio": average(
                [record.hidden_inference_ratio for record in records]
            ),
            "average_effective_total_ms_per_chunk": (
                max(0.0, arithmetic_total_ms - overlap_total_ms) / chunk_count
                if chunk_count
                else 0.0
            ),
            "inference_hidden_ms_per_chunk": (
                overlap_total_ms / chunk_count if chunk_count else 0.0
            ),
        }

    def as_dict(self, *, command_wall_ms: float) -> dict[str, Any]:
        count = len(self.chunks)

        def average(values: list[float]) -> float:
            return sum(values) / count if count else 0.0

        average_inference_ms = average(
            [chunk.inference_ms for chunk in self.chunks]
        )
        summary: dict[str, Any] = {
            "tasks_executed": int(self.task_count),
            "trials_executed": int(self.trial_count),
            "chunks_executed": count,
            "chunk_time_ms": average_inference_ms,
            "average_inference_ms_per_chunk": average_inference_ms,
            "average_communication_ms_per_chunk": average(
                [chunk.communication_ms for chunk in self.chunks]
            ),
            "average_action_execution_ms_per_chunk": average(
                [chunk.action_execution_ms for chunk in self.chunks]
            ),
            "average_total_ms_per_chunk": average(
                [chunk.total_ms for chunk in self.chunks]
            ),
            "average_episode_wall_ms": (
                sum(self.episode_wall_ms) / len(self.episode_wall_ms)
                if self.episode_wall_ms
                else 0.0
            ),
            "evaluation_workload_wall_ms": sum(self.episode_wall_ms),
            "command_wall_ms": float(command_wall_ms),
        }
        if self.ac_stream_enabled:
            overlap = self._ac_stream_overlap_dict()
            summary["ac_stream_overlap"] = overlap
            summary["chunk_time_ms"] = (
                overlap["average_inference_wall_ms"]
                if overlap["async_d8_inferences"]
                else None
            )
        return summary

    def format_summary(self, *, command_wall_ms: float) -> str:
        summary = self.as_dict(command_wall_ms=command_wall_ms)
        chunk_time_ms = summary["chunk_time_ms"]
        chunk_time = "N/A" if chunk_time_ms is None else f"{chunk_time_ms:.2f} ms"
        lines = [
            "========== LIBERO Timing Summary ==========",
            f"tasks executed                 : {summary['tasks_executed']}",
            f"trials executed                : {summary['trials_executed']}",
            f"chunks executed                : {summary['chunks_executed']}",
            f"Chunk Time                      : {chunk_time}",
            "average inference/chunk        : "
            f"{summary['average_inference_ms_per_chunk']:.2f} ms",
            "average communication/chunk    : "
            f"{summary['average_communication_ms_per_chunk']:.2f} ms",
            "average action execution/chunk : "
            f"{summary['average_action_execution_ms_per_chunk']:.2f} ms",
            "average total/chunk            : "
            f"{summary['average_total_ms_per_chunk']:.2f} ms",
            "average episode wall time      : "
            f"{summary['average_episode_wall_ms'] / 1000.0:.2f} s",
            "evaluation workload wall time  : "
            f"{summary['evaluation_workload_wall_ms'] / 1000.0:.2f} s",
            f"complete command wall time     : {summary['command_wall_ms'] / 1000.0:.2f} s",
            "===========================================",
        ]
        if self.ac_stream_enabled:
            overlap = summary["ac_stream_overlap"]
            boundary_count = overlap["boundary_evaluated_inferences"]
            lines.extend(
                [
                    "========== AC-Stream Async Overlap ==========",
                    "async D8 inferences             : "
                    f"{overlap['async_d8_inferences']}",
                    "ready before chunk boundary     : "
                    f"{overlap['ready_before_boundary']}/{boundary_count} "
                    f"({overlap['ready_before_boundary_rate'] * 100.0:.2f}%)",
                    "average inference wall time     : "
                    f"{overlap['average_inference_wall_ms']:.2f} ms",
                    "first background D8 inference  : "
                    f"{overlap['first_background_d8_inference_ms']:.2f} ms",
                    "steady-state D8 inference       : "
                    f"mean={overlap['steady_state_d8_mean_ms']:.2f} ms "
                    f"p50={overlap['steady_state_d8_p50_ms']:.2f} ms "
                    f"p90={overlap['steady_state_d8_p90_ms']:.2f} ms "
                    f"(n={overlap['steady_state_d8_count']})",
                    "average action overlap time     : "
                    f"{overlap['average_action_overlap_ms']:.2f} ms",
                    "average boundary wait time      : "
                    f"{overlap['average_boundary_wait_ms']:.2f} ms",
                    "average hidden inference ratio  : "
                    f"{overlap['average_hidden_inference_ratio'] * 100.0:.2f}%",
                    "deadline misses                 : "
                    f"{overlap['deadline_misses']}/{boundary_count} "
                    f"({overlap['deadline_miss_rate'] * 100.0:.2f}%)",
                    "average effective time/chunk    : "
                    f"{overlap['average_effective_total_ms_per_chunk']:.2f} ms",
                    "inference hidden by actions     : "
                    f"{overlap['inference_hidden_ms_per_chunk']:.2f} ms/chunk",
                    "==========================================",
                ]
            )
        return "\n".join(lines)

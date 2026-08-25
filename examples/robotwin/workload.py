"""Deterministic RoboTwin 2.0 evaluation workload construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


ROBOTWIN_TASKS = (
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb",
    "blocks_ranking_size", "click_alarmclock", "click_bell", "dump_bin_bigbin",
    "grab_roller", "handover_block", "handover_mic", "lift_pot", "move_can_pot",
    "move_playingcard_away", "move_stapler_pad", "hanging_mug", "open_laptop",
    "open_microwave", "pick_diverse_bottles", "pick_dual_bottles",
    "place_a2b_left", "place_a2b_right", "place_bread_basket",
    "place_bread_skillet", "place_can_basket", "place_cans_plasticbox",
    "place_container_plate", "place_dual_shoes", "place_empty_cup", "place_fan",
    "place_burger_fries", "place_mouse_pad", "place_object_basket",
    "place_object_scale", "place_object_stand", "place_phone_stand",
    "move_pillbottle_pad", "place_shoe", "press_stapler", "put_bottles_dustbin",
    "put_object_cabinet", "rotate_qrcode", "scan_object", "shake_bottle",
    "shake_bottle_horizontally", "stack_blocks_three", "stack_blocks_two",
    "stack_bowls_three", "stack_bowls_two", "stamp_seal", "turn_switch",
)
ROBOTWIN_CONFIGS = ("demo_clean", "demo_randomized")


@dataclass(frozen=True)
class RoboTwinJob:
    task: str
    config: str
    trial: int

    @property
    def identity(self) -> tuple[str, str, int]:
        return self.task, self.config, self.trial


def _validate_selection(selected: Sequence[str], allowed: Sequence[str], label: str) -> None:
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown RoboTwin {label}: {', '.join(unknown)}")


def build_workload(
    *,
    num_trials: int,
    tasks: Sequence[str] | None = None,
    configs: Sequence[str] | None = None,
) -> list[RoboTwinJob]:
    if num_trials < 1:
        raise ValueError("num_trials must be at least 1")
    selected_tasks = tuple(tasks or ROBOTWIN_TASKS)
    selected_configs = tuple(configs or ROBOTWIN_CONFIGS)
    _validate_selection(selected_tasks, ROBOTWIN_TASKS, "tasks")
    _validate_selection(selected_configs, ROBOTWIN_CONFIGS, "configs")
    return [
        RoboTwinJob(task=task, config=config, trial=trial)
        for trial in range(num_trials)
        for config in selected_configs
        for task in selected_tasks
    ]


def distribute_workload(
    jobs: Iterable[RoboTwinJob], gpu_ids: Sequence[str]
) -> dict[str, list[RoboTwinJob]]:
    if not gpu_ids:
        raise ValueError("At least one GPU ID is required")
    assignments = {str(gpu): [] for gpu in gpu_ids}
    for index, job in enumerate(jobs):
        assignments[str(gpu_ids[index % len(gpu_ids)])].append(job)
    return assignments

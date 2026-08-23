import json

import pytest

from examples.libero.rollout import _build_arg_parser, _build_evaluation_assignments
from examples.libero.workload import (
    build_worker_manifests,
    iter_manifest_trials,
    load_worker_manifest,
)


@pytest.mark.parametrize(("gpu_count", "expected_per_gpu"), [(4, 500), (8, 250)])
def test_full_libero_50_trials_is_evenly_balanced(
    gpu_count: int,
    expected_per_gpu: int,
) -> None:
    gpus = [str(index) for index in range(gpu_count)]

    manifests = build_worker_manifests(
        suites=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        num_trials=50,
        gpus=gpus,
    )

    assert [manifest["workload_size"] for manifest in manifests] == [
        expected_per_gpu
    ] * gpu_count
    assigned = [trial for manifest in manifests for trial in iter_manifest_trials(manifest)]
    assert len(assigned) == 2000
    assert len(set(assigned)) == 2000


def test_non_divisible_workload_differs_by_at_most_one_trial() -> None:
    manifests = build_worker_manifests(
        suites=["tiny"],
        num_trials=5,
        gpus=["2", "3", "7", "9"],
        task_counts={"tiny": 3},
    )

    sizes = [manifest["workload_size"] for manifest in manifests]
    assert sizes == [4, 4, 4, 3]
    assert max(sizes) - min(sizes) == 1


def test_single_trial_workers_mix_suites_for_runtime_balance() -> None:
    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    manifests = build_worker_manifests(
        suites=suites,
        num_trials=1,
        gpus=["0", "1", "2", "3"],
    )

    assert [manifest["workload_size"] for manifest in manifests] == [10, 10, 10, 10]
    for manifest in manifests:
        assigned_suites = {suite for suite, _, _ in iter_manifest_trials(manifest)}
        assert assigned_suites == set(suites)


def test_duplicate_gpu_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        build_worker_manifests(
            suites=["libero_spatial"],
            num_trials=1,
            gpus=["0", "0"],
        )


def test_duplicate_suite_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="suite names must be unique"):
        build_worker_manifests(
            suites=["libero_spatial", " libero_spatial "],
            num_trials=1,
            gpus=["0"],
        )


def test_worker_manifest_round_trip_preserves_explicit_trial_ids(tmp_path) -> None:
    manifest_path = tmp_path / "worker.json"
    manifest_path.write_text(
        json.dumps(
            {
                "worker_id": 2,
                "gpu": "7",
                "workload_size": 3,
                "assignments": [
                    {
                        "task_suite_name": "libero_goal",
                        "task_id": 4,
                        "trial_ids": [7, 8, 9],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_worker_manifest(manifest_path)

    assert list(iter_manifest_trials(manifest)) == [
        ("libero_goal", 4, 7),
        ("libero_goal", 4, 8),
        ("libero_goal", 4, 9),
    ]


def test_worker_manifest_rejects_declared_workload_size_mismatch(tmp_path) -> None:
    manifest_path = tmp_path / "worker.json"
    manifest_path.write_text(
        json.dumps(
            {
                "worker_id": 0,
                "gpu": "0",
                "workload_size": 2,
                "assignments": [
                    {
                        "task_suite_name": "libero_spatial",
                        "task_id": 0,
                        "trial_ids": [0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="workload_size"):
        load_worker_manifest(manifest_path)


def test_rollout_parser_accepts_internal_worker_mode() -> None:
    args = _build_arg_parser().parse_args(
        [
            "--config",
            "recipe.yaml",
            "--work-manifest",
            "worker.json",
            "--worker-id",
            "worker_2",
            "--suppress-final-summary",
        ]
    )

    assert args.work_manifest == "worker.json"
    assert args.worker_id == "worker_2"
    assert args.suppress_final_summary is True


def test_rollout_worker_uses_only_manifest_assignments(tmp_path) -> None:
    manifest_path = tmp_path / "worker.json"
    manifest_path.write_text(
        json.dumps(
            {
                "worker_id": 1,
                "gpu": "3",
                "workload_size": 2,
                "assignments": [
                    {
                        "task_suite_name": "libero_object",
                        "task_id": 8,
                        "trial_ids": [11, 12],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args = _build_arg_parser().parse_args(
        ["--config", "recipe.yaml", "--work-manifest", str(manifest_path)]
    )

    assignments, manifest = _build_evaluation_assignments(args, {})

    assert assignments == [
        {
            "task_suite_name": "libero_object",
            "task_id": 8,
            "trial_ids": [11, 12],
        }
    ]
    assert manifest["worker_id"] == 1

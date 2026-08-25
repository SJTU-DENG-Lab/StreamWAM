from examples.robotwin.workload import (
    ROBOTWIN_CONFIGS,
    ROBOTWIN_TASKS,
    build_workload,
    distribute_workload,
)


def test_one_trial_is_all_50_tasks_times_two_configs() -> None:
    jobs = build_workload(num_trials=1)

    assert len(ROBOTWIN_TASKS) == 50
    assert ROBOTWIN_CONFIGS == ("demo_clean", "demo_randomized")
    assert len(jobs) == 100
    assert len({job.identity for job in jobs}) == 100


def test_workload_filters_and_balances_arbitrary_gpu_count() -> None:
    jobs = build_workload(
        num_trials=1,
        tasks=[ROBOTWIN_TASKS[0], ROBOTWIN_TASKS[1], ROBOTWIN_TASKS[2]],
        configs=["demo_clean"],
    )
    assignments = distribute_workload(jobs, ["2", "5"])

    assert [len(worker_jobs) for worker_jobs in assignments.values()] == [2, 1]
    assert [job.task for job in jobs] == list(ROBOTWIN_TASKS[:3])

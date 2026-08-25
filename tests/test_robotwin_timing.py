from examples.robotwin.timing import aggregate_evaluation


def test_timing_excludes_warmup_and_averages_all_episodes() -> None:
    records = [
        {"record_type": "inference", "warmup": True, "model_inference_ms": 999.0},
        {"record_type": "inference", "warmup": False, "model_inference_ms": 40.0},
        {"record_type": "inference", "warmup": False, "model_inference_ms": 44.0},
        {"record_type": "episode", "success": True, "total_time_s": 4.0},
        {"record_type": "episode", "success": False, "total_time_s": 6.0},
    ]

    summary = aggregate_evaluation(records)

    assert summary["chunk_time_ms"] == 42.0
    assert summary["total_time_per_episode_s"] == 5.0
    assert summary["successes"] == 1
    assert summary["episodes"] == 2


def test_skipped_timeout_is_not_an_episode_or_accuracy_failure() -> None:
    records = [
        {"record_type": "episode", "success": True, "total_time_s": 4.0},
        {"record_type": "job", "status": "skipped_timeout", "success": None},
    ]

    summary = aggregate_evaluation(records)

    assert summary["successes"] == 1
    assert summary["episodes"] == 1
    assert summary["total_time_per_episode_s"] == 4.0

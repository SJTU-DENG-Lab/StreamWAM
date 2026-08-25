import json
from pathlib import Path

import pytest

from examples.robotwin.robotwin_worker import load_single_job, write_job_status


def test_single_job_runner_rejects_multi_job_payload(tmp_path: Path) -> None:
    path = tmp_path / "job.json"
    path.write_text(json.dumps([{"task": "a"}, {"task": "b"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="one job object"):
        load_single_job(path)


def test_job_status_is_atomic_and_contains_phase_identity(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    job = {"task": "adjust_bottle", "config": "demo_clean", "trial": 0}

    write_job_status(path, job, phase="environment_setup")

    status = json.loads(path.read_text(encoding="utf-8"))
    assert status["phase"] == "environment_setup"
    assert status["task"] == "adjust_bottle"
    assert status["config"] == "demo_clean"
    assert not list(tmp_path.glob("*.tmp.*"))

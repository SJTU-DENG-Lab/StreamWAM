from __future__ import annotations

import importlib.util
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_streamwam_is_the_only_package_identity() -> None:
    project = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    assert project["name"] == "streamwam"
    assert importlib.util.find_spec("streamwam") is not None
    assert importlib.util.find_spec("starwam") is None

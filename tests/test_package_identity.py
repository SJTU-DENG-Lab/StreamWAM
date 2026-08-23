from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

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


def test_default_inference_import_does_not_require_pyarrow() -> None:
    code = """
import importlib.abc
import sys

class BlockPyArrow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyarrow" or fullname.startswith("pyarrow."):
            raise ModuleNotFoundError("pyarrow intentionally unavailable")
        return None

sys.meta_path.insert(0, BlockPyArrow())
import streamwam
assert callable(streamwam.build_framework)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

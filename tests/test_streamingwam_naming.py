from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".html", ".md", ".py", ".sh", ".svg", ".toml", ".yaml", ".yml"}
IGNORED_PARTS = {".git", ".pytest_cache", ".venv", "__pycache__", "build", "superpowers"}
OLD_NAMES = ("Stream" + "WAM", "Stream-" + "WAM", "stream" + "wam", "stream-" + "wam")


def _tracked_source_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        if any(part in IGNORED_PARTS for part in path.relative_to(ROOT).parts):
            continue
        yield path


def test_python_distribution_and_package_use_streamingwam() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "streamingwam"' in pyproject
    assert 'include = ["streamingwam*"]' in pyproject
    assert (ROOT / "streamingwam" / "__init__.py").is_file()
    assert not (ROOT / OLD_NAMES[2]).exists()


def test_no_old_brand_or_package_names_remain() -> None:
    failures: list[str] = []
    for path in _tracked_source_files():
        relative = path.relative_to(ROOT)
        path_text = relative.as_posix()
        contents = path.read_text(encoding="utf-8")
        for old_name in OLD_NAMES:
            if old_name in path_text or old_name in contents:
                failures.append(f"{relative}: contains {old_name!r}")
    assert not failures, "\n".join(failures)

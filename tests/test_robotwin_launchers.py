from pathlib import Path


def test_robotwin_launchers_are_thin_and_forward_arguments() -> None:
    scripts = Path("examples/robotwin/scripts")
    expected = {
        "launch_streamwam_robotwin_baseline_4gpu.sh": "--inference-mode baseline",
        "launch_streamwam_robotwin_cd_4gpu.sh": "--inference-mode cd",
        "launch_streamwam_robotwin_ac_stream_4gpu.sh": "--inference-mode ac-stream",
    }
    for name, mode in expected.items():
        text = (scripts / name).read_text(encoding="utf-8")
        assert "python" not in text.split("PYTHON_BIN", 1)[0]
        assert "examples/robotwin/multigpu_rollout.py" in text
        assert mode in text
        assert '"$@"' in text
        assert text.count("if ") == 0

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "index.html"
README = ROOT / "README.md"
JOINT_VIDEO = ROOT / "docs" / "assets" / "real-robot" / "joint-wam.mp4"
PROMO_BUILDER = ROOT / "docs" / "video" / "build_streamingwam_promo.py"


def test_real_robot_joint_metrics_and_speedups_are_consistent() -> None:
    page = PAGE.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    promo_builder = PROMO_BUILDER.read_text(encoding="utf-8")

    assert page.count("682.1 ms") == 2
    assert page.count("90 s") >= 3
    assert "5.6×</strong> Chunk" in page
    assert "2.4×</strong> Total" in page
    assert "5.6 times faster Chunk Time and 2.4 times faster Total Time" in page
    assert "from 90 s to 38 s (2.4× faster)" in page
    assert "667.1 ms" not in page
    assert "from 68 s to 38 s" not in page

    assert "| Joint WAM | 682.1 ms | 90 s |" in readme
    assert "from 90 s to 38 s (2.4× faster)" in readme
    assert "5.6× Chunk Time speedup" in readme
    assert "667.1 ms" not in readme
    assert "from 68 s to 38 s" not in readme
    assert "Completed · 2.4× Faster" in promo_builder
    assert "Completed · 1.8× Faster" not in promo_builder


def test_joint_rollout_remains_click_to_load_with_a_fresh_cache_token() -> None:
    page = PAGE.read_text(encoding="utf-8")
    token = "assets/real-robot/joint-wam.mp4?v=joint-full-20260830"

    assert page.count(token) == 2
    assert f'<source data-src="{token}" type="video/mp4">' in page
    assert f'<source src="{token}"' not in page
    assert 'preload="none"' in page


def test_joint_rollout_asset_is_the_full_90_second_two_view_video() -> None:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,r_frame_rate",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(JOINT_VIDEO),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    media = json.loads(result.stdout)
    video = next(stream for stream in media["streams"] if stream["codec_type"] == "video")

    assert video["codec_name"] == "h264"
    assert (video["width"], video["height"]) == (640, 720)
    assert video["r_frame_rate"] == "24/1"
    assert abs(float(media["format"]["duration"]) - 90.0) < 0.05

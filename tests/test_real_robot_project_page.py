import json
import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "index.html"
README = ROOT / "README.md"
JOINT_VIDEO = ROOT / "docs" / "assets" / "real-robot" / "joint-wam.mp4"
JOINT_POSTER = ROOT / "docs" / "assets" / "real-robot" / "joint-wam.jpg"
JOINT_MANIFEST = ROOT / "docs" / "assets" / "real-robot" / "joint-wam.json"
STREAMING_VIDEO = ROOT / "docs" / "assets" / "real-robot" / "streaming-wam.mp4"
STREAMING_POSTER = ROOT / "docs" / "assets" / "real-robot" / "streaming-wam.jpg"
STREAMING_MANIFEST = ROOT / "docs" / "assets" / "real-robot" / "streaming-wam.json"
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


def test_streaming_rollout_uses_the_new_side_view_and_remains_click_to_load() -> None:
    page = PAGE.read_text(encoding="utf-8")
    video_token = "assets/real-robot/streaming-wam.mp4?v=streaming-side-20260831"
    poster_token = "assets/real-robot/streaming-wam.jpg?v=streaming-side-20260831"

    assert page.count(video_token) == 2
    assert page.count(poster_token) == 1
    assert f'<source data-src="{video_token}" type="video/mp4">' in page
    assert f'<source src="{video_token}"' not in page


def test_real_robot_table_combines_timing_and_30_trial_success_results() -> None:
    page = PAGE.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    section = page[page.index('id="benchmark-real-robot"') : page.index('class="efficiency-intro')]

    assert 'href="#benchmark-real-robot"' in page
    assert 'id="real-robot-success"' not in page
    assert "success-rate-card" not in page
    assert "Real-robot success rate" not in page
    assert "Real robot inference, rollout time, and task success" in section
    assert "over 30 trials per method" in section
    assert "over 30 trials per method" in readme
    assert "30 Hz control frequency" in section
    assert "30 Hz control frequency" in readme
    assert "25 Hz control frequency" not in page
    assert "25 Hz control frequency" not in readme
    assert "<th scope=\"col\">Successes</th>" in section
    assert "<th scope=\"col\">Success Rate ↑</th>" in section
    assert "<th scope=\"row\">Joint WAM</th><td>682.1 ms</td><td>90 s</td><td>26 / 30</td><td>86.67%</td>" in section
    assert "<th scope=\"row\">Distilled WAM (1V10A)</th><td>402.7 ms</td><td>60 s</td><td>19 / 30</td><td>63.33%</td>" in section
    assert "<th scope=\"row\">Distilled WAM (1V2A)</th><td>150.3 ms</td><td>61 s</td><td>17 / 30</td><td>56.67%</td>" in section
    assert "<strong>27 / 30</strong>" in section
    assert "<strong>90.00%</strong>" in section
    assert "| Joint WAM | 682.1 ms | 90 s | 26 / 30 | 86.67% |" in readme
    assert "| Distilled WAM (1V10A) | 402.7 ms | 60 s | 19 / 30 | 63.33% |" in readme
    assert "| Distilled WAM (1V2A) | 150.3 ms | 61 s | 17 / 30 | 56.67% |" in readme
    assert "| Streaming-WAM (Ours) | **122.62 ms** | **38 s** | **27 / 30** | **90.00%** |" in readme


def test_libero_optimization_chart_shows_cumulative_effective_cycle_speedups() -> None:
    page = PAGE.read_text(encoding="utf-8")
    section = page[page.index('id="optimization-stack"') : page.index('id="discussion"')]

    assert page.index("Inference efficiency.") < page.index('id="optimization-stack"')
    assert page.index('id="optimization-stack"') < page.index('id="discussion"')
    assert 'href="#optimization-stack"' in page
    assert "FastWAM-Joint" in section
    assert "FastWAM-Joint-40K" not in section
    assert section.index("Model level") < section.index("One-step consistency distillation")
    assert section.index("System level") < section.index("Asynchronous overlap")
    assert section.index("Implementation level") < section.index("KV cache + computation reuse")
    assert section.count('class="optimization-level-row ') == 3
    assert "One-step consistency distillation" in section
    assert "Asynchronous overlap" in section
    assert "KV cache + computation reuse" in section
    assert "torch.compile + CUDA Graphs" in section
    assert section.count('class="optimization-row ') == 5
    for speedup in ("1.00×", "1.25×", "2.31×", "2.82×", "2.66×", "2.95×", "3.48×", "3.12×", "3.79×"):
        assert speedup in section
    assert "Each row includes all optimizations above it." in section
    assert "<strong>Cumulative single-GPU end-to-end effective-cycle speedups on LIBERO.</strong>" in section
    assert "the interval between consecutive inference returns" in section
    assert "the interval between consecutive action-chunk installations" in section
    assert "capturing inference–execution overlap" not in section
    assert "We progressively stack optimizations across the model, system, and implementation levels." in section
    assert "model-, system-, and implementation-level" not in section


def test_robotwin_streaming_success_results_are_consistent() -> None:
    page = PAGE.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    expected_row_html = '<tr><th scope="row">Streaming-WAM (Ours)</th><td><strong>91.68</strong></td><td><strong>91.80</strong></td><td><strong>91.74</strong></td></tr>'

    assert expected_row_html in page
    assert "| Streaming-WAM (Ours) | **91.68** | **91.80** | **91.74** |" in readme
    assert page.count("by 4.74 percentage points, from 87.0 to 91.74") == 2
    assert readme.count("by 4.74 percentage points, from 87.0 to 91.74") == 2
    assert "maintaining or improving task success" in page
    assert "maintaining or improving task success" in readme
    for stale_value in ("90.40", "90.80", "90.60", "from 87.0 to 90.6"):
        assert stale_value not in page
        assert stale_value not in readme


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


def test_joint_rollout_manifest_locks_the_reviewed_sources_layout_and_sync() -> None:
    manifest = json.loads(JOINT_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["layout"] == "vertical"
    assert manifest["duration_seconds"] == 90.0
    assert manifest["views"] == [
        {
            "position": "top",
            "source": "demo1.rar/joint-1.mp4",
            "source_sha256": "d559ecac36dd18084445d03515a906aacbbaeecaf3bac8ff4b82102228281ff8",
            "start_seconds": 0.0,
        },
        {
            "position": "bottom",
            "source": "demo1.rar/joint-2.mp4",
            "source_sha256": "dacd691ae7c72f3ea63d7edf9345df593ce02a90272000d4f93e217f24abb2ce",
            "start_seconds": 0.0,
        },
    ]
    assert hashlib.sha256(JOINT_VIDEO.read_bytes()).hexdigest() == manifest["output_sha256"]
    assert hashlib.sha256(JOINT_POSTER.read_bytes()).hexdigest() == manifest["poster_sha256"]


def test_streaming_rollout_manifest_locks_the_new_side_view_and_sync() -> None:
    manifest = json.loads(STREAMING_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["layout"] == "vertical"
    assert manifest["duration_seconds"] == 38.0
    assert manifest["views"] == [
        {
            "position": "top",
            "source": "demo1.rar/1v2a-rtc-1.mp4",
            "source_sha256": "74872b10a1f5f7dea8bc4330026f597586e1a56184d505f9c772ab8fc60c6bf9",
            "start_seconds": 0.0,
        },
        {
            "position": "bottom",
            "source": "飞书20260831-114524.mp4",
            "source_sha256": "fb707837b357411654d384469a9df6a0bc23ad56ebde27c66e952d210706ed0a",
            "start_seconds": 0.0,
        },
    ]
    assert hashlib.sha256(STREAMING_VIDEO.read_bytes()).hexdigest() == manifest["output_sha256"]
    assert hashlib.sha256(STREAMING_POSTER.read_bytes()).hexdigest() == manifest["poster_sha256"]

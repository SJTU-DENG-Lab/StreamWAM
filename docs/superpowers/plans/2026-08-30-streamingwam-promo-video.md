# Streaming-WAM Project Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible 1080p project video that compares Joint WAM and Streaming-WAM at real speed and closes with the method and timing figures.

**Architecture:** A Python build tool will validate/extract the source media, synthesize an original music bed, generate an FFmpeg filter graph, render the final MP4, and validate its streams. FFmpeg performs all spatial composition, timed overlays, transitions, audio mixing, and delivery encoding.

**Tech Stack:** Python 3, NumPy, FFmpeg/FFprobe, bsdtar, H.264, AAC

## Global Constraints

- Final output is 1920×1080 at 30 fps with H.264 video, AAC audio, and fast-start metadata.
- Robot footage runs at real speed and both methods start simultaneously.
- Comparison stops when Streaming-WAM finishes; it does not wait for Joint WAM.
- Layout uses two stacked camera views per method, with Joint WAM on the left and Streaming-WAM on the right.
- Music is original, instrumental, unobtrusive, and mixed with quiet robot ambience.

---

### Task 1: Reproducible media builder

**Files:**
- Create: `docs/video/build_streamingwam_promo.py`
- Test: `docs/video/test_build_streamingwam_promo.py`

**Interfaces:**
- Consumes: the three supplied still images and four raw rollout videos, either from an extracted source directory or `demo1.rar`.
- Produces: `docs/video/outputs/streamingwam-project-video.mp4`, `docs/video/outputs/streamingwam-project-video-music.wav`, and a JSON validation report.

- [ ] **Step 1: Write failing unit tests for timeline and FFmpeg graph construction**

```python
def test_timeline_ends_after_results_fade():
    timeline = Timeline()
    assert timeline.total_seconds == 51.0

def test_comparison_starts_both_methods_together():
    graph = build_filter_graph(Timeline())
    assert "setpts=PTS-STARTPTS" in graph
    assert "xstack" in graph
```

- [ ] **Step 2: Run the tests and verify they fail before the builder exists**

Run: `python -m unittest docs.video.test_build_streamingwam_promo -v`

Expected: FAIL because `docs.video.build_streamingwam_promo` does not exist.

- [ ] **Step 3: Implement source validation, extraction, deterministic music synthesis, filter-graph generation, rendering, and FFprobe validation**

The builder must use a 4-second intro, a 38-second rollout comparison, a 4.5-second method figure, and a 5-second results figure. It must add a concise completion cue at the end of the comparison and encode with `libx264 -crf 17 -preset slow -pix_fmt yuv420p -movflags +faststart` plus AAC audio.

- [ ] **Step 4: Run unit tests**

Run: `python -m unittest docs.video.test_build_streamingwam_promo -v`

Expected: all tests PASS.

### Task 2: Render and visual validation

**Files:**
- Create: `docs/video/outputs/streamingwam-project-video.mp4`
- Create: `docs/video/outputs/streamingwam-project-video-validation.json`
- Create: `docs/video/outputs/contact-sheet.jpg`

**Interfaces:**
- Consumes: `docs/video/build_streamingwam_promo.py` and the source assets.
- Produces: the final reviewable MP4 and inspection artifacts.

- [ ] **Step 1: Render the final video**

Run: `python docs/video/build_streamingwam_promo.py`

Expected: a complete MP4 in `docs/video/outputs/` without FFmpeg errors.

- [ ] **Step 2: Validate delivery metadata**

Run: `ffprobe -v error -show_entries stream=index,codec_name,width,height,r_frame_rate -show_entries format=duration -of json docs/video/outputs/streamingwam-project-video.mp4`

Expected: H.264 1920×1080 at 30 fps, AAC audio, and approximately 51.5 seconds total duration including transitions.

- [ ] **Step 3: Generate and inspect representative frames**

Run: `ffmpeg -i docs/video/outputs/streamingwam-project-video.mp4 -vf "fps=1/8,scale=480:-1,tile=4x2" -frames:v 1 docs/video/outputs/contact-sheet.jpg`

Expected: the contact sheet shows the opening, synchronized four-panel comparison, Streaming-WAM completion cue, method figure, and timing figure in the correct order.

- [ ] **Step 4: Review repository changes without pushing generated source footage**

Run: `git status --short && git diff --check`

Expected: only the reproducible builder, tests, plan/spec, and explicitly selected delivery artifacts appear; extracted raw videos remain outside version control.

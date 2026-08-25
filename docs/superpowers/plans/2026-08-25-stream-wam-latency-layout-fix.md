# Stream-WAM Latency Layout Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the compressed four-panel latency image with two readable `2400 × 900` figures and identify both LIBERO ablation rows as Stream-WAM variants.

**Architecture:** Keep one deterministic Matplotlib generator, but give it separate `render_chunk_time()` and `render_episode_time()` outputs. Each output uses three independent axes with width ratios `2:1:1`; the HTML stacks both images and preserves their intrinsic ratio inside a narrow-screen scrolling viewport.

**Tech Stack:** Python 3.10, Matplotlib, NumPy, static HTML/CSS, pytest, Pillow.

## Global Constraints

- Generate exactly two PNG files at `2400 × 900`.
- Show benchmark panels in LIBERO, RoboTwin 2.0, RoboCasa order.
- Use independent continuous y-axes and no broken-axis treatment.
- Preserve every authoritative metric value.
- Use `Stream-WAM w/o Action Conditioning` and `Stream-WAM w/o Slot Encoder` consistently.
- Prevent page-level horizontal overflow on narrow screens.

---

### Task 1: Lock the page and image contracts

**Files:**
- Modify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: the public static page and generator CLI.
- Produces: regression coverage for two image assets, exact dimensions, naming, data, and mobile-safe markup.

- [ ] **Step 1: Write failing tests**

Update the page assertions to require `assets/stream-wam-chunk-time.png` and `assets/stream-wam-episode-time.png`, each declared as `width="2400" height="900"`. Assert that the generated images open as `(2400, 900)` and that both `w/o` rows start with `Stream-WAM` in the visible and accessible tables.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest -q tests/test_academic_project_page.py -k 'latency or benchmark_tables'`

Expected: failures report the old single image, `(2400, 1800)` size, and old ablation labels.

- [ ] **Step 3: Keep the expected metric literals unchanged**

The expected values remain the existing LIBERO, RoboTwin, and RoboCasa tuples; only asset structure and method labels change.

---

### Task 2: Generate two correctly proportioned figures

**Files:**
- Modify: `academic_project_page/generate_latency_figure.py`
- Create: `academic_project_page/assets/stream-wam-chunk-time.png`
- Create: `academic_project_page/assets/stream-wam-episode-time.png`
- Delete: `academic_project_page/assets/stream-wam-latency.png`

**Interfaces:**
- Consumes: the authoritative latency tuples already defined by the generator.
- Produces: `render_chunk_time(path: Path)` and `render_episode_time(path: Path)`, plus a CLI that writes both assets to an output directory.

- [ ] **Step 1: Split the renderer by metric**

Create a shared wide-figure helper using `figsize=(12, 4.5)` and `dpi=200`. Use a `1 × 3` grid with `width_ratios=(2, 1, 1)` and one axis per benchmark.

- [ ] **Step 2: Draw Chunk Time panels**

Render LIBERO, RoboTwin 2.0, and RoboCasa on separate continuous axes. Retain exact labels and values, teal Stream-WAM bars, and hatched Stream-WAM ablations.

- [ ] **Step 3: Draw Episode Time panels**

Render LIBERO Long/Short paired bars in the first panel and the single RoboTwin/RoboCasa episode-time series in their own panels.

- [ ] **Step 4: Generate both production assets**

Run: `python academic_project_page/generate_latency_figure.py`

Expected: two deterministic `2400 × 900` PNGs are written under `academic_project_page/assets/`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `pytest -q tests/test_academic_project_page.py -k latency`

Expected: all latency tests pass.

---

### Task 3: Update page layout and method-family naming

**Files:**
- Modify: `academic_project_page/index.html`
- Modify: `academic_project_page/styles.css`
- Modify: `academic_project_page/README.md`
- Modify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: the two generated image assets.
- Produces: two stacked, linked figures and consistent Stream-WAM ablation names.

- [ ] **Step 1: Replace the single image markup**

Add separate Chunk Time and Episode Time figure blocks with distinct captions and descriptive alternative text. Wrap each link in a `.latency-viewport` element.

- [ ] **Step 2: Preserve ratio and mobile readability**

Set each image to `width: 100%; height: auto`. At narrow widths, give the inner link a minimum width and the viewport `overflow-x: auto`, without causing body-level overflow.

- [ ] **Step 3: Rename the ablations everywhere user-visible**

Change both LIBERO table rows and the exact-data table rows to `Stream-WAM w/o Action Conditioning` and `Stream-WAM w/o Slot Encoder`.

- [ ] **Step 4: Update asset-generation documentation**

Document that the generator writes both wide images and give the single command used to regenerate them.

- [ ] **Step 5: Run the full page suite**

Run: `pytest -q tests/test_academic_project_page.py`

Expected: all tests pass.

---

### Task 4: Verify presentation and publish

**Files:**
- Verify only: `academic_project_page/`

**Interfaces:**
- Consumes: completed static page.
- Produces: reproducible validation evidence and deployed GitHub Pages output.

- [ ] **Step 1: Run static checks**

Run: `python -m py_compile academic_project_page/generate_latency_figure.py`, `node --check academic_project_page/script.js`, and `git diff --check`.

- [ ] **Step 2: Verify deterministic output**

Generate both files into a temporary directory and compare their SHA-256 hashes with the committed assets.

- [ ] **Step 3: Inspect desktop and mobile rendering**

Confirm both figures retain a `8:3` ratio, the three benchmark panels are readable, narrow screens scroll only inside `.latency-viewport`, and the document has no horizontal overflow.

- [ ] **Step 4: Commit and push**

Commit the implementation as `fix: correct latency figure layout`, push `main`, and verify the Pages workflow succeeds for that commit.

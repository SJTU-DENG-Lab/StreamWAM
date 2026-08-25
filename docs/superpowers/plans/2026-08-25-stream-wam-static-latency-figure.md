# Stream-WAM Static Latency Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the academic page's HTML latency bars with a reproducible Python-generated vertical-bar figure while reordering benchmark sections and removing LingBot-VA rows.

**Architecture:** A focused Python generator under `academic_project_page/` owns the plotted latency data and writes one committed PNG asset. The static HTML embeds that asset and keeps success tables semantic. Existing repository-local page tests validate the public structure and a subprocess test validates the generator output.

**Tech Stack:** Python 3.10+, matplotlib, static HTML/CSS, pytest, agent-browser

## Global Constraints

- Benchmark order is LIBERO, RoboTwin 2.0, RoboCasa.
- The public success tables contain no LingBot-VA rows.
- The latency figure contains LIBERO and RoboTwin 2.0 only.
- The public label is Episode Time, never Total Time.
- Stream-WAM is teal; complete baselines are muted; ablations are hatched and labeled.
- All reported numeric values remain unchanged.

---

### Task 1: Lock the public page contract

**Files:**
- Modify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: `academic_project_page/index.html`
- Produces: failing assertions describing the new benchmark order and static-figure contract

- [ ] **Step 1: Write failing benchmark-order and row tests**

Update the literal expected table rows to omit both LingBot-VA rows and assert this exact section order:

```python
assert benchmark_order == [
    "benchmark-libero",
    "benchmark-robotwin",
    "benchmark-robocasa",
]
```

- [ ] **Step 2: Write a failing static-figure test**

```python
images = [attrs for tag, attrs in parser.attributes if tag == "img"]
latency = [item for item in images if item.get("class") == "latency-plot"]
assert len(latency) == 1
assert latency[0]["src"] == "assets/stream-wam-latency.png"
assert "LIBERO" in latency[0]["alt"]
assert "RoboTwin" in latency[0]["alt"]
assert "latency-bar" not in html
assert "Episode Time" in visible_text
assert "Total Time" not in visible_text
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run: `pytest -q tests/test_academic_project_page.py`

Expected: failures for old benchmark order, LingBot rows, and missing static image.

### Task 2: Build the reproducible Python figure

**Files:**
- Create: `academic_project_page/generate_latency_figure.py`
- Create: `academic_project_page/assets/stream-wam-latency.png`
- Modify: `tests/test_academic_project_page.py`

**Interfaces:**
- Produces: `render(output_path: pathlib.Path) -> None`
- Output: a valid PNG at `academic_project_page/assets/stream-wam-latency.png`

- [ ] **Step 1: Add a failing generator integration test**

```python
def test_latency_generator_writes_a_png(tmp_path: Path) -> None:
    output = tmp_path / "latency.png"
    subprocess.run([sys.executable, str(GENERATOR_PATH), "--output", str(output)], check=True)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 50_000
```

- [ ] **Step 2: Run the generator test and verify RED**

Run: `pytest -q tests/test_academic_project_page.py::test_latency_generator_writes_a_png`

Expected: failure because `generate_latency_figure.py` does not exist.

- [ ] **Step 3: Implement the generator**

Define the literal LIBERO and RoboTwin latency data, create a 2×2 matplotlib figure, render vertical bars with the approved colors and hatching, annotate every bar, use a broken y-axis for LIBERO chunk time, and expose:

```python
def render(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=PAPER)
```

The CLI accepts `--output`; its default is `assets/stream-wam-latency.png` relative to the script.

- [ ] **Step 4: Install build-only plotting dependencies and generate the asset**

Run:

```bash
python -m pip install matplotlib
python academic_project_page/generate_latency_figure.py
```

- [ ] **Step 5: Run the generator test and verify GREEN**

Run: `pytest -q tests/test_academic_project_page.py::test_latency_generator_writes_a_png`

Expected: pass with a valid non-empty PNG.

### Task 3: Integrate and visually verify the figure

**Files:**
- Modify: `academic_project_page/index.html`
- Modify: `academic_project_page/styles.css`
- Modify: `academic_project_page/README.md`
- Modify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: `assets/stream-wam-latency.png`
- Produces: responsive image link and reordered semantic benchmark sections

- [ ] **Step 1: Apply the minimal HTML changes**

Move the complete RoboTwin section before RoboCasa, remove both LingBot rows, replace the legacy latency markup with:

```html
<figure class="latency-figure breakout" aria-labelledby="latency-caption">
  <a href="assets/stream-wam-latency.png" target="_blank">
    <img class="latency-plot" src="assets/stream-wam-latency.png"
      alt="Chunk and episode time comparisons for LIBERO and RoboTwin 2.0.">
  </a>
  <figcaption id="latency-caption">Latency comparison for LIBERO and RoboTwin 2.0. Lower is better.</figcaption>
</figure>
```

- [ ] **Step 2: Replace chart CSS with responsive image CSS**

```css
.latency-figure { margin-top: 42px; }
.latency-figure a { display: block; }
.latency-plot { width: 100%; height: auto; display: block; border-radius: 18px; }
```

- [ ] **Step 3: Run all static checks**

Run:

```bash
pytest -q tests/test_academic_project_page.py
node --check academic_project_page/script.js
git diff --check
```

Expected: all tests pass and both static checks exit zero.

- [ ] **Step 4: Verify desktop and mobile rendering**

Serve `academic_project_page/`, inspect at 1280 px and 390 px widths, and confirm the plot is readable, links to the original PNG, and creates no page-level horizontal overflow.

- [ ] **Step 5: Request release review and commit**

Request a read-only review of the diff, resolve Critical and Important findings, then commit only the page, test, generator, asset, spec, and plan files.


# Stream-WAM Academic Spacetime Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current illustrated method graphic with a camera-ready asynchronous spacetime diagram that explains the real AC-Stream prediction/execution loop.

**Architecture:** Keep the method figure as one standalone, accessible SVG embedded by the existing HTML figure shell. The SVG owns the static time/cycle diagram and its sole non-semantic time-cursor animation; page CSS owns sizing, viewport containment, and caption presentation. Python page tests parse the real HTML and SVG to protect causal structure, timing geometry, academic styling, and reduced-motion behavior.

**Tech Stack:** Semantic HTML, responsive CSS, standalone SVG/CSS animation, Python `pytest`, `xml.etree.ElementTree`, FFmpeg/librsvg.

## Global Constraints

- Use a `1600 × 860` SVG view box on a white or near-white background.
- Show `t₀`, `t₁`, `t₂`, and `t₃` on a left-to-right time axis with vertical rules and three horizontal update rows.
- Show cold start, execution overlap, aligned action-prefix conditioning of the next video only, forward handoff, and one lighter repeated update.
- Omit exact action counts, delay values, latency numbers, `VM`, `IDM`, `FDM`, and cache terminology.
- Use thin charcoal connectors, square or slightly rounded modules, muted gray/blue/green/gold fills, and no gradients, shadows, glow, illustrations, or decorative checkmarks.
- Animate only a thin time cursor; the complete method must remain visible at every animation frame.
- Preserve the existing narrow-screen contained horizontal viewport.

---

### Task 1: Replace the old illustration contract with academic spacetime tests

**Files:**
- Modify: `tests/test_academic_project_page.py:461-670`
- Test: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: `docs/assets/stream-wam-method.svg` and the `docs/index.html` figure shell.
- Produces: parsed structural and geometric contracts for the replacement SVG.

- [ ] **Step 1: Replace the HTML description expectation**

Require this screen-reader description:

```python
description = (
    "A spacetime diagram shows Stream-WAM predicting each next video-action "
    "chunk during execution. The aligned action prefix conditions only the next "
    "visual prediction, and each prepared action chunk begins at the following "
    "handoff without a planned pause."
)
assert description in " ".join(parser.text_parts)
```

- [ ] **Step 2: Replace old SVG ID and label expectations**

Parse the SVG and require:

```python
required_ids = {
    "time-axis", "time-t0", "time-t1", "time-t2", "time-t3",
    "row-cold-start", "row-stream-1", "row-stream-2",
    "observation-0", "observation-1", "observation-2",
    "joint-wam-0", "ac-update-1", "ac-update-2",
    "video-0", "video-1", "video-2",
    "action-0", "action-1", "action-2",
    "execution-0", "execution-1",
    "aligned-prefix-1", "aligned-prefix-2",
    "prefix-to-video-1", "prefix-to-video-2", "time-cursor",
}
assert required_ids <= ids
```

Also require `Cold start`, `Streaming update 1`, `Streaming update 2`, `Observation O₀`, `Joint WAM`, `AC-Stream update`, `Aligned action prefix`, `Action-conditioned video V₁`, `Next action chunk A₁`, and `Executing A₀`.

- [ ] **Step 3: Add causal geometry assertions**

Parse numeric `x`, `width`, and path terminal coordinates. Assert that `ac-update-1` begins inside `execution-0` and ends no later than the `t₂` handoff, `execution-1` begins exactly at `t₂`, and `prefix-to-video-1` terminates on the left boundary and within the vertical bounds of `video-1`.

```python
by_id = {element.attrib["id"]: element for element in root.iter() if "id" in element.attrib}
execution_start = float(by_id["execution-0"].attrib["x"])
execution_end = execution_start + float(by_id["execution-0"].attrib["width"])
update_start = float(by_id["ac-update-1"].attrib["x"])
update_end = update_start + float(by_id["ac-update-1"].attrib["width"])
handoff_x = float(by_id["time-t2"].attrib["x1"])
assert execution_start < update_start < update_end <= handoff_x == execution_end
assert float(by_id["execution-1"].attrib["x"]) == handoff_x
video = by_id["video-1"]
path_end = re.search(r"L\s*([0-9.]+)\s+([0-9.]+)\s*$", by_id["prefix-to-video-1"].attrib["d"])
assert float(path_end.group(1)) == float(video.attrib["x"])
assert float(video.attrib["y"]) <= float(path_end.group(2)) <= float(video.attrib["y"]) + float(video.attrib["height"])
```

- [ ] **Step 4: Add academic-style and animation assertions**

Require no SVG `filter`, `linearGradient`, or `radialGradient` elements; require every module rectangle radius to be at most `6`; require `method-time-cursor` to be the only keyframe; and require reduced motion to set `animation: none !important` for `.time-cursor`.

```python
svg_namespace = "{http://www.w3.org/2000/svg}"
for tag in ("filter", "linearGradient", "radialGradient"):
    assert root.findall(f".//{svg_namespace}{tag}") == []
for rectangle in root.findall(f".//{svg_namespace}rect"):
    assert float(rectangle.attrib.get("rx", "0")) <= 6
assert set(re.findall(r"@keyframes\s+([\w-]+)", svg_source)) == {"method-time-cursor"}
assert re.search(
    r"@media\s*\(prefers-reduced-motion:\s*reduce\).*?\.time-cursor\s*\{[^}]*animation:\s*none\s*!important",
    svg_source,
    re.DOTALL,
)
```

- [ ] **Step 5: Run the focused tests and verify RED**

```bash
pytest -q tests/test_academic_project_page.py -k "academic_spacetime or method_svg"
```

Expected: failures identify the old `0 0 1600 760` view box, missing spacetime IDs, old description, gradients/filters, and multiple method animations.

---

### Task 2: Draw and integrate the academic spacetime SVG

**Files:**
- Replace: `docs/assets/stream-wam-method.svg`
- Modify: `docs/index.html:17-18,187-192`
- Modify: `docs/styles.css:181-185,284-287`
- Test: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: the semantic IDs and geometric contracts defined in Task 1.
- Produces: a standalone `1600 × 860` SVG and updated page integration.

- [ ] **Step 1: Build the static chart skeleton**

Draw the top time axis, four vertical landmarks, and three dashed horizontal row dividers. Use flat fills and 1–1.5 px lines and attach every required ID.

- [ ] **Step 2: Draw the cold-start row**

Place `Observation O₀`, `Joint WAM`, `Predicted video V₀`, and `Action chunk A₀` before `t₁`, then connect `A₀` to `Executing A₀` across `t₁ → t₂`.

- [ ] **Step 3: Draw the first streaming update**

During `Executing A₀`, route `Observation O₁` and `Aligned action prefix` into `AC-Stream update`. Produce `Action-conditioned video V₁` and `Next action chunk A₁` before `t₂`; terminate the gold prefix path only on `video-1`; connect `A₁` forward to `execution-1` at `t₂`.

- [ ] **Step 4: Draw the lighter repeated update**

Repeat the causal structure for `O₂`, `V₂`, and `A₂` at reduced opacity while preserving readable labels and forward path direction.

- [ ] **Step 5: Add the sole animation and accessibility**

Use only:

```css
.time-cursor { animation: method-time-cursor 8s linear infinite; }
@keyframes method-time-cursor {
  from { transform: translateX(0); }
  to { transform: translateX(1110px); }
}
@media (prefers-reduced-motion: reduce) {
  .time-cursor { animation: none !important; }
}
```

Keep a complete SVG `<title>` and `<desc>`.

- [ ] **Step 6: Update page copy and shell styling**

Change the hidden description to the Task 1 text, bump the shared CSS/JS and SVG asset versions, and remove the method shell's rounded card/shadow treatment. Retain `overflow-x: auto`, focus outline, caption, and mobile `min-width: 900px`.

- [ ] **Step 7: Run focused and full page tests until GREEN**

```bash
pytest -q tests/test_academic_project_page.py -k "academic_spacetime or method_svg"
pytest -q tests/test_academic_project_page.py
```

Expected: all tests pass.

---

### Task 3: Render, inspect, review, and commit

**Files:**
- Verify: `docs/assets/stream-wam-method.svg`
- Verify: `docs/index.html`
- Verify: `docs/styles.css`
- Verify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: the completed SVG and page integration from Task 2.
- Produces: a reviewed commit and renderable figure for user inspection.

- [ ] **Step 1: Rasterize at both target widths**

```bash
ffmpeg -loglevel error -y -i docs/assets/stream-wam-method.svg -frames:v 1 /tmp/streamwam-academic-1600.png
ffmpeg -loglevel error -y -i docs/assets/stream-wam-method.svg -vf scale=900:-1 -frames:v 1 /tmp/streamwam-academic-900.png
```

- [ ] **Step 2: Inspect causal and typographic readability**

Check both renders for flat academic styling, forward-only handoffs, prediction inside execution windows, gold paths ending only on video outputs, readable labels, and no card-like decoration.

- [ ] **Step 3: Run final verification**

```bash
pytest -q tests/test_academic_project_page.py
git diff --check
git status --short
```

- [ ] **Step 4: Request focused review**

Ask a reviewer to inspect AC-Stream semantic fidelity, SVG geometry, the single-animation constraint, accessibility, and responsive containment. Address every high-confidence issue and rerun Step 3.

- [ ] **Step 5: Commit the implementation**

```bash
git add docs/assets/stream-wam-method.svg docs/index.html docs/styles.css tests/test_academic_project_page.py
git commit -m "feat: redraw Stream-WAM method figure"
```

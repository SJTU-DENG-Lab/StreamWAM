# Stream-WAM Action Overlap Detail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the abstract animated Stream-WAM method timeline with a static, publication-style diagram that concretely explains the 32-action A₀/A₁ overlap and the 16 AC-Stream condition slots.

**Architecture:** Keep the figure as one accessible SVG asset referenced by the existing project page. Encode action ribbons as explicit SVG cell groups with stable IDs so tests can verify counts, alignment, handoff geometry, conditioning inputs, and the absence of animation. Update page metadata and the cache key for the taller artwork.

**Tech Stack:** SVG/XML, HTML5, CSS, Python `pytest`, `xml.etree.ElementTree`.

## Global Constraints

- The figure is completely static: no animation, `@keyframes`, moving cursor, gradient, filter, shadow, or glow.
- A₀ and A₁ each contain exactly 32 visible action cells.
- A₀ cells 0–7 execute before observation; cells 8–15 execute during inference; cells 16–31 are unused look-ahead.
- `Observe O₁` and inference begin at the A₀[7]/A₀[8] boundary.
- The unlabeled inference end tick occurs before the action-16 handoff.
- A₀[8:16] and A₁[0:8] are vertically aligned and shown as one shared eight-action interval.
- The handoff resumes at A₁[8]; the overlap is not executed twice.
- The condition strip has 8 known slots copied from the overlap and 8 unknown slots.
- Observation O₁ and all 16 slots enter the same update; the gold conditioning path ends at V₁.
- Do not show latency values, a fixed completion index, `Next chunk ready`, or `completion time varies`.
- Preserve a 1600 px-wide viewBox and the page's 1100 px minimum mobile artwork width.

---

### Task 1: Lock the detailed semantics in regression tests

**Files:**
- Modify: `tests/test_academic_project_page.py:516-710`

**Interfaces:**
- Consumes: `docs/assets/stream-wam-method.svg` as XML.
- Produces: stable expectations for SVG IDs, cell counts, alignment, slot construction, static presentation, and intrinsic size.

- [ ] **Step 1: Write failing detailed-figure assertions**

Index elements by `id` and assert the following exact behavior:

```python
assert root.attrib["viewBox"] == "0 0 1600 980"
assert len(list(by_id["action-ribbon-a0"])) == 32
assert len(list(by_id["action-ribbon-a1"])) == 32
assert len(list(by_id["condition-known-slots"])) == 8
assert len(list(by_id["condition-unknown-slots"])) == 8
assert float(by_id["a0-cell-8"].attrib["x"]) == float(by_id["a1-cell-0"].attrib["x"])
assert float(by_id["a0-cell-15"].attrib["x"]) == float(by_id["a1-cell-7"].attrib["x"])
assert float(by_id["handoff-to-a1-8"].attrib["data-target-x"]) == float(by_id["a1-cell-8"].attrib["x"])
```

Also assert the labels `Observe O₁`, `A₀[8:16] = A₁[0:8]`, `16 condition slots`, `8 known action slots`, and `8 unknown slots`; reject the obsolete abstract labels and forbidden completion copy.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `pytest -q tests/test_academic_project_page.py -k 'streamwam_method_svg or streamwam_second_update or method_figure_scrolls'`

Expected: FAIL because the existing asset is 860 px tall, animated, and has no cell ribbons or slot groups.

- [ ] **Step 3: Commit the failing tests**

Run: `git add tests/test_academic_project_page.py && git commit -m "test: specify Stream-WAM action overlap figure"`

### Task 2: Draw the static detailed method figure

**Files:**
- Modify: `docs/assets/stream-wam-method.svg`

**Interfaces:**
- Consumes: the stable IDs specified by Task 1.
- Produces: a valid 1600×980 accessible SVG with explicit ribbons and slots.

- [ ] **Step 1: Draw the detailed static geometry**

Use `viewBox="0 0 1600 980"`, retain `<title>` and `<desc>`, and draw a compact cold start plus these explicit groups:

```xml
<g id="action-ribbon-a0">…32 rect children id="a0-cell-0" through "a0-cell-31"…</g>
<g id="action-ribbon-a1">…32 rect children id="a1-cell-0" through "a1-cell-31"…</g>
<g id="condition-known-slots">…8 rect children…</g>
<g id="condition-unknown-slots">…8 rect children…</g>
```

Use a constant cell pitch so A₁ begins exactly eight cells to the right of A₀. Give A₀[8:16] and A₁[0:8] matching gold styling; use green for executed windows and pale gray for look-ahead. Add range brackets, a camera-frame `Observe O₁` marker, an inference bar beginning at the same x coordinate, and an unlabeled end tick before the action-16 handoff.

- [ ] **Step 2: Add overlap, handoff, and conditioning paths**

Add `#overlap-connectors`, `#handoff-to-a1-8`, `#overlap-to-known-slots`, `#observation-to-update`, `#slots-to-update`, and `#condition-to-video-v1`. Store `data-target-x` on the handoff path. The gold path must end at `#video-1`; observation and slot paths must terminate on `#ac-update-1`.

- [ ] **Step 3: Add a light repeated continuation**

Keep `#row-stream-2` at opacity 0.65–0.8 and show only a compact A₁-to-A₂ continuation cue, without repeating all 32 cells and 16 slots.

- [ ] **Step 4: Validate and run focused tests**

Run: `xmllint --noout docs/assets/stream-wam-method.svg` and the focused command from Task 1. Expected: all pass.

- [ ] **Step 5: Commit the SVG**

Run: `git add docs/assets/stream-wam-method.svg && git commit -m "feat: detail Stream-WAM action overlap"`

### Task 3: Integrate and visually verify the taller figure

**Files:**
- Modify: `docs/index.html:180-195`
- Modify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: the 1600×980 SVG from Task 2.
- Produces: correct intrinsic dimensions, refreshed cache key, and accurate accessible prose.

- [ ] **Step 1: Write a failing page-integration assertion**

Assert `width="1600"`, `height="980"`, a cache key newer than `v=20260826-2`, and accessible prose mentioning 32 actions, observation after 8 actions, eight shared actions, and 16 condition slots.

- [ ] **Step 2: Run the page-integration test and verify failure**

Run: `pytest -q tests/test_academic_project_page.py -k 'method_figure'`. Expected: FAIL on the old 860 px metadata and abstract prose.

- [ ] **Step 3: Update page metadata and accessible copy**

Set `height="980"` and `src="assets/stream-wam-method.svg?v=20260826-3"`. Rewrite the hidden description and caption to explain the 32-action A₀, O₁ after eight actions, A₀[8:16]/A₁[0:8] overlap, and the eight-known plus eight-unknown condition slots.

- [ ] **Step 4: Run all page tests**

Run: `pytest -q tests/test_academic_project_page.py`. Expected: PASS.

- [ ] **Step 5: Rasterize and inspect both required widths**

Run `rsvg-convert` at widths 1600 and 1100, then inspect both PNGs for readable labels, exact alignment, unclipped arrows, and balanced whitespace. If unavailable, use Chromium screenshots at equivalent widths.

- [ ] **Step 6: Verify and commit integration**

Run `pytest -q tests/test_academic_project_page.py`, `git diff --check`, and `git status --short`. Then commit with `git commit -m "docs: integrate detailed Stream-WAM method figure"`.

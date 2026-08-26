# Stream-WAM Overview + Inset Method Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the oversized 32-cell method diagram with one compact static SVG containing a Figure-6-style global streaming overview and one magnified single-window inset.

**Architecture:** Preserve the existing external SVG and page figure container. Give the global timeline and inset stable semantic IDs so XML tests can verify overview/inset separation, execution–prediction overlap, inset alignment, update inputs, and output routing without relying on decorative metadata. Update the page’s intrinsic height, cache key, caption, and hidden description.

**Tech Stack:** SVG/XML, HTML5, Python `pytest`, `xml.etree.ElementTree`, Chromium screenshots.

## Global Constraints

- One 1600 px-wide SVG with a target height of 720–760 px.
- Global overview uses `t₀ → t₃`, long bars, and no per-action cells or slot details.
- Exactly one inset sits inside overview whitespace and magnifies one `t → t+1` overlap.
- The inset uses continuous A₀/A₁ segments, not 32-cell ribbons.
- A₀’s gold overlap aligns with A₁’s gold shared prefix; handoff enters A₁ continuation.
- Observation and action context enter the same AC-Stream update; the gold path ends at V₁.
- Static flat academic style only: no animation, gradients, filters, shadows, glow, or cartoon illustration.
- Mobile artwork remains horizontally scrollable at a 1100 px minimum width.

---

### Task 1: Specify the overview + inset semantics with failing tests

**Files:**
- Modify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: `docs/assets/stream-wam-method.svg` as XML.
- Produces: assertions for the global overview, exactly one inset, continuous segment geometry, inset placement, path routing, static style, and absence of action cells.

- [ ] **Step 1: Replace 32-cell expectations with overview/inset expectations**

Assert `viewBox="0 0 1600 740"`; require IDs `global-overview`, `overview-cycle-1`, `overview-cycle-2`, `detail-inset`, `inset-selection`, `inset-callout`, `inset-a0-before`, `inset-a0-overlap`, `inset-a0-lookahead`, `inset-a1-prefix`, `inset-a1-continuation`, `inset-observation`, `inset-update`, `inset-video-1`, and `inset-action-1`. Assert no `action-cell` or `condition-slot` class appears.

- [ ] **Step 2: Add real-geometry assertions**

Parse rectangle bounds and path endpoints. Verify prediction boxes lie inside their current execution windows, the inset is inside the SVG and does not intersect the overview’s occupied upper band, A₀ overlap and A₁ prefix have identical x/width values, the handoff path ends above A₁ continuation, both inset inputs terminate at the update, and output paths terminate at V₁/A₁.

- [ ] **Step 3: Run focused tests and verify failure**

Run: `pytest -q tests/test_academic_project_page.py -k 'streamwam_method_svg or streamwam_second_update'`.

Expected: FAIL because the current SVG is 980 px tall and contains 64 action cells plus explicit 8+8 slot cells instead of an overview/inset composition.

- [ ] **Step 4: Commit failing tests**

Run: `git add tests/test_academic_project_page.py && git commit -m "test: specify Stream-WAM overview inset figure"`.

### Task 2: Draw the compact global overview and inset

**Files:**
- Modify: `docs/assets/stream-wam-method.svg`

**Interfaces:**
- Consumes: stable IDs and geometry contracts from Task 1.
- Produces: one accessible 1600×740 static SVG.

- [ ] **Step 1: Draw the global spacetime overview**

Keep the cold-start stage and draw two long execution windows across `t₁ → t₂` and `t₂ → t₃`. Place `AC-Stream update 1` inside execution A₀ and `AC-Stream update 2` inside execution A₁. Route each update to paired V/A outputs before its handoff. Give cycle 2 opacity 0.70–0.78.

- [ ] **Step 2: Draw and connect the single magnified inset**

Place a bordered inset in lower-right whitespace. Draw three A₀ continuous segments and two A₁ continuous segments. Align the A₀ overlap and A₁ prefix exactly. Mark observation/inference at the left overlap boundary, handoff at the right boundary, and connect the selected overview overlap to the inset with a thin dashed callout.

- [ ] **Step 3: Draw the inset conditioning path**

Use one gold `known action context` block and one dashed `unknown future slots` block, not individual slot cells. Route O₁ and the combined context into one update. Route gold to V₁ and neutral to A₁.

- [ ] **Step 4: Validate and run focused tests**

Run: `xmllint --noout docs/assets/stream-wam-method.svg` and the focused test command from Task 1. Expected: PASS.

- [ ] **Step 5: Commit SVG implementation**

Run: `git add docs/assets/stream-wam-method.svg && git commit -m "feat: combine Stream-WAM overview and overlap inset"`.

### Task 3: Integrate, render, and verify the compact figure

**Files:**
- Modify: `docs/index.html`
- Modify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: the 1600×740 SVG from Task 2.
- Produces: matching page dimensions, refreshed cache key, and accessible two-scale explanation.

- [ ] **Step 1: Write failing page-integration assertions**

Assert the figure image is `1600×740`, uses cache key `v=20260826-4`, and the caption/description explain that the overview shows repeated streaming while the inset magnifies one overlap and its action-conditioned update.

- [ ] **Step 2: Run the page test and verify failure**

Run: `pytest -q tests/test_academic_project_page.py -k 'academic_spacetime_method_figure'`. Expected: FAIL on the old 980 px metadata and prior caption.

- [ ] **Step 3: Update page metadata and copy**

Set the intrinsic image size to `1600×740`, advance the cache key to `v=20260826-4`, and replace the hidden description/caption with the overview-plus-inset explanation.

- [ ] **Step 4: Run all page tests**

Run: `pytest -q tests/test_academic_project_page.py`. Expected: PASS.

- [ ] **Step 5: Render and inspect at 1600 and 1100 px**

Use Chromium to capture the SVG at both widths. Check label legibility, inset whitespace placement, callout clarity, uninterrupted overview execution, and absence of collisions or clipping.

- [ ] **Step 6: Verify and commit integration**

Run `xmllint --noout`, all page tests, `git diff --check`, and `git status --short`. Commit tracked changes with `git commit -m "docs: integrate compact Stream-WAM method figure"`.

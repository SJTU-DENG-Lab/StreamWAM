# Stream-WAM Readable Hero Visual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the hero runtime/attention graphic wider and its attention labels legible while removing the section number badges.

**Architecture:** Keep the current semantic HTML sections and 10×10 matrix cell markup. Change the decorative heading markup, desktop grid allocation, targeted typography sizing, and runtime-track geometry while retaining the seven-second reveal and the 1040px stacked breakpoint.

**Tech Stack:** Static HTML, CSS Grid, pytest page-contract tests, agent-browser responsive QA.

## Global Constraints

- Do not change the two-chunk attention topology.
- Keep the seven-second runtime reveal timing; make synchronous prediction/execution boundaries contiguous and split the Stream-WAM execution rail into three uniformly spaced chunk segments with later prediction windows centered over the segment boundaries.
- Preserve exact desktop top/bottom alignment between `.hero-main` and `.hero-figure`.
- Preserve zero page-level horizontal overflow at 390px and 320px.
- At desktop widths, use `16px` matrix rows/cells and approximately `10px` token labels so the attention comparison consumes the unused lower space.

---

### Task 1: Readable hero graphic

**Files:**
- Modify: `tests/test_academic_project_page.py`
- Modify: `docs/index.html`
- Modify: `docs/styles.css`

**Interfaces:**
- Consumes: Existing `.hero`, `.visual-section-heading`, `.attention-panel-heading`, `.attention-path-badge`, `.mask-chunk-heads`, `.mask-column-labels`, and `.mask-row-labels` selectors.
- Produces: The same accessible headings and matrix DOM without decorative numeric spans.

- [ ] **Step 1: Write the failing test**

Add a page contract that asserts the pipeline has no `>01<` or `>02<`, the desktop hero uses a right graphic column with a minimum of at least `680px`, and CSS declares larger floors for the path badge, panel titles, chunk headings, and matrix labels.

- [ ] **Step 2: Run the focused test to verify RED**

Run: `pytest -q tests/test_academic_project_page.py -k readable_attention_visual`

Expected: FAIL because the numeric spans remain, the graphic minimum is `620px`, and the typography remains below the new floors.

- [ ] **Step 3: Implement the minimal HTML/CSS changes**

Remove only the two number spans; use a desktop hero grid of `minmax(0,1fr) minmax(700px,1.18fr)`; increase the path badge, panel heading, chunk heading, and matrix label font sizes without changing mask markup. Make both synchronous prediction/execution transitions contiguous. Replace the continuous Stream-WAM execution rail with three evenly spaced segments and center later prediction windows over the two narrow chunk boundaries.

- [ ] **Step 4: Verify GREEN and run full regression**

Run: `pytest -q tests/test_academic_project_page.py && node --check docs/script.js && git diff --check`

Expected: all tests pass and both syntax checks exit successfully.

- [ ] **Step 5: Browser QA**

At 1920px and 1440px, measure the right graphic width and verify its outer top/bottom deltas from the left hero are at most 2px. At 390px and 320px, verify `document.documentElement.scrollWidth === document.documentElement.clientWidth` and visually inspect the attention labels.

- [ ] **Step 6: Review, commit, and deploy**

Request read-only code review, fix any Critical or Important findings, commit the implementation, push `main`, wait for the Pages workflow, and verify the live asset version and removed badges.

---

### Task 2: Use the attention module's unused vertical space

**Files:**
- Modify: `tests/test_academic_project_page.py`
- Modify: `docs/styles.css`
- Modify: `docs/index.html`

**Interfaces:**
- Consumes: Existing `.attention-panel-heading`, `.attention-path-badge`, `.mask-chunk-heads`, `.mask-column-labels`, `.mask-row-labels`, and `.mask-grid-10` selectors.
- Produces: The same two 10×10 matrices at a readable desktop scale with no topology or runtime changes.

- [ ] **Step 1: Write the failing test**

Extend the readable-attention contract to require `.mask-row-labels` and `.mask-grid-10` to use ten `minmax(16px,1fr)` rows, matrix cells to have `min-height: 16px`, token labels to use `.62rem`, chunk headings to use `.65rem`, attention panel titles to use `.82rem`, and the path badge to use `.74rem`.

- [ ] **Step 2: Run the focused test to verify RED**

Run: `pytest -q tests/test_academic_project_page.py -k readable_attention_visual`

Expected: FAIL because desktop rows remain `10px` and the label/title sizes remain below the new readable scale.

- [ ] **Step 3: Implement the minimal CSS and cache-key changes**

Set both desktop mask row grids and cells to `16px`, widen the row-label gutter and matching column-label offset, apply the approved font sizes, and bump both HTML asset keys from `20260826-9` to `20260826-10`.

- [ ] **Step 4: Verify behavior and deployment**

Run the full pytest, Node syntax, and diff checks. In a real browser at 1440px, confirm computed token labels are at least `9.5px`, both panels remain inside the hero card, and the attention comparison uses the former bottom gap. Recheck 390px and 320px containment, request read-only review, then commit, push, wait for Pages, and verify live asset key `20260826-10`.

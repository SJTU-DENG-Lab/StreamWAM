# Stream-WAM Readable Hero Visual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the hero runtime/attention graphic wider and its attention labels legible while removing the section number badges.

**Architecture:** Keep the current semantic HTML sections, runtime tracks, and 10×10 matrix cell markup. Change only the decorative heading markup, desktop grid allocation, and targeted typography sizing; retain the 1040px stacked breakpoint.

**Tech Stack:** Static HTML, CSS Grid, pytest page-contract tests, agent-browser responsive QA.

## Global Constraints

- Do not change the two-chunk attention topology.
- Do not change runtime animation timing or track placement.
- Preserve exact desktop top/bottom alignment between `.hero-main` and `.hero-figure`.
- Preserve zero page-level horizontal overflow at 390px and 320px.

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

Remove only the two number spans; use a desktop hero grid of `minmax(0,1fr) minmax(700px,1.16fr)`; increase the path badge, panel heading, chunk heading, and matrix label font sizes without changing mask markup.

- [ ] **Step 4: Verify GREEN and run full regression**

Run: `pytest -q tests/test_academic_project_page.py && node --check docs/script.js && git diff --check`

Expected: all tests pass and both syntax checks exit successfully.

- [ ] **Step 5: Browser QA**

At 1920px and 1440px, measure the right graphic width and verify its outer top/bottom deltas from the left hero are at most 2px. At 390px and 320px, verify `document.documentElement.scrollWidth === document.documentElement.clientWidth` and visually inspect the attention labels.

- [ ] **Step 6: Review, commit, and deploy**

Request read-only code review, fix any Critical or Important findings, commit the implementation, push `main`, wait for the Pages workflow, and verify the live asset version and removed badges.

# Stream-WAM Compact Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compact the hero figure and replace the oversized single attention graph with a direct standard-versus-action-conditioned comparison.

**Architecture:** Keep semantic runtime HTML and the existing CSS scan animation unchanged in behavior. Replace only the attention markup and its CSS with two small, topology-matched matrices, then validate size and overflow in a real browser.

**Tech Stack:** HTML5, CSS Grid, CSS keyframes, pytest/`HTMLParser`, agent-browser.

## Global Constraints

- Runtime labels are exactly `Model Prediction` and `Robot Execution` in normal title case.
- `World-Action Prediction × Robot Execution`, visible `Committed Actions`, `Two-chunk directed attention`, `Keys`, and `Queries` are absent from the hero figure.
- Runtime scan-reveal timing and fixed-bar behavior remain unchanged.
- Standard and action-conditioned matrices use identical topology except for the `Aₖ → Vₖ₊₁` cross-chunk cell.
- The `Aₖ → Vₖ₊₁` badge is visible outside the matrix cell.
- The figure does not impose a 680 px minimum height and remains overflow-free down to 320 px.

---

### Task 1: Lock the compact visual contract

**Files:**
- Modify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: the `.pipeline-visual` HTML substring and `docs/styles.css`.
- Produces: assertions for exact labels, removed copy, paired matrices, matching topology, and compact outer sizing.

- [ ] **Step 1:** Change runtime assertions to require `Model Prediction` and `Robot Execution`, reject removed labels and visible committed-action markup.
- [ ] **Step 2:** Require exactly two compact attention matrices and assert that only the second matrix contains `.cross-chunk-condition` at row `Vₖ₊₁`, column `Aₖ`.
- [ ] **Step 3:** Require the external `Aₖ → Vₖ₊₁` badge and reject the old heading/callout/axis copy.
- [ ] **Step 4:** Run `pytest -q tests/test_academic_project_page.py -k 'runtime_visual or attention_matrix'` and verify RED.

---

### Task 2: Compact runtime framing and terminology

**Files:**
- Modify: `docs/index.html`
- Modify: `docs/styles.css`

**Interfaces:**
- Consumes: existing `.runtime-comparison`, timeline bars, curtains, and cursors.
- Produces: the same horizontal schedules in a shorter outer frame with the revised legend.

- [ ] **Step 1:** Remove the large pipeline heading and committed-action timeline segment; update the legend.
- [ ] **Step 2:** Remove the outer minimum height and reduce panel/timeline vertical spacing while preserving all horizontal percentages and the seven-second reveal.
- [ ] **Step 3:** Run the runtime-focused test and verify GREEN.

---

### Task 3: Build the paired compact attention matrices

**Files:**
- Modify: `docs/index.html`
- Modify: `docs/styles.css`

**Interfaces:**
- Consumes: the two-chunk topology established by the current faithful matrix.
- Produces: `.attention-comparison` with `.attention-standard` and `.attention-conditioned`, each containing one `.compact-attention-matrix`.

- [ ] **Step 1:** Add matched token labels and sixteen cells to each panel.
- [ ] **Step 2:** Keep the standard cross-chunk cell masked and highlight only the corresponding action-conditioned cell.
- [ ] **Step 3:** Add the visible `.attention-path-badge` and remove all old explanatory/axis markup.
- [ ] **Step 4:** Add desktop side-by-side and mobile stacked CSS; increment matching asset versions.
- [ ] **Step 5:** Run focused tests and verify GREEN.

---

### Task 4: Verify and deploy

**Files:**
- Verify: `docs/index.html`
- Verify: `docs/styles.css`
- Verify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: completed page source.
- Produces: tested and deployed GitHub Pages output.

- [ ] **Step 1:** Run `pytest -q tests/test_academic_project_page.py && node --check docs/script.js && git diff --check`.
- [ ] **Step 2:** Measure figure height and overflow at 1920, 1440, 1024, 390, and 320 px with agent-browser; inspect desktop and mobile screenshots.
- [ ] **Step 3:** Request read-only code review and fix all Critical/Important findings.
- [ ] **Step 4:** Commit, push `main`, wait for Pages success, and verify the live HTML serves the new asset version.

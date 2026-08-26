# Stream-WAM Two-Chunk Hero Visual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the hero figure as a scan-revealed runtime schedule and a large two-chunk action-conditioned attention diagram.

**Architecture:** Semantic HTML defines the complete schedules and the grouped attention matrix. CSS owns the synchronized curtain/cursor animation, responsive stacked layout, and highlighted cross-chunk connection; JavaScript remains unchanged.

**Tech Stack:** HTML5, CSS Grid, CSS keyframes, pytest/`HTMLParser`, vanilla JavaScript syntax validation.

## Global Constraints

- Both runtime rows begin with `World-Action Prediction` before `Robot Execution`.
- The yellow cursor reveals a pre-rendered schedule by moving a black curtain; bars do not grow or disappear.
- Synchronous execution is interrupted by later prediction; Stream-WAM execution remains continuous after startup.
- Use `World-Action Prediction`, `Robot Execution`, and `Committed Actions`; remove `Model update`, `Robot motion`, and `Action prefix` from the figure.
- Attention shows `Chunk k` and `Chunk k+1`, with committed action keys from `Chunk k` conditioning future visual queries in `Chunk k+1`.
- Preserve the existing hero width, article width, title, lead, controls, and compact metrics.

---

### Task 1: Lock the corrected runtime and attention contracts

**Files:**
- Modify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: the hero figure substring between `.pipeline-visual` and `#pipeline-caption`.
- Produces: regression assertions for `.timeline-curtain`, `.execution-continuous`, `.chunk-group`, and `.cross-chunk-condition`.

- [ ] **Step 1: Replace the old animation assertions**

Assert that each runtime row contains a curtain, that CSS defines one
`timeline-reveal` animation for curtain and cursor, and that former
`generation-reveal-*` keyframes are absent.

- [ ] **Step 2: Add runtime vocabulary and ordering assertions**

Assert the new three labels are present, the old three labels are absent, and
the Stream-WAM generation block appears before its continuous execution rail in
markup.

- [ ] **Step 3: Add two-chunk attention assertions**

Assert `Chunk k`, `Chunk k+1`, `Future Visual`, `Committed Actions`, and at
least one `.cross-chunk-condition` cell are present.

- [ ] **Step 4: Run the focused tests and verify RED**

Run: `pytest -q tests/test_academic_project_page.py -k 'runtime_visual or attention_matrix'`

Expected: FAIL because the old block-growth animation and single-chunk matrices remain.

---

### Task 2: Implement the scan-revealed runtime schedule

**Files:**
- Modify: `docs/index.html`
- Modify: `docs/styles.css`

**Interfaces:**
- Consumes: `.runtime-comparison`, `.runtime-row`, and the existing 7-second timeline cycle.
- Produces: fixed schedule bars beneath `.timeline-curtain` and a cursor synchronized through `@keyframes timeline-reveal`.

- [ ] **Step 1: Replace the runtime labels and markup**

Use `World-Action Prediction`, `Robot Execution`, and `Committed Actions`.
Place first prediction blocks at the start of both rows; place the continuous
Stream-WAM execution rail only after its first prediction block.

- [ ] **Step 2: Replace block-growth animation with a curtain reveal**

Keep all prediction and execution bars at full width and opacity. Add a black
absolute curtain above each schedule with `left` animated from the timeline
start to end, and apply the identical animation to the yellow cursor.

- [ ] **Step 3: Encode the two schedule shapes**

Give the synchronous row two non-overlapping prediction/execution pairs. Give
Stream-WAM one startup prediction, a continuous execution rail, and later short
prediction blocks overlapping that rail.

- [ ] **Step 4: Run the runtime test and verify GREEN**

Run: `pytest -q tests/test_academic_project_page.py -k runtime_visual`

Expected: PASS.

---

### Task 3: Implement the large two-chunk attention diagram

**Files:**
- Modify: `docs/index.html`
- Modify: `docs/styles.css`

**Interfaces:**
- Consumes: the implementation contract in `streamwam/modules/ac_stream.py`, where condition actions are injected into `z1` future visual queries.
- Produces: a single grouped `.two-chunk-matrix` with a `.cross-chunk-condition` region at query `Future Visual (Chunk k+1)` × key `Committed Actions (Chunk k)`.

- [ ] **Step 1: Replace the two small matrix panels**

Create one panel with explicit query/key axes and four groups: visual/action for
`Chunk k`, followed by future-visual/next-action for `Chunk k+1`.

- [ ] **Step 2: Highlight only the directed cross-chunk condition**

Render the previous action-to-next visual cell with the mixed action/violet
style and a pulse. Keep all other allowed cells blue/yellow and masked cells
outlined.

- [ ] **Step 3: Stack both hero figure panels at full width**

Set `.pipeline-layout` to one column and increase timeline/matrix cell sizing.
Add responsive rules that keep labels and the matrix within 390 px.

- [ ] **Step 4: Update accessibility and cache version**

Rewrite `#pipeline-description` and `#pipeline-caption` to describe the two
runtime schedules and exact cross-chunk connection. Increment both CSS and JS
asset query versions together.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `pytest -q tests/test_academic_project_page.py -k 'runtime_visual or attention_matrix or release_versions'`

Expected: PASS.

---

### Task 4: Verify, review, and deploy

**Files:**
- Verify: `docs/index.html`
- Verify: `docs/styles.css`
- Verify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: the completed static page.
- Produces: tested, reviewed, pushed GitHub Pages source.

- [ ] **Step 1: Run all automated checks**

Run: `pytest -q tests/test_academic_project_page.py && node --check docs/script.js && git diff --check`

Expected: PASS with no warnings.

- [ ] **Step 2: Run browser checks**

Serve `docs/`, load at 1920, 1440, 1024, and 390 px widths, assert no horizontal
overflow, and capture the desktop hero for visual inspection.

- [ ] **Step 3: Request code review**

Provide the reviewer the user requirements, design spec, base SHA, and head SHA.
Fix all Critical and Important findings.

- [ ] **Step 4: Commit and push**

Commit the tested implementation, push `main`, and verify the Pages deployment
serves the new versioned assets.

# Stream-WAM Faithful Two-Chunk Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the simplified attention comparison with faithful two-chunk FastWAM-style masks and structurally align the hero figure from the left eyebrow to the LIBERO metric baseline.

**Architecture:** Wrap the left hero content below the DENG Lab lockup in `.hero-main` and use a two-row desktop grid so the right figure shares its exact row. Render each method as a 10×10 CSS mask built from two repeated 5×5 within-chunk blocks; Stream-WAM differs only by the aligned prior-action-prefix to next-`f₁` cross-chunk cells.

**Tech Stack:** Semantic HTML5, CSS Grid, CSS keyframes, pytest/`HTMLParser`, vanilla JavaScript asset versioning, agent-browser.

## Global Constraints

- Each matrix exposes two token groups of `f₀`, `f₁`, `fₕ`, `a₁`, and `aₕ`.
- Standard Joint WAM has two repeated 5×5 diagonal blocks and 50 masked off-diagonal cells.
- Stream-WAM has the same topology plus only the aligned action-prefix-to-next-`f₁` condition cells.
- No previous-chunk path opens into next `f₀`, `fₕ`, or action rows.
- Runtime comparison markup, schedules, seven-second curtain animation, and reduced-motion behavior remain unchanged.
- At desktop widths, `.pipeline-visual` top and bottom align with `.hero-main` within two CSS pixels.
- Below 1040 px, hero content and figure return to normal single-column flow.
- Both matrices remain fully contained at 320 px.
- CSS and JavaScript asset query versions always match.

---

### Task 1: Lock the hero-row and 10×10 mask contracts

**Files:**
- Modify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: the hero and pipeline HTML substrings plus `docs/styles.css`.
- Produces: topology tests for `.hero-main`, `.mask-grid-10`, `.chunk-mask-block`, `.cross-chunk-condition`, and desktop row placement.

- [ ] **Step 1: Add the failing hero structure test**

Assert that `.hero-main` wraps the eyebrow, title, lead, actions, and metrics;
the DENG Lab lockup remains outside it; and desktop CSS places `.hero-main` and
`.hero-figure` in grid row two.

- [ ] **Step 2: Add the failing 10×10 topology test**

Parse each `.mask-grid-10` panel and assert 100 cells, 10 column labels, 10 row
labels, two 5×5 diagonal blocks, and completely masked off-diagonal quadrants in
the standard panel.

- [ ] **Step 3: Add the failing Stream-WAM delta test**

Assert that the Stream-WAM matrix matches the standard classes at every cell
except the prior-action-prefix key positions on the next-`f₁` row. Assert no
cross-chunk condition class occurs on next-`f₀`, next-`fₕ`, or next-action rows.

- [ ] **Step 4: Run the focused tests and verify RED**

Run: `pytest -q tests/test_academic_project_page.py -k 'hero_main or attention_matrix'`

Expected: FAIL because `.hero-main` and 10×10 matrices do not exist.

---

### Task 2: Establish exact desktop hero alignment

**Files:**
- Modify: `docs/index.html`
- Modify: `docs/styles.css`

**Interfaces:**
- Consumes: the existing `.hero-copy` child elements and `.hero-figure`.
- Produces: `.hero-main` as the left row-two content and a row-two `.hero-figure` with an assistive-only caption.

- [ ] **Step 1: Wrap left main content**

Leave `.lab-lockup` as the first `.hero-copy` child. Wrap `.eyebrow` through
`.headline-results` in `<div class="hero-main">` without changing their order or
copy.

- [ ] **Step 2: Convert the wide hero to a two-row grid**

At widths above 1040 px, make `.hero-copy` use `display: contents`, place the lab
lockup in column one/row one, place `.hero-main` in column one/row two, and place
`.hero-figure` in column two/row two with `align-self: stretch`.

- [ ] **Step 3: Make the figure fill row two**

Set `.hero-figure` to a vertical flex container and `.pipeline-visual` to flex
within it. Visually hide the figcaption while retaining it in the accessibility
tree so the dark figure itself reaches the metric baseline.

- [ ] **Step 4: Restore stacked responsive flow**

Inside the existing `max-width: 1040px` media query, restore `.hero-copy` to
`display: block`, reset explicit grid rows/columns, and return `.hero-figure` to
its intrinsic height.

- [ ] **Step 5: Run the hero structure test and verify GREEN**

Run: `pytest -q tests/test_academic_project_page.py -k hero_main`

Expected: PASS.

---

### Task 3: Render the faithful paired attention masks

**Files:**
- Modify: `docs/index.html`
- Modify: `docs/styles.css`

**Interfaces:**
- Consumes: the current paired attention panel container.
- Produces: two `.mask-grid-10` panels with fixed row-major 100-cell topology and a visible `a-prefixᵏ → f₁ᵏ⁺¹` annotation on Stream-WAM.

- [ ] **Step 1: Replace compact four-group labels**

For both panels, render ten column labels and ten row labels in this order:
`f₀ᵏ`, `f₁ᵏ`, `fₕᵏ`, `a₁ᵏ`, `aₕᵏ`, `f₀ᵏ⁺¹`, `f₁ᵏ⁺¹`, `fₕᵏ⁺¹`, `a₁ᵏ⁺¹`,
`aₕᵏ⁺¹`. Add subtle chunk brackets over columns 1–5 and 6–10.

- [ ] **Step 2: Render the standard 10×10 cells**

Repeat the FastWAM-style 5×5 mask at row/column offsets zero and five. Use
`.masked-cell` for every off-diagonal cell; preserve blue visual allowed cells
and yellow action allowed cells inside each diagonal block.

- [ ] **Step 3: Render the Stream-WAM delta**

Copy the standard topology and replace only the next-`f₁` row versus prior
action-prefix key cells with `.cross-chunk-condition`. Add a concise badge
`a-prefixᵏ → f₁ᵏ⁺¹`; do not put text inside the tiny cells.

- [ ] **Step 4: Add compact 10×10 CSS**

Use CSS grid with ten `minmax(0,1fr)` tracks, two-pixel gaps, and small square
cells. Keep both panels side by side on desktop and stack them at the current
mobile breakpoint.

- [ ] **Step 5: Update accessibility and cache version**

Rewrite `#pipeline-description` and `#pipeline-caption` for the exact block-
diagonal versus AC-Stream delta. Increment the matching local CSS and JavaScript
query versions.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run: `pytest -q tests/test_academic_project_page.py -k 'attention_matrix or release_versions'`

Expected: PASS.

---

### Task 4: Browser verification, review, and deployment

**Files:**
- Verify: `docs/index.html`
- Verify: `docs/styles.css`
- Verify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: completed static page.
- Produces: aligned, reviewed, deployed GitHub Pages artifact.

- [ ] **Step 1: Run automated checks**

Run: `pytest -q tests/test_academic_project_page.py && node --check docs/script.js && git diff --check`

Expected: 30 page tests or more PASS, JavaScript parses, and no whitespace errors.

- [ ] **Step 2: Verify desktop edge alignment**

At 1920×1080 and 1440×1000, measure `.eyebrow`, `.pipeline-visual`, and
`.headline-results`. Require absolute top and bottom deltas no greater than two
pixels.

- [ ] **Step 3: Verify responsive containment**

At widths 1024, 390, and 320 px, assert page scroll width does not exceed the
viewport and each mask grid's client width equals its scroll width. Capture one
desktop and one mobile screenshot for visual inspection.

- [ ] **Step 4: Request read-only review**

Give the reviewer the source-of-truth files, design specification, base SHA,
head SHA, browser measurements, and test results. Fix every Critical and
Important issue.

- [ ] **Step 5: Commit and deploy**

Commit the implementation, push `main`, wait for the Pages workflow to finish,
and verify the live HTML serves the new asset version and both `.mask-grid-10`
panels.

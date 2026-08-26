# Stream-WAM Overlap + Attention Inset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the lower Stream-WAM inset into a left temporal-overlap view with direct segment correspondence and a right single Stream-WAM 10×10 action-conditioned attention mask.

**Architecture:** Preserve the existing global overview and lower inset frame. Replace the lower inset contents with two stable SVG groups separated by one divider; encode attention cells with row/column metadata so tests can verify the exact two cross-chunk connections. Keep all geometry and accessible copy in the existing SVG/page files.

**Tech Stack:** SVG/XML, HTML5, Python `pytest`, `xml.etree.ElementTree`, Chromium screenshots.

## Global Constraints

- The overview contains no `action-conditioned` or `handoff` text.
- The lower inset contains exactly two columns titled `Temporal overlap` and `Action-conditioned attention`.
- `shared prefix` and `known action context` share x and width; `look-ahead` and `unknown future slots` share x and width.
- Both mapping arrows are straight vertical paths.
- O₁ reaches `Stream Update` through one straight horizontal path.
- The inset uses `Stream Update`, not `AC-Stream update`.
- The right side contains one Stream-WAM 10×10 mask and no Standard Joint WAM comparison.
- Token order is `f₀, f₁, fₕ, a₁, aₕ` for each of two chunks.
- Exactly two cells, row 6 with columns 3 and 4, are cross-chunk action-conditioned.
- The figure remains static, flat, legible at 1100 px, and 1600×740 unless collision-free layout requires a small height increase.

---

### Task 1: Specify the split inset and mask semantics

**Files:**
- Modify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: `docs/assets/stream-wam-method.svg` as XML.
- Produces: geometry and attention-mask contracts for Task 2.

- [ ] **Step 1: Write failing split-layout assertions**

Require IDs `temporal-overlap-panel`, `attention-panel`, `inset-divider`, `inset-shared-prefix`, `inset-known-context`, `inset-a0-lookahead`, `inset-unknown-slots`, `shared-to-known`, `lookahead-to-unknown`, `inset-observation-input`, and `inset-update`. Assert the two panel headings and reject `One overlap window`, `handoff`, and overview `action-conditioned` text.

- [ ] **Step 2: Write failing correspondence/path assertions**

Parse visible bounds and path commands. Assert the shared/known pair and look-ahead/unknown pair have identical x/width values; `shared-to-known` and `lookahead-to-unknown` contain only `M` and `V` commands; and `inset-observation-input` contains only `M` and `H`, preserving one y coordinate from O₁ to the update.

- [ ] **Step 3: Write failing attention-mask assertions**

Require `streamwam-attention-mask` to contain exactly 100 direct `rect` children with unique `data-row="0"…"9"` and `data-col="0"…"9"`. Require exactly two `cross-chunk-condition` cells at `(6,3)` and `(6,4)`. Assert ten row labels and ten column labels in the established two-chunk order and no `Standard Joint WAM` text.

- [ ] **Step 4: Run focused tests and verify failure**

Run: `pytest -q tests/test_academic_project_page.py -k 'streamwam_method_svg or streamwam_second_update'`.

Expected: FAIL because the current inset has one temporal view, no right mask, non-aligned look-ahead/unknown blocks, and old update/overview wording.

- [ ] **Step 5: Commit failing tests**

Run: `git add tests/test_academic_project_page.py && git commit -m "test: specify Stream-WAM overlap attention inset"`.

### Task 2: Draw the left temporal view and right attention mask

**Files:**
- Modify: `docs/assets/stream-wam-method.svg`

**Interfaces:**
- Consumes: the stable IDs and geometry contracts from Task 1.
- Produces: a valid two-column lower inset with one exact 10×10 mask.

- [ ] **Step 1: Simplify overview annotations**

Remove the overview `action-conditioned` and `handoff` labels while preserving the continuous A₀/A₁ execution bars, update boxes, output arrows, selection, and callout.

- [ ] **Step 2: Draw the left 55% temporal-overlap panel**

Use continuous A₀/A₁ blocks. Align `known action context` directly below `shared prefix`; align `unknown future slots` directly below `look-ahead`. Add only vertical mapping arrows. Place O₁ and condition context on two input rows, each connected horizontally to a block labelled `Stream Update`.

- [ ] **Step 3: Draw the right 45% attention panel**

Add one 10×10 matrix with 100 square cells and row/column token labels. Reproduce the hero Stream-WAM mask: allowed within-chunk visual/action cells, masked cross-chunk cells, and only `(row=6,col=3)` plus `(row=6,col=4)` in gold. Add `action prefix → next visual future` below the matrix.

- [ ] **Step 4: Validate and run focused tests**

Run `xmllint --noout docs/assets/stream-wam-method.svg` and the focused tests from Task 1. Expected: PASS.

- [ ] **Step 5: Commit SVG implementation**

Run: `git add docs/assets/stream-wam-method.svg && git commit -m "feat: add Stream-WAM attention inset"`.

### Task 3: Integrate copy, render, and verify

**Files:**
- Modify: `docs/index.html`
- Modify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: the final SVG from Task 2.
- Produces: updated cache key, caption, hidden description, and visual verification.

- [ ] **Step 1: Write failing page-integration assertions**

Require cache key `v=20260826-5`; require caption/description to mention the temporal alignment on the left and the two cross-chunk action-to-visual attention cells on the right.

- [ ] **Step 2: Run the integration test and verify failure**

Run: `pytest -q tests/test_academic_project_page.py -k 'academic_spacetime_method_figure'`. Expected: FAIL on the old cache key and copy.

- [ ] **Step 3: Update page metadata and accessible copy**

Advance the SVG cache key and rewrite the caption/description. Keep intrinsic dimensions synchronized with the SVG viewBox.

- [ ] **Step 4: Run all page tests and XML validation**

Run `pytest -q tests/test_academic_project_page.py`, `xmllint --noout docs/assets/stream-wam-method.svg`, and `git diff --check`. Expected: PASS.

- [ ] **Step 5: Render at 1600 and 1100 px**

Use Chromium to inspect the global overview, direct left-column mappings, matrix token labels, gold cells, divider, and absence of collisions/clipping.

- [ ] **Step 6: Commit integration**

Commit tracked changes with `git commit -m "docs: integrate overlap attention method figure"`.

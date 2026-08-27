# Stream-WAM Code-Faithful Slot Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redraw the method inset so A₁ shared actions visibly feed both the slot assembly and Stream Update as separate inputs.

**Architecture:** Keep the existing SVG and page shell. Protect the code-faithful labels, routing, accessibility copy, and removal of misleading arrows with the existing XML/HTML test suite, then minimally update the SVG and page copy.

**Tech Stack:** Static SVG, HTML, Python `pytest`, `xml.etree.ElementTree`, `xmllint`.

## Global Constraints

- Do not use the label `known action context`.
- Use `shared actions`, `shared action slots`, `unknown action slots`, and `condition slots`.
- Do not display `· 8`, `· 16`, or `· 24` count suffixes.
- O₁ follows A₀[0:8]; A₀[8:16] aligns with A₁[0:8].
- A₁ shared actions feed the shared-action slots and Stream Update through two separate visible paths.
- The combined condition slots also feed Stream Update independently.
- The lower O₁ input box center must equal the A₀ observation-boundary x-coordinate.
- A₁[8:32] must not point to the unknown slots.
- Keep both slot labels on one line.
- Preserve the existing 1600×740 canvas and overall two-scale layout.

---

### Task 1: Lock the corrected method semantics

**Files:**
- Modify: `tests/test_academic_project_page.py`
- Test: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: SVG IDs and visible HTML copy.
- Produces: failing expectations for the corrected labels, two-branch routing, and cache-busted asset URL.

- [ ] **Step 1: Write the failing test**

Update the existing method-figure tests to require `shared actions`, `shared action slots`, `unknown action slots`, `condition slots`, `shared-to-condition-slots`, and `shared-actions-to-update`. Require count suffixes and the old A₀-to-slots bypass path to be absent. Verify that the slot path begins at A₁ shared actions, both slot labels contain no `tspan`, and both the direct action path and condition path terminate at `Stream Update`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_academic_project_page.py -k 'method_figure or spacetime'`

Expected: FAIL because the current SVG still uses count suffixes, wrapped slot labels, and routes slots from A₀ instead of A₁.

- [ ] **Step 3: Commit with the implementation**

The tests and static asset form one user-visible unit and will be committed together after Task 2 passes.

### Task 2: Redraw the inset and update accessible copy

**Files:**
- Modify: `docs/assets/stream-wam-method.svg`
- Modify: `docs/index.html`
- Test: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: expectations from Task 1.
- Produces: a code-faithful SVG and matching page description/caption.

- [ ] **Step 1: Implement the minimal SVG change**

Label both overlap blocks `shared actions`; use `executed actions`, `remaining actions`, and `predicted actions` elsewhere. Widen the slot blocks so `shared action slots` and `unknown action slots` remain on one line. Route A₁ shared actions directly downward into the shared-action slots and separately into Stream Update. Keep the combined condition-slot route into Stream Update.

- [ ] **Step 2: Update accessible HTML copy**

Describe O₁ after A₀[0:8], the shared A₀/A₁ region, and the separate action-prefix and condition-slot inputs. Bump the SVG query string to `v=20260827-2`.

- [ ] **Step 3: Verify the focused tests pass**

Run: `pytest -q tests/test_academic_project_page.py -k 'method_figure or spacetime'`

Expected: PASS.

- [ ] **Step 4: Verify the complete static page**

Run: `pytest -q tests/test_academic_project_page.py && xmllint --noout docs/assets/stream-wam-method.svg && git diff --check`

Expected: all checks pass with no XML or whitespace errors.

- [ ] **Step 5: Render and inspect**

Render the SVG to PNG at full width and inspect it for readable labels, clean arrow routing, and a balanced inset.

- [ ] **Step 6: Commit**

Run: `git add docs/assets/stream-wam-method.svg docs/index.html tests/test_academic_project_page.py docs/superpowers/specs/2026-08-27-streamwam-code-faithful-slot-routing-design.md docs/superpowers/plans/2026-08-27-streamwam-code-faithful-slot-routing.md && git commit -m "fix: simplify Stream-WAM shared-action routing"`

### Task 3: Align the repeated O₁ input with its timeline boundary

**Files:**
- Modify: `docs/assets/stream-wam-method.svg`
- Test: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: `inset-observation` as the A₀ observation-boundary line.
- Produces: `inset-observation-input-box` centered on that boundary and a connector beginning at its right edge.

- [ ] **Step 1: Write the failing alignment test**

Add `inset-observation-input-box` to the required SVG IDs. Assert that the box center equals `inset-observation.x1` and that `inset-observation-input` begins at the box's right edge.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest -q tests/test_academic_project_page.py::test_streamwam_method_svg_places_prediction_inside_execution_and_inset_in_whitespace`

Expected: FAIL because the lower O₁ box does not yet have the required ID or alignment.

- [ ] **Step 3: Move the box and reconnect its path**

Set the 80-pixel-wide O₁ box to `x="288"`, keep the observation boundary at `x="328"`, center its label at `x="328"`, and start the horizontal connector at `x="368"`.

- [ ] **Step 4: Verify the full page**

Run: `pytest -q tests/test_academic_project_page.py && xmllint --noout docs/assets/stream-wam-method.svg && git diff --check`

Expected: 48 tests pass with valid XML and no whitespace errors.

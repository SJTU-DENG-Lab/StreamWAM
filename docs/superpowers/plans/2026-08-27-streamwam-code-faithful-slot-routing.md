# Stream-WAM Code-Faithful Slot Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redraw the method inset so A₀[8:16] visibly feeds both the A₁ hard prefix and the 8+8 condition-slot assembly used by AC-Stream.

**Architecture:** Keep the existing SVG and page shell. Protect the code-faithful labels, routing, accessibility copy, and removal of misleading arrows with the existing XML/HTML test suite, then minimally update the SVG and page copy.

**Tech Stack:** Static SVG, HTML, Python `pytest`, `xml.etree.ElementTree`, `xmllint`.

## Global Constraints

- Do not use the label `known action context`.
- Use `shared actions`, `shared action slots · 8`, `unknown slots · 8`, and `16 condition slots`.
- O₁ follows A₀[0:8]; A₀[8:16] feeds both A₁[0:8] and the shared-action slots.
- A₁[8:32] must not point to the unknown slots.
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

Update the existing method-figure tests to require `shared actions`, `shared action slots · 8`, `unknown slots · 8`, `16 condition slots`, `shared-to-action-prefix`, and `shared-to-condition-slots`. Require the old labels and false slot arrow to be absent.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_academic_project_page.py -k 'method_figure or spacetime'`

Expected: FAIL because the current SVG still contains `known action context`, `shared prefix`, and `continuation-to-unknown`.

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

Rename the overlap and prefix labels to `shared actions`; label A₀[0:8] as executed and A₀[16:32] as remaining A₀. Replace the lower context blocks with two adjacent eight-slot blocks under one `16 condition slots` label. Draw separate routes from the shared A₀ segment to the A₁ prefix and the condition-slot strip. Remove decorative overview arrows and the obsolete attention prose.

- [ ] **Step 2: Update accessible HTML copy**

Describe O₁ after A₀[0:8], the reuse of A₀[8:16] as A₁[0:8], and the 8+8 slot assembly. Bump the SVG query string to `v=20260827-1`.

- [ ] **Step 3: Verify the focused tests pass**

Run: `pytest -q tests/test_academic_project_page.py -k 'method_figure or spacetime'`

Expected: PASS.

- [ ] **Step 4: Verify the complete static page**

Run: `pytest -q tests/test_academic_project_page.py && xmllint --noout docs/assets/stream-wam-method.svg && git diff --check`

Expected: all checks pass with no XML or whitespace errors.

- [ ] **Step 5: Render and inspect**

Render the SVG to PNG at full width and inspect it for readable labels, clean arrow routing, and a balanced inset.

- [ ] **Step 6: Commit**

Run: `git add docs/assets/stream-wam-method.svg docs/index.html tests/test_academic_project_page.py docs/superpowers/specs/2026-08-27-streamwam-code-faithful-slot-routing-design.md docs/superpowers/plans/2026-08-27-streamwam-code-faithful-slot-routing.md && git commit -m "fix: clarify Stream-WAM slot routing"`


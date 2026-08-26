# Stream-WAM Method Figure Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Left-align the lower method figure with `Cold start`, restore conditioning slots below A₁, and remove redundant overview and temporal annotations without changing attention semantics.

**Architecture:** Keep the existing 1600×740 static SVG and its semantic IDs. Encode the requested layout as SVG geometry, protect it with XML-based regression tests, and update the page cache key so GitHub Pages serves the new asset.

**Tech Stack:** SVG, HTML, Python `xml.etree.ElementTree`, pytest, agent-browser.

## Global Constraints

- Preserve the existing A₀ labels `execute`, `shared prefix`, and `look-ahead`.
- Preserve the existing A₁ labels `shared prefix` and `new actions`.
- Preserve all 100 attention-cell classes and the two conditioned cells `(6,3)` and `(6,4)`.
- Keep the SVG viewBox and page dimensions at `1600 × 740`.

---

### Task 1: Specify the revised geometry and removed annotations

**Files:**
- Modify: `tests/test_academic_project_page.py`
- Test: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: existing SVG IDs in `docs/assets/stream-wam-method.svg`
- Produces: regression assertions for frame alignment, vertical ordering, removed labels, removed return arrow, and the next cache key

- [ ] **Step 1: Write failing assertions**

Add assertions equivalent to:

```python
assert float(by_id["detail-inset-frame"].attrib["x"]) == 58
assert bounds("inset-shared-prefix")[3] < bounds("inset-known-context")[1]
assert bounds("inset-a1-continuation")[3] < bounds("inset-unknown-slots")[1]
assert "Observe O₁" not in text
assert "predict A₁ while A₀ continues" not in text
assert "overview-action-to-execution" not in ids
```

Update the expected image source to `assets/stream-wam-method.svg?v=20260826-6`.

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
pytest -q tests/test_academic_project_page.py -k 'streamwam_method_svg or academic_spacetime_method_figure'
```

Expected: failures for the old frame x-coordinate, old slot ordering, old annotations, and cache key.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_academic_project_page.py
git commit -m "test: specify method figure alignment cleanup"
```

### Task 2: Implement the SVG and page integration

**Files:**
- Modify: `docs/assets/stream-wam-method.svg`
- Modify: `docs/index.html`
- Test: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: Task 1 geometry and text assertions
- Produces: a left-aligned lower frame, reordered conditioning slots, clean overview, and cache-busted page asset

- [ ] **Step 1: Reposition the lower frame and panels**

Set the lower frame to `x="58" width="1462"`, move the temporal panel and its contents left, retain the attention panel on the right, and place the divider between the two content columns.

- [ ] **Step 2: Reorder the temporal rows**

Keep the A₀ row above the A₁ row. Place `inset-known-context` below `inset-shared-prefix` and `inset-unknown-slots` below `inset-a1-continuation`, preserving column alignment and current labels.

- [ ] **Step 3: Remove redundant annotations**

Delete the `Observe O₁` text, the `predict A₁ while A₀ continues` text and rule, and the return connector from `overview-action-1` to `overview-execution-1`. Preserve the temporal boundary line and O₁ input block.

- [ ] **Step 4: Update the cache key**

Change the page image source to:

```html
<img class="method-figure-artwork" src="assets/stream-wam-method.svg?v=20260826-6" alt="" width="1600" height="740">
```

- [ ] **Step 5: Run the page test suite**

```bash
pytest -q tests/test_academic_project_page.py
```

Expected: 48 tests pass.

- [ ] **Step 6: Commit the implementation**

```bash
git add docs/assets/stream-wam-method.svg docs/index.html tests/test_academic_project_page.py
git commit -m "fix: align and simplify Stream-WAM method figure"
```

### Task 3: Render, review, and publish

**Files:**
- Verify: `docs/assets/stream-wam-method.svg`
- Verify: `docs/index.html`

**Interfaces:**
- Consumes: Task 2 implementation
- Produces: reviewed and pushed `main`

- [ ] **Step 1: Check repository hygiene**

```bash
git diff --check
git status --short
```

- [ ] **Step 2: Render at desktop and narrow widths**

Use agent-browser against the local docs server at widths 1600 and 1100. Confirm that the lower frame aligns with `Cold start`, the two lower columns are visually balanced, labels do not overlap, and the attention mask remains legible.

- [ ] **Step 3: Request code review**

Review the final git range against `docs/superpowers/specs/2026-08-26-streamwam-method-figure-alignment-design.md`; fix all Critical and Important findings.

- [ ] **Step 4: Push**

```bash
git push origin main
```

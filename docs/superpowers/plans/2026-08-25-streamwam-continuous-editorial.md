# StreamWAM Continuous Editorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove visible section-label scaffolding and reduce the remaining benchmark identifiers so the project page reads as continuous research prose.

**Architecture:** Keep the existing semantic sections, IDs, prose, method figure, and benchmark tables. Change only their visible labels and typography; sections without visual headings use `aria-label`, while benchmark sections retain compact `h3` names.

**Tech Stack:** Static HTML, CSS, Python `html.parser` regression tests, agent-browser visual QA.

## Global Constraints

- Keep the masthead and its headline results unchanged.
- Preserve every benchmark value, unit, protocol, and table row in source order.
- Do not hide content behind JavaScript or tabs.
- Keep desktop and mobile layouts usable.

---

### Task 1: Lock the continuous-editorial contract

**Files:**
- Modify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: `parse_page()` and the real `academic_project_page/index.html`.
- Produces: regression coverage for removed visual labels and retained benchmark identifiers.

- [ ] **Step 1: Write the failing test**

Add a parser assertion that the visible text omits the obsolete labels and headings, while all three benchmark sections remain named. Add a CSS assertion that benchmark `h3` typography has a compact maximum size rather than display typography.

```python
for removed in (
    "Research notes · August 2026",
    "04 · Evidence",
    "What the Current Results Show",
    "05 · Discussion",
    "Where This Leaves Us",
    "Read, run, and revisit.",
    "Benchmark 01",
    "Benchmark 02",
    "Benchmark 03",
):
    assert removed not in visible_text
assert {"LIBERO", "RoboCasa", "RoboTwin 2.0"} <= set(parser.text_parts)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest -q tests/test_academic_project_page.py -k continuous`

Expected: FAIL because the current page still renders the labels and headings.

- [ ] **Step 3: Commit together with Task 2 after GREEN**

Do not commit a failing test separately.

### Task 2: Remove visible headings and compact benchmark names

**Files:**
- Modify: `academic_project_page/index.html`
- Modify: `academic_project_page/styles.css`
- Modify: `academic_project_page/README.md`
- Test: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: existing section IDs used by the article TOC.
- Produces: the same navigable article with continuous visual hierarchy.

- [ ] **Step 1: Simplify HTML without changing content**

Remove `.article-label`, `.act-label`, `.section-number`, and `.benchmark-label` elements. Remove the visible `h2` elements in experiments, discussion, and resources; replace their `aria-labelledby` attributes with stable `aria-label` values. Keep benchmark `h3` elements as their accessible names.

- [ ] **Step 2: Normalize the typography**

Set `.act-opening` to body-scale typography and compact benchmark names:

```css
.editorial-act .act-opening {
  margin-bottom: 27px;
  font-family: inherit;
  font-size: 1.09rem;
  font-weight: 650;
  line-height: 1.9;
  letter-spacing: 0;
}
.benchmark-intro h3 { font-size: clamp(1.15rem, 1.8vw, 1.4rem); }
```

Remove obsolete heading-size and label selectors. Preserve responsive body sizing.

- [ ] **Step 3: Update editing guidance**

Document that the article deliberately uses no visible sectional labels, and that only compact benchmark names identify the three result tables.

- [ ] **Step 4: Run verification**

Run:

```bash
pytest -q tests/test_academic_project_page.py
node --check academic_project_page/script.js
git diff --check
```

Expected: all page tests pass, JavaScript syntax succeeds, and the diff has no whitespace errors.

- [ ] **Step 5: Perform browser QA**

Serve `academic_project_page/`, inspect desktop and 390 px mobile layouts, confirm three tables remain visible/scrollable, and confirm no browser errors.

- [ ] **Step 6: Commit and push**

```bash
git add academic_project_page/README.md academic_project_page/index.html academic_project_page/styles.css tests/test_academic_project_page.py
git commit -m "style: simplify StreamWAM editorial hierarchy"
git push origin main
```

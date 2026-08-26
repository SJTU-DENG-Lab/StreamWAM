# Stream-WAM Results Copy and Table Typography Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the project page results introduction, simplify the success tables, improve numeric typography, and explain the measured inference speedups with concrete values.

**Architecture:** Keep the static page structure intact and limit HTML edits to `#experiments`. Express the reader-facing contract through the existing parser-based project page tests, then make the smallest HTML and CSS changes needed to satisfy it.

**Tech Stack:** Static HTML, CSS, Python standard library HTML parsers, pytest

## Global Constraints

- Do not modify the hero or title section.
- Use the capitalized, unhyphenated form `World Action Models` consistently.
- All evaluation numbers and speedups must match the values already embedded in the page.
- Numeric cells must use the page's inherited sans serif font while retaining tabular numeral alignment.
- Remove all three prose summaries beneath the success tables.

---

### Task 1: Results narrative and table presentation

**Files:**
- Modify: `tests/test_academic_project_page.py:824-860`
- Modify: `docs/index.html:185-263`
- Modify: `docs/styles.css:180-207`

**Interfaces:**
- Consumes: the existing `parse_page()` and `parse_benchmarks()` test helpers and the `#experiments` HTML structure
- Produces: a results section with approved copy, a `Long` LIBERO column, no `.benchmark-reading` elements, and sans serif numeric cells

- [ ] **Step 1: Write the failing tests**

Update the LIBERO header expectation and add a focused contract test:

```python
def test_results_copy_and_numeric_typography_are_concrete() -> None:
    parser, html = parse_page()
    visible_text = " ".join(parser.text_parts)
    css = (PAGE_ROOT / "styles.css").read_text(encoding="utf-8")

    for copy in (
        "further training FastWAM-Joint",
        "X-WAM on RoboCasa",
        "StarWAM on RoboTwin 2.0",
        "four NVIDIA H100 GPUs",
        "Task success alone does not reveal how much inference interrupts robot execution",
        "12.0× on LIBERO",
        "4.0× on RoboTwin 2.0",
        "3.7× on RoboCasa",
        "3.0× and 2.6× on long and short LIBERO tasks",
        "1.4× on RoboTwin 2.0",
        "3.2× on RoboCasa",
    ):
        assert copy in visible_text

    for removed_copy in (
        "Stream-WAM reaches 98.20% average success",
        "Stream-WAM reaches 87.6 total success",
        "Stream-WAM reports 75.35% average task success",
        "Every bar is labeled with its reported value",
    ):
        assert removed_copy not in visible_text

    assert 'class="benchmark-reading"' not in html
    assert ".benchmark-reading" not in css
    numeric_rule = re.search(r"tbody td\s*\{([^}]*)\}", css)
    assert numeric_rule is not None
    assert "font-family: inherit" in numeric_rule.group(1)
    assert "monospace" not in numeric_rule.group(1)
    assert "font-variant-numeric: tabular-nums" in css
```

Change the first LIBERO expected row to:

```python
["Method", "Long", "Spatial", "Goal", "Object", "Average ↑"],
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
.venv/bin/pytest -q tests/test_academic_project_page.py::test_benchmark_tables_and_protocols_match_the_authoritative_results tests/test_academic_project_page.py::test_results_copy_and_numeric_typography_are_concrete
```

Expected: FAIL because the page still uses `LIBERO-10`, the old narrative and summaries remain, and numeric cells still use `ui-monospace`.

- [ ] **Step 3: Implement the approved HTML and CSS changes**

In `docs/index.html`:

- Replace the three introductory paragraphs with the two approved results paragraphs from the design spec.
- Change the LIBERO table header and nearby description from `LIBERO-10` to `Long`.
- Delete all three `<p class="benchmark-reading">` elements.
- Replace the single efficiency paragraph with the two approved efficiency paragraphs from the design spec.

In `docs/styles.css`, replace the numeric cell declaration with:

```css
tbody td { color: var(--muted); font-family: inherit; font-size: .84rem; font-weight: 520; line-height: 1.4; }
```

Delete the unused `.benchmark-reading` rule. Keep `font-variant-numeric: tabular-nums` on the shared table rule.

- [ ] **Step 4: Run the focused project page tests**

Run:

```bash
.venv/bin/pytest -q tests/test_academic_project_page.py
```

Expected: all tests pass.

- [ ] **Step 5: Verify scope and commit**

Run:

```bash
git diff --check
git diff -- docs/index.html docs/styles.css tests/test_academic_project_page.py
```

Confirm that `docs/index.html` changes are confined to `#experiments`, `docs/styles.css` changes are confined to table and results rules, and the tests cover only the approved results contract. Then commit:

```bash
git add docs/index.html docs/styles.css tests/test_academic_project_page.py
git commit -m "refactor: clarify project page results"
```

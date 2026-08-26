# Stream-WAM Results Copy Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mechanical ablation explanation with the approved natural wording and remove the unwanted RoboCasa source-provenance sentence.

**Architecture:** Make a copy-only change in the existing static project page. Update the existing narrative regression test first, then replace the two affected HTML sentences without changing layout, tables, scripts, styles, or asset versions.

**Tech Stack:** Static HTML and Python `pytest`.

## Global Constraints

- Use lowercase `consistency distillation`.
- Introduce the two component removals as Stream-WAM ablation studies, not as consecutive `removes` definitions.
- Remove the complete RoboCasa sentence beginning `Published policy results from`.
- Keep the RoboCasa 24-task protocol, 50 trials per task, and average-success sentence.
- Do not change benchmark values, table markup, Citation, styles, JavaScript, or asset versions.
- Commit and push the correction to `origin/main`.

---

### Task 1: Correct results and RoboCasa copy

**Files:**
- Modify: `tests/test_academic_project_page.py`
- Modify: `docs/index.html`

**Interfaces:**
- Consumes: Existing Task performance and RoboCasa introduction paragraphs.
- Produces: The approved results legend and protocol-only RoboCasa introduction.

- [ ] **Step 1: Write the failing regression assertions**

Require the page to contain:

```python
"CD refers to one-step consistency distillation"
"We also conduct ablation studies on Stream-WAM by removing action conditioning or the slot encoder to evaluate the contribution of each component"
```

Require the page not to contain:

```python
"Consistency Distillation"
"Stream-WAM w/o Action Conditioning removes Action Conditioning"
"Published policy results from"
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest -q tests/test_academic_project_page.py -k 'narrative or benchmark'
```

Expected: FAIL because the current page contains the old capitalized and mechanical copy plus the unwanted RoboCasa provenance sentence.

- [ ] **Step 3: Replace only the two affected HTML passages**

Use the exact approved Task performance paragraph from the design spec and reduce the RoboCasa introduction to:

```html
<p>RoboCasa evaluation follows the standard 24-task protocol, covering 24 kitchen manipulation tasks with 50 trials per task and reporting average success.</p>
```

- [ ] **Step 4: Verify focused and full tests**

Run:

```bash
pytest -q tests/test_academic_project_page.py -k 'narrative or benchmark'
pytest -q tests/test_academic_project_page.py
git diff --check
```

Expected: all commands pass.

- [ ] **Step 5: Review, commit, and push**

Confirm the diff changes only the approved copy and its assertions, then run a read-only review and commit:

```bash
git add docs/index.html tests/test_academic_project_page.py
git commit -m "fix: refine results and RoboCasa copy"
git push origin main
```

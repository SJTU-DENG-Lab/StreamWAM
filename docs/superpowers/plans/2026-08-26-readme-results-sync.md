# README Results Synchronization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Synchronize the GitHub README's current-results section with the task-performance and inference-efficiency results on the Stream-WAM project page.

**Architecture:** Keep README results as static Markdown tables. Separate task performance from inference efficiency, mirror the project page's three benchmark tables and accessible latency table, and protect the exact published values with literal regression fixtures.

**Tech Stack:** Markdown, HTML inline emphasis, Python 3, pytest

## Global Constraints

- The project page is the source of truth for current method names, values, ordering, and best/second-best styles.
- Visible performance rows use `Stream-WAM (Ours)`; efficiency rows use `Stream-WAM`.
- Best performance values use Markdown bold and second-best values use HTML `<u>` tags.
- Task performance and inference efficiency remain separate.
- Installation, runtime, citation, license, and acknowledgements remain unchanged.

---

### Task 1: Synchronize README Results

**Files:**
- Modify: `README.md`
- Modify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: current result rows in `docs/index.html`, including its hidden `latency-data` table
- Produces: a self-contained `## Current results` README section with exact benchmark and runtime tables

- [ ] **Step 1: Write the failing regression expectations**

Update `test_readme_leads_with_project_page_and_has_current_citation` to require exact multiline table blocks for:

- LIBERO performance: OpenVLA through both Stream-WAM ablations.
- RoboTwin 2.0 performance: π₀ through Stream-WAM (Ours).
- RoboCasa performance: π₀.₅ through Stream-WAM (Ours), with X-WAM at `75.42%`, X-WAM-CD at `75.33%`, and Stream-WAM at `75.35%`.
- Inference efficiency: all exact LIBERO, RoboTwin 2.0, and RoboCasa rows from the project page's accessible latency table.

Add scoped negative checks for stale values `75.83`, `189.3`, `81.6`, and `112.2` in the README current-results section.

- [ ] **Step 2: Verify the expectations fail**

Run:

```bash
pytest -q tests/test_academic_project_page.py -k readme
```

Expected: FAIL because the README still contains abbreviated performance tables and stale RoboTwin timing values.

- [ ] **Step 3: Replace the README current-results section**

In `README.md`:

- Retain the model-lineage introduction.
- Add a `### Task performance` heading.
- Add complete `#### LIBERO`, `#### RoboTwin 2.0`, and `#### RoboCasa` performance tables copied from the current project page.
- Preserve bold/underline result styling.
- Add `### Inference efficiency`, the project page's concise measurement explanation, and one exact table with columns `Benchmark`, `Method`, `Chunk Time`, and `Episode Time`.
- Include the current speedup summary: `12.0×`, `4.0×`, `3.2×`, `3.0×`, `2.6×`, `1.4×`, and `1.8×`.
- Remove the old combined tables and stale result commentary.

- [ ] **Step 4: Verify focused and full tests pass**

Run:

```bash
pytest -q tests/test_academic_project_page.py -k readme
pytest -q tests/test_academic_project_page.py
git diff --check
```

Expected: focused tests pass, all 37 project-page tests pass, and the diff has no whitespace errors.

- [ ] **Step 5: Commit and push**

```bash
git add README.md tests/test_academic_project_page.py docs/superpowers/plans/2026-08-26-readme-results-sync.md
git commit -m "docs: synchronize README results"
git push origin main
```

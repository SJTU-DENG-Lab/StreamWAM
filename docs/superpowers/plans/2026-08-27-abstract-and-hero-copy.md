# Abstract and Hero Copy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the project-page hero description and Abstract with concise, paper-aligned Stream-WAM copy without including unrelated working-tree changes.

**Architecture:** Keep the existing static HTML structure and update only three paragraphs in `docs/index.html`. Add a focused parser-based regression test in a new test file so the copy can be staged independently from the currently modified shared page test.

**Tech Stack:** Static HTML, Python standard-library `html.parser`, pytest, Git selective staging.

## Global Constraints

- Use `World Action Models (WAMs)`, not `world-action models`.
- Use `future visual observations`, `future video generation`, `action chunk`, and `action continuation`.
- Avoid `video-model priors`, `future-video`, `world-action chunk`, `action postfix`, and `state–prediction mismatch`.
- Retain `action-conditioned` because it names the central mechanism.
- Preserve all uncommitted method-figure changes and exclude them from the implementation commit.

---

### Task 1: Lock the Hero and Abstract Copy

**Files:**
- Create: `tests/test_project_page_copy.py`
- Modify: `docs/index.html:47`
- Modify: `docs/index.html:154-155`

**Interfaces:**
- Consumes: The hero paragraph with class `hero-lede` and the two paragraphs immediately following the Abstract heading.
- Produces: Exact rendered hero and Abstract strings protected by a focused pytest regression test.

- [ ] **Step 1: Write the failing test**

Create a small `HTMLParser` subclass that records the normalized text of `p.hero-lede` and paragraphs inside `div.article-header`, then assert these exact values:

```python
EXPECTED_HERO = (
    "Stream-WAM introduces action-conditioned streaming for World Action Models. "
    "It overlaps inference with robot execution and conditions future video generation "
    "on the committed action prefix, aligning the predicted visual trajectory with the "
    "motion underway. The robot continues acting while the next prediction is prepared."
)

EXPECTED_ABSTRACT = [
    (
        "World Action Models (WAMs) jointly generate future visual observations and robot "
        "actions, allowing policies to reason about how the scene may evolve under interaction. "
        "Their iterative generation, however, is often slower than the robot control cycle: "
        "synchronous execution leaves the robot idle during inference, while naive asynchronous "
        "switching can create inconsistency between successive predictions."
    ),
    (
        "We introduce Stream-WAM, an action-conditioned streaming framework that overlaps WAM "
        "inference with robot execution. Actions already committed to execution condition future "
        "video generation, aligning the predicted visual trajectory with the motion underway; "
        "this action-conditioned future then guides a consistent action continuation. Stream-WAM "
        "therefore brings streaming into world prediction rather than treating continuity only as "
        "an action space constraint. We evaluate the method on LIBERO, RoboCasa, and RoboTwin 2.0. "
        "On LIBERO, Stream-WAM achieves 98.20% average success with 41.0 ms chunk latency and 4.74 s "
        "total time, reducing chunk latency by 12.0× and total time by 3.4× relative to FastWAM."
    ),
]
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest -q tests/test_project_page_copy.py`

Expected: FAIL because the current page still says `world-action models`, `future-video generation`, and `world-action chunk`.

- [ ] **Step 3: Replace only the target HTML paragraphs**

Set `p.hero-lede` to `EXPECTED_HERO`, preserving `<strong>` around `Stream-WAM`, `action-conditioned streaming`, and `committed action prefix`. Set the two Abstract paragraphs to the two `EXPECTED_ABSTRACT` strings, preserving `class="abstract-lead"` on the first paragraph.

- [ ] **Step 4: Run focused and page-level verification**

Run: `pytest -q tests/test_project_page_copy.py`

Expected: PASS.

Run: `pytest -q tests/test_academic_project_page.py -k 'hero or abstract or headline'`

Expected: PASS. The known in-progress method-figure test is excluded by the expression.

- [ ] **Step 5: Stage only the copy implementation**

Stage `tests/test_project_page_copy.py` in full. Interactively stage only the hero and Abstract hunks from `docs/index.html`, rejecting the unrelated method-figure hunk. Confirm with `git diff --cached --stat` and `git diff --cached` that no SVG, shared test, or method-figure HTML changes are included.

- [ ] **Step 6: Commit and verify from a clean checkout**

```bash
git commit -m "docs: refine abstract and hero copy"
```

Create a detached temporary worktree at the new commit and run:

```bash
pytest -q tests/test_academic_project_page.py tests/test_project_page_copy.py
git diff --check HEAD^..HEAD
```

Expected: all tests pass and the commit has no whitespace errors.

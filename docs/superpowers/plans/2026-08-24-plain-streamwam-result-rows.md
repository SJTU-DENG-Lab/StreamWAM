# Plain StreamWAM Result Rows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all highlighting from the three StreamWAM result rows while preserving their plain naming and exact values.

**Architecture:** Replace only the `<mark>`-wrapped cells in the three result rows with ordinary Markdown cells. Selectively stage the committed README view so concurrent literal interface and runtime-rename changes remain in the worktree.

**Tech Stack:** GitHub-flavored Markdown, Git selective staging, pytest

## Global Constraints

- Use exactly the three plain rows from `docs/superpowers/specs/2026-08-24-plain-streamwam-result-rows-design.md`.
- Do not add bold, `(Ours)`, HTML, icons, badges, or custom styling.
- Preserve every result value, table column, row order, protocol, and surrounding paragraph.
- Preserve the committed RTC-AC executable literals in the published README.
- Exclude all concurrent source, script, test, model-card, and runtime-rename changes.

---

### Task 1: Replace and validate the result rows

**Files:**
- Modify: `README.md`
- Reference: `docs/superpowers/specs/2026-08-24-plain-streamwam-result-rows-design.md`

**Interfaces:**
- Consumes: the three marked StreamWAM rows in the committed README
- Produces: three unstyled StreamWAM result rows with identical values

- [ ] **Step 1: Replace the marked rows**

Replace the LIBERO, RoboCasa, and RoboTwin rows with the exact plain rows in the
specification. Make no other content change.

- [ ] **Step 2: Selectively stage README**

Build the staged README from the intended working-tree presentation while
retaining the three committed RTC-AC literal command tokens. Assert
`git diff --cached --name-only` contains only `README.md`.

- [ ] **Step 3: Validate the staged README**

Read `git show :README.md` and assert exactly three rows begin with
`| StreamWAM |`, no `<mark>` or `(Ours)` remains, and all three rows exactly
match the specification.

- [ ] **Step 4: Run repository checks**

```bash
git diff --cached --check
/inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/wyx/FastWAM/.venv/bin/python -m pytest tests/test_package_identity.py -q
```

Expected: no whitespace errors and `2 passed`.

- [ ] **Step 5: Commit README only**

```bash
git commit -m "docs: remove StreamWAM result highlighting"
```

### Task 2: Review and publish

**Files:**
- Review: committed `README.md`
- Review: this task's design and plan

**Interfaces:**
- Consumes: the documentation-only commit
- Produces: a reviewed fast-forward update to `origin/main`

- [ ] **Step 1: Request independent read-only review**

Require checks for exact plain rows, absence of styling and `(Ours)`, unchanged
values and table structure, and exclusion of concurrent changes.

- [ ] **Step 2: Address Critical and Important findings**

Apply only required README corrections and repeat Task 1 Steps 3–4.

- [ ] **Step 3: Fetch and verify scope**

```bash
git fetch origin main
git diff --name-only origin/main..HEAD
```

Rebase normally if the remote advanced; never force-push.

- [ ] **Step 4: Push and verify SHA**

```bash
git push origin main
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

Expected: local and remote SHA values match exactly.

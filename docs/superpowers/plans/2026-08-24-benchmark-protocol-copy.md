# Benchmark Protocol Copy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add concise RoboCasa and RoboTwin evaluation protocols immediately before their existing result tables.

**Architecture:** Make two localized prose insertions in `README.md`. Protect the approved tables by capturing their text before editing and asserting that it remains unchanged afterward.

**Tech Stack:** GitHub-flavored Markdown, Python read-only assertions, pytest

## Global Constraints

- Insert only the two approved protocol paragraphs.
- Keep `Clean` and `Random` as the RoboTwin table headers.
- Do not alter any benchmark value, unit, row label, emphasis, or LIBERO text.
- Do not stage or edit the untracked `docs/huggingface/README.md` file.

---

### Task 1: Add benchmark protocol copy

**Files:**
- Modify: `README.md`
- Reference: `docs/superpowers/specs/2026-08-24-benchmark-protocol-copy-design.md`

**Interfaces:**
- Consumes: the existing RoboCasa and RoboTwin subsection/table boundaries
- Produces: two public evaluation-protocol paragraphs without table changes

- [ ] **Step 1: Capture the existing tables for comparison**

Use a read-only script to extract the RoboCasa table from `### RoboCasa` to
`### RoboTwin` and the RoboTwin table from `### RoboTwin` to
`## Runtime layout`. Keep these strings in memory or a temporary directory
outside the repository for post-edit comparison.

- [ ] **Step 2: Insert the RoboCasa paragraph**

Immediately before the RoboCasa table, insert:

```markdown
We evaluate on the RoboCasa target benchmark across 50 target tasks, with 50
trials per task, and report the average task success rate.
```

- [ ] **Step 3: Insert the RoboTwin paragraph**

Immediately before the RoboTwin table, insert:

```markdown
We evaluate 50 RoboTwin 2.0 tasks with 100 rollout episodes per task. `Clean`
reports the success rate under the easy setting, while `Random` reports the
success rate under the hard domain-randomization setting.
```

- [ ] **Step 4: Validate scope and content**

Run read-only assertions that each paragraph appears exactly once, occurs
between its subsection heading and table, and that both result tables match
their pre-edit rows exactly. Confirm the two table headers still contain
`Clean` and `Random`.

- [ ] **Step 5: Run repository checks**

```bash
git diff --check
/inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/wyx/FastWAM/.venv/bin/python -m pytest tests/test_package_identity.py -q
```

Expected: no whitespace errors and `2 passed`.

- [ ] **Step 6: Commit only the README**

```bash
git add README.md
git commit -m "docs: document RoboCasa and RoboTwin protocols"
```

### Task 2: Review and publish

**Files:**
- Review: `README.md`
- Review: `docs/superpowers/specs/2026-08-24-benchmark-protocol-copy-design.md`

**Interfaces:**
- Consumes: the committed README copy change
- Produces: a reviewed fast-forward update to `origin/main`

- [ ] **Step 1: Request independent read-only review**

Give the reviewer the design path, base SHA, and head SHA. Require checks for
exact protocol counts, placement, wording, unchanged result tables, and the
absence of unrelated changes.

- [ ] **Step 2: Address Critical and Important findings**

Apply only required corrections, rerun Task 1 Steps 4–5, and commit them.

- [ ] **Step 3: Fetch before publishing**

```bash
git fetch origin main
git status --short --branch
```

If `origin/main` advanced, rebase normally and rerun validation. Never
force-push.

- [ ] **Step 4: Push and verify**

```bash
git push origin main
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

Expected: local and remote SHA values match exactly.

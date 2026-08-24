# StreamWAM Public Name and Result Highlight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish StreamWAM as the sole public method name and highlight its three result rows without bold text or an `(Ours)` suffix.

**Architecture:** Update reader-facing README copy and Markdown-table cells while preserving the executable RTC-AC literals currently present on `origin/main`. Because concurrent runtime-rename work also modifies `README.md`, build and review a documentation-only staged diff that excludes all uncommitted source, test, and interface rename changes.

**Tech Stack:** GitHub-flavored Markdown, GitHub Markdown API, Git selective staging, pytest

## Global Constraints

- Use `StreamWAM` as the only reader-facing method name.
- Emphasize `action-conditioned streaming formulation` and action-conditioned future video generation in the introduction.
- Remove `(Ours)` and bold markup from all three StreamWAM result rows.
- Wrap every StreamWAM result cell in `<mark>...</mark>`.
- Preserve all benchmark values, protocols, units, row order, and ablations.
- Keep the published literal commands on the executable `origin/main` contract: `rtc_ac_checkpoint.pt`, `launch_streamwam_libero_rtc_ac_4gpu.sh`, and `--rtc-ac-accelerated`.
- Preserve and exclude all concurrent runtime-rename, source, example, test, and Hugging Face changes.

---

### Task 1: Build the publication README content

**Files:**
- Modify: `README.md`
- Reference: `docs/superpowers/specs/2026-08-24-streamwam-public-name-and-result-highlight-design.md`

**Interfaces:**
- Consumes: current benchmark tables and the executable interface committed on `origin/main`
- Produces: reader-facing StreamWAM naming and highlighted result rows

- [ ] **Step 1: Replace the introduction**

Use the exact approved StreamWAM/action-conditioned paragraph from the design
specification.

- [ ] **Step 2: Normalize reader-facing names**

Change release-status, Quick Start, launch-heading, checkpoint prose, LIBERO
analysis, and runtime-layout references to `StreamWAM`. Retain the three
lowercase/literal RTC-AC command tokens in Global Constraints.

- [ ] **Step 3: Replace the three proposed-method rows**

Insert the exact three `<mark>StreamWAM</mark>` rows from the specification and
remove all `(Ours)` and bold markup from those rows.

- [ ] **Step 4: Construct a documentation-only staged diff**

Compare the intended README against `HEAD:README.md`. Stage only public naming,
introduction, result-row, and result-analysis changes. Do not stage concurrent
literal `ac_stream` interface changes or any other worktree file.

- [ ] **Step 5: Validate staged content**

Read the staged README with `git show :README.md` and assert:

- the exact action-conditioned introduction is present;
- reader-facing `AC-StreamWAM`, `AC-Stream`, and `(Ours)` are absent;
- the three RTC-AC executable literals remain;
- exactly three table rows start with `<mark>StreamWAM</mark>`;
- every approved result number remains present;
- `git diff --cached --name-only` contains only `README.md`.

- [ ] **Step 6: Render-check the highlight**

Send a representative marked row to `https://api.github.com/markdown/raw` and
assert the returned HTML retains one `<mark>` element for each cell and contains
no `<strong>` elements.

- [ ] **Step 7: Run repository validation**

```bash
git diff --check -- README.md
/inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/wyx/FastWAM/.venv/bin/python -m pytest tests/test_package_identity.py -q
```

Expected: no README whitespace errors and `2 passed`.

- [ ] **Step 8: Commit only the staged README**

```bash
git commit -m "docs: unify StreamWAM result presentation"
```

### Task 2: Review and publish

**Files:**
- Review: `README.md` from the committed tree
- Review: the design and plan for this task

**Interfaces:**
- Consumes: the committed documentation-only change
- Produces: a reviewed fast-forward update to `origin/main`

- [ ] **Step 1: Request independent read-only review**

Require checks for the action-conditioned emphasis, sole public StreamWAM name,
exact marked rows, absence of bold/Ours, executable command validity, and
exclusion of concurrent changes.

- [ ] **Step 2: Address Critical and Important findings**

Apply only required README corrections, repeat Task 1 Steps 5–7, and commit.

- [ ] **Step 3: Fetch and verify publication scope**

```bash
git fetch origin main
git diff --name-only origin/main..HEAD
```

Expected: only this task's spec, plan, and README documentation commits. Rebase
normally if the remote advanced; never force-push.

- [ ] **Step 4: Push and verify**

```bash
git push origin main
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

Expected: local and remote SHA values match exactly.

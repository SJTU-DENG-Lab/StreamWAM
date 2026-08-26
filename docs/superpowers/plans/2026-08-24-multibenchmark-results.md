# Multi-Benchmark README Results Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish clean LIBERO, RoboCasa, and RoboTwin result tables in the root README while removing temporary runtime and citation content and crediting the benchmark implementation bases.

**Architecture:** Make one focused documentation change in `README.md`. Preserve the existing LIBERO content under a new subsection, add benchmark-specific lineage text and two independent result tables, then remove obsolete sections and extend acknowledgements.

**Tech Stack:** GitHub-flavored Markdown, shell validation, pytest

## Global Constraints

- Keep exactly one `## Current results` section with `### LIBERO`, `### RoboCasa`, and `### RoboTwin` in that order.
- State that StreamWAM models initialize from FastWAM-Joint checkpoints and are further trained.
- State that RoboCasa builds on X-WAM and RoboTwin builds on StarWAM.
- Preserve every approved result value and place measurement units in table headers.
- Represent unavailable measurements with an em dash (`—`).
- Remove the complete `## Accelerated runtime contract` and `## Citation` sections.
- Add direct StarWAM and X-WAM links to the existing acknowledgements.
- Do not modify runtime code, configuration, dependency metadata, or examples.

---

### Task 1: Reorganize the public README

**Files:**
- Modify: `README.md`
- Reference: `docs/superpowers/specs/2026-08-24-multibenchmark-results-design.md`

**Interfaces:**
- Consumes: the existing README section boundaries and approved benchmark values
- Produces: the final public README rendered by GitHub

- [ ] **Step 1: Add the result hierarchy and lineage statement**

Under `## Current results`, add an introductory paragraph that distinguishes
FastWAM-Joint checkpoint initialization from the X-WAM and StarWAM
benchmark-specific implementation lineage. Add `### LIBERO` before the existing
LIBERO protocol and table. Use this copy:

```markdown
All StreamWAM models are initialized from FastWAM-Joint checkpoints and then
further trained. The RoboCasa implementation builds on X-WAM, while the
RoboTwin implementation builds on StarWAM.
```

- [ ] **Step 2: Add the RoboCasa result table**

Append `### RoboCasa` and this table after the LIBERO analysis:

```markdown
| Method | Accuracy (%) ↑ | Chunk Time (ms) ↓ | Total Time (s) ↓ |
|---|---:|---:|---:|
| X-WAM | 75.42 | 504.00 | 37.31 |
| X-WAM-CD | 75.33 | 135.21 | 33.60 |
| **StreamWAM (Ours)** | **75.35** | **136.76** | **11.76** |
```

- [ ] **Step 3: Add the RoboTwin result table**

Append `### RoboTwin` and this table after RoboCasa:

```markdown
| Method | Clean (%) ↑ | Random (%) ↑ | Total (%) ↑ | Chunk Time (ms) ↓ | Total Time (s) ↓ |
|---|---:|---:|---:|---:|---:|
| StarWAM | 84.8 | 86.0 | 85.4 | 189.3 | — |
| StarWAM-CD | 79.0 | 79.2 | 79.1 | 81.6 | — |
| **StreamWAM (Ours)** | **87.2** | **88.8** | **87.6** | — | **112.2** |
```

- [ ] **Step 4: Remove obsolete sections and extend acknowledgements**

Delete everything from `## Accelerated runtime contract` up to, but not
including, `## Runtime layout`. Delete the complete `## Citation` section.
Keep the existing acknowledgement links and add:

```markdown
[StarWAM](https://github.com/shaohua-pan/StarWAM)
[X-WAM](https://github.com/sharinka0715/X-WAM)
```

- [ ] **Step 5: Run structural and content validation**

Run a read-only script that asserts:

- one `## Current results` heading;
- benchmark subsections occur in the approved order;
- every RoboCasa and RoboTwin value appears;
- `## Accelerated runtime contract` and `## Citation` are absent;
- both new acknowledgement URLs appear;
- every row in each new Markdown table has the expected column count.

Expected: all assertions pass with exit code 0.

- [ ] **Step 6: Run repository validation**

Run:

```bash
git diff --check
/inspire/qb-ilm/project/qproject-fundationmodel/yangyi-253108120173/wyx/FastWAM/.venv/bin/python -m pytest tests/test_package_identity.py -q
```

Expected: no whitespace errors and `2 passed`.

- [ ] **Step 7: Commit the README**

```bash
git add README.md
git commit -m "docs: add RoboCasa and RoboTwin results"
```

### Task 2: Review and publish

**Files:**
- Review: `README.md`
- Review: `docs/superpowers/specs/2026-08-24-multibenchmark-results-design.md`

**Interfaces:**
- Consumes: the committed README change from Task 1
- Produces: a reviewed fast-forward update to `origin/main`

- [ ] **Step 1: Request independent review**

Provide the reviewer with the design specification, base SHA, head SHA, and a
request to classify findings as Critical, Important, or Minor. Require checks
for model-lineage accuracy, table values and columns, removed sections, and
acknowledgement links.

- [ ] **Step 2: Address review findings**

Fix every Critical or Important finding, rerun Task 1 Steps 5–6, and commit the
correction. Record Minor findings that do not affect public accuracy.

- [ ] **Step 3: Confirm the remote branch has not advanced**

```bash
git fetch origin main
git status --short --branch
```

Expected: a clean worktree with local `main` only ahead of `origin/main`. If the
remote advanced, rebase normally and rerun validation; never force-push.

- [ ] **Step 4: Push and verify the remote SHA**

```bash
git push origin main
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

Expected: local and remote SHA values match exactly.

# LIBERO Current Results Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the README's single AC-StreamWAM latency row with the approved six-method LIBERO success-and-efficiency comparison and publish the FastWAM-Joint-CD checkpoint status.

**Architecture:** Keep the change documentation-only. One Markdown table reports all main methods and ablations under a shared evaluation protocol, followed by a concise speed/accuracy interpretation; the release-status table gains one Hugging Face checkpoint row.

**Tech Stack:** Markdown, Git, existing repository documentation checks.

## Global Constraints

- Use one results table, not separate comparison and ablation tables.
- Preserve all supplied success, chunk-time, and long/short episode-time values exactly.
- Define 50 trials per task, 10 tasks per suite, arithmetic mean success, chunk latency, and long/short episode wall time.
- Use `AC-StreamWAM (Ours)`, `w/o Action Conditioning`, and `w/o Slot Encoder` as the public row labels.
- Add `FastWAM-Joint-CD checkpoint` immediately before `AC-StreamWAM checkpoint` in Release status.
- Link both checkpoint rows to `https://huggingface.co/SJTU-DENG-Lab/StreamWAM`.
- Leave the accelerated runtime contract and implementation interfaces unchanged.

---

### Task 1: Publish the Combined LIBERO Results

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: exact table and interpretation in `docs/superpowers/specs/2026-08-24-libero-current-results-design.md`.
- Produces: a single GitHub Markdown table with six methods, seven reported metrics, and two released checkpoint rows.

- [ ] **Step 1: Add the checkpoint release row**

Insert `FastWAM-Joint-CD checkpoint` directly before `AC-StreamWAM checkpoint`, using the same official Hugging Face link and available status.

- [ ] **Step 2: Replace Current results**

Replace the D8-only description, one-row table, and 45.20 ms paragraph with the approved evaluation definition, six-row result table, and speed/ablation interpretation. Do not change the following `Accelerated runtime contract` section.

- [ ] **Step 3: Verify the published values and structure**

Run:

```bash
git diff --check
rg -n "FastWAM-Joint-CD checkpoint|AC-StreamWAM checkpoint|50 trials per task|Chunk Time|Episode Time|w/o Action Conditioning|w/o Slot Encoder" README.md
rg -n "493.0|114.2|142.3|41.0|35.1|36.3|16.31 / 8.25|5.36 / 3.15" README.md
! rg "45.20 ms|Steady-state D8 statistics exclude" README.md
```

Expected: both checkpoint rows and the full evaluation definition are present, representative values from every row are present, obsolete single-run wording is absent, and the Markdown diff has no whitespace errors.

- [ ] **Step 4: Commit, review, and push**

```bash
git add README.md
git commit -m "docs: publish LIBERO success and efficiency results"
git push origin main
```

After review and push, verify the remote `main` SHA equals local `HEAD`.

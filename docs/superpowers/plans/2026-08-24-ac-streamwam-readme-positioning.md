# AC-StreamWAM README Positioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish AC-StreamWAM as the paper-facing method name throughout the root README while preserving every existing `rtc_ac` implementation interface.

**Architecture:** This is a documentation-only naming layer. Reader-facing prose, headings, badges, tables, and layout comments use AC-StreamWAM; literal script names, CLI flags, configuration identifiers, and D0/D8 diagnostic keys remain unchanged so documented commands continue to execute.

**Tech Stack:** Markdown, Git, existing Bash/Python command interfaces.

## Global Constraints

- Use **Action-Conditioned Streaming WAM (AC-StreamWAM)** as the paper-facing method name.
- Keep `RTC-AC`/`rtc_ac` only where it is part of a literal implementation interface.
- The introduction contains exactly the two approved paragraphs and no acceleration paragraph.
- The release-status table contains exactly five approved rows in the approved order.
- The AC-StreamWAM checkpoint links to `https://huggingface.co/SJTU-DENG-Lab/StreamWAM`.
- Do not rename code, scripts, CLI flags, configuration keys, or D0/D8 diagnostics.
- Push only after the README diff and compatibility checks pass.

---

### Task 1: Publish the AC-StreamWAM README Naming Layer

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the approved copy and naming map in `docs/superpowers/specs/2026-08-24-ac-streamwam-readme-positioning-design.md`.
- Produces: reader-facing AC-StreamWAM terminology while retaining the literal `launch_streamwam_libero_rtc_ac_4gpu.sh`, `--rtc-ac-accelerated`, `rtc_ac_checkpoint.pt`, `prewarmed_d0`, and `prewarmed_d8` tokens.

- [ ] **Step 1: Replace the introduction and release metadata**

Insert the two approved opening paragraphs. Replace the checkpoint badge with a Hugging Face badge linking to the official StreamWAM repository. Replace the release-status table with the five approved rows and remove the legacy ModelScope note.

- [ ] **Step 2: Apply the README-wide reader-facing naming map**

Rename the Quick Start heading, launch subsection, checkpoint prose, measurement prose, result model label, geometry label, and runtime-layout comments from RTC-AC to AC-StreamWAM. Preserve all literal implementation tokens listed in the Interfaces block.

- [ ] **Step 3: Verify copy, naming, and command compatibility**

Run:

```bash
git diff --check
rg -n "Action-Conditioned Streaming WAM|AC-StreamWAM|huggingface.co/SJTU-DENG-Lab/StreamWAM" README.md
rg -n "launch_streamwam_libero_rtc_ac_4gpu.sh|--rtc-ac-accelerated|rtc_ac_checkpoint.pt|prewarmed_d0|prewarmed_d8" README.md
! rg "Accelerated RTC-AC|Launch RTC-AC|RTC-AC measurements|RTC-AC benchmark checkpoint|MoT and Shared-DiT checkpoints|ModelScope-Checkpoints" README.md
```

Expected: all approved public terms and implementation tokens are present, obsolete reader-facing terminology is absent, and the diff has no whitespace errors.

- [ ] **Step 4: Commit and push**

```bash
git add README.md
git commit -m "docs: introduce AC-StreamWAM positioning"
git push origin main
```

After pushing, verify `git ls-remote origin refs/heads/main` matches `git rev-parse HEAD`.

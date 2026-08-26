# Full-Video Attention Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the existing hero attention matrices so action queries read the full visual future within each chunk.

**Architecture:** Preserve the current 10×10 HTML matrices and change only cell classes. Update the canonical five-token topology in the existing regression test first, then apply the same allowed cells to both panels while retaining Stream-WAM's two cross-chunk condition cells.

**Tech Stack:** Semantic HTML, CSS class-based matrix cells, Python, pytest

## Global Constraints

- Add no visible characters, labels, rows, columns, or explanatory copy.
- Keep the token order `f₀`, `f₁`, `fₕ`, `a₁`, `aₕ`.
- Within each chunk, both action rows read all five token groups.
- Standard Joint WAM remains fully masked across chunks.
- Stream-WAM differs only at the existing two previous-action-to-next-`f₁` cells.
- Do not change CSS, JavaScript, captions, runtime animation, method figure, prose, or results.

---

### Task 1: Correct the Matrix Topology

**Files:**
- Modify: `tests/test_academic_project_page.py:973-1012`
- Modify: `docs/index.html:110-140`

**Interfaces:**
- Consumes: the full-video mask contract in `MoTWAM._build_mot_attention_mask`
- Produces: two 100-cell row-major matrices with the corrected repeated five-token blocks

- [ ] **Step 1: Update the failing topology fixture**

In `test_attention_matrix_compares_visual_and_action_attention`, change the
canonical action-row loop to allow every within-chunk key:

```python
for row in (offset + 3, offset + 4):
    for column in range(offset, offset + 5):
        expected[row * 10 + column] = action
```

Keep `cross_indices = {(6 * 10) + 3, (6 * 10) + 4}` unchanged.

- [ ] **Step 2: Verify the topology test fails**

Run:

```bash
pytest -q tests/test_academic_project_page.py -k attention_matrix
```

Expected: FAIL because the current HTML masks the `a₁/aₕ` query cells at the
`f₁/fₕ` key columns.

- [ ] **Step 3: Correct only the HTML cell classes**

For both matrices and both five-token diagonal blocks, replace the two
`masked-cell` entries at the `f₁` and `fₕ` columns of each `a₁` and `aₕ` row
with `action-token`. Do not change the two `cross-chunk-condition` cells.

- [ ] **Step 4: Verify focused and full tests**

Run:

```bash
pytest -q tests/test_academic_project_page.py -k attention_matrix
pytest -q tests/test_academic_project_page.py
git diff --check
```

Expected: the focused test and full project-page suite pass with no whitespace
errors.

- [ ] **Step 5: Commit without pushing**

```bash
git add docs/index.html tests/test_academic_project_page.py docs/superpowers/plans/2026-08-26-full-video-attention-correction.md
git commit -m "fix: correct full-video attention masks"
```

Do not push because the branch already contains another agent's unpushed
method-figure commits and the user requested only the local correction in this
turn.

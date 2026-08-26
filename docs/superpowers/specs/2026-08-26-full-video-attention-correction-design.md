# Full-Video Attention Correction Design

## Goal

Correct the existing Standard Joint WAM and Stream-WAM hero matrices without
adding labels, annotations, rows, columns, or explanatory copy.

## Source of Truth

- The LIBERO FastWAM-Joint and AC-Stream recipes use `full_video` action
  conditioning.
- `MoTWAM._build_mot_attention_mask` therefore allows every action query to
  read `f0`, `f1`, `fh`, and the complete action stream.
- AC-Stream adds the aligned action-condition path into the next `f1` visual
  query. That highlighted cross-chunk delta remains unchanged.

## Matrix Correction

- Keep both existing 10×10 matrices, their two five-token chunks, and the
  labels `f₀`, `f₁`, `fₕ`, `a₁`, and `aₕ`.
- Within each five-token diagonal block:
  - `f₀` reads only `f₀`.
  - `f₁` and `fₕ` read `f₀`, `f₁`, and `fₕ`.
  - `a₁` and `aₕ` read all five token groups.
- The Standard Joint WAM off-diagonal regions remain fully masked.
- Stream-WAM remains identical to Standard Joint WAM except for the existing
  two highlighted previous-action-to-next-`f₁` cells.

## Constraints

- Add no visible characters or labels.
- Do not change layout, colors, captions, runtime animation, method figure,
  prose, or result tables.
- Update the exact topology regression test before changing the HTML.

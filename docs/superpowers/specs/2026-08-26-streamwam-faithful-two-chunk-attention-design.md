# Stream-WAM Faithful Two-Chunk Attention Design

## Goal

Replace the incorrect simplified attention comparison with a two-chunk mask
that follows the FastWAM token vocabulary and the repository's AC-Stream
condition path exactly. Align the right hero figure with the left hero content
from the `Stream-WAM` eyebrow through the LIBERO metric baseline.

## Source of truth

- FastWAM's training mask supplies the within-chunk token groups and visual
  language: `f₀`, `f₁`, `fₕ`, `a₁`, and `aₕ`.
- `streamwam/modules/ac_stream.py` is authoritative for the cross-chunk
  addition. `forward_ac_stream` computes `z1_condition` from condition-stream
  keys and adds it only to the `z1` future-video slice.
- `build_ac_stream_prev_action_target` and `apply_ac_stream_hard_prefix_`
  establish that the condition content is the aligned known prefix of the
  active action chunk, not the entire prior visual/action state.

## Standard Joint WAM mask

- Draw one 10×10 grouped matrix with two five-token chunks:
  - Chunk `k`: `f₀ᵏ`, `f₁ᵏ`, `fₕᵏ`, `a₁ᵏ`, `aₕᵏ`.
  - Chunk `k+1`: `f₀ᵏ⁺¹`, `f₁ᵏ⁺¹`, `fₕᵏ⁺¹`, `a₁ᵏ⁺¹`, `aₕᵏ⁺¹`.
- Each diagonal 5×5 block repeats the FastWAM-style within-chunk mask:
  - `f₀` reads only `f₀`.
  - `f₁` and `fₕ` read `f₀`, `f₁`, and `fₕ`.
  - `a₁` and `aₕ` read `f₀`, `a₁`, and `aₕ`.
- Every cell in both off-diagonal 5×5 regions is masked. The standard joint
  model therefore has no attention between chunk `k` and chunk `k+1`.

## Stream-WAM mask

- Begin with the identical two diagonal 5×5 blocks and identical masked
  off-diagonal regions.
- Add only the AC-Stream condition path in the lower-left off-diagonal region:
  the aligned action-prefix portion of chunk `k` conditions `f₁ᵏ⁺¹`.
- Do not open paths from the previous chunk into `f₀ᵏ⁺¹`, `fₕᵏ⁺¹`, or the next
  action groups.
- Represent the prefix without an exact action count. Highlight the relevant
  `f₁ᵏ⁺¹` query versus prior-action key region and label it with a compact
  `a-prefixᵏ → f₁ᵏ⁺¹` annotation.
- Use the same cell positions, scale, and legend for both methods so the added
  cross-chunk path is the only visual difference.

## Hero geometry

- Wrap the left content from the eyebrow through the metrics in a `.hero-main`
  container while leaving the DENG Lab lockup above it.
- On wide screens, use a two-column, two-row grid:
  - DENG Lab lockup occupies column one, row one.
  - `.hero-main` occupies column one, row two.
  - `.hero-figure` occupies column two, row two.
- Stretch `.hero-figure` to the row-two height. The right figure therefore
  starts exactly with the `Stream-WAM` eyebrow and ends exactly with the bottom
  of the LIBERO metrics, independent of title wrapping at 1440 or 1920 px.
- Keep the figure caption available to assistive technology without letting a
  visible caption extend below the metric baseline.
- Restore the existing single-column document flow below 1040 px.

## Runtime and sizing

- Preserve the approved runtime schedule, curtain animation, labels, and
  reduced-motion behavior unchanged.
- Fit the two 10×10 attention matrices below the runtime comparison within the
  row-two height. Use compact grouped cells and mathematical labels rather than
  explanatory prose.
- At mobile widths, stack the two masks and keep every cell inside its panel;
  the full page must remain overflow-free at 320 px.

## Accessibility and testing

- Update the screen-reader description to state that standard joint WAM has no
  cross-chunk attention and Stream-WAM adds only the aligned action-prefix to
  next-`f₁` condition path.
- Automated tests assert the 10 labels per matrix, two repeated 5×5 diagonal
  patterns, fully masked standard off-diagonal quadrants, and the single
  Stream-WAM cross-chunk addition.
- Browser tests measure the eyebrow/figure top delta and metrics/figure bottom
  delta at 1920 and 1440 px; both must be within two CSS pixels.
- Recheck 1024, 390, and 320 px for stacking and horizontal overflow.
- Increment the matching CSS and JavaScript asset version before deployment.

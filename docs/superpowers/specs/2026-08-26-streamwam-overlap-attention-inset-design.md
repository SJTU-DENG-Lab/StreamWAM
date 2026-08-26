# Stream-WAM Overlap + Attention Inset Design

## Goal

Refine the lower inset of the Stream-WAM method figure into two coordinated
views:

- a left `Temporal overlap` view explains how the executing action segments
  align with the next chunk and with the conditioning inputs;
- a right `Action-conditioned attention` view shows the Stream-WAM attention
  mask already established in the hero visual.

The global `t₀ → t₃` streaming overview remains above both views.

## Global Overview Changes

Preserve the cold start and two overlapped streaming cycles. Remove the
`action-conditioned` label from the overview because the right inset now
explains that mechanism. Remove every `handoff` label; continuous A₀/A₁ bars
and arrows carry the transition semantics without naming the boundary.

The overview remains visually primary and unchanged in height unless the lower
split requires a small spacing adjustment.

## Lower Inset Layout

Keep one bordered lower inset in the existing whitespace. Split it into two
columns with a subtle vertical divider:

- left column: approximately 55% of the inset width;
- right column: approximately 45% of the inset width.

Use `Temporal overlap` as the left heading and `Action-conditioned attention`
as the right heading. Do not use `One overlap window`.

## Left: Temporal Overlap

### Action segments

Draw A₀ with three continuous segments:

- executed action segment;
- gold shared/overlap segment that executes while inference runs;
- pale look-ahead segment.

Draw A₁ below it with:

- a gold `shared prefix` aligned exactly under the A₀ overlap;
- a green continuation aligned after the shared prefix.

Retain the compact equality `A₀[8:16] = A₁[0:8]`. Do not show the word
`handoff`.

### Direct condition correspondence

Place `known action context` directly below `shared prefix`, with identical x
position and width. Connect them using a straight vertical arrow.

Place `unknown future slots` directly below the A₀ `look-ahead` segment, with
the same x position and width. Connect them using a straight vertical arrow.

This alignment must make both correspondences visible without curved,
dog-legged, or diagonal routing.

### Observation and update

Place O₁ on the lower input row. Connect O₁ to the update with one straight
horizontal arrow. Connect the combined condition input to the same update with
another straight horizontal arrow.

Rename the model block from `AC-Stream update` to `Stream Update` everywhere in
the inset. Its neutral output continues to A₁ and its gold output continues to
V₁, but do not add an `action-conditioned` text label to those paths.

## Right: Action-Conditioned Attention

Show only the Stream-WAM mask, not the Standard Joint WAM comparison. This
keeps the inset legible and avoids duplicating the hero comparison.

Use the same two-chunk, ten-token ordering as the hero:

`f₀, f₁, fₕ, a₁, aₕ | f₀, f₁, fₕ, a₁, aₕ`

The mask preserves the established within-chunk structure. Highlight only the
two cross-chunk condition cells that connect previous-chunk `a₁/aₕ` keys to the
next-chunk `f₁` query. Use gold for these cells, blue for allowed visual
attention, green/teal for allowed action attention, and pale outlined cells for
masked positions.

Add a restrained gold annotation such as `action prefix → next visual future`
below the matrix. Do not introduce new mathematical claims or additional
cross-chunk connections.

## Visual Language

- Static flat SVG only; no animation or cursor.
- Preserve the current white background, academic thin strokes, muted green,
  gold, blue, and gray palette.
- The attention matrix uses square cells with small gaps and readable token
  labels at the 1100 px display width.
- No gradients, filters, shadows, glow, or cartoon illustration.
- Keep the inset visually subordinate to the global overview.

## Page Integration

Keep the SVG in its current article position. Advance the cache key and update
the hidden description and caption so they mention both the temporal-alignment
view and the Stream-WAM attention mask.

The desktop breakout and mobile 1100 px minimum artwork width remain unchanged.

## Verification

- The overview contains no `action-conditioned` or `handoff` text.
- The inset contains exactly two titled columns.
- `shared prefix` and `known action context` have the same x position and width.
- `look-ahead` and `unknown future slots` have the same x position and width.
- Both correspondence arrows are straight and vertical.
- O₁ reaches `Stream Update` through a straight horizontal path.
- The update is labelled `Stream Update`, not `AC-Stream update`.
- The right column contains one 10×10 Stream-WAM mask.
- Token ordering matches the hero mask in both rows and columns.
- Exactly two cells are marked as cross-chunk action conditioning.
- Those cells correspond to previous `a₁/aₕ` keys and next `f₁` query.
- No Standard Joint WAM comparison appears in the inset.
- The 1600 px and 1100 px renders remain legible and collision-free.
- XML validation, page tests, and `git diff --check` pass.

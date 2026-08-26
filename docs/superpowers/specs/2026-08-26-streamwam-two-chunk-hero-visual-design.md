# Stream-WAM Two-Chunk Hero Visual Design

## Goal

Correct the hero visual so its animation reads as elapsed time, both runtime
tracks begin with prediction, and the attention panel faithfully communicates
cross-chunk action conditioning.

## Runtime timeline

- Keep the comparison between `Synchronous WAM` and `Stream-WAM`, but give the
  runtime panel the full figure width so the timing structure is legible.
- Rename the visual vocabulary to `World-Action Prediction`, `Robot Execution`,
  and `Committed Actions`. Remove `Model update`, `Robot motion`, and `Action
  prefix` from this figure.
- Pre-render each complete schedule beneath a black curtain. A yellow time
  cursor and the curtain edge move together from left to right, progressively
  uncovering the schedule. Bars never scale, pop in, or disappear during a
  cycle.
- Both rows begin with a prediction interval before execution begins.
- `Synchronous WAM` alternates prediction and execution, so robot execution is
  interrupted while the next chunk is predicted.
- `Stream-WAM` begins with prediction, then shows one continuous execution rail
  while shorter later prediction intervals overlap it.
- In reduced-motion mode, remove the curtain and cursor and display the complete
  schedules.

## Two-chunk attention

- Give the attention panel the full figure width beneath the runtime panel.
- Replace the two small single-chunk matrices with one larger two-chunk diagram.
- Group the query and key axes into `Chunk k` and `Chunk k+1`, each containing a
  visual group and an action group.
- Label the aligned previous-chunk action group `Committed Actions`, matching
  the runtime vocabulary.
- Highlight the directed block from the committed action keys in `Chunk k` to
  the future-visual queries in `Chunk k+1`. This represents the implementation:
  aligned actions from the active chunk are placed in the condition stream and
  injected only into the next prediction's future visual tokens (`z1`).
- Keep the remaining allowed and masked regions visually secondary, with row
  and column labels so the diagram is understandable without guessing colors.
- Animate only the highlighted cross-chunk path with a restrained pulse; do not
  animate the matrix geometry.

## Layout and accessibility

- Stack runtime and attention panels inside the hero figure. This makes both
  diagrams materially larger without widening the long-form article.
- Preserve the existing wide hero shell and compact benchmark rail.
- Update the screen-reader description and caption to explain prediction-first
  startup, interrupted synchronous execution, continuous Stream-WAM execution,
  and the `Chunk k` action-to-`Chunk k+1` visual connection.
- Preserve no-horizontal-overflow behavior at desktop, tablet, and mobile
  breakpoints.

## Verification

- Tests require the black curtain, synchronized time cursor, prediction-first
  markup, continuous Stream-WAM execution, new vocabulary, and a two-chunk
  attention block.
- The former per-block reveal keyframes and the labels `Model update`, `Robot
  motion`, and `Action prefix` must be absent from the hero figure.
- Run the page test suite, JavaScript syntax check, whitespace check, and local
  browser checks at 1920, 1440, 1024, and 390 px widths.

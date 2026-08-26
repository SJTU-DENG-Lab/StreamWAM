# Stream-WAM Compact Comparison Design

## Goal

Reduce the hero figure to the visual weight of the left hero copy while
preserving the approved runtime animation and making the attention difference
immediately understandable.

## Runtime panel

- Preserve the two-row scan-reveal timeline and its synchronous-versus-streaming
  timing semantics.
- Remove the visible `Committed Actions` segment from the Stream-WAM timeline.
- Use the legend labels `Model Prediction` and `Robot Execution` in normal title
  case.
- Remove the heading `World-Action Prediction × Robot Execution`; retain only a
  small `Streaming control loop` context label.
- Compress vertical spacing without changing the relative horizontal timing of
  the prediction and execution rails.

## Attention comparison

- Replace the single large matrix with two equal compact matrices shown side by
  side on desktop: `Standard attention` and `Action-conditioned attention`.
- Both matrices represent the same two chunks using the compact labels `Vₖ`,
  `Aₖ`, `Vₖ₊₁`, and `Aₖ₊₁` on both axes.
- In the standard matrix, the `Aₖ` key to `Vₖ₊₁` query cell remains masked. In
  the action-conditioned matrix, the same cell is highlighted.
- Show `Aₖ → Vₖ₊₁` as a prominent badge beside the action-conditioned panel
  title instead of tiny text inside a matrix cell.
- Remove `Two-chunk directed attention`, the sentence-style committed-action
  callout, and the visible `Keys` / `Queries` labels.
- Stack the two compact panels on narrow screens without horizontal overflow.

## Size and accessibility

- Remove the 680 px minimum height from the outer visual and reduce panel,
  timeline, and matrix spacing so the right figure is close to the height of
  the left hero content on wide screens.
- Keep the scan curtain, time cursor, reduced-motion behavior, screen-reader
  explanation, and figure caption.
- Version both local CSS and JavaScript URLs together to avoid stale Pages CSS.

## Verification

- Tests assert exact runtime labels, absence of removed copy, two attention
  matrices, and the differing cross-chunk cell at the same topology position.
- Browser checks cover 1920, 1440, 1024, 390, and 320 px widths, including outer
  figure height and horizontal overflow.

# Stream-WAM Readable Hero Visual Design

## Goal

Improve the first-screen runtime and attention graphic at desktop widths without changing its scheduling or attention semantics.

## Approved design

- Remove the decorative `01` and `02` badges while retaining the two section headings.
- Render `Streaming control loop` as a clear interface heading rather than a code label: use the site's sans-serif stack at `14px`, weight `760`, compact tracking, and a brighter neutral color while leaving the adjacent legend unchanged.
- Give the graphic more desktop width by shifting the hero column ratio toward the right and increasing the graphic column's minimum width.
- Enlarge the `a-prefixᵏ → f₁ᵏ⁺¹` badge, both attention panel titles, chunk headings, and row/column token labels.
- Use the unused space below the attention comparison: at desktop widths, each of the ten mask rows and cells must be at least `16px` high, token labels must be approximately `10px`, and the attention comparison should extend toward the bottom of the hero graphic with only normal card padding remaining.
- Preserve the existing two-chunk 10×10 masks exactly.
- In Synchronous WAM, each prediction must meet the following execution segment without an empty interval.
- In Stream-WAM, show three evenly spaced execution chunks with narrow boundaries. For every execution chunk, start the next model prediction at the chunk's 50% point and finish it exactly at that execution chunk's end; retain the initial prediction that produces the first action chunk.
- Preserve the existing reveal animation and the graphic's top/bottom alignment with the left hero content.
- At widths of 1040px and below, keep the existing stacked layout with no horizontal page overflow at 390px or 320px.

## Verification

- The page test must reject the old numeric badges and enforce the wider desktop column rule and readable typography floors.
- All existing topology and runtime tests must continue to pass.
- Browser QA must cover 1920px, 1440px, 390px, and 320px viewports.
- At 1440px, the attention comparison must remain fully inside `.pipeline-visual`, its labels must resolve to at least `9.5px`, and its bottom gap must be materially smaller than before the change.

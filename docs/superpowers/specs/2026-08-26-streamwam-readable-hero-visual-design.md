# Stream-WAM Readable Hero Visual Design

## Goal

Improve the first-screen runtime and attention graphic at desktop widths without changing its scheduling or attention semantics.

## Approved design

- Remove the decorative `01` and `02` badges while retaining the two section headings.
- Give the graphic more desktop width by shifting the hero column ratio toward the right and increasing the graphic column's minimum width.
- Enlarge the `a-prefixᵏ → f₁ᵏ⁺¹` badge, both attention panel titles, chunk headings, and row/column token labels.
- Preserve the existing two-chunk 10×10 masks exactly.
- Preserve the existing runtime animation and the graphic's top/bottom alignment with the left hero content.
- At widths of 1040px and below, keep the existing stacked layout with no horizontal page overflow at 390px or 320px.

## Verification

- The page test must reject the old numeric badges and enforce the wider desktop column rule and readable typography floors.
- All existing topology and runtime tests must continue to pass.
- Browser QA must cover 1920px, 1440px, 390px, and 320px viewports.

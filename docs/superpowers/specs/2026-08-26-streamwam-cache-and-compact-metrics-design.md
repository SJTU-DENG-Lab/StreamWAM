# Stream-WAM Cache and Compact Metrics Design

## Root cause

The deployed HTML and CSS match the repository, but GitHub Pages serves both
with `Cache-Control: max-age=600`. Because `index.html` referenced the stable
`styles.css` URL, an existing browser could combine newly deployed HTML with a
cached pre-deployment stylesheet. That mismatch rendered the new diagram as
unformatted text and removed the title accent colors.

## Static asset versioning

- Add an explicit release query to the stylesheet and script URLs in
  `docs/index.html`.
- Tests require both local asset references to carry the same nonempty version.
- Every project-page visual release increments that version, forcing browsers
  to request the matching CSS and JavaScript while preserving simple filenames.

## Compact metrics

- Keep all three metrics immediately below the View Code, Get Models, and Paper
  controls.
- Each metric should have approximately the same height and visual weight as a
  project action button, rather than a large result card.
- Remove the large paper panel, rounded outer card, shadow, serif display
  numbers, and 100+ px minimum height.
- Use a compact inline rail with small separators:
  - `98.20%` with `LIBERO average success`;
  - `41.0 ms` with `LIBERO chunk time` and a small `12.0× faster vs FastWAM`
    badge;
  - `4.74 s` with `LIBERO total time` and a small `3.4× faster vs FastWAM`
    badge.
- At narrow widths the items may wrap, but no item becomes a tall standalone
  card and the page must not overflow horizontally.

## Hero-only width

- Restore the shared shell used by lower-page breakout content to its prior
  maximum width of 1220 px.
- Introduce a separate Hero width of approximately 1680 px with 32 px viewport
  gutters, leaving the 800 px reading column unchanged.
- Increase the desktop title ceiling modestly and allocate enough Hero width to
  keep the runtime and attention graphics readable.
- The fixed top navigation and all article text retain their current widths.

## Existing visual

- Preserve the two-track synchronous/Stream-WAM animation and action-conditioned
  attention matrix.
- Do not change their information content in this pass.
- The versioned stylesheet must make the title colors and both diagrams appear
  immediately after deployment without a manual hard refresh.

## Verification

- Automated tests cover versioned local asset URLs, compact metric CSS, separate
  Hero/shared shell variables, exact metric copy, and the existing diagram.
- Browser checks run at 1920×1080, 1440×1000, 1024×900, and 390×844.
- Verify computed title colors, grid layout for the diagram, compact metric
  height near the project-button height, no horizontal overflow, and a successful
  GitHub Pages deployment.

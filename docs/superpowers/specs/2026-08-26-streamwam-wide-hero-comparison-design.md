# Stream-WAM Wide Hero Comparison Design

## Goal

Make the project-page hero wider, faster to read, and more faithful to the
Stream-WAM runtime: compare a synchronous world-action model with Stream-WAM,
then explain action-conditioned attention with a compact token matrix.

## Repository and deployment layout

- Remove the `academic_project_page/` directory name from the public repository.
- Move the deployable static site to the conventional `docs/` directory.
- Update the GitHub Pages workflow, tests, local preview commands, asset paths,
  and figure-generation defaults so the public URL remains
  `https://sjtu-deng-lab.github.io/StreamWAM/`.
- Preserve the existing long-form article and benchmark content unless a change
  is explicitly listed below.

## Hero layout

- Increase the main page shell to approximately 1480 px and reduce unused side
  margins at large viewports.
- Use a wide, responsive hero with three information zones:
  1. project identity, title, lead, actions, and headline metrics;
  2. a two-row animated runtime comparison;
  3. an action-conditioned attention matrix.
- Collapse the comparison and matrix beneath the copy at narrower widths. Keep
  the page free of horizontal overflow at 390 px, 1024 px, and 1440 px.

## Title and metrics

- The exact title remains “Streaming Your World-Action Model for Real-Time Robot
  Manipulation.”
- Color only “Streaming” in teal and “Real-Time” in violet. All remaining words
  use the primary ink color.
- Place the three headline metrics directly below the View Code, Get Models,
  and Paper controls:
  - `98.20%` — LIBERO average success;
  - `41.0 ms` — LIBERO chunk time, `12.0× faster vs FastWAM`;
  - `4.74 s` — LIBERO total time, `3.4× faster vs FastWAM`.
- Remove the previous RoboCasa total-time headline from the hero. Detailed
  benchmark tables remain unchanged.

## Runtime comparison animation

- Compare only `Synchronous WAM` and `Stream-WAM`; remove `Naive Async`.
- Do not label the animated generation block as “Inference” or “Video & Action
  Chunk.” A violet block and a small generative shimmer visually encode model
  computation.
- Animate each violet block as a left-to-right reveal coupled to a moving time
  cursor, rather than moving a fully formed block across the timeline.
- The synchronous row alternates a relatively long violet generation interval
  with a shorter execution interval.
- The Stream-WAM row shows a shorter violet generation interval overlapping a
  longer, continuous teal execution rail. The committed action prefix is shown
  as a compact orange segment within the active execution rail.
- No exact action counts or timing numbers appear inside the animation.
- Reduced-motion mode displays the final comparison state without sweep or
  shimmer animations.

## Attention diagram

- Replace the former text flow (`Current Observation → Action Prefix → ...`)
  with a compact matrix inspired by FastWAM’s attention-mask visual language.
- Use blue rounded cells for visual tokens and yellow rounded cells for action
  tokens. Use filled or glowing cells to show allowed directed attention and
  muted outlined cells for masked connections.
- The diagram compares standard separated attention with Stream-WAM’s directed
  action-to-future-video conditioning, emphasizing the newly enabled action
  prefix to future-video region.
- Provide a concise visible legend and a complete screen-reader description;
  the cell grid itself remains decorative.
- Recreate the diagram in repository-native HTML/CSS. Do not copy or ship
  FastWAM’s raster assets.

## README

- Move the complete Citation section immediately after `Runtime layout` and
  before License/Acknowledgements.
- Keep the existing Project Page badge before GitHub Code.

## Verification

- Automated tests assert the site lives in `docs/`, the Pages workflow deploys
  `docs/`, only the requested title words are accented, the metric values and
  speedups are exact, Naive Async and the old text-flow labels are absent, and
  the README Citation follows Runtime layout.
- Run the full project-page test module, JavaScript syntax check, and diff
  whitespace check.
- Render at 1440×1000, 1024×900, and 390×844; verify no horizontal overflow,
  readable metric cards, correct animation proportions, and an intelligible
  attention matrix.


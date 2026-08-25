# Stream-WAM latency layout fix

## Goal

Replace the compressed four-panel latency figure with two readable, correctly proportioned figures and make the LIBERO ablations visibly part of the Stream-WAM method family.

## Figure structure

- Generate two static PNG files: one for Chunk Time and one for Episode Time.
- Each image uses a wide `2400 × 900` canvas and is displayed at its intrinsic aspect ratio.
- Each image contains three adjacent benchmark panels in this order: LIBERO, RoboTwin 2.0, and RoboCasa.
- Each benchmark panel has its own continuous y-axis so values with different scales remain legible without broken axes.
- LIBERO receives more horizontal space because it contains six methods; RoboTwin and RoboCasa each contain three.
- Stream-WAM uses the existing teal highlight. Its two LIBERO ablations use related light colors and hatching.
- Exact values remain printed above the bars. Lower-is-better language remains visible.

## Method naming

Use the following names consistently in the LIBERO success table, latency figures, and accessible latency-data table:

- `Stream-WAM`
- `Stream-WAM w/o Action Conditioning`
- `Stream-WAM w/o Slot Encoder`

The names make it explicit that both `w/o` rows are Stream-WAM variants rather than independent methods.

## Page layout

- Stack the Chunk Time and Episode Time figures vertically below the benchmark result tables.
- Preserve the source image ratio with `width: 100%` and `height: auto`.
- Use a horizontally scrollable figure viewport on narrow screens so the chart is not compressed into illegibility.
- Each image links to its full-resolution asset.
- Keep a screen-reader-only exact-data table adjacent to the figures.

## Verification

- Tests assert the two image sources, intrinsic dimensions, ordering, alternative text, and updated ablation names.
- Generator tests assert both PNGs are non-empty and have the expected `2400 × 900` dimensions.
- Existing benchmark values must remain unchanged.
- Desktop and mobile checks verify preserved aspect ratio, readable layout, and no page-level horizontal overflow.

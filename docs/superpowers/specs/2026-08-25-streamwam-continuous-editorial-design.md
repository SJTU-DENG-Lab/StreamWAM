# StreamWAM Continuous Editorial Design

## Goal

Make the project page read as one continuous research article instead of a stack of visibly titled sections.

## Visual structure

- Keep the masthead and its headline results unchanged.
- Remove the visible `Research notes · August 2026` label.
- Remove all numbered act, evidence, discussion, resources, and benchmark labels.
- Remove the visible Evidence, Discussion, and Resources display headings.
- Render the three act-opening sentences at normal body scale; they may retain modest weight or color but must not resemble section headings.
- Keep `LIBERO`, `RoboCasa`, and `RoboTwin 2.0` as necessary table identifiers, using compact typography close to body size.
- Preserve the existing prose order, method figure, resource links, and benchmark tables.

## Semantics and accessibility

- Sections without visible headings receive stable accessible names through `aria-label`.
- Benchmark sections remain named by their compact benchmark headings.
- Table captions and horizontal-scroll behavior remain unchanged.

## Regression constraints

- All benchmark values, units, protocols, and source order remain unchanged.
- The page remains usable without JavaScript and responsive at desktop and mobile widths.
- Automated tests verify that removed labels/headings do not return and that the three benchmark names remain compact.

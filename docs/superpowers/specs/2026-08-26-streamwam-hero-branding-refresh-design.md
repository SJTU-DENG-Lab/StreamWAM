# Stream-WAM Hero Branding Refresh Design

## Goal

Refresh the project page masthead and hero copy so the page opens with clear
DENG Lab ownership and a concise Stream-WAM identity instead of an editorial
section label.

## Scope

This change is limited to the site header and the opening hero copy in
`academic_project_page/`. The hero image, project-resource buttons, headline
results, article body, benchmark tables, figures, and footer remain unchanged.

## Header

- Replace the current Stream-WAM wordmark at the upper left with a local DENG
  Lab logo and a visible `DENG Lab ↗` link.
- Link the logo and label to `https://sjtu-deng-lab.github.io/home/` and open
  the destination in a new tab with the existing external-link protections.
- Do not add an MLSys Team label or link.
- Preserve the Code, Models, and Paper items on the right and preserve the
  existing mobile navigation behavior.

The DENG Lab image will be stored under `academic_project_page/assets/` so the
published page remains self-contained.

## Hero

- Remove `00 · Abstract · Research preview` from the hero.
- Add a compact green status dot followed by the eyebrow label `Stream-WAM`.
- Keep `Stream-WAM:` as the large project name.
- Replace the current tagline with:

  `Streaming Your World-Action Model for Real-Time Robot Manipulation.`

- Replace the two small introductory paragraphs with one larger lead paragraph:

  > **Stream-WAM** introduces **action-conditioned streaming** for world-action
  > models. It overlaps world-action inference with robot execution and feeds
  > the **committed action prefix** back into future-video generation, so the
  > model imagines what comes next with knowledge of the motion already
  > underway. The robot keeps acting while its next world-action chunk is
  > prepared.

The three marked concepts will use semantic `<strong>` elements. The paragraph
will be visually larger than the current summary and detail copy while keeping
the existing responsive width and readable line height.

## Accessibility and Responsive Behavior

- The DENG Lab logo will have an informative accessible name on its link; any
  purely decorative status dot will be hidden from assistive technology.
- Keyboard focus styles and external-link safety attributes will be retained.
- At narrow widths, the DENG Lab lockup must fit beside the menu control, and
  the hero typography must continue to scale without horizontal overflow.

## Verification

- Update the project-page tests to assert the DENG Lab link, new hero title,
  new lead copy, and absence of the old editorial label.
- Run `pytest -q tests/test_academic_project_page.py`.
- Run `node --check academic_project_page/script.js`.
- Preview the page locally at desktop and mobile viewport widths and check the
  opening layout for overflow and unintended navigation changes.

# Stream-WAM Results Legend and End Matter Design

## Goal

Clarify the meanings of CD and the two Stream-WAM ablations, then replace the detached green Resources section with a stronger Citation block integrated directly beneath the concluding discussion.

## Results Legend

Replace the second Task performance paragraph with:

> We compare Stream-WAM with general purpose robot policies and World Action Model baselines in task performance and inference efficiency. CD refers to one-step consistency distillation. We also conduct ablation studies on Stream-WAM by removing action conditioning or the slot encoder to evaluate the contribution of each component. The tables report results on LIBERO, RoboTwin 2.0, and RoboCasa; best and second best results are shown in bold and underlined, respectively.

This paragraph must define the method labels without turning into a benchmark-by-benchmark list. Do not change table names, values, or emphasis.

In the RoboCasa introduction, keep only the 24-task protocol, 50 trials per task, and average-success description. Remove the sentence beginning `Published policy results from` and do not replace it with another source-provenance sentence.

## End Matter Structure

- Delete the detached green Resources section and its green background treatment entirely.
- Place a semantic Citation subsection immediately after the final Discussion paragraph, inside the same concluding section.
- Keep `id="resources"` on the integrated subsection so existing internal references and project-page structure remain stable.
- Remove the Open source prose sentence.
- Place the exact existing Stream-WAM BibTeX citation inside the Citation subsection.
- Add a compact `Copy` control at the upper-right corner of the citation card.
- Add two open-source buttons directly below the citation card:
  - `Code ↗` linking to `https://github.com/SJTU-DENG-Lab/StreamWAM`
  - `Models ↗` linking to `https://huggingface.co/SJTU-DENG-Lab/StreamWAM`
- Both external buttons open in a new tab with `rel="noopener noreferrer"`.

## Copy Interaction

- The copy button is a real `type="button"` control with the accessible label `Copy BibTeX citation`.
- The BibTeX code element has a stable identifier used by the copy handler.
- Clicking the button copies the citation text exactly as displayed, without HTML indentation or additional whitespace.
- After a successful copy, the visible label changes from `Copy` to `Copied` and then returns to `Copy` after a short delay.
- Use the Clipboard API when available. If it is unavailable, fall back to a temporary textarea and `document.execCommand("copy")` so the control also works in local previews.
- If both copy mechanisms fail, keep the citation visible and change the button label to `Select text`; do not hide or replace the citation.
- Add the handler to the existing `docs/script.js`; do not add a separate script or external dependency.

## Visual Design

- Use a large `Citation` heading consistent with the page's main editorial section headings.
- Present the BibTeX in a dark, high-contrast, rounded code card inspired by the FastWAM project page.
- Keep the card within the existing reading column and retain horizontal scrolling on narrow screens.
- Give the card restrained depth through a subtle shadow rather than a new colored section background.
- Position the copy button inside the card at its upper-right corner without covering the BibTeX text.
- Render Code and Models as compact outlined action buttons below the card, aligned with the page's existing button language.
- Keep the spacing from the final conclusion to Citation visibly connected: no full article-section gap and no intervening border.

## Responsive Behavior

- The citation card remains full width within the reading column.
- Long BibTeX lines scroll horizontally rather than forcing page overflow.
- Reserve enough top and right padding for the copy button at desktop and mobile widths.
- The buttons wrap on narrow screens.

## Verification

- Update tests to require the exact CD, Action Conditioning, and Slot Encoder explanations.
- Require Citation to be nested inside the Discussion section and immediately follow its concluding copy.
- Require the exact BibTeX citation and the two button links.
- Require the copy button semantics and test successful Clipboard API behavior plus the local-preview fallback.
- Assert the detached `article-section resources reading-column` markup and green `.resources` background rule are absent.
- Assert the citation card uses dark styling, rounded corners, and horizontal overflow.
- Run the full academic project-page test module and inspect the rendered ending at desktop and mobile widths.

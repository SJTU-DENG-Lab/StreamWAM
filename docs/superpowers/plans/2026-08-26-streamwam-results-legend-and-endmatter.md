# Stream-WAM Results Legend and End Matter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clarify the results legend and integrate a copyable FastWAM-inspired Citation card with Code and Models buttons directly beneath the concluding discussion.

**Architecture:** Preserve the static HTML/CSS/JavaScript page architecture. Move the existing citation markup into a nested end-matter section under Discussion, add a dependency-free clipboard handler to the existing script, and verify both document structure and copy behavior with the existing pytest suite plus a Node harness.

**Tech Stack:** Static HTML, CSS, browser JavaScript, Python `pytest`, Node.js `vm` test harness, agent-browser visual verification.

## Global Constraints

- Delete the detached green Resources section and its `.resources` background treatment.
- Citation follows the final Discussion paragraph without a full section gap or border.
- The citation code must remain exactly unchanged.
- The citation card is dark, rounded, horizontally scrollable, and includes a top-right Copy button.
- Copy uses the Clipboard API first, a temporary textarea fallback second, and `Select text` failure feedback last.
- Code and Models are the only open-source buttons below the card.
- Do not change benchmark table data, labels, or emphasis.
- Bump matching `styles.css` and `script.js` release query versions together.
- Commit and push the finished change to `origin/main`.

---

### Task 1: Results legend and integrated copyable end matter

**Files:**
- Create: `tests/citation_copy_harness.js`
- Modify: `tests/test_academic_project_page.py`
- Modify: `docs/index.html`
- Modify: `docs/styles.css`
- Modify: `docs/script.js`

**Interfaces:**
- Consumes: `#discussion`, the exact `#citation-bibtex` text, existing `.button` styling, and the current deferred `docs/script.js` load.
- Produces: `#resources` nested under Discussion, `.citation-card`, `.citation-copy`, `.resource-actions`, and a click handler that copies the complete citation.

- [ ] **Step 1: Write failing structure, copy, and styling tests**

Update the page tests to require the exact legend sentences, the Citation subsection inside `#discussion`, absence of the detached Resources section and green `.resources` rule, exact Code and Models buttons, a `type="button"` copy control with `aria-label="Copy BibTeX citation"`, and dark-card CSS containing `overflow-x: auto`, `border-radius`, and a dark background.

Add `tests/citation_copy_harness.js` that loads the real `docs/script.js` with Node's `vm`, supplies a minimal document/navigator fixture, triggers the registered citation-copy click handler, and exits nonzero unless:

```javascript
await clickHandler();
assert.equal(copiedText, expectedCitation);
assert.equal(copyButton.textContent, "Copied");
```

Run the harness a second time without `navigator.clipboard.writeText`, require the temporary textarea to receive the exact citation, and require `document.execCommand("copy")` to be called.

- [ ] **Step 2: Run focused tests to verify they fail**

Run:

```bash
pytest -q tests/test_academic_project_page.py -k 'resources or research_story or release_versions'
node tests/citation_copy_harness.js
```

Expected: pytest fails on the old detached green Resources markup and missing copy control; Node fails because `docs/script.js` does not register a citation-copy handler.

- [ ] **Step 3: Implement the legend, end matter, copy behavior, and styles**

Replace the results paragraph with the exact approved legend. Move `#resources` beneath the final Discussion paragraph, use an `h2` Citation heading, wrap the exact code in a positioned dark `.citation-card`, add the Copy button, and place Code and Models button links in `.resource-actions` below it.

In `docs/script.js`, register the click handler, copy `citation.textContent.trim()`, use `navigator.clipboard.writeText` when available, otherwise copy through a temporary readonly textarea, and update the label to `Copied`, `Select text`, then back to `Copy` after the appropriate delay.

In `docs/styles.css`, remove `.resources` background/link rules and add connected end-matter spacing, large heading, dark card, top-right copy button, readable light monospace text, and wrapping resource actions. Bump both asset query strings from `v=20260826-13` to `v=20260826-14`.

- [ ] **Step 4: Run focused and full tests**

Run:

```bash
node tests/citation_copy_harness.js
pytest -q tests/test_academic_project_page.py
git diff --check
```

Expected: all commands pass with clean output.

- [ ] **Step 5: Verify the rendered ending at desktop and mobile widths**

Serve `docs/` locally, inspect the Discussion/Citation ending at 1280×900 and 390×844 with agent-browser, click Copy, and verify the visible label changes to `Copied`. Confirm there is no green detached region or excessive section gap and that the citation card does not overflow the page at mobile width.

- [ ] **Step 6: Review, commit, and push**

Review the diff against `docs/superpowers/specs/2026-08-26-streamwam-results-legend-and-endmatter-design.md`, run the full test module once more, request a read-only code review, address Critical and Important issues, then commit:

```bash
git add docs/index.html docs/styles.css docs/script.js tests/test_academic_project_page.py tests/citation_copy_harness.js
git commit -m "feat: integrate copyable citation end matter"
git push origin main
```

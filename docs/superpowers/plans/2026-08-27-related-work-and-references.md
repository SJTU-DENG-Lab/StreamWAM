# Related Work and References Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the first two editorial sections with concise, source-grounded copy, align the page on `shared actions` terminology, and add a numbered References section above Citation.

**Architecture:** Preserve the existing static-page section structure and method figure. Update copy and semantic assertions in place, then add a sibling References section with small dedicated styles and structural regression coverage.

**Tech Stack:** Static HTML, CSS, Python standard-library `HTMLParser`, pytest.

## Global Constraints

- Use `shared actions`, `shared action slots`, `unknown action slots`, and `condition slots` for Stream-WAM.
- Do not use `committed action prefix` or `shared prefix` in visible project-page copy.
- Describe FDM as adding a future-state prediction before standard video and action prediction; do not claim that its architecture prevents transfer to FastWAM-Joint.
- Present Prefix-Conditioned Methods as a family and Training-Time RTC as its representative implementation.
- Keep the Streaming overview figure, task results, latency results, Citation BibTeX, and open-source actions unchanged.
- Place References immediately before Citation without introducing a colored background region.

---

### Task 1: Concise Related Work and Shared-Actions Terminology

**Files:**
- Modify: `tests/test_academic_project_page.py`
- Modify: `docs/index.html:47`
- Modify: `docs/index.html:158-203`

**Interfaces:**
- Consumes: Existing `#act-wam`, `#act-async`, and `#act-streamwam` editorial sections plus hero copy.
- Produces: Two concise source-linked editorial sections and project-page copy consistent with the Streaming overview.

- [ ] **Step 1: Update semantic assertions before page copy**

Update the page-story test to require the new headings and the core method statements:

```python
assert "World Action Models and the Real-Time Gap" in visible_text
assert "Asynchronous Strategies and the Missing World Coupling" in visible_text
assert "Prefix-conditioned methods" in visible_text
assert "Training-Time RTC" in visible_text
assert "FDM grounding" in visible_text
assert "additional future-state prediction" in visible_text
assert "shared actions" in visible_text
assert "committed action prefix" not in visible_text
assert "shared prefix" not in visible_text
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest -q tests/test_academic_project_page.py -k 'research_story'`

Expected: FAIL because the current page has the former headings and terminology.

- [ ] **Step 3: Replace the two editorial sections and align adjacent copy**

Use the exact approved Section 1 and Section 2 copy from `docs/superpowers/specs/2026-08-27-related-work-and-references-design.md`. Add inline reference links with stable IDs:

```html
<a class="reference-cite" href="#ref-fast-wam" aria-label="Reference 1">[1]</a>
```

Replace remaining visible `committed action prefix` occurrences in the hero and third editorial section with `shared actions`. Preserve the method figure markup and all experiment content.

- [ ] **Step 4: Run the focused semantic tests**

Run: `pytest -q tests/test_academic_project_page.py -k 'research_story or abstract or hero'`

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add docs/index.html tests/test_academic_project_page.py
git commit -m "docs: tighten real-time WAM discussion"
```

### Task 2: Numbered References Before Citation

**Files:**
- Modify: `tests/test_academic_project_page.py`
- Modify: `docs/index.html:320-342`
- Modify: `docs/styles.css:221-228`

**Interfaces:**
- Consumes: Inline links `#ref-fast-wam`, `#ref-wam-real-time`, `#ref-rtc`, `#ref-training-time-rtc`, and `#ref-lingbot-va` from Task 1.
- Produces: A keyboard-accessible numbered bibliography immediately before the unchanged Citation section.

- [ ] **Step 1: Write the structural References assertions**

Add a test that parses the page and asserts:

```python
references_start = html.index('id="references"')
citation_start = html.index('id="resources"')
assert references_start < citation_start
for reference_id in (
    "ref-fast-wam",
    "ref-wam-real-time",
    "ref-rtc",
    "ref-training-time-rtc",
    "ref-lingbot-va",
):
    assert f'id="{reference_id}"' in html
    assert f'href="#{reference_id}"' in html
```

Also assert the five official arXiv URLs and the absence of a background declaration in `.reference-section`.

- [ ] **Step 2: Run the new test and verify RED**

Run: `pytest -q tests/test_academic_project_page.py -k 'references_precede_citation'`

Expected: FAIL because `#references` does not yet exist.

- [ ] **Step 3: Add References HTML and restrained CSS**

Insert a sibling section above `#resources`:

```html
<section class="reference-section" id="references" aria-labelledby="references-title">
  <h2 id="references-title">References</h2>
  <ol class="reference-list">
    <li id="ref-fast-wam">T. Yuan, Z. Dong, Y. Liu, and H. Zhao, “Fast-WAM: Do World Action Models Need Test-Time Future Imagination?” <em>arXiv preprint arXiv:2603.16666</em>, 2026. <a href="https://arxiv.org/abs/2603.16666">arXiv</a>.</li>
    <li id="ref-wam-real-time">Motubrain Team, “World Action Models in Real Time: An Empirical Study of Smooth Execution via Asynchronous Deployment,” <em>arXiv preprint arXiv:2608.01880</em>, 2026. <a href="https://arxiv.org/abs/2608.01880">arXiv</a>.</li>
    <li id="ref-rtc">K. Black, M. Y. Galliker, and S. Levine, “Real-Time Execution of Action Chunking Flow Policies,” <em>arXiv preprint arXiv:2506.07339</em>, 2025. <a href="https://arxiv.org/abs/2506.07339">arXiv</a>.</li>
    <li id="ref-training-time-rtc">K. Black, A. Z. Ren, M. Equi, and S. Levine, “Training-Time Action Conditioning for Efficient Real-Time Chunking,” <em>arXiv preprint arXiv:2512.05964</em>, 2025. <a href="https://arxiv.org/abs/2512.05964">arXiv</a>.</li>
    <li id="ref-lingbot-va">Q. Zhang et al., “Native Video-Action Pretraining for Generalizable Robot Control,” <em>arXiv preprint arXiv:2607.08639</em>, 2026. <a href="https://arxiv.org/abs/2607.08639">arXiv</a>.</li>
  </ol>
</section>
```

Add only spacing, typography, hanging indentation, and link styles. Do not set a section background or alter `.citation-card`.

- [ ] **Step 4: Run complete verification**

Run: `pytest -q tests/test_academic_project_page.py`

Expected: all tests pass.

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 5: Commit Task 2**

```bash
git add docs/index.html docs/styles.css tests/test_academic_project_page.py
git commit -m "docs: add project references"
```

### Task 3: Visual and Source Review

**Files:**
- Verify: `docs/index.html`
- Verify: `docs/styles.css`

**Interfaces:**
- Consumes: Completed Tasks 1 and 2.
- Produces: Evidence that References, Citation, and revised editorial copy remain readable on desktop and mobile.

- [ ] **Step 1: Serve the static page locally**

Run from `docs/`: `python -m http.server 8000 --bind 127.0.0.1`

Expected: the project page is available at `http://127.0.0.1:8000/`.

- [ ] **Step 2: Inspect desktop and mobile layouts**

At approximately 1440 px and 390 px viewport widths, verify that inline citations wrap with their sentences, reference entries use readable hanging indentation, References precedes Citation, and no new background block separates the ending from Discussion.

- [ ] **Step 3: Review source claims against primary papers**

Verify RTC and Prefix-Conditioned Methods against arXiv:2608.01880 Sections 2 and 4, and FDM grounding against arXiv:2607.08639 Section 2.3.7 and Figure 6. Confirm no unsupported architectural-transfer claim remains.

- [ ] **Step 4: Request independent review and push**

Request a read-only reviewer for the completed commit range. Resolve every Critical or Important finding, rerun the page test module, and push `main` only after the reviewer reports readiness.

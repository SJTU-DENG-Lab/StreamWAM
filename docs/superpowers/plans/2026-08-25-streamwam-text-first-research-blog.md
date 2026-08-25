# StreamWAM Text-First Research Blog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the current StreamWAM masthead while replacing the visual project-page body with a linear, text-first research blog.

**Architecture:** Keep the dependency-free static site and its existing deployment workflow. `index.html` becomes a semantic article whose full scientific narrative and all benchmark tables are visible without JavaScript; `styles.css` supplies a restrained reading system; `script.js` handles only the progressively enhanced mobile menu.

**Tech Stack:** Semantic HTML5, responsive CSS, vanilla JavaScript, Python `pytest`, GitHub Pages.

## Global Constraints

- Preserve all current benchmark values, protocols, model lineage, public links, and release status.
- Public method naming is always `StreamWAM`; never publish internal method names or an `Ours` label.
- Keep Paper and the final rollout film as non-interactive `Coming Soon` statuses.
- Add no framework, build step, CDN, analytics, backend, remote font, or new media source.
- Do not modify training, inference, evaluation, checkpoint, or repository-root README code.
- Prose must be useful draft content without invented authors, venue, citation, results, limitations, failure cases, or ablations.

---

### Task 1: Lock the Text-First Article Contract

**Files:**
- Modify: `tests/test_academic_project_page.py`
- Test: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: existing `PageParser`, `parse_page()`, and static page files.
- Produces: assertions defining linear section order, paragraph density, visible result tables, and removed visual-project UI.

- [ ] **Step 1: Add a failing linear-article test**

Add a test that collects section IDs in document order and verifies the article contains `motivation`, `overlap`, `method`, `execution`, `testbed`, `experiments`, and `discussion`. Count non-empty paragraph elements inside the article and require at least 24.

```python
def test_page_is_a_linear_text_first_research_article() -> None:
    parser, html = parse_page()
    article_ids = [
        attrs["id"]
        for tag, attrs in parser.attributes
        if tag == "section" and attrs.get("id") in ARTICLE_SECTION_IDS
    ]
    assert article_ids == list(ARTICLE_SECTION_IDS)
    assert html.count("<p") >= 24
    assert "chapter-index" not in html
    assert "future-slots" not in html
```

- [ ] **Step 2: Add a failing always-visible-results test**

Require three benchmark sections with ordinary headings and tables, and prohibit tab-only attributes.

```python
def test_all_benchmark_tables_are_visible_without_tabs() -> None:
    parser, html = parse_page()
    assert html.count("<table") == 3
    assert "data-tabs" not in html
    assert "data-panel" not in html
    assert "role=\"tab\"" not in html
```

- [ ] **Step 3: Run the focused tests and observe failure**

Run: `pytest -q tests/test_academic_project_page.py`

Expected: the new article-order and always-visible-table tests fail against the current card-and-tab page.

- [ ] **Step 4: Commit the failing contract tests with the implementation only after they pass**

Do not make a red-test-only commit on public `main`; keep the tests unstaged until Task 3 completes.

---

### Task 2: Rewrite the Page as a Research Narrative

**Files:**
- Modify: `academic_project_page/index.html`
- Test: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: existing rollout images, exact result tables, links, protocols, and naming constraints.
- Produces: a semantic `article.longform` with seven ordered content sections and three directly visible benchmark tables.

- [ ] **Step 1: Preserve and simplify the masthead**

Keep the title, subtitle, concise abstract, links, one rollout figure, and three headline result values. Remove duplicate controls or cards below the masthead.

- [ ] **Step 2: Write the problem-driven article introduction**

Create complete multi-paragraph sections for `motivation` and `overlap`. Explain sequential waiting, streaming overlap, and the mismatch created when generated futures do not account for actions already underway.

- [ ] **Step 3: Write the method and execution chapters**

Create multi-paragraph `method` and `execution` sections. Retain one three-stage action-conditioned method figure and one compact chronological execution timeline. Remove the strategy-card grid, paired feature cards, sticky chapter rail, and future-artifact placeholder cards.

- [ ] **Step 4: Write the unified-testbed chapter**

Explain sequential, overlap-only, and action-conditioned streaming in prose. Use one compact semantic comparison table only if it materially improves clarity.

- [ ] **Step 5: Make all experiments part of the document flow**

Place LIBERO, RoboCasa, and RoboTwin as three visible subsections under `experiments`. Precede each unchanged table with its protocol paragraph and follow it with one interpretation derived directly from existing values. Preserve all current values exactly.

- [ ] **Step 6: Finish with discussion and resources**

Write a prose conclusion covering action conditioning, joint success/latency evaluation, and the provisional report scope. Follow it with compact inline Code, Models, Paper, rollout-film, lineage, and acknowledgements content; retain no resource-card grid.

---

### Task 3: Build the Reading-Focused Visual System

**Files:**
- Modify: `academic_project_page/styles.css`
- Modify: `academic_project_page/script.js`
- Modify: `academic_project_page/README.md`
- Test: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: the semantic class names and IDs created by Task 2.
- Produces: a warm-white article layout, responsive tables, mobile navigation, and editing guidance.

- [ ] **Step 1: Replace card-layout CSS with article typography**

Use a 760–820 px reading column, 1.75–1.85 body line height, clear serif headings, restrained teal/orange accents, and generous paragraph spacing. Allow figures and tables to break out to the existing 1180–1220 px canvas.

- [ ] **Step 2: Limit visual components**

Style only the masthead image, one method figure, one execution timeline, three result tables, and an optional rollout strip. Remove unused chapter-rail, tab, resource-card, future-slot, and large strategy-card selectors.

- [ ] **Step 3: Simplify progressive JavaScript**

Keep the mobile menu behavior. Remove result-tab activation and chapter-rail tracking because all article content is now visible and linear.

- [ ] **Step 4: Update page editing documentation**

Document the new section order and state that scientific prose must remain visible without JavaScript; note that benchmark tables are intentionally not tabbed.

- [ ] **Step 5: Run all page checks**

Run: `pytest -q tests/test_academic_project_page.py`

Expected: all tests pass.

Run: `node --check academic_project_page/script.js`

Expected: exit code 0.

Run: `git diff --check`

Expected: exit code 0.

- [ ] **Step 6: Commit the implementation**

```bash
git add academic_project_page tests/test_academic_project_page.py
git commit -m "feat: make project page a text-first research blog"
```

---

### Task 4: Browser QA, Review, and Deployment

**Files:**
- Modify if required: `academic_project_page/index.html`
- Modify if required: `academic_project_page/styles.css`
- Modify: `academic_project_page/assets/streamwam-social-preview.jpg`

**Interfaces:**
- Consumes: the complete static research article from Tasks 2–3.
- Produces: verified desktop/mobile presentation and a matching 1200×630 social preview.

- [ ] **Step 1: Serve and inspect the page at three widths**

Test at 1440 px desktop, 1100 px intermediate, and 390 px mobile. Confirm paragraph measure, section rhythm, visible tables, horizontal table scrolling, header menu, focus states, and absence of horizontal page overflow.

- [ ] **Step 2: Verify the no-JavaScript reading path**

Confirm the header links and every benchmark table remain present in source order when `script.js` is unavailable.

- [ ] **Step 3: Regenerate the social preview**

Capture the light masthead at exactly 1200×630 and replace `academic_project_page/assets/streamwam-social-preview.jpg`. Confirm dimensions with `file`.

- [ ] **Step 4: Request independent review**

Ask the reviewer to check scientific restraint, prose-first hierarchy, exact values, accessibility, responsive behavior, and dirty-worktree scope. Fix every Critical or Important issue.

- [ ] **Step 5: Re-run verification and commit any QA fixes**

Run the focused pytest suite, JavaScript syntax check, and `git diff --check`; commit only the project-page files and focused tests.

- [ ] **Step 6: Push and verify GitHub Pages**

Push `main`, monitor the `Deploy StreamWAM project page` workflow through `completed/success`, then open `https://sjtu-deng-lab.github.io/StreamWAM/` and verify the long-form section headings, three tables, warm-white theme, and lack of browser errors.

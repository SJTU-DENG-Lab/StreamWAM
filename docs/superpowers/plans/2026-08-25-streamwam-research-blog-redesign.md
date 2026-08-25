# StreamWAM Research Blog Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dark StreamWAM project showcase with a bright, long-form research blog that can grow into the full paper narrative while remaining accurate and useful as a preview.

**Architecture:** Preserve the dependency-free static site and deployment boundary. Semantic HTML provides the complete article and no-JavaScript navigation, CSS supplies a warm editorial reading system with breakout figures, and vanilla JavaScript progressively enhances chapter tracking, compact navigation, and benchmark tabs.

**Tech Stack:** HTML5, CSS3, vanilla JavaScript, Python standard library, pytest, agent-browser, GitHub Actions/Pages

## Global Constraints

- Public method naming is always `StreamWAM`; `RTC-AC`, `AC-StreamWAM`, and `Ours` must not appear.
- Benchmark values, protocols, model lineage, public URLs, and release states remain unchanged.
- No `TODO`, `TBD`, Lorem Ipsum, invented authors, citation, venue, experimental result, limitation, or unpublished comparison appears publicly.
- The page uses a warm-white academic editorial design; there is no full-width dark section or dark-mode variant.
- Existing real rollout frames remain the only qualitative visual evidence; no fake video control is shown.
- The page remains readable and navigable without JavaScript and honors `prefers-reduced-motion`.
- No frontend dependency, build step, CDN, remote font, analytics, backend, or CMS is introduced.
- Only `academic_project_page/`, its focused tests, the plan, and deployment workflow may be staged from the dirty worktree.

---

### Task 1: Research-blog content contract

**Files:**
- Modify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: the approved redesign spec and the existing `PageParser`
- Produces: assertions for the seven stable chapter anchors, light theme metadata, draft-content policy, existing result values, and local-reference integrity

- [ ] **Step 1: Add failing narrative and light-theme tests**

Add a test equivalent to:

```python
def test_page_is_a_bright_long_form_research_story() -> None:
    parser, html = parse_page()
    assert {
        "motivation", "testbed", "method", "execution",
        "experiments", "discussion", "resources",
    } <= parser.ids
    assert "#f7f5ef" in (PAGE_ROOT / "styles.css").read_text(encoding="utf-8").lower()
    assert "Why Streaming WAM?" in " ".join(parser.text_parts)
    assert "A Unified Streaming Testbed" in " ".join(parser.text_parts)
    assert "Streaming While Acting" in " ".join(parser.text_parts)
    assert not any(token in html for token in ("TODO", "TBD", "Lorem ipsum"))
```

Extend no-JavaScript navigation assertions so the static chapter index contains anchor links to all seven chapter IDs.

- [ ] **Step 2: Run the focused tests and verify the expected failure**

Run: `pytest -q tests/test_academic_project_page.py`

Expected: the new test fails because the current page lacks `motivation`, `testbed`, `execution`, `experiments`, and `discussion` anchors and the warm-white theme token.

### Task 2: Semantic long-form article

**Files:**
- Modify: `academic_project_page/index.html`

**Interfaces:**
- Consumes: existing public copy, results, links, rollout assets, and Task 1 chapter IDs
- Produces: semantic article sections `motivation`, `testbed`, `method`, `execution`, `experiments`, `discussion`, and `resources`; `.chapter-index`; `.article-prose`; `.breakout`; and existing `data-tabs` hooks

- [ ] **Step 1: Rewrite the masthead and article skeleton**

Replace the cinematic hero with an editorial masthead containing the project title, subtitle, framework summary, action-conditioned summary, Code/Models/Paper controls, a wide rollout figure and caption, and three compact headline metrics. Add the static chapter index immediately after the masthead.

- [ ] **Step 2: Write concise preview chapters**

Implement complete preview prose for the motivation, unified testbed, action-conditioned method, streaming execution, experiments, discussion, and resources chapters. Each chapter starts with a numbered eyebrow, heading, and one-sentence conclusion. Add no final-paper claims beyond the approved spec.

- [ ] **Step 3: Add semantic figures and evidence**

Add an accessible sequential-versus-streaming timeline, a strategy comparison matrix, the three-stage action-conditioned figure, the asynchronous execution timeline, the existing result tabs/tables, and the rollout gallery. Use HTML/CSS shapes and text labels rather than new synthetic images.

- [ ] **Step 4: Run content and reference tests**

Run: `pytest -q tests/test_academic_project_page.py`

Expected: chapter/content assertions pass; light-theme assertions remain red until Task 3.

### Task 3: Bright editorial design and chapter tracking

**Files:**
- Modify: `academic_project_page/styles.css`
- Modify: `academic_project_page/script.js`
- Modify: `academic_project_page/README.md`

**Interfaces:**
- Consumes: Task 2 semantic classes and chapter IDs
- Produces: warm-white visual tokens, responsive prose/breakout layout, `.chapter-link.is-current` state, no-JavaScript chapter navigation, and documented editing boundaries

- [ ] **Step 1: Replace the dark token system**

Define `--paper: #f7f5ef`, `--paper-bright: #fffdf8`, charcoal text tokens, teal method accents, muted orange observation accents, cool-gray rules, and light card surfaces. Update `meta[name="theme-color"]` to match the light page.

- [ ] **Step 2: Implement the editorial layout**

Style the masthead, reading column, wide figures, chapter rail, numbered headings, key-idea note, comparison/timeline diagrams, result tables, rollout figures, discussion notes, and resources. Brighten images by removing dark saturation filters. Keep contrast at WCAG AA for normal text and preserve visible keyboard focus.

- [ ] **Step 3: Implement responsive behavior**

At desktop widths, place the sticky chapter rail beside the reading column. At tablet/mobile widths, turn it into a horizontal scrollable index, stack figures, preserve table scrolling, and retain the existing accessible compact menu.

- [ ] **Step 4: Add progressive chapter tracking**

Use `IntersectionObserver` only to update `.chapter-link.is-current`; never hide article content. If the observer is unavailable or reduced motion is preferred, the full document remains visible and all anchors work normally.

- [ ] **Step 5: Update the maintainer guide**

Document the seven chapter anchors, prose and breakout conventions, safe locations for future paper text/figures/videos, and the existing result-update and deployment commands.

- [ ] **Step 6: Run the complete static checks**

Run: `pytest -q tests/test_academic_project_page.py && node --check academic_project_page/script.js && git diff --check -- academic_project_page tests/test_academic_project_page.py`

Expected: every command exits successfully.

### Task 4: Social preview, browser QA, review, and deployment

**Files:**
- Modify: `academic_project_page/assets/streamwam-social-preview.jpg`
- Modify: `.github/workflows/pages.yml` only if deployment verification identifies a workflow defect

**Interfaces:**
- Consumes: the completed light page and existing Pages deployment
- Produces: a matching 1200×630 social card, verified desktop/mobile pages, reviewed commits, and a successful live deployment

- [ ] **Step 1: Capture a light social-preview asset**

Serve the page locally, set a 1200×630 browser viewport, capture the redesigned masthead, and replace the existing Open Graph image with the resulting local JPEG. Do not synthesize new robot imagery.

- [ ] **Step 2: Run desktop and mobile browser QA**

At approximately 1440×900 and 390×844, verify the masthead, chapter rail/index, method figures, result tabs, table overflow, gallery, mobile menu, and resource links. Check browser errors and capture full-page screenshots for visual inspection.

- [ ] **Step 3: Verify no-JavaScript behavior**

Block `script.js` in a browser session and confirm all chapter links are visible, all result panels remain readable, and no essential content depends on JavaScript.

- [ ] **Step 4: Run final automated verification**

Run: `pytest -q tests/test_academic_project_page.py && node --check academic_project_page/script.js && git diff --check -- academic_project_page tests/test_academic_project_page.py`

Expected: all tests and checks pass.

- [ ] **Step 5: Request independent review**

Request read-only review for scientific-copy accuracy, public naming, accessibility, responsive behavior, social metadata, and exact deployment scope. Resolve all Critical and Important findings.

- [ ] **Step 6: Commit, push, and verify production**

Stage only the plan, `academic_project_page/`, `tests/test_academic_project_page.py`, and any necessary Pages workflow change. Commit with `feat: redesign project page as research blog`, push `main`, monitor `Deploy StreamWAM project page` to success, then open `https://sjtu-deng-lab.github.io/StreamWAM/` and repeat the title, asset, desktop, mobile, tab, and browser-error checks.

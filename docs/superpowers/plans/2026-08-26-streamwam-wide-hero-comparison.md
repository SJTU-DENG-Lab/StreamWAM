# Stream-WAM Wide Hero Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a wider Stream-WAM hero with exact title accents, benchmark speedup cards, a two-track runtime animation, and an action-conditioned attention matrix while migrating the project page to `docs/`.

**Architecture:** Keep the project page dependency-free and repository-native: semantic HTML supplies the comparison and matrix content, CSS owns layout and animation, and the existing minimal JavaScript remains limited to navigation behavior. Move the complete static-site bundle into `docs/`, update its deploy workflow and tests atomically, then validate the public artifact through the same paths GitHub Pages uses.

**Tech Stack:** HTML5, CSS animations and media queries, vanilla JavaScript, pytest/`HTMLParser`, GitHub Actions Pages.

## Global Constraints

- The public URL remains `https://sjtu-deng-lab.github.io/StreamWAM/`.
- The exact title is `Streaming Your World-Action Model for Real-Time Robot Manipulation.`
- Only `Streaming` is teal and only `Real-Time` is violet.
- Headline metrics are exactly `98.20%`, `41.0 ms`, `12.0×`, `4.74 s`, and `3.4×`, with both speedups labeled against FastWAM.
- The animation compares only `Synchronous WAM` and `Stream-WAM` and contains no numeric action/timing annotations.
- The generation blocks are visual and must not contain the labels `Inference` or `Video & Action Chunk`.
- The attention diagram uses repository-native HTML/CSS; no FastWAM raster asset is copied.
- Reduced-motion mode presents a complete static state.

---

### Task 1: Migrate the Pages artifact and README Citation

**Files:**
- Move: `academic_project_page/*` to `docs/*`
- Modify: `.github/workflows/pages.yml`
- Modify: `README.md`
- Modify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: the existing static site rooted at `academic_project_page/` and the Pages artifact workflow.
- Produces: `PAGE_ROOT = REPO_ROOT / "docs"`, a deploy artifact at `./docs`, and a Citation section immediately after Runtime layout.

- [ ] **Step 1: Write the failing migration and README-order tests**

```python
PAGE_ROOT = REPO_ROOT / "docs"

def test_project_page_uses_standard_docs_directory() -> None:
    assert PAGE_ROOT.joinpath("index.html").is_file()
    assert not REPO_ROOT.joinpath("academic_project_page").exists()
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert '"docs/**"' in workflow
    assert "path: ./docs" in workflow

def test_readme_citation_immediately_follows_runtime_layout() -> None:
    readme = REPO_ROOT.joinpath("README.md").read_text(encoding="utf-8")
    runtime = readme.index("## Runtime layout")
    citation = readme.index("## Citation")
    license_heading = readme.index("## License")
    assert runtime < citation < license_heading
    assert readme[runtime:citation].rstrip().endswith("```")
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest -q tests/test_academic_project_page.py -k 'standard_docs_directory or citation_immediately'`

Expected: FAIL because `docs/index.html` does not exist, `academic_project_page/` still exists, and Citation remains after Acknowledgements.

- [ ] **Step 3: Move the static site and update all path consumers**

```bash
git mv academic_project_page/.nojekyll docs/.nojekyll
git mv academic_project_page/index.html docs/index.html
git mv academic_project_page/styles.css docs/styles.css
git mv academic_project_page/script.js docs/script.js
git mv academic_project_page/generate_latency_figure.py docs/generate_latency_figure.py
git mv academic_project_page/assets docs/assets
git mv academic_project_page/README.md docs/PROJECT_PAGE.md
```

Update `.github/workflows/pages.yml` to watch `docs/**` and upload `./docs`. Replace test constants and documentation commands from `academic_project_page` to `docs`. Move the existing complete Citation block to immediately after the Runtime layout code fence and before License.

- [ ] **Step 4: Run migration tests and verify GREEN**

Run: `pytest -q tests/test_academic_project_page.py -k 'standard_docs_directory or citation_immediately or pages_workflow'`

Expected: PASS.

- [ ] **Step 5: Commit the migration**

```bash
git add README.md .github/workflows/pages.yml docs tests/test_academic_project_page.py
git commit -m "refactor: move project page to docs"
```

---

### Task 2: Rebuild the wide hero and metric strip

**Files:**
- Modify: `docs/index.html`
- Modify: `docs/styles.css`
- Modify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: existing `.hero`, `.hero-copy`, `.hero-actions`, and `.headline-results` elements.
- Produces: exact accent classes `.title-accent-streaming` and `.title-accent-realtime`, metric cards nested below `.hero-actions`, and a shell near 1480 px.

- [ ] **Step 1: Write failing hero structure tests**

```python
def test_wide_hero_accents_only_streaming_and_real_time() -> None:
    _, html = parse_page()
    css = PAGE_ROOT.joinpath("styles.css").read_text(encoding="utf-8")
    assert '<span class="title-accent-streaming">Streaming</span>' in html
    assert '<span class="title-accent-realtime">Real-Time</span>' in html
    assert "title-accent-model" not in html
    assert "title-accent-control" not in html
    assert "--shell: min(1480px" in css

def test_headline_metrics_follow_actions_and_use_libero_speedups() -> None:
    _, html = parse_page()
    actions_end = html.index("</div>", html.index('class="hero-actions"'))
    metrics_start = html.index('class="headline-results"')
    figure_start = html.index('class="hero-figure')
    assert actions_end < metrics_start < figure_start
    for text in ("98.20%", "41.0 ms", "12.0×", "4.74 s", "3.4×"):
        assert text in html[metrics_start:figure_start]
    assert "RoboCasa total time" not in html[metrics_start:figure_start]
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest -q tests/test_academic_project_page.py -k 'wide_hero or headline_metrics_follow'`

Expected: FAIL because the old title accents, shell width, metric position, and RoboCasa metric remain.

- [ ] **Step 3: Implement the title, layout, and metrics**

Use this title structure:

```html
<h1 id="hero-title"><span class="title-accent-streaming">Streaming</span> Your World-Action Model for <span class="title-accent-realtime">Real-Time</span> Robot Manipulation.</h1>
```

Place `.headline-results` directly after `.hero-actions` inside `.hero-copy`. Each latency card contains a value, label, and comparison line such as `<small><b>12.0× faster</b> vs FastWAM</small>`. Change `--shell` to `min(1480px, calc(100vw - 48px))`, use a wider hero grid, and keep responsive one-column fallbacks.

- [ ] **Step 4: Run hero tests and verify GREEN**

Run: `pytest -q tests/test_academic_project_page.py -k 'wide_hero or headline_metrics_follow or masthead_and_hero'`

Expected: PASS.

- [ ] **Step 5: Commit the hero structure**

```bash
git add docs/index.html docs/styles.css tests/test_academic_project_page.py
git commit -m "feat: widen Stream-WAM hero metrics"
```

---

### Task 3: Replace the comparison animation and add the attention matrix

**Files:**
- Modify: `docs/index.html`
- Modify: `docs/styles.css`
- Modify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: the current `.pipeline-visual` hero figure and reduced-motion media query.
- Produces: `.runtime-comparison` with two `.runtime-row` children, CSS-revealed `.generation-window` blocks, a `.attention-matrix` with visual/action cell classes, and full accessible descriptions.

- [ ] **Step 1: Write failing comparison and matrix tests**

```python
def test_runtime_visual_has_two_tracks_and_no_old_flow_copy() -> None:
    _, html = parse_page()
    visual = html[html.index('class="pipeline-visual"'):html.index('<figcaption id="pipeline-caption"')]
    assert visual.count('class="runtime-row') == 2
    assert "Synchronous WAM" in visual and "Stream-WAM" in visual
    for forbidden in ("Naive Async", ">Inference<", "Video & Action Chunk", "Current Observation", "Directed feedback", "Visual Future + Next Action Chunk"):
        assert forbidden not in visual
    assert 'class="generation-window' in visual
    assert not re.search(r"\b\d+\s*(?:actions?|ms|seconds?)\b", visual, re.I)

def test_attention_matrix_uses_visual_and_action_tokens() -> None:
    _, html = parse_page()
    css = PAGE_ROOT.joinpath("styles.css").read_text(encoding="utf-8")
    assert 'class="attention-matrix"' in html
    assert html.count('class="matrix-cell visual-token') >= 4
    assert html.count('class="matrix-cell action-token') >= 4
    assert "Action-conditioned attention" in html
    assert "@keyframes generation-reveal" in css
    reduced = css.split("@media (prefers-reduced-motion: reduce)", 1)[1]
    assert ".generation-window" in reduced
    assert "animation: none" in reduced
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest -q tests/test_academic_project_page.py -k 'runtime_visual_has_two or attention_matrix'`

Expected: FAIL because Naive Async and the text-flow panel remain and no matrix/reveal animation exists.

- [ ] **Step 3: Implement the two-track runtime visual**

Create two semantic rows. Each row has an accessible label and a decorative timeline containing a violet `.generation-window`, teal `.execution-rail`, time cursor, and optional `.action-prefix-segment`. Implement `@keyframes generation-reveal` with `transform: scaleX(0)` to `scaleX(1)` and `transform-origin: left`, using shorter width/duration values for `.runtime-stream` than `.runtime-sync`.

- [ ] **Step 4: Implement the action-conditioned attention matrix**

Use a compact grid whose blue cells represent visual tokens and yellow cells represent action tokens. A comparison pair labels standard separated attention and Stream-WAM directed attention; the latter highlights the action-to-future-video quadrant. Add a visible two-item legend and a hidden paragraph that explains the directed connection without relying on color.

- [ ] **Step 5: Add responsive and reduced-motion behavior**

At 1040 px, place the runtime comparison and matrix beneath the copy. At 760 px, stack both diagrams, keep every cell at least 20 px, and remove nonessential captions. In reduced-motion mode set `.generation-window`, `.timeline-cursor`, and glow elements to `animation: none` and render their final state.

- [ ] **Step 6: Run all page tests and syntax checks**

Run: `pytest -q tests/test_academic_project_page.py && node --check docs/script.js && git diff --check`

Expected: all checks PASS with no warnings.

- [ ] **Step 7: Render responsive screenshots**

Open the local site at 1440×1000, 1024×900, and 390×844. Assert `document.documentElement.scrollWidth <= innerWidth` at every size and visually verify the generation reveal, continuous Stream-WAM execution rail, metric placement, and matrix readability.

- [ ] **Step 8: Commit, review, and deploy**

```bash
git add docs/index.html docs/styles.css tests/test_academic_project_page.py
git commit -m "feat: compare Stream-WAM runtime and attention"
git push origin main
```

Request a read-only review across the implementation commits before pushing. After push, wait for the Pages workflow to succeed and verify the live HTML contains `attention-matrix`, `4.74 s`, and `3.4×`.

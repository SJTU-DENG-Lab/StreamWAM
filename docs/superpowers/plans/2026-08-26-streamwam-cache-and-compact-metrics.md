# Stream-WAM Cache and Compact Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent stale project-page CSS after deployment, compress the three LIBERO metrics to project-button scale, and widen only the Hero while preserving lower-page widths.

**Architecture:** Version the existing local CSS and JavaScript URLs at the HTML boundary so a deployed HTML revision always requests matching assets. Keep the current semantic metric markup and diagrams, changing only CSS layout tokens and metric presentation. Separate `--hero-shell` from the shared `--shell` so the Hero can grow without changing article or chart widths.

**Tech Stack:** HTML5, CSS, vanilla JavaScript, pytest/`HTMLParser`, GitHub Pages.

## Global Constraints

- Preserve all metric values and speedup copy exactly.
- Preserve the two-track runtime and attention-matrix HTML.
- Metric items should be approximately the height and weight of project buttons, not result cards.
- Hero maximum width is approximately 1680 px; shared shell returns to 1220 px; reading column remains 800 px.
- Both local CSS and JavaScript URLs carry the same explicit asset version.
- No horizontal overflow at 1920, 1440, 1024, or 390 px.

---

### Task 1: Version project-page assets

**Files:**
- Modify: `docs/index.html`
- Modify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: `styles.css` and `script.js` local references.
- Produces: matching nonempty `?v=` query strings that change the browser cache key.

- [ ] **Step 1: Write the failing cache-key test**

```python
def test_local_css_and_script_use_matching_release_versions() -> None:
    parser, _ = parse_page()
    local = [
        attrs.get("href") or attrs.get("src")
        for tag, attrs in parser.attributes
        if tag in {"link", "script"}
        and (attrs.get("href", "").startswith("styles.css") or attrs.get("src", "").startswith("script.js"))
    ]
    assert len(local) == 2
    versions = [urlparse(value).query for value in local]
    assert versions[0] == versions[1]
    assert re.fullmatch(r"v=\d{8}-\d+", versions[0])
```

- [ ] **Step 2: Run the test and verify RED**

Run: `pytest -q tests/test_academic_project_page.py::test_local_css_and_script_use_matching_release_versions`

Expected: FAIL because both URLs have empty query strings.

- [ ] **Step 3: Add the matching release query**

```html
<link rel="stylesheet" href="styles.css?v=20260826-3">
<script src="script.js?v=20260826-3" defer></script>
```

- [ ] **Step 4: Run the test and verify GREEN**

Run: `pytest -q tests/test_academic_project_page.py::test_local_css_and_script_use_matching_release_versions`

Expected: PASS.

---

### Task 2: Make metrics project-button scale

**Files:**
- Modify: `docs/styles.css`
- Modify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: `.headline-results` and its `p`, `strong`, `span`, `small`, and `b` descendants.
- Produces: an unboxed compact rail with a maximum item height close to the 44 px button minimum.

- [ ] **Step 1: Write the failing compact-metric test**

```python
def test_headline_metrics_use_compact_button_scale() -> None:
    css = (PAGE_ROOT / "styles.css").read_text(encoding="utf-8")
    container = re.search(r"\.headline-results\s*\{([^}]*)\}", css)
    item = re.search(r"\.headline-results p\s*\{([^}]*)\}", css)
    assert container and item
    assert "box-shadow: none" in container.group(1)
    assert "background: transparent" in container.group(1)
    assert "min-height: 54px" in item.group(1)
    assert "max-height: 68px" in item.group(1)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `pytest -q tests/test_academic_project_page.py::test_headline_metrics_use_compact_button_scale`

Expected: FAIL because the current rail has a paper background, shadow, and 108 px items.

- [ ] **Step 3: Implement the compact rail**

Set the container background to transparent, remove shadow and rounded card treatment, keep only subtle dividers, and set each item to `min-height: 54px; max-height: 68px`. Use sans-serif numbers around `1.05rem`, inline labels around `.56rem`, and speedups as small violet badges rather than separate display lines. At 390 px keep three columns with abbreviated visual weight and wrapping limited to each item.

- [ ] **Step 4: Run metric and page tests**

Run: `pytest -q tests/test_academic_project_page.py -k 'headline_metrics or benchmark_results'`

Expected: PASS.

---

### Task 3: Separate Hero width from content width

**Files:**
- Modify: `docs/styles.css`
- Modify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: root `--shell`, `.shell`, `.breakout`, and `.hero.shell`.
- Produces: `--hero-shell: min(1680px, calc(100vw - 32px))`, restored shared shell at 1220 px, and a larger desktop title ceiling.

- [ ] **Step 1: Write the failing Hero-width test**

```python
def test_hero_width_is_independent_from_article_breakouts() -> None:
    css = (PAGE_ROOT / "styles.css").read_text(encoding="utf-8")
    assert "--shell: min(1220px" in css
    assert "--hero-shell: min(1680px" in css
    hero_shell = re.search(r"\.hero\.shell\s*\{([^}]*)\}", css)
    assert hero_shell is not None
    assert "width: var(--hero-shell)" in hero_shell.group(1)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `pytest -q tests/test_academic_project_page.py::test_hero_width_is_independent_from_article_breakouts`

Expected: FAIL because only the shared 1480 px shell exists.

- [ ] **Step 3: Implement Hero-only width**

Add `--hero-shell`, restore `--shell`, override `.hero.shell`, and raise the desktop title clamp ceiling from 66 px to 72 px while keeping the existing 1040 px and 760 px one-column breakpoints.

- [ ] **Step 4: Run all automated checks**

Run: `pytest -q tests/test_academic_project_page.py && node --check docs/script.js && git diff --check`

Expected: all checks PASS.

- [ ] **Step 5: Run browser verification**

At 1920×1080 and 1440×1000 verify the Hero expands while the article remains 800 px and breakout content remains at most 1220 px. At 1024×900 and 390×844 verify no horizontal overflow, metric items remain compact, computed title colors are teal/violet, and `.pipeline-layout` computes to `grid`.

- [ ] **Step 6: Review and deploy**

Commit implementation changes, request read-only review, push `main`, wait for Pages success, and verify the live HTML requests `styles.css?v=20260826-3` and the live computed metric height is within 54–68 px.

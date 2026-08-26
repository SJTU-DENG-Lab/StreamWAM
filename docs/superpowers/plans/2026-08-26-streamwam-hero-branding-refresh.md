# Stream-WAM Hero Branding Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the project page's editorial-style opening with DENG Lab ownership and the approved Stream-WAM hero message.

**Architecture:** Keep the dependency-free static page structure intact. Update the existing header and hero markup in `index.html`, add narrowly scoped responsive styles in `styles.css`, and store an optimized copy of the official DENG Lab lockup under the page's local assets so GitHub Pages remains self-contained.

**Tech Stack:** Static HTML5, CSS, existing vanilla JavaScript navigation, pytest, ffmpeg

## Global Constraints

- Limit visible changes to the site header and opening hero copy.
- Preserve the hero image, project-resource buttons, headline results, article body, benchmark tables, figures, footer, and mobile navigation behavior.
- The header must link only to DENG Lab; do not add an MLSys Team label or link.
- Use the approved hero tagline exactly: `Streaming Your World-Action Model for Real-Time Robot Manipulation.`
- Keep all published assets local to `academic_project_page/`.

---

### Task 1: Refresh the Header and Hero

**Files:**
- Create: `academic_project_page/assets/deng-lab.webp`
- Modify: `academic_project_page/index.html:23-48`
- Modify: `academic_project_page/styles.css:31-55`
- Test: `tests/test_academic_project_page.py:228-258`
- Test: `tests/test_academic_project_page.py:364-383`

**Interfaces:**
- Consumes: the existing `.site-header`, `.site-nav`, `.hero`, `.hero-copy`, `.eyebrow`, `.hero-name`, and `.hero-tagline` page structure.
- Produces: `.lab-lockup`, `.deng-lab-logo`, `.live-dot`, and `.hero-lede` markup and styles; no JavaScript API changes.

- [ ] **Step 1: Update the page contract with failing assertions**

Change the expected hero title and add explicit branding and copy assertions:

```python
assert parser.hero_title == (
    "Stream-WAM: Streaming Your World-Action Model for Real-Time Robot Manipulation."
)
assert "Streaming Your World-Action Model for Real-Time Robot Manipulation." in visible_text
assert "committed action prefix" in visible_text
assert "00 · Abstract" not in visible_text
assert "Research preview" not in visible_text
assert "MLSys Team" not in visible_text
assert "https://sjtu-deng-lab.github.io/home/" in links
```

Add a local-brand-asset assertion beside the link assertions:

```python
deng_logos = [
    attrs
    for tag, attrs in parser.attributes
    if tag == "img" and "deng-lab-logo" in attrs.get("class", "").split()
]
assert deng_logos == [
    {
        "class": "deng-lab-logo",
        "src": "assets/deng-lab.webp",
        "alt": "",
        "width": "420",
        "height": "155",
    }
]
```

Replace the later positive assertion for `00 · Abstract` with:

```python
assert "00 · Abstract" not in visible_text
```

- [ ] **Step 2: Run the focused test to verify the new contract fails**

Run:

```bash
pytest -q tests/test_academic_project_page.py::test_page_exposes_the_research_preview_and_available_artifacts \
  tests/test_academic_project_page.py::test_page_exposes_a_complete_research_story_without_draft_placeholders
```

Expected: FAIL because the old hero title and `00 · Abstract · Research preview` are still present and no DENG Lab link or logo exists.

- [ ] **Step 3: Create an optimized local DENG Lab lockup**

Use the same official lockup published by the DENG Lab WaveForcing page. Crop its large whitespace margins and convert it to a compact WebP asset:

```bash
asset_tmp_dir=$(mktemp -d)
curl -Ls --max-time 20 \
  https://sjtu-deng-lab.github.io/WaveForcing/static/images/sjtu-deng-lab.png \
  -o "$asset_tmp_dir/sjtu-deng-lab.png"
ffmpeg -y -loglevel error -i "$asset_tmp_dir/sjtu-deng-lab.png" \
  -vf "crop=1900:700:150:320,scale=420:155" \
  academic_project_page/assets/deng-lab.webp
```

Verify the generated dimensions:

```bash
file academic_project_page/assets/deng-lab.webp
```

Expected: WebP image data with dimensions `420 x 155`.

- [ ] **Step 4: Replace the header wordmark and hero copy**

Replace the header's current `.wordmark` anchor with:

```html
<a class="lab-lockup" href="https://sjtu-deng-lab.github.io/home/" target="_blank" rel="noopener noreferrer" aria-label="Visit DENG Lab">
  <img class="deng-lab-logo" src="assets/deng-lab.webp" alt="" width="420" height="155">
  <span>DENG Lab ↗</span>
</a>
```

Replace the hero eyebrow, title, summary, and detail paragraphs with:

```html
<p class="eyebrow"><span class="live-dot" aria-hidden="true"></span>Stream-WAM</p>
<h1 id="hero-title"><span class="hero-name">Stream-WAM:</span><span class="hero-tagline">Streaming Your World-Action Model for Real-Time Robot Manipulation.</span></h1>
<p class="hero-lede"><strong>Stream-WAM</strong> introduces <strong>action-conditioned streaming</strong> for world-action models. It overlaps world-action inference with robot execution and feeds the <strong>committed action prefix</strong> back into future-video generation, so the model imagines what comes next with knowledge of the motion already underway. The robot keeps acting while its next world-action chunk is prepared.</p>
```

Do not change the `.hero-actions`, `.hero-figure`, or `.headline-results` markup.

- [ ] **Step 5: Implement the new lockup, status dot, and larger lead styling**

Add the new header lockup rules and replace the hero summary/detail rules with
styles equivalent to:

```css
.lab-lockup { display: inline-flex; align-items: center; gap: 12px; color: var(--muted); font-size: .7rem; font-weight: 760; letter-spacing: .07em; text-decoration: none; text-transform: uppercase; }
.deng-lab-logo { width: auto; height: 34px; mix-blend-mode: multiply; object-fit: contain; }
.lab-lockup:hover, .lab-lockup:focus-visible { color: var(--teal-deep); }
.eyebrow { display: flex; align-items: center; gap: 9px; margin: 0 0 20px; color: var(--teal-deep); font-size: .7rem; font-weight: 790; letter-spacing: .17em; text-transform: uppercase; }
.live-dot { width: 8px; height: 8px; flex: 0 0 auto; border-radius: 50%; background: var(--teal); box-shadow: 0 0 0 4px var(--teal-soft); }
.hero-lede { max-width: 660px; margin: 30px 0 0; color: var(--muted); font-size: clamp(1.08rem,1.35vw,1.24rem); line-height: 1.68; }
.hero-lede strong { color: var(--ink); font-weight: 760; }
```

Retain the existing `.wordmark` rules because the unchanged footer still uses them. Remove only `.eyebrow span`, `.hero-summary`, `.hero-detail`, and `.hero-detail strong`, which no longer match live hero markup. Under the existing `max-width: 760px` media query, add:

```css
.deng-lab-logo { height: 29px; }
.lab-lockup { gap: 8px; font-size: .64rem; }
```

- [ ] **Step 6: Run the page regression tests and JavaScript syntax check**

Run:

```bash
pytest -q tests/test_academic_project_page.py
node --check academic_project_page/script.js
git diff --check
```

Expected: all pytest tests pass, Node exits successfully without output, and `git diff --check` exits successfully without output.

- [ ] **Step 7: Preview the result at desktop and mobile sizes**

Start the static server:

```bash
python -m http.server 8000 --directory academic_project_page
```

Inspect `http://127.0.0.1:8000/` at a desktop viewport near `1440 x 1000` and a mobile viewport near `390 x 844`. Verify the DENG Lab lockup is readable, the green dot aligns with `Stream-WAM`, the approved title wraps cleanly, the larger lead remains legible, and neither header variant overflows.

- [ ] **Step 8: Commit the completed refresh**

```bash
git add academic_project_page/assets/deng-lab.webp \
  academic_project_page/index.html \
  academic_project_page/styles.css \
  tests/test_academic_project_page.py
git commit -m "feat: refresh Stream-WAM hero branding"
```

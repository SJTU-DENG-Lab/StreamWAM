# StreamWAM Academic Project Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a polished, video-first StreamWAM research preview page with honest coming-soon states, current benchmark results, and automatic GitHub Pages deployment.

**Architecture:** A dependency-free static site lives under `academic_project_page/`, with semantic HTML as the source of truth, CSS for responsive presentation, and a small progressive-enhancement script. A focused pytest file validates public content and local references, while a GitHub Actions workflow deploys only that directory.

**Tech Stack:** HTML5, CSS3, vanilla JavaScript, Python standard library, pytest, GitHub Actions/Pages

## Global Constraints

- The public method name is always `StreamWAM`; do not publish `RTC-AC` or `AC-StreamWAM`.
- Paper and final rollout films are visibly marked `Coming Soon` and have no dead links or fake playback controls.
- Use only local assets and relative URLs; add no frontend packages, CDNs, remote fonts, analytics, cookies, or build step.
- Preserve essential content when JavaScript is unavailable and honor `prefers-reduced-motion`.
- StreamWAM result rows receive no bold text or unique background.
- Deploy only `academic_project_page/`; never upload repository outputs, checkpoints, datasets, or source code as the Pages artifact.
- Do not stage or modify unrelated dirty-worktree files.

---

### Task 1: Public-content contract and generated poster assets

**Files:**
- Create: `tests/test_academic_project_page.py`
- Create: `academic_project_page/assets/libero-drawer.webp`
- Create: `academic_project_page/assets/libero-basket.webp`
- Create: `academic_project_page/assets/libero-stove.webp`
- Create: `academic_project_page/assets/streamwam-social-preview.jpg`

**Interfaces:**
- Consumes: current README benchmark values and existing local LIBERO success videos under `outputs/`
- Produces: `PAGE_ROOT: pathlib.Path`, three rollout poster assets, one social preview asset, and tests that define the static page contract

- [ ] **Step 1: Write failing page-contract tests**

Create tests that parse `academic_project_page/index.html` with `html.parser.HTMLParser`, assert one each of `header`, `main`, and `footer`, assert the title/copy/public URLs/current metric strings, reject `RTC-AC`, `AC-StreamWAM`, `Ours`, and non-functional `href="#"`, verify local `src`/`href` references exist, and parse the Pages workflow to ensure its artifact path is exactly `./academic_project_page`.

- [ ] **Step 2: Run the contract test and verify failure**

Run: `pytest -q tests/test_academic_project_page.py`

Expected: failure because `academic_project_page/index.html` does not exist.

- [ ] **Step 3: Extract and optimize the approved rollout posters**

Use `ffmpeg` to extract representative frames from the three approved successful LIBERO videos and encode WebP images around 1600 px wide. Compose the social preview from those local frames at 1200×630 with readable StreamWAM title text. Do not modify or stage source videos under `outputs/`.

- [ ] **Step 4: Verify asset dimensions and size**

Run `file academic_project_page/assets/*` and `du -h academic_project_page/assets/*`; confirm all four images decode, poster widths are suitable for high-density displays, and total tracked image size remains appropriate for a project page.

### Task 2: Semantic page and robotics-cinema visual system

**Files:**
- Create: `academic_project_page/index.html`
- Create: `academic_project_page/styles.css`

**Interfaces:**
- Consumes: the local assets and content contract from Task 1
- Produces: a complete no-JavaScript-readable single-page site with IDs `overview`, `method`, `results`, `gallery`, and `resources`

- [ ] **Step 1: Implement semantic HTML**

Add SEO/Open Graph metadata, skip link, accessible sticky navigation, hero copy, active Code/Models links, text-only Paper and Rollout Film coming-soon states, three-metric ribbon, action-conditioned method flow, streaming-strategy comparison, full LIBERO/RoboCasa/RoboTwin result tables and protocols, model-lineage note, poster gallery, resource cards, acknowledgements, and footer.

- [ ] **Step 2: Implement responsive CSS**

Define a near-black navy visual system, mint/cyan accents, limited orange status accents, editorial type scale using system fonts, cinematic poster treatments, accessible controls/tables/focus states, desktop and mobile layouts, horizontal table overflow, and reduced-motion overrides. Keep all StreamWAM table rows visually neutral.

- [ ] **Step 3: Run page-contract tests**

Run: `pytest -q tests/test_academic_project_page.py`

Expected: page-content assertions pass; script/workflow assertions may still fail because those files are added in later tasks.

### Task 3: Progressive enhancements and editing guide

**Files:**
- Create: `academic_project_page/script.js`
- Create: `academic_project_page/README.md`

**Interfaces:**
- Consumes: data attributes and IDs defined in `index.html`
- Produces: resilient navigation/result-tab/reveal behavior plus a maintainer guide for preview, editing, and future video replacement

- [ ] **Step 1: Add dependency-free progressive enhancements**

Implement compact-menu open/close behavior, ARIA-correct result tabs, Escape/outside-click menu dismissal, and IntersectionObserver reveal classes. Gate motion through `matchMedia('(prefers-reduced-motion: reduce)')`; leave all result panels visible when JavaScript is absent.

- [ ] **Step 2: Document the editing interface**

Document `python -m http.server`, exact files for copy/styles/assets, the future `assets/videos/<slug>.mp4` convention and `<video controls preload="metadata" poster="…">` replacement markup, result-update locations, and first-time GitHub Pages enablement.

- [ ] **Step 3: Run targeted tests and JavaScript syntax check**

Run: `pytest -q tests/test_academic_project_page.py && node --check academic_project_page/script.js`

Expected: all implemented page assertions pass and Node exits 0.

### Task 4: GitHub Pages deployment and end-to-end verification

**Files:**
- Create: `.github/workflows/pages.yml`
- Modify: `tests/test_academic_project_page.py`

**Interfaces:**
- Consumes: the complete static site directory
- Produces: a Pages artifact containing only `academic_project_page/` and final HTTP/reference checks

- [ ] **Step 1: Add the least-privilege Pages workflow**

Trigger on pushes to `main` affecting the page or workflow plus `workflow_dispatch`. Grant `contents: read`, `pages: write`, and `id-token: write`; configure Pages, upload `./academic_project_page`, and deploy it with concurrency cancellation enabled.

- [ ] **Step 2: Complete workflow and reference tests**

Assert the workflow contains official current Pages action majors, exact artifact directory, and required permissions. Extend the HTML parser test so each local stylesheet, script, image, and internal fragment target resolves.

- [ ] **Step 3: Run structural verification**

Run: `pytest -q tests/test_academic_project_page.py && node --check academic_project_page/script.js && git diff --check -- academic_project_page .github/workflows/pages.yml tests/test_academic_project_page.py`

Expected: all checks pass with no whitespace errors.

- [ ] **Step 4: Run HTTP smoke test**

Serve `academic_project_page/` on a free localhost port, request `/`, `/styles.css`, `/script.js`, and every tracked asset with `curl --fail`, then stop the server. Expected: every request returns successfully.

- [ ] **Step 5: Check staged scope, review, commit, and push**

Stage only `academic_project_page/`, `.github/workflows/pages.yml`, `tests/test_academic_project_page.py`, and this plan. Request independent review, resolve blocking findings, inspect `git diff --cached --name-status`, commit with `feat: add StreamWAM academic project page`, and push `main` to `origin`.

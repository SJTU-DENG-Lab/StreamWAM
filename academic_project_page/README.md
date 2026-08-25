# Stream-WAM academic project page

This directory contains the dependency-free, long-form research preview published with GitHub Pages.
The page is organized as an editable research story rather than a fixed paper template.

## Editing the story

The public article opens with three continuous editorial acts: `act-wam`, `act-async`, and
`act-streamwam`, followed by `experiments`, `discussion`, and `resources`. Keep these IDs stable
because the table of contents and external links use them. These sections intentionally have no
visible numbered labels or display-sized headings; their accessible names live on the section
elements. Only the compact `LIBERO`, `RoboCasa`, and `RoboTwin 2.0` headings identify the result
tables. Put ordinary prose inside `.reading-column`; use `.breakout` only for the method figure or
benchmark tables that need more width.

The asynchronous-strategy discussion links to the relevant primary source and distinguishes action
prefix conditioning from action-conditioned future-video generation. Preserve that distinction when
editing the method narrative or the `w/o Action Conditioning` ablation description.

All three benchmark tables are intentionally visible in source order. Do not hide scientific content
behind tabs or require JavaScript to read the article. JavaScript is limited to the mobile header menu.

The page intentionally uses only local HTML, CSS, JavaScript, and media. Paper and rollout-video
entries should remain non-interactive `Coming Soon` notices until real public URLs are available.

This directory is a dependency-free preview site for Stream-WAM. GitHub Pages publishes this directory as-is.

## Preview locally

From the repository root:

```bash
python -m http.server 8000 --directory academic_project_page
```

Then open `http://localhost:8000`.

## Edit the page

- Update public copy, links, benchmark values, and task captions in `index.html`.
- Update colors, typography, spacing, and responsive layout in `styles.css`.
- Update the progressively enhanced mobile navigation in `script.js`.
- Replace rollout posters in `assets/` while keeping their existing filenames, or update the corresponding `src` and social metadata in `index.html`.

The page intentionally uses only local assets and system fonts, so no package installation or build command is required.

## Add the rollout film later

Put the optimized MP4 file under `assets/videos/` using a descriptive lowercase name, for example `assets/videos/streamwam-rollouts.mp4`. Add it only in an approved linear media position near the article conclusion, or update the masthead rollout status to link to the public film. A local video element should use an existing rollout image as its poster:

```html
<video controls preload="metadata" poster="assets/libero-drawer.webp">
  <source src="assets/videos/libero-drawer.mp4" type="video/mp4">
  Your browser does not support embedded video.
</video>
```

Remove the `Rollout film · Coming Soon` status only after the video or public film URL is available and the local reference test passes. Keep the masthead image for fast loading and social sharing.

## Update results safely

The summary metrics appear near the top of `index.html`; the detailed values live in the three visible benchmark sections (`benchmark-libero`, `benchmark-robocasa`, and `benchmark-robotwin`). Update both locations when a headline metric changes, then run:

```bash
pytest -q tests/test_academic_project_page.py
node --check academic_project_page/script.js
```

## Deploy

The workflow at `.github/workflows/pages.yml` publishes only this directory after changes reach `main`. For the first deployment, a repository administrator may need to open **Settings → Pages** and set **Source** to **GitHub Actions**. The expected project URL is:

`https://sjtu-deng-lab.github.io/StreamWAM/`

No repository outputs, checkpoints, or datasets are included in the Pages artifact.

# StreamWAM academic project page

This directory is a dependency-free preview site for StreamWAM. GitHub Pages publishes this directory as-is.

## Preview locally

From the repository root:

```bash
python -m http.server 8000 --directory academic_project_page
```

Then open `http://localhost:8000`.

## Edit the page

- Update public copy, links, benchmark values, and task captions in `index.html`.
- Update colors, typography, spacing, and responsive layout in `styles.css`.
- Update mobile navigation or result-tab behavior in `script.js`.
- Replace rollout posters in `assets/` while keeping their existing filenames, or update the corresponding `src` and social metadata in `index.html`.

The page intentionally uses only local assets and system fonts, so no package installation or build command is required.

## Add rollout videos later

Put optimized MP4 files under `assets/videos/` using descriptive lowercase names, for example `assets/videos/libero-drawer.mp4`. Replace a gallery poster's `<img>` element with:

```html
<video controls preload="metadata" poster="assets/libero-drawer.webp">
  <source src="assets/videos/libero-drawer.mp4" type="video/mp4">
  Your browser does not support embedded video.
</video>
```

Remove that card's `Coming Soon` badge only after the video file is committed and the local reference test passes. Keep the poster for fast loading and social sharing.

## Update results safely

The summary metrics appear near the top of `index.html`; the detailed values live in the three result panels (`panel-libero`, `panel-robocasa`, and `panel-robotwin`). Update both locations when a headline metric changes, then run:

```bash
pytest -q tests/test_academic_project_page.py
node --check academic_project_page/script.js
```

## Deploy

The workflow at `.github/workflows/pages.yml` publishes only this directory after changes reach `main`. For the first deployment, a repository administrator may need to open **Settings → Pages** and set **Source** to **GitHub Actions**. The expected project URL is:

`https://sjtu-deng-lab.github.io/StreamWAM/`

No repository outputs, checkpoints, or datasets are included in the Pages artifact.

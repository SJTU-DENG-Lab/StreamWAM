# Streaming-WAM academic project page

This directory contains the dependency-free, long-form research preview published with GitHub Pages.
The page is organized as an editable research story rather than a fixed paper template.

## Editing the story

The public article opens with `insights` and three continuous editorial acts: `act-wam`, `act-async`,
and `act-streamingwam`, followed by `experiments`, `discussion`, and `resources`. Keep these IDs stable
because the table of contents and external links use them. These sections intentionally have no
visible numbered labels or display-sized headings; their accessible names live on the section
elements. Only the compact `LIBERO`, `RoboTwin 2.0`, `RoboCasa`, and `Real robot evaluation`
headings identify the result tables. Put ordinary prose inside `.reading-column`; use `.breakout`
only for the method figure or benchmark tables that need more width.

The asynchronous-strategy discussion links to the relevant primary source and distinguishes action
prefix conditioning from action-conditioned future-video generation. Preserve that distinction when
editing the method narrative or the `Streaming-WAM w/o Action Conditioning` ablation description.

All four result tables are intentionally visible in source order. Do not hide scientific content behind
tabs or require JavaScript to read the article. JavaScript progressively enhances the mobile header,
desktop table of contents, citation copying, and lazy-loaded rollout videos.

The page intentionally uses only local HTML, CSS, JavaScript, and media. Four real-robot videos are
published as lazy-loaded local MP4 assets; keep their captions and timing values synchronized with
the real-robot result table.

This directory is a dependency-free preview site for Streaming-WAM. GitHub Pages publishes this directory as-is.

## Preview locally

From the repository root:

```bash
python -m http.server 8000 --directory docs
```

Then open `http://localhost:8000`.

## Edit the page

- Update public copy, links, benchmark values, and task captions in `index.html`.
- Update colors, typography, spacing, and responsive layout in `styles.css`.
- Update the progressively enhanced mobile navigation in `script.js`.
- Update latency values or plotting styles in `generate_latency_figure.py`, then regenerate the `2400 × 900` Chunk Time and Episode Time figures with `python docs/generate_latency_figure.py`.
- Replace rollout posters in `assets/` while keeping their existing filenames, or update the corresponding `src` and social metadata in `index.html`.

The deployed page intentionally uses only local assets and system fonts. Regenerating the committed latency figure requires matplotlib, but viewing and deploying the page require no Python packages or build command.

## Update the real-robot rollouts

Store optimized MP4 files under `assets/real-robot/` and keep their poster images beside them. Each
rollout card uses a `data-src` source so the browser downloads video only after the reader presses Play:

```html
<video class="real-robot-video" controls preload="none" poster="assets/real-robot/example-poster.webp">
  <source data-src="assets/real-robot/example.mp4" type="video/mp4">
  Your browser does not support embedded video.
</video>
```

After changing a rollout, verify playback, poster loading, caption metrics, and the real-robot result
table at desktop and mobile widths.

## Update results safely

The summary metrics appear near the top of `index.html`; detailed values live in `benchmark-libero`,
`benchmark-robotwin`, `benchmark-robocasa`, and `benchmark-real-robot`. Update every corresponding
location when a headline metric changes, then run:

```bash
node --check docs/script.js
python -m http.server 8000 --directory docs
```

Open `http://localhost:8000`, verify the headline and detailed metrics agree, and check that the local images, navigation links, and external project links resolve correctly.

## Deploy

The workflow at `.github/workflows/pages.yml` publishes only this directory after changes reach `main`. For the first deployment, a repository administrator may need to open **Settings → Pages** and set **Source** to **GitHub Actions**. The expected project URL is:

`https://sjtu-deng-lab.github.io/Streaming-WAM/`

No repository outputs, checkpoints, or datasets are included in the Pages artifact.

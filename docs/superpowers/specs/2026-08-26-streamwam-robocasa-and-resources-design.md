# Stream-WAM RoboCasa and Resources Design

## Goal

Correct the RoboCasa protocol and comparison table, mark Stream-WAM as the authors' method in every visible task performance table, and reduce the final resources section to a short open source statement and a citation block.

## RoboCasa Protocol

- The Stream-WAM RoboCasa evaluation follows the standard 24-task RoboCasa protocol, not a 50 target-task protocol.
- Each task uses 50 trials and the table reports average success.
- Correct the protocol in the project page, README, and tests.
- The table header is `Average Success ↑`.
- The table contains only the following recognizable baselines and project methods, in this order:

| Method | Average Success |
|---|---:|
| π₀.₅ | 41.4% |
| π₀-FAST | 61.2% |
| π₀ | 62.5% |
| Cosmos Policy | 67.1% |
| X-WAM | 75.42% |
| X-WAM-CD | **75.83%** |
| Stream-WAM (Ours) | 75.35% |

- The π₀.₅ and π₀-FAST values follow the supplied SOTA2 RoboCasa leaderboard.
- The π₀ and Cosmos Policy values use the 24-task main table in the Cosmos Policy paper; use the official 67.1% Cosmos Policy main result rather than the 66.6% ablation value displayed by SOTA2.
- Keep X-WAM-CD at 75.83% and retain its best-result bold styling. Keep X-WAM at 75.42% and retain its second-best underline.
- Link the benchmark description to the supplied SOTA2 leaderboard so readers can inspect the external reference results.

## Ours Labels and Hardware Wording

- Rename `Stream-WAM` to `Stream-WAM (Ours)` in the three visible task performance tables: LIBERO, RoboTwin 2.0, and RoboCasa.
- Do not rename Stream-WAM in the hidden latency data table, figures, navigation, prose, metadata, or resource names.
- Change `All evaluations use four NVIDIA H100 GPUs` to `Our evaluations use four NVIDIA H100 GPUs`, because the expanded RoboCasa table includes externally published baseline results.

## Resources and Citation

Replace the current resources content with:

1. One concise open source sentence with inline links to GitHub and Hugging Face.
2. A semantic `Citation` heading.
3. A horizontally scrollable BibTeX code block containing exactly:

```bibtex
@misc{denglab2026streamwam,
  title        = {Stream-WAM: Streaming Your World-Action Model for Real-Time Robot Manipulation},
  author       = {{DENG Lab}},
  year         = {2026},
  howpublished = {Project page},
  organization = {Shanghai Jiao Tong University},
  url          = {https://sjtu-deng-lab.github.io/StreamWAM/}
}
```

The open source sentence is:

> **Open source.** Stream-WAM training and inference code and evaluation recipes are available on GitHub, with released checkpoints hosted on Hugging Face.

Remove the Code/Models/Paper/Rollout link strip, Model lineage paragraph, and Acknowledgements paragraph. Remove their unused CSS rules. Do not add a copy button or JavaScript.

## Styling

- Keep the existing resources background and reading-column width.
- Style the Citation heading consistently with the page's compact editorial headings.
- Give the BibTeX block a restrained paper-colored panel, border, rounded corners, readable monospace type, and horizontal overflow for narrow screens.
- Style the two inline resource links consistently with existing editorial links.

## Verification

- Add failing tests for the corrected 24-task RoboCasa protocol, exact method rows, and visible `Stream-WAM (Ours)` labels.
- Add a failing resources test for the exact citation, inline GitHub/Hugging Face links, and removal of the obsolete resource strip, lineage, and acknowledgements.
- Update existing tests that intentionally encode the old 50-task protocol or old RoboCasa rows.
- Run the full academic project page test module.
- Review the diff to ensure hero/title changes from the other agent are not included in this implementation.

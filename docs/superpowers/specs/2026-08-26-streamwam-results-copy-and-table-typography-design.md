# Stream-WAM Results Copy and Table Typography Design

## Goal

Make the project page results section explain the experimental setup and efficiency findings directly, remove redundant commentary beneath the tables, and improve the readability of numeric table cells without changing the reported results.

## Scope

- Replace the three introductory results paragraphs with two concise paragraphs.
- Present LIBERO with FastWAM-Joint as the primary setting, then explain that X-WAM on RoboCasa and StarWAM on RoboTwin 2.0 test the approach across benchmarks and World Action Model families.
- State once that all evaluations use four NVIDIA H100 GPUs.
- Use the capitalized, unhyphenated form `World Action Models` consistently and avoid unnecessary hyphenated compounds.
- Rename the displayed LIBERO success column from `LIBERO-10` to `Long`, including the nearby table description.
- Remove the prose summaries beneath all three success tables.
- Replace the numeric cells' monospace face with the page's inherited sans serif face while retaining tabular numeral alignment.
- Expand the inference efficiency introduction to explain why chunk latency and end to end episode time are both measured, followed by the reported speedups.
- Do not modify the hero or title section.

## Approved Results Introduction

> To evaluate Stream-WAM across different World Action Model families, we begin by further training FastWAM-Joint with our streaming approach on LIBERO. We then extend the same design to X-WAM on RoboCasa and StarWAM on RoboTwin 2.0 to examine its applicability across benchmarks and model families. All evaluations use four NVIDIA H100 GPUs.
>
> We compare Stream-WAM with general purpose robot policies and World Action Model baselines in both task performance and inference efficiency. The performance tables summarize success across Long, Spatial, Goal, and Object in LIBERO, the clean and randomized settings of RoboTwin 2.0, and the RoboCasa target tasks. Best and second best results are shown in bold and underlined, respectively.

## Approved Efficiency Copy

> **Inference efficiency.** Task success alone does not reveal how much inference interrupts robot execution. We therefore measure chunk latency, the time required to prepare the next action chunk, together with end to end episode time, which captures the accumulated cost of inference, execution, and replanning over a complete rollout.
>
> Stream-WAM reduces chunk latency by 12.0× on LIBERO relative to FastWAM, 4.0× on RoboTwin 2.0 relative to StarWAM-Joint, and 3.7× on RoboCasa relative to X-WAM. These latency gains yield end to end speedups of 3.0× and 2.6× on long and short LIBERO tasks, 1.4× on RoboTwin 2.0, and 3.2× on RoboCasa.

## Styling

Numeric table cells will inherit the page's sans serif font stack, use a slightly larger readable size and moderate weight, and continue using `font-variant-numeric: tabular-nums` from the table rule. The now-unused `.benchmark-reading` rule will be removed.

## Verification

- Update the academic project page tests to expect `Long` and the new copy.
- Add coverage ensuring the removed summaries are absent and numeric cells no longer use a monospace font.
- Run the focused project page test module.
- Confirm the implementation diff remains confined to the results section, its table styles, and relevant tests.

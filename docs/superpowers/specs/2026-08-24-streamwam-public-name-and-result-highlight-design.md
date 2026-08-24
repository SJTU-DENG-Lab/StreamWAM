# StreamWAM Public Name and Result Highlight Design

## Goal

Use `StreamWAM` as the only reader-facing method name in the root README,
emphasize action conditioning as the core modeling design, and visually
highlight the StreamWAM result rows without bold text or an `(Ours)` suffix.

## Introduction

Replace the second introductory paragraph with:

> Building on this framework, we introduce **StreamWAM**, an
> **action-conditioned streaming formulation** that feeds the prefix of actions
> currently being executed by the robot back into the world model. This
> explicitly conditions future video generation on ongoing robot actions.
> Rather than treating inference–execution overlap merely as a systems
> optimization, StreamWAM couples the two processes: the executed action prefix
> shapes the predicted visual future, while the model asynchronously infers the
> next world-action chunk as the robot continues executing the current chunk.

The copy makes action conditioning prominent while avoiding a separate
`Action-Conditioned Streaming WAM`, `AC-StreamWAM`, or `AC-Stream` method name.

## Reader-facing naming

Replace reader-facing `AC-StreamWAM` and `AC-Stream` references in `README.md`
with `StreamWAM`, including:

- release-status runtime and checkpoint names;
- Quick Start and launch headings;
- checkpoint prose;
- the LIBERO result row and result analysis;
- runtime-layout descriptions.

Preserve literal executable interfaces containing `ac_stream`, including the
checkpoint filename example, launcher filename, and `--ac-stream-accelerated`
flag. These literals must continue to match the renamed runtime implementation.

## Result-row presentation

In the LIBERO, RoboCasa, and RoboTwin tables:

- use exactly `StreamWAM` as the method label;
- remove `(Ours)`;
- remove all bold markup from the method label and values;
- wrap every cell in the StreamWAM row with `<mark>...</mark>`.

The approved rows are:

```markdown
| <mark>StreamWAM</mark> | <mark>96.60</mark> | <mark>98.80</mark> | <mark>97.40</mark> | <mark>100.00</mark> | <mark>98.20</mark> | <mark>41.0</mark> | <mark>5.36 / 3.15</mark> |
| <mark>StreamWAM</mark> | <mark>75.35</mark> | <mark>136.76</mark> | <mark>11.76</mark> |
| <mark>StreamWAM</mark> | <mark>87.2</mark> | <mark>88.8</mark> | <mark>87.6</mark> | <mark>—</mark> | <mark>112.2</mark> |
```

GitHub removes inline CSS, `style`, and `bgcolor` attributes from rendered
README HTML. `<mark>` is therefore the stable semantic fallback: it supplies a
visible background behind each cell's content while retaining searchable,
copyable Markdown table data. No SVG or image table will be introduced.

## Scope and concurrent changes

- Modify only `README.md` for the implementation.
- Preserve all supplied benchmark values, protocols, units, and table order.
- Do not modify source code, scripts, examples, tests, or the Hugging Face
  model-card directory as part of this task.
- The worktree contains concurrent AC-stream runtime rename changes. Preserve
  them and stage only the intended README result/naming change plus this task's
  specification and plan.

## Validation

- Confirm no reader-facing `AC-StreamWAM`, `AC-Stream`, or `(Ours)` remains in
  `README.md`.
- Confirm literal `ac_stream_checkpoint.pt`,
  `launch_streamwam_libero_ac_stream_4gpu.sh`, and
  `--ac-stream-accelerated` remain unchanged.
- Confirm exactly three result rows begin with `<mark>StreamWAM</mark>` and that
  their numeric values match the approved rows.
- Render a representative row through GitHub's Markdown API and confirm every
  `<mark>` tag survives sanitization while no bold tags are present.
- Run `git diff --check` and the package-identity tests in the validated
  FastWAM environment.

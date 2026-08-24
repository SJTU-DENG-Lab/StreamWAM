# AC-StreamWAM README Positioning Design

**Date:** 2026-08-24

## Scope

Update the opening project description, release-status table, and all reader-facing method references in the root README. Keep `RTC-AC`/`rtc_ac` only where it is part of a literal implementation interface such as a script filename, CLI flag, configuration key, or runtime diagnostic; use **Action-Conditioned Streaming WAM (AC-StreamWAM)** everywhere else in the README.

## Opening Copy

The introduction will contain exactly these two paragraphs, with no following acceleration paragraph:

> StreamWAM is a research framework for **streaming World-Action Models (WAMs)**. It provides a unified testbed for systematically studying and comparing different streaming strategies for WAM-based robot control.
>
> Building on this framework, we introduce **Action-Conditioned Streaming WAM (AC-StreamWAM)**, a streaming formulation that feeds the prefix of actions currently being executed by the robot back into the world model, conditioning future video generation on ongoing actions. Rather than treating inference–execution overlap merely as a systems optimization, AC-StreamWAM couples the two processes: the actions being executed shape the predicted visual future, while the model asynchronously infers the next world-action chunk as the robot continues executing the current action chunk.

## Release Status

The table will contain exactly five rows in this order:

| Asset | Status |
|---|---|
| StreamWAM inference and training code | ✅ Available in this repository |
| Accelerated AC-StreamWAM runtime | ✅ Available in this repository |
| LIBERO and RoboTwin recipes | ✅ Available in this repository |
| AC-StreamWAM checkpoint | ✅ [Available on Hugging Face](https://huggingface.co/SJTU-DENG-Lab/StreamWAM) |
| Technical report | ⏳ Coming soon |

The existing note about legacy ModelScope checkpoint names will be removed because the release-status checkpoint entry now points to the official StreamWAM Hugging Face repository.

The header checkpoint badge will also point to `https://huggingface.co/SJTU-DENG-Lab/StreamWAM` and use a Hugging Face checkpoint label instead of the legacy ModelScope badge.

## README-Wide Naming

Reader-facing headings and prose will use AC-StreamWAM:

- `Quick start: accelerated AC-StreamWAM on LIBERO`;
- `Launch AC-StreamWAM`;
- `AC-StreamWAM checkpoint` and `AC-StreamWAM measurements`;
- `Wan2.2-TI2V-5B AC-StreamWAM` in the results table;
- `AC-StreamWAM geometry`, model wrapper, and runtime in explanatory labels.

Technical implementation tokens remain unchanged so every documented command continues to run:

- `examples/libero/scripts/launch_streamwam_libero_rtc_ac_4gpu.sh`;
- `--rtc-ac-accelerated`;
- placeholder/config identifiers containing `rtc_ac`;
- D0/D8 runtime diagnostics such as `prewarmed_d0` and `prewarmed_d8`.

## Verification and Delivery

- Confirm the README contains the approved method name and Hugging Face link.
- Confirm the old accelerated-introduction paragraph and obsolete release rows are absent.
- Run Markdown/diff checks and the relevant documentation-facing test suite.
- Commit and push the change to `origin/main` after verification.

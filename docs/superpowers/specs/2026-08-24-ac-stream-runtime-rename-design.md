# AC-Stream Runtime Rename Design

## Goal

Replace the active RTC-AC name with AC-Stream consistently across StreamWAM code, configuration, CLI, tests, launchers, metrics, documentation, and the Hugging Face model card while preserving checkpoint compatibility and inference behavior.

## Public contract

- Sampling method: `ac-stream`.
- Acceleration flag: `--ac-stream-accelerated`.
- Framework variant: `ac-stream`.
- Configuration prefix: `ac_stream_`.
- JSON fields: `ac_stream_overlap` and `ac_stream_acceleration`.
- User-facing display name: `AC-Stream`.
- Launcher/config filenames use `ac_stream` because repository filenames use underscores.

The old `rtc_ac`, `--rtc-ac-accelerated`, and RTC-AC public names are not compatibility aliases. They are removed from active interfaces so incorrect old commands fail explicitly.

## Python contract

Python modules and identifiers use snake/camel case:

- `streamwam.inference.ac_stream`
- `streamwam.modules.ac_stream`
- `streamwam.wam.ac_stream_wam`
- `ACStream*` classes
- `ac_stream_*` functions, fields, and variables

## Compatibility boundary

Raw FastWAM checkpoint tensor keys and payload metadata are not rewritten. The loader continues to accept the same checkpoint bytes directly and performs the same strict pre-mutation validation. Joint CD, Euler, and standard FastWAM behavior remain unchanged.

Historical documents under `docs/superpowers/specs/` and `docs/superpowers/plans/` remain historical records. Active README, LIBERO documentation, model card, code, tests, recipes, scripts, runtime logs, and result schemas use AC-Stream.

## Verification

- New public-interface tests must fail before implementation because `ac-stream` and `--ac-stream-accelerated` do not exist yet.
- Focused AC-Stream, checkpoint, timing, and multi-GPU tests must pass.
- Full pytest, shell syntax checks, and `git diff --check` must pass.
- An active-tree scan excluding historical specs/plans and binary/cache/output directories must find no RTC-AC public names.
- The Hugging Face README must match the active interface and retain `ac-stream/` model paths.

# LIBERO Current Results README Design

**Date:** 2026-08-24

## Scope

Replace the root README's single-row latency result with one combined LIBERO success-and-efficiency table. Add the released FastWAM-Joint-CD checkpoint to the release-status table. Do not change code, runtime interfaces, or the accelerated runtime contract below the results section.

## Evaluation Description

The results section will state that every method is evaluated on the four LIBERO suites with 50 trials per task and 10 tasks per suite. Success values are percentages. `Average` is the arithmetic mean across LIBERO-10, LIBERO-Spatial, LIBERO-Goal, and LIBERO-Object. `Chunk Time` is average inference latency per action chunk. `Episode Time` is average wall-clock duration for long- and short-horizon tasks.

## Results Table

| Method | LIBERO-10 | Spatial | Goal | Object | Average (%) ↑ | Chunk Time (ms) ↓ | Episode Time (s) ↓ Long / Short |
|---|---:|---:|---:|---:|---:|---:|---:|
| FastWAM | 96.20 | 96.20 | 94.20 | 96.20 | 95.70 | 493.0 | 16.31 / 8.25 |
| FastWAM-Joint-CD | 97.20 | 99.60 | 98.60 | 100.00 | 98.85 | 114.2 | 6.89 / 3.74 |
| FastWAM-RTC | 58.40 | 76.20 | 77.00 | 83.40 | 73.75 | 142.3 | 6.23 / 3.20 |
| **AC-StreamWAM (Ours)** | 96.60 | 98.80 | 97.40 | 100.00 | 98.20 | 41.0 | 5.36 / 3.15 |
| w/o Action Conditioning | 94.40 | 96.40 | 96.60 | 97.60 | 96.25 | 35.1 | 5.20 / 2.92 |
| w/o Slot Encoder | 95.60 | 98.40 | 96.80 | 99.80 | 97.65 | 36.3 | 5.31 / 3.01 |

## Interpretation

The table will be followed by this concise interpretation:

> AC-StreamWAM achieves a 98.20% average success rate with a chunk latency of 41.0 ms, providing a strong balance between control performance and streaming efficiency. It reduces chunk latency by approximately 12.0× compared with FastWAM and 2.8× compared with FastWAM-Joint-CD. Removing action conditioning decreases the average success rate by 1.95 percentage points, while removing the slot encoder results in a 0.55-point drop.

## Release Status

Insert the following row immediately before the AC-StreamWAM checkpoint row:

| FastWAM-Joint-CD checkpoint | ✅ [Available on Hugging Face](https://huggingface.co/SJTU-DENG-Lab/StreamWAM) |

Both released checkpoint rows use the same official Hugging Face repository.

## Verification

- Confirm every supplied numeric value appears exactly once in the new table.
- Confirm the success-rate and timing definitions are present.
- Confirm both checkpoint rows link to the official Hugging Face repository.
- Confirm the obsolete single-row 45.20 ms result and its D8-specific description are removed from `Current results`.
- Run Markdown/diff checks before commit and push.

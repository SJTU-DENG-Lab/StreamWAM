# FastWAM Runtime Checkpoint Loader Design

> **Architecture update:** The loader behavior in this document is unchanged,
> but its implementation has moved into `streamwam/checkpointing/`. See
> `docs/superpowers/plans/2026-08-21-checkpointing-package-refactor.md`; the
> earlier `streamwam/utils/checkpoint.py` placement described below is superseded.

## Goal

Allow StreamWAM inference entrypoints to load original FastWAM release checkpoints directly at runtime with:

```bash
--checkpoint-format fastwam
```

The loader must not write a converted checkpoint or require a second on-disk copy. Existing StreamWAM checkpoint behavior remains the default.

## Scope

The first implementation supports the released FastWAM MoT checkpoints for LIBERO and RoboTwin:

- `libero_uncond_2cam224.pt`
- `robotwin_uncond_3cam_384.pt`

It also reads their accompanying FastWAM `dataset_stats.json` files in memory. Training/resume checkpoint loading is unchanged.

## Architecture

Checkpoint responsibilities live in the focused `streamwam/checkpointing/`
package:

- `core.py` contains generic training checkpoint, backbone inference, and
  ActionDiT initialization helpers.
- `streamwam_format.py` contains native StreamWAM inference loading.
- `fastwam_format.py` contains FastWAM checkpoint, statistics, and config
  adaptation.
- `loader.py` performs explicit source-format dispatch.
- `__init__.py` is the only public checkpoint API.

The former `streamwam/utils/checkpoint.py` module is deleted without a
compatibility shim.

The package exposes three inference-oriented functions:

```python
load_inference_checkpoint(model, path, checkpoint_format="streamwam") -> dict
load_inference_stats(path, checkpoint_format="streamwam") -> dict
prepare_inference_config(config, checkpoint_format="streamwam") -> config
```

Format dispatch is explicit. Supported values initially are `streamwam` and `fastwam`. Unknown formats fail immediately.

Both LIBERO rollout and the benchmark-neutral `StreamWAMPolicy` call these functions, preventing duplicate format logic.

## Standard StreamWAM Loading

The `streamwam` branch preserves current behavior:

- accept direct files and supported checkpoint directories;
- extract `model_state_dict`, `module`, or `state_dict` payloads;
- strip known `module.`, `model.`, and `_orig_mod.` prefixes;
- retain current non-strict loading behavior for compatibility.

## FastWAM Checkpoint Loading

The FastWAM loader reads the original `.pt` directly with CPU mapping and memory mapping where supported:

```python
torch.load(path, map_location="cpu", weights_only=True, mmap=True)
```

It requires a payload containing:

```text
mot
proprio_encoder
step
torch_dtype
```

The `mot` state is split without copying tensor storage:

```text
mixtures.video.*  -> model.mot.experts["video"]
mixtures.action.* -> model.mot.experts["action"]
```

Prefix removal produces the submodule-local parameter names. `payload["proprio_encoder"]` loads into `model.proprio_encoder`.

Loading targets the expert submodules instead of the whole `MoTWAM`. The video expert is the same module instance as `model.backbone.get_dit()`, and the action expert is the same instance as `model.action_expert`; this avoids alias-related false missing keys.

## Strict Validation

FastWAM loading is fail-fast. Before any parameter is copied, the loader validates:

- the model family is `mot_wam`;
- `model.mot.experts` contains `video` and `action`;
- `model.proprio_encoder` exists;
- all checkpoint `mot` keys begin with `mixtures.video.` or `mixtures.action.`;
- source and target key sets match exactly for each expert;
- every matching tensor has the same shape;
- proprio encoder keys and shapes match exactly;
- no source tensor remains unused.

After validation, all three submodules load with `strict=True`. The loader returns metadata including format, step, dtype, and per-submodule tensor counts.

Any incompatibility raises a descriptive exception. FastWAM loading never continues with warnings or partially initialized weights.

## FastWAM Stats Loading

FastWAM stats are converted only in memory. No output JSON is written.

For each of `action.default` and `state.default`, the canonical StreamWAM view is:

```text
global_min  -> min
global_max  -> max
global_mean -> mean
global_std  -> std
```

This supports LIBERO global min/max normalization and RoboTwin global z-score normalization. Missing groups or fields fail immediately.

The standard `streamwam` branch continues using the existing `load_lerobot_stats()` format.

## Model Construction

The released FastWAM checkpoint contains the complete video expert, action expert, and proprio encoder. When `checkpoint_format=fastwam`, inference entrypoints set `config.framework.action_expert_init_from = None` before building the model so a separate ActionDiT initialization payload is not required.

The Wan2.2 base directory is still required for architecture inference, VAE weights, T5 weights, and tokenizer files. The initial implementation may load the base video DiT before the FastWAM checkpoint overwrites it; skipping that redundant I/O is a later optimization, not part of correctness.

## CLI Changes

The following entrypoints gain `--checkpoint-format`, defaulting to `streamwam`:

- `examples/libero/rollout.py`
- `examples/robotwin/policy_server.py`

`StreamWAMPolicy.__init__` gains `checkpoint_format="streamwam"`.

Launch scripts pass `CHECKPOINT_FORMAT` when set:

- `examples/libero/scripts/launch_streamwam_libero_mot_rollout.sh`
- `examples/robotwin/scripts/launch_streamwam_robotwin_policy_server.sh`

The selected format controls both checkpoint and stats loading.

## LIBERO Runtime Flow

With `--checkpoint-format fastwam`, LIBERO inference performs:

1. Load recipe and overrides.
2. Select the FastWAM format and disable ActionDiT init payload loading.
3. Build the StreamWAM Wan2.2 MoT structure, VAE, and text-conditioning resources.
4. Memory-map the original FastWAM checkpoint.
5. Validate all video, action, and proprio keys and shapes.
6. Strictly load the three submodules.
7. Read FastWAM stats and create the canonical in-memory view.
8. Load or generate task text embeddings.
9. Create LIBERO environments and run the existing action-chunk rollout loop.

## Testing

Tests use small fake MoT modules rather than the 12 GB release checkpoint.

Required coverage:

- standard StreamWAM format behavior remains compatible;
- FastWAM video/action/proprio loading succeeds with exact keys and shapes;
- missing payload groups fail;
- unknown `mixtures.*` branches fail;
- missing, unexpected, and shape-mismatched tensors fail before mutation;
- FastWAM LIBERO stats map global min/max correctly;
- FastWAM RoboTwin stats map global mean/std correctly;
- CLI defaults remain `streamwam`;
- CLI `fastwam` selection reaches the shared loader.

A final integration check should load the real LIBERO release checkpoint on a machine with the Wan2.2 base model and sufficient RAM/VRAM, then verify the reported tensor counts and run one LIBERO task trial.

## Non-Goals

- Converting or saving a second checkpoint.
- Supporting FastWAM training resume.
- Automatically detecting checkpoint formats.
- Silently accepting partial FastWAM weights.
- Guaranteeing bitwise identity with the original FastWAM runtime.

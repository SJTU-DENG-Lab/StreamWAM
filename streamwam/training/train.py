"""Generic training entry point for StreamWAM recipes."""

import argparse
import logging
from typing import Any

import torch

from streamwam.utils.config_cli import apply_overrides


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("streamwam.train")


def _configure_rank_logging():
    import os
    if int(os.environ.get("RANK", "0")) != 0:
        logging.getLogger("streamwam").setLevel(logging.WARNING)
        logging.getLogger("streamwam.train").setLevel(logging.WARNING)


def _maybe_prepare_streamwam_text_cache(cfg: Any) -> None:
    data = getattr(cfg, "data", None)
    if getattr(data, "dataset_type", None) != "lerobot":
        return
    roots = list(getattr(data, "dataset_dirs", None) or ([] if getattr(data, "root", None) is None else [data.root]))
    roots = [str(root) for root in roots if root]
    if roots:
        from streamwam.builder import _ensure_text_caches
        _ensure_text_caches(cfg, roots)


def main():
    _configure_rank_logging()
    parser = argparse.ArgumentParser(description="StreamWAM training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Config overrides as dot.notation=value (e.g. training.batch_size=2)",
    )
    args = parser.parse_args()

    from streamwam import build_framework, build_trainer
    from streamwam.config import config_to_dict, load_config

    config = load_config(args.config)
    if args.override:
        config = apply_overrides(config, args.override)

    taxonomy = getattr(config, "taxonomy", None)
    logger.info(f"Config: framework={config.framework.type}, backbone={config.backbone.type}")
    if taxonomy is not None:
        logger.info(
            f"taxonomy.package={taxonomy.package}, "
            f"model_family={taxonomy.model_family}, "
            f"preset={taxonomy.preset}"
        )
    logger.info(
        f"action_dim={config.framework.action_dim}, "
        f"chunk_size={config.framework.chunk_size}, "
        f"strategy={config.training.strategy}"
    )

    _maybe_prepare_streamwam_text_cache(config)

    logger.info("Building model...")
    if torch.cuda.is_available():
        import os as _os
        local_rank = int(_os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    mp = (config.training.mixed_precision or "no").lower()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mp, torch.float32)
    logger.info(f"Build device={device}, dtype={dtype}")
    model = build_framework(config, device=device, dtype=dtype).to(device)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model built: {total/1e6:.1f}M params total")

    logger.info("Building trainer...")
    trainer = build_trainer(model, config)

    wandb_run = getattr(trainer, "_wandb_run", None) or getattr(trainer, "wandb_run", None)
    if wandb_run is not None:
        try:
            wandb_run.config.update(config_to_dict(config), allow_val_change=True)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to log config to wandb: {e}")

    logger.info("Starting training...")
    trainer.train()


if __name__ == "__main__":
    main()

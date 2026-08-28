"""Streaming-WAM trainer: unified training loop with HuggingFace Accelerate + DeepSpeed."""

import json
import os
import re
import time
import logging
from contextlib import nullcontext
from math import ceil
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from streamingwam.config import TrainingConfig, StreamingWAMConfig
from streamingwam.wam.base import WAMModel

logger = logging.getLogger(__name__)


def _checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match:
        return int(match.group(1))
    meta_path = path / "trainer_state.json"
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            return int(json.load(f)["global_step"])
    raise ValueError(f"Cannot infer global step from checkpoint path: {path}")


class StreamingWAMTrainer:
    """Unified trainer for Streaming-WAM models.

    Features:
    - HuggingFace Accelerate for distributed training (DeepSpeed ZeRO-2)
    - Freezes VAE + text encoder, trains only MoT (video + action experts)
    - Gradient accumulation, mixed precision, gradient clipping
    - Periodic eval, checkpointing, logging
    """

    def __init__(
        self,
        model: WAMModel,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.config = config or TrainingConfig()
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Setup accelerator
        try:
            from accelerate import Accelerator
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                mixed_precision=self.config.mixed_precision,
                log_with="wandb" if self.config.wandb_enabled else None,
                step_scheduler_with_optimizer=False,
            )
        except ImportError:
            # Fallback: no accelerator (single GPU, no mixed precision management)
            self.accelerator = None
            logger.warning("accelerate not installed, running without distributed support")

        if self.accelerator is not None:
            zero_stage = "none"
            deepspeed_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
            if deepspeed_plugin is not None:
                zero_stage = deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown")
            logger.info(
                "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d mixed_precision=%s grad_accum=%d",
                self.accelerator.distributed_type,
                zero_stage,
                self.accelerator.num_processes,
                self.accelerator.mixed_precision,
                self.config.gradient_accumulation_steps,
            )

        self.global_step = 0
        self._wandb_run = None
        self._resume_step: Optional[int] = None
        self._load_model_only_if_requested()
        self._setup()
        self._resume_if_requested()
        self._apply_resume_lr_override_if_requested()
        self._init_wandb()

    def _init_wandb(self):
        """Initialise a wandb run on the main process if enabled. Failures
        are non-fatal (logged as warnings)."""
        if not getattr(self.config, "wandb_enabled", False):
            return
        if not self._is_main_process():
            return
        try:
            import wandb  # type: ignore
        except ImportError:
            logger.warning("wandb_enabled=True but wandb not installed; skipping.")
            return
        try:
            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=getattr(self.config, "wandb_run_name", None),
                config={
                    "learning_rate": self.config.learning_rate,
                    "batch_size": self.config.batch_size,
                    "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                    "max_steps": self.max_steps,
                    "mixed_precision": self.config.mixed_precision,
                    "strategy": self.config.strategy,
                },
                reinit=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"wandb.init failed ({e}); continuing without wandb.")
            self._wandb_run = None

    def _wandb_log(self, payload: dict):
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.log(payload, step=self.global_step)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"wandb.log failed ({e})")

    @staticmethod
    def _train_wandb_payload(loss_dict: dict, lr: float, steps_per_sec: float) -> dict:
        payload = {
            "train/loss_total": loss_dict["loss_total"],
            "train/loss_video": loss_dict["loss_video"],
            "train/loss_action": loss_dict["loss_action"],
            "train/lr": lr,
            "train/steps_per_sec": steps_per_sec,
        }
        optional_keys = (
            "loss_action_eef",
            "loss_action_gripper",
            "action_target_gripper_mean",
            "action_target_gripper_open_rate",
        )
        for key in optional_keys:
            if key in loss_dict:
                payload[f"train/{key}"] = loss_dict[key]
        return payload

    @staticmethod
    def _format_action_monitor(loss_dict: dict) -> str:
        if "loss_action_gripper" not in loss_dict:
            return ""
        return (
            f" | action_eef={loss_dict['loss_action_eef']:.4f}"
            f" | gripper={loss_dict['loss_action_gripper']:.4f}"
            f" | grip_open={loss_dict['action_target_gripper_open_rate']:.2f}"
        )

    def _setup(self):
        """Setup optimizer, scheduler, dataloader, freeze strategy."""
        # Freeze strategy: only train the MoT (contains both video + action experts)
        self._apply_freeze_strategy()

        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Dataloader
        dataloader_kwargs = {}
        if self.config.num_workers > 0:
            dataloader_kwargs.update({
                "timeout": 120,
                "persistent_workers": True,
                "prefetch_factor": 1,
            })

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            **dataloader_kwargs,
        )

        # LR scheduler. When max_steps is not explicit, match FastWAM's
        # optimizer-step horizon: epochs over the dataset at global batch size,
        # then divide by gradient accumulation.
        world_size = self.accelerator.num_processes if self.accelerator is not None else 1
        if self.config.max_steps is not None:
            max_steps = max(int(self.config.max_steps), 1)
        else:
            global_batch_size = max(int(self.config.batch_size) * max(int(world_size), 1), 1)
            micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
            max_steps = max(
                ceil(micro_steps_per_epoch / max(int(self.config.gradient_accumulation_steps), 1))
                * int(self.config.num_epochs),
                1,
            )
        if self.config.warmup_steps is not None:
            warmup_steps = int(self.config.warmup_steps)
        else:
            warmup_steps = int(max_steps * self.config.warmup_ratio)

        sched_type = str(self.config.lr_scheduler_type).strip().lower()
        if sched_type in {"cosine", "cosine_with_min_lr"}:
            min_lr = self.config.learning_rate * 0.01 if sched_type == "cosine" else float(self.config.min_lr)
            warmup_steps = min(max(int(warmup_steps), 0), max_steps - 1)
            remaining_steps = max(max_steps - warmup_steps, 1)
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=min_lr,
            )
            if warmup_steps > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1.0 / max(warmup_steps, 1),
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
                self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[warmup_steps],
                )
            else:
                self.lr_scheduler = main_scheduler
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)

        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.world_size = world_size
        self.effective_global_batch_size = (
            int(self.config.batch_size)
            * max(int(world_size), 1)
            * max(int(self.config.gradient_accumulation_steps), 1)
        )
        logger.info(
            "Training schedule: dataset_size=%d world_size=%d per_device_batch=%d grad_accum=%d "
            "effective_global_batch=%d max_steps=%d warmup_steps=%d scheduler=%s lr=%.2e",
            len(self.train_dataset),
            world_size,
            self.config.batch_size,
            self.config.gradient_accumulation_steps,
            self.effective_global_batch_size,
            self.max_steps,
            self.warmup_steps,
            self.config.lr_scheduler_type,
            self.config.learning_rate,
        )

        # Prepare with accelerator
        if self.accelerator:
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = (
                self.accelerator.prepare(
                    self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
                )
            )

    def _apply_freeze_strategy(self):
        """Apply the configured training strategy.

        Supported strategies:
        - ``full``: train the MoT (or action_expert) end-to-end. Everything
          else (VAE, text encoder, optional shared backbone) is frozen.
        - ``lora``: freeze the entire base model and inject LoRA adapters
          (via ``peft``) into all DiT linear layers matching
          ``config.lora_target_modules``. Action-specific heads are kept
          trainable as full-rank parameters.
        - ``staged``: phase 1 -- train only the small action heads
          (action_encoder/decoder/embedder, state_encoder, action_proj_out)
          for ``config.staged_warmup_steps`` steps; phase 2 -- unfreeze
          MoT / action_expert / backbone DiT and continue full fine-tuning.
          The phase-2 transition is triggered from the training loop via
          :meth:`_maybe_unfreeze_staged`.
        """
        strategy = getattr(self.config, "strategy", "full")
        self._staged_unfrozen = False
        if strategy == "lora":
            self._apply_lora_strategy()
            return
        if strategy == "staged":
            self._apply_staged_phase1()
            return

        # Default: freeze everything, then unfreeze MoT / action_expert.
        self.model.requires_grad_(False)
        if hasattr(self.model, "mot"):
            self.model.mot.requires_grad_(True)
        elif hasattr(self.model, "shared_dit"):
            self.model.shared_dit.requires_grad_(True)
        elif hasattr(self.model, "action_expert"):
            self.model.action_expert.requires_grad_(True)
            model_config = getattr(self.model, "config", None)
            if bool(getattr(model_config, "feature_condition_train_backbone", False)):
                dit = self.model.backbone.get_dit()
                dit.requires_grad_(True)
        else:
            # Shared-DiT WAM variants train the world DiT directly; VAE/text encoder stay frozen.
            if hasattr(self.model, "backbone"):
                dit = self.model.backbone.get_dit()
                dit.requires_grad_(True)
        # Always keep small action heads trainable.
        containers = [self.model]
        if hasattr(self.model, "shared_dit"):
            containers.append(self.model.shared_dit)
        for container in containers:
            for name in (
                "action_encoder", "action_decoder",
                "action_embedder", "action_proj_out",
                "state_encoder", "proprio_encoder",
                "feature_projector", "feature_timestep_embedding",
            ):
                mod = getattr(container, name, None)
                if mod is not None:
                    mod.requires_grad_(True)

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"Strategy=full | params: {total:,} total, {trainable:,} trainable "
            f"({100*trainable/total:.1f}%)"
        )

    def _apply_lora_strategy(self):
        """Inject LoRA adapters into the DiT (and action_expert if present)."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise RuntimeError(
                "training.strategy='lora' requires the `peft` package. "
                "Install with: pip install peft"
            ) from e

        self.model.requires_grad_(False)
        target_modules = list(self.config.lora_target_modules)

        if hasattr(self.model, "mot"):
            wrap_target = self.model.mot
            wrap_attr = "mot"
        elif hasattr(self.model, "shared_dit"):
            wrap_target = self.model.shared_dit
            wrap_attr = "shared_dit"
        elif hasattr(self.model, "backbone"):
            wrap_target = self.model.backbone.get_dit()
            wrap_attr = "_lora_dit"
        else:
            raise RuntimeError(
                "Cannot apply LoRA: model has no `.mot`, `.shared_dit`, or `.backbone`."
            )

        lora_cfg = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        peft_module = get_peft_model(wrap_target, lora_cfg)
        if wrap_attr == "mot":
            self.model.mot = peft_module
        elif wrap_attr == "shared_dit":
            self.model.shared_dit = peft_module
        else:
            # peft wraps the DiT in-place; track it on the WAM model so its
            # adapter params are visible to the optimizer.
            self.model._lora_dit = peft_module

        # Keep small action heads trainable as full params.
        containers = [self.model]
        if hasattr(self.model, "shared_dit"):
            containers.append(self.model.shared_dit)
        for container in containers:
            for name in (
                "action_encoder", "action_decoder",
                "action_embedder", "action_proj_out",
                "state_encoder", "proprio_encoder",
                "feature_projector", "feature_timestep_embedding",
            ):
                mod = getattr(container, name, None)
                if mod is not None:
                    mod.requires_grad_(True)

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"Strategy=lora (r={self.config.lora_r}, alpha={self.config.lora_alpha}) "
            f"| params: {total:,} total, {trainable:,} trainable "
            f"({100*trainable/total:.2f}%)"
        )

    def _apply_staged_phase1(self):
        """Phase 1 of the staged strategy: only action heads are trainable."""
        self.model.requires_grad_(False)
        unfrozen = []
        containers = [("model", self.model)]
        if hasattr(self.model, "shared_dit"):
            containers.append(("shared_dit", self.model.shared_dit))
        for prefix, container in containers:
            for name in (
                "action_encoder", "action_decoder",
                "action_embedder", "action_proj_out",
                "state_encoder", "proprio_encoder",
                "feature_projector", "feature_timestep_embedding",
            ):
                mod = getattr(container, name, None)
                if mod is not None:
                    mod.requires_grad_(True)
                    unfrozen.append(f"{prefix}.{name}")

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        warmup = getattr(self.config, "staged_warmup_steps", 1000)
        logger.info(
            f"Strategy=staged | phase=1/2 (action heads only) | warmup_steps={warmup} "
            f"| unfrozen={unfrozen} | params: {total:,} total, {trainable:,} trainable "
            f"({100*trainable/max(total,1):.2f}%)"
        )

    def _maybe_unfreeze_staged(self):
        """If running under ``staged`` and warmup is complete, unfreeze the
        backbone / MoT / action_expert in-place. Idempotent and safe to call
        every step. Existing optimizer param groups continue holding action-
        head params; newly-unfrozen weights are added as a fresh param group
        so their grads start being applied immediately.
        """
        if getattr(self, "_staged_unfrozen", True):
            return
        if getattr(self.config, "strategy", "full") != "staged":
            return
        warmup = getattr(self.config, "staged_warmup_steps", 1000)
        if self.global_step < warmup:
            return

        added_params = []
        if hasattr(self.model, "mot"):
            for p in self.model.mot.parameters():
                if not p.requires_grad:
                    p.requires_grad_(True)
                    added_params.append(p)
        elif hasattr(self.model, "shared_dit"):
            for p in self.model.shared_dit.parameters():
                if not p.requires_grad:
                    p.requires_grad_(True)
                    added_params.append(p)
        elif hasattr(self.model, "action_expert"):
            for p in self.model.action_expert.parameters():
                if not p.requires_grad:
                    p.requires_grad_(True)
                    added_params.append(p)
        elif hasattr(self.model, "backbone"):
            dit = self.model.backbone.get_dit()
            for p in dit.parameters():
                if not p.requires_grad:
                    p.requires_grad_(True)
                    added_params.append(p)

        if added_params:
            self.optimizer.add_param_group({
                "params": added_params,
                "lr": self.optimizer.param_groups[0]["lr"],
                "weight_decay": self.config.weight_decay,
            })

        self._staged_unfrozen = True
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"Strategy=staged | phase=2/2 unfrozen at step={self.global_step} "
            f"| params: {total:,} total, {trainable:,} trainable "
            f"({100*trainable/max(total,1):.2f}%)"
        )

    def _unwrap(self):
        """Return the underlying model, unwrapping DDP/Accelerate wrappers."""
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(self.model)
        return self.model

    def _model_compute_param(self):
        for param in self.model.parameters():
            if param.requires_grad:
                return param
        return next(self.model.parameters())

    def _load_model_only_if_requested(self):
        resume = getattr(self.config, "resume", None)
        if not resume or not bool(getattr(self.config, "resume_model_only", False)):
            return
        resume_dir = Path(resume)
        if not resume_dir.is_dir():
            raise FileNotFoundError(f"training.resume must point to a checkpoint directory: {resume}")
        if self.accelerator is None:
            model_path = resume_dir / "model.pt"
            if not model_path.is_file():
                raise FileNotFoundError(f"single-GPU checkpoint missing model.pt: {model_path}")
            payload = torch.load(model_path, map_location="cpu", weights_only=False)
            state = payload.get("model_state_dict", payload)
        else:
            model_path = resume_dir / "pytorch_model" / "mp_rank_00_model_states.pt"
            if not model_path.is_file():
                raise FileNotFoundError(f"DeepSpeed checkpoint missing model state: {model_path}")
            payload = torch.load(model_path, map_location="cpu", weights_only=False)
            state = payload.get("module")
            if not isinstance(state, dict):
                raise KeyError(f"No module state dict found in {model_path}")
        result = self.model.load_state_dict(state, strict=False)
        if result.missing_keys:
            logger.warning("Missing model-only resume keys, first 20: %s", result.missing_keys[:20])
        if result.unexpected_keys:
            logger.warning("Unexpected model-only resume keys, first 20: %s", result.unexpected_keys[:20])
        if self._is_main_process():
            logger.info(
                "Loaded model weights from %s; restarting optimizer/scheduler with lr=%.2e",
                resume_dir,
                self.config.learning_rate,
            )

    def _resume_if_requested(self):
        resume = getattr(self.config, "resume", None)
        if not resume or bool(getattr(self.config, "resume_model_only", False)):
            return
        resume_dir = Path(resume)
        if not resume_dir.is_dir():
            raise FileNotFoundError(f"training.resume must point to a checkpoint directory: {resume}")
        if self.accelerator is None:
            model_path = resume_dir / "model.pt"
            if not model_path.is_file():
                raise FileNotFoundError(f"single-GPU checkpoint missing model.pt: {model_path}")
            payload = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(payload["model_state_dict"])
            self.global_step = int(payload.get("step", _checkpoint_step(resume_dir)))
        else:
            self.accelerator.load_state(str(resume_dir))
            self.global_step = int(_checkpoint_step(resume_dir))
        self._resume_step = self.global_step
        if self._is_main_process():
            logger.info(f"Resumed training state from {resume_dir} at step={self.global_step}")

    def _apply_resume_lr_override_if_requested(self):
        resume_lr = getattr(self.config, "resume_lr", None)
        resume_scheduler = getattr(self.config, "resume_lr_scheduler_type", None)
        if resume_lr is None and not resume_scheduler:
            return
        if not getattr(self.config, "resume", None):
            raise ValueError("training.resume_lr/resume_lr_scheduler_type require training.resume")
        if resume_lr is not None:
            lr_value = float(resume_lr)
            for group in self.optimizer.param_groups:
                group["lr"] = lr_value
                group["initial_lr"] = lr_value
        sched_type = str(resume_scheduler or "").strip().lower()
        if sched_type:
            if sched_type != "constant":
                raise ValueError("training.resume_lr_scheduler_type currently supports only 'constant'")
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
        if self.accelerator is not None:
            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)
        if self._is_main_process():
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "Applied resume LR override: lr=%.2e scheduler=%s",
                current_lr,
                sched_type or "restored",
            )

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training: max_steps={self.max_steps}, batch_size={self.config.batch_size}")
        logger.info(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"Effective global batch size: {self.effective_global_batch_size}")

        self.model.train()
        # Use a natural ``yield from`` infinite iterator so that accelerate's
        # DataLoaderShard can complete each epoch normally (including its
        # end-of-epoch broadcast to all ranks).  Catching StopIteration and
        # calling iter() again bypasses that broadcast, causing a NCCL hang
        # at every epoch boundary.
        def _inf_loader(dl):
            while True:
                yield from dl

        data_iter = _inf_loader(self.train_dataloader)
        start_time = time.time()

        stop_step = self._target_stop_step()
        while self.global_step < stop_step:
            batch = next(data_iter)

            # Move to device if no accelerator
            if not self.accelerator:
                model_param = self._model_compute_param()
                device = model_param.device
                model_dtype = model_param.dtype
                new_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        v = v.to(device)
                        if v.dtype.is_floating_point and v.dtype != model_dtype:
                            v = v.to(model_dtype)
                    new_batch[k] = v
                batch = new_batch

            # Forward + backward
            if self.accelerator:
                stepped = False
                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        loss, loss_dict = self._unwrap().training_step(batch)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.global_step += 1
                        self._maybe_unfreeze_staged()
                        stepped = True

                # Logging / save / eval must happen OUTSIDE the
                # ``accelerator.accumulate`` context so that DDP gradient-sync
                # state and the eval-time forward passes do not interleave —
                # otherwise subsequent training steps deadlock on a stale
                # collective op.
                if stepped:
                    if self.global_step % self.config.log_every == 0:
                        elapsed = time.time() - start_time
                        steps_per_sec = max(0, self.global_step - (self._resume_step or 0)) / max(elapsed, 1e-9)
                        lr = self.optimizer.param_groups[0]["lr"]
                        if self._is_main_process():
                            logger.info(
                                f"step={self.global_step} | loss={loss_dict['loss_total']:.4f} "
                                f"| video={loss_dict['loss_video']:.4f} "
                                f"| action={loss_dict['loss_action']:.4f}"
                                f"{self._format_action_monitor(loss_dict)} "
                                f"| lr={lr:.2e} | steps/s={steps_per_sec:.2f}"
                            )
                            self._wandb_log(self._train_wandb_payload(loss_dict, lr, steps_per_sec))

                    # Save
                    if self.global_step % self.config.save_every == 0:
                        self._save_checkpoint()

                    # Eval
                    if (
                        self.val_dataset is not None
                        and self.global_step % self.config.eval_every == 0
                    ):
                        self._run_eval()
            else:
                # Simple single-GPU path
                loss, loss_dict = self._unwrap().training_step(batch)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                self.global_step += 1
                self._maybe_unfreeze_staged()

                if self.global_step % self.config.log_every == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = max(0, self.global_step - (self._resume_step or 0)) / max(elapsed, 1e-9)
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"step={self.global_step} | loss={loss_dict['loss_total']:.4f} "
                        f"| video={loss_dict['loss_video']:.4f} | action={loss_dict['loss_action']:.4f}"
                        f"{self._format_action_monitor(loss_dict)} "
                        f"| lr={lr:.2e} | steps/s={steps_per_sec:.2f}"
                    )
                    self._wandb_log(self._train_wandb_payload(loss_dict, lr, steps_per_sec))

                if self.global_step % self.config.save_every == 0:
                    self._save_checkpoint()

                if (
                    self.val_dataset is not None
                    and self.global_step % self.config.eval_every == 0
                ):
                    self._run_eval()

        if self.global_step >= self.max_steps:
            logger.info(f"Training complete. Total steps: {self.global_step}")
        else:
            logger.info(
                f"Debug stop reached at step={self.global_step} "
                f"(scheduler max_steps={self.max_steps})"
            )

    def _target_stop_step(self) -> int:
        debug_stop = getattr(self.config, "debug_stop_after_steps", None)
        if debug_stop is None:
            return self.max_steps
        if debug_stop <= 0:
            raise ValueError("training.debug_stop_after_steps must be positive when set")
        return min(self.max_steps, (self._resume_step or self.global_step) + int(debug_stop))

    def _is_main_process(self) -> bool:
        """Return True if this is the rank-0 process (or no accelerator)."""
        if self.accelerator is None:
            return True
        return getattr(self.accelerator, "is_main_process", True)

    def _save_checkpoint(self):
        """Save model checkpoint."""
        save_dir = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
        os.makedirs(save_dir, exist_ok=True)

        if self.accelerator:
            # `save_state` internally synchronises across ranks; only rank-0
            # writes the meta files but all ranks must call into it.
            self.accelerator.save_state(save_dir)
        else:
            torch.save(
                {"model_state_dict": self.model.state_dict(), "step": self.global_step},
                os.path.join(save_dir, "model.pt"),
            )
        if self._is_main_process():
            with open(os.path.join(save_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
                json.dump({"global_step": self.global_step}, f, indent=2)
            logger.info(f"Saved checkpoint to {save_dir}")
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Keep only the last ``save_total_limit`` checkpoints (by step)."""
        limit = getattr(self.config, "save_total_limit", None)
        if not limit or limit <= 0:
            return
        try:
            import re
            import shutil
            out_dir = self.config.output_dir
            if not os.path.isdir(out_dir):
                return
            ckpts = []
            for name in os.listdir(out_dir):
                m = re.fullmatch(r"checkpoint-(\d+)", name)
                if m:
                    ckpts.append((int(m.group(1)), os.path.join(out_dir, name)))
            ckpts.sort(key=lambda x: x[0])
            to_delete = ckpts[: max(0, len(ckpts) - int(limit))]
            for _, p in to_delete:
                shutil.rmtree(p, ignore_errors=True)
                logger.info(f"Removed old checkpoint: {p}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"checkpoint cleanup failed: {e}")

    def _run_eval(self):
        """Evaluate the model on a small slice of val_dataset.

        ALL ranks run eval simultaneously (no barriers, no rank-0-only path).
        This is critical: any barrier asymmetry between rank 0 and other ranks
        desynchronises the NCCL sequence counter, causing the next epoch-start
        ``synchronize_rng_states`` broadcast (NumelIn=5056) to hang.
        Only rank 0 logs the results.
        """
        from streamingwam.training.metrics import action_dim_mse, action_mse, video_psnr

        if self.val_dataset is None:
            return

        was_training = self.model.training
        self.model.eval()
        try:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )
            max_samples = max(1, getattr(self.config, "eval_max_samples", 4))
            compute_video = getattr(self.config, "eval_compute_video_psnr", False)
            mse_list: list[float] = []
            mse_per_step: list[torch.Tensor] = []
            mse_per_dim: list[torch.Tensor] = []
            pred_gripper_values: list[torch.Tensor] = []
            target_gripper_values: list[torch.Tensor] = []
            psnr_list: list[float] = []
            n_seen = 0
            model_param = self._model_compute_param()
            device = model_param.device
            model_dtype = model_param.dtype

            for vb in val_loader:
                if n_seen >= max_samples:
                    break
                batch = {}
                for k, v in vb.items():
                    if isinstance(v, torch.Tensor):
                        v = v.to(device)
                        if v.dtype.is_floating_point and v.dtype != model_dtype:
                            v = v.to(model_dtype)
                    batch[k] = v
                video = batch["video"]
                action_gt = batch["action"]
                ctx = batch["context"]
                cmask = batch.get("context_mask")
                is_pad = batch.get("action_is_pad")
                proprio = batch.get("proprio")
                T_a = action_gt.shape[1]
                first_frame = video[:, :, 0]
                eval_extra_kwargs = {
                    "proprio": None if proprio is None else proprio[:, 0, :],
                }
                if hasattr(self._unwrap(), "shared_dit"):
                    eval_extra_kwargs["action_num_inference_steps"] = int(
                        self.config.eval_action_num_inference_steps or self.config.eval_num_inference_steps
                    )
                ac_ctx = self.accelerator.autocast() if self.accelerator is not None else nullcontext()
                with torch.no_grad(), ac_ctx:
                    pred_a = self._unwrap().infer_action(
                        first_frame, ctx, cmask,
                        action_horizon=T_a,
                        num_inference_steps=self.config.eval_num_inference_steps,
                        seed=self.config.seed,
                        num_video_frames=video.shape[2],
                        **eval_extra_kwargs,
                    )
                target_a = action_gt[:, :T_a]
                am = action_mse(pred_a, target_a, is_pad=is_pad, per_step=True)
                mse_list.append(float(am["mse"].item()))
                mse_per_step.append(am["mse_per_step"].detach().cpu())
                mse_per_dim.append(action_dim_mse(pred_a, target_a, is_pad=is_pad).detach().cpu())
                if is_pad is None:
                    pred_gripper_values.append(pred_a[..., -1].detach().float().cpu().reshape(-1))
                    target_gripper_values.append(target_a[..., -1].detach().float().cpu().reshape(-1))
                else:
                    keep = (~is_pad).detach().cpu()
                    pred_gripper_values.append(pred_a[..., -1].detach().float().cpu()[keep])
                    target_gripper_values.append(target_a[..., -1].detach().float().cpu()[keep])
                if compute_video:
                    ac_ctx2 = self.accelerator.autocast() if self.accelerator is not None else nullcontext()
                    with torch.no_grad(), ac_ctx2:
                        out = self._unwrap().infer_joint(
                            first_frame, ctx, cmask,
                            num_video_frames=video.shape[2],
                            action_horizon=T_a,
                            num_inference_steps=self.config.eval_num_inference_steps,
                            seed=self.config.seed,
                            **eval_extra_kwargs,
                        )
                    pv = out["video"]
                    psnr_list.append(float(
                        video_psnr(pv, video[:, :, : pv.shape[2]]).item()
                    ))
                n_seen += 1

            # Only rank 0 logs — no distributed ops needed
            if self._is_main_process():
                if not mse_list:
                    logger.warning("[eval] no samples produced metrics")
                else:
                    avg_mse = sum(mse_list) / len(mse_list)
                    log_msg = f"[eval] step={self.global_step} | action_mse={avg_mse:.4f} (n={len(mse_list)})"
                    wandb_payload = {"eval/action_mse": avg_mse, "eval/n_samples": len(mse_list)}
                    if psnr_list:
                        avg_psnr = sum(psnr_list) / len(psnr_list)
                        log_msg += f" | video_psnr={avg_psnr:.2f}dB"
                        wandb_payload["eval/video_psnr"] = avg_psnr
                    if mse_per_dim:
                        dim_mse = torch.stack(mse_per_dim, dim=0).mean(dim=0)
                        eef_mse = dim_mse[:-1].mean().item()
                        gripper_mse = dim_mse[-1].item()
                        log_msg += f" | action_eef_mse={eef_mse:.4f} | gripper_mse={gripper_mse:.4f}"
                        wandb_payload["eval/action_eef_mse"] = float(eef_mse)
                        wandb_payload["eval/action_gripper_mse"] = float(gripper_mse)
                        for i, v in enumerate(dim_mse.tolist()):
                            wandb_payload[f"eval/mse_dim_{i}"] = float(v)
                    if pred_gripper_values:
                        pred_gripper = torch.cat(pred_gripper_values)
                        target_gripper = torch.cat(target_gripper_values)
                        wandb_payload["eval/pred_gripper_mean"] = float(pred_gripper.mean().item())
                        wandb_payload["eval/pred_gripper_open_rate"] = float((pred_gripper > 0).float().mean().item())
                        wandb_payload["eval/target_gripper_mean"] = float(target_gripper.mean().item())
                        wandb_payload["eval/target_gripper_open_rate"] = float((target_gripper > 0).float().mean().item())
                    if mse_per_step:
                        T = min(t.shape[0] for t in mse_per_step)
                        stacked = torch.stack([t[:T] for t in mse_per_step], dim=0).mean(dim=0)
                        for i, v in enumerate(stacked.tolist()):
                            wandb_payload[f"eval/mse_step_{i}"] = float(v)
                    logger.info(log_msg)
                    self._wandb_log(wandb_payload)
        except Exception as e:  # noqa: BLE001
            if self._is_main_process():
                logger.warning(f"[eval] skipped due to error: {e}")
        finally:
            if was_training:
                self.model.train()

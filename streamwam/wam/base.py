"""Base WAM taxonomy classes."""

from __future__ import annotations

import torch.nn as nn


class WAMModel(nn.Module):
    """Base class for StreamWAM methods."""

    def training_step(self, sample: dict):
        raise NotImplementedError

    def infer_action(self, *args, **kwargs):
        raise NotImplementedError

    def infer_joint(self, *args, **kwargs):
        raise NotImplementedError

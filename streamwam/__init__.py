"""StreamWAM: taxonomy-oriented WAM entry points."""

from streamwam.builder import build_framework, build_trainer
from streamwam.taxonomy import (
    ACTION_HEAD,
    FEATURE_CONDITIONED_ACTION_MODEL,
    LATENT_ACTION,
    MOT_WAM,
    SHARED_DIT_WAM,
    TOKEN_ACTION,
    normalize_model_family,
    validate_taxonomy,
)
from streamwam.wam import FeatureConditionedActionModel, MoTWAM, SharedDiTWAM, WAMModel

__all__ = [
    "build_framework",
    "build_trainer",
    "WAMModel",
    "FeatureConditionedActionModel",
    "MoTWAM",
    "SharedDiTWAM",
    "FEATURE_CONDITIONED_ACTION_MODEL",
    "MOT_WAM",
    "SHARED_DIT_WAM",
    "ACTION_HEAD",
    "TOKEN_ACTION",
    "LATENT_ACTION",
    "normalize_model_family",
    "validate_taxonomy",
]

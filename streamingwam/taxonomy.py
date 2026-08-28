"""Taxonomy helpers for Streaming-WAM model families."""

from __future__ import annotations

from typing import Any

FEATURE_CONDITIONED_ACTION_MODEL = "feature_conditioned_action_model"
MOT_WAM = "mot_wam"
SHARED_DIT_WAM = "shared_dit_wam"

ACTION_HEAD = "action_head"
TOKEN_ACTION = "token_action"
LATENT_ACTION = "latent_action"

_MODEL_FAMILY_ALIASES = {
    FEATURE_CONDITIONED_ACTION_MODEL: FEATURE_CONDITIONED_ACTION_MODEL,
    MOT_WAM: MOT_WAM,
    SHARED_DIT_WAM: SHARED_DIT_WAM,
}

_SUPPORTED_ACTION_REPRESENTATIONS = {
    ACTION_HEAD,
    TOKEN_ACTION,
    LATENT_ACTION,
}


def _taxonomy(config: Any) -> Any:
    return getattr(config, "taxonomy", None)


def normalize_model_family(model_family: str | None) -> str:
    key = (model_family or MOT_WAM).strip().lower()
    try:
        return _MODEL_FAMILY_ALIASES[key]
    except KeyError as e:
        allowed = sorted(set(_MODEL_FAMILY_ALIASES.values()))
        raise ValueError(f"Unknown Streaming-WAM model_family={model_family!r}; allowed families: {allowed}") from e


def validate_taxonomy(config: Any) -> str:
    taxonomy = _taxonomy(config)
    if taxonomy is None:
        return MOT_WAM

    package = getattr(taxonomy, "package", "streamingwam")
    if package != "streamingwam":
        raise ValueError(f"taxonomy.package must be 'streamingwam', got {package!r}")

    model_family = normalize_model_family(getattr(taxonomy, "model_family", MOT_WAM))
    action_representation = getattr(taxonomy, "action_representation", TOKEN_ACTION)
    if action_representation not in _SUPPORTED_ACTION_REPRESENTATIONS:
        raise ValueError(
            f"taxonomy.action_representation={action_representation!r} is not supported; "
            f"allowed: {sorted(_SUPPORTED_ACTION_REPRESENTATIONS)}"
        )

    if model_family == FEATURE_CONDITIONED_ACTION_MODEL and action_representation != ACTION_HEAD:
        raise ValueError(
            "feature_conditioned_action_model expects taxonomy.action_representation='action_head'"
        )
    if model_family == MOT_WAM and action_representation not in (TOKEN_ACTION, ACTION_HEAD):
        raise ValueError("mot_wam expects token_action/action_head style action representation")
    if model_family == SHARED_DIT_WAM and action_representation not in (TOKEN_ACTION, LATENT_ACTION):
        raise ValueError("shared_dit_wam expects token_action or latent_action")

    return model_family

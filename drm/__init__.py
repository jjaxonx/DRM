from typing import Any

__all__ = ["DRMPipeline", "build_reward_model", "REWARD_HEADS", "HEAD_LAYER_MISSING"]


def __getattr__(name: str) -> Any:
    if name in ("REWARD_HEADS", "HEAD_LAYER_MISSING"):
        from drm.utils import HEAD_LAYER_MISSING, REWARD_HEADS

        return {"REWARD_HEADS": REWARD_HEADS, "HEAD_LAYER_MISSING": HEAD_LAYER_MISSING}[name]
    if name == "DRMPipeline":
        from drm.pipeline import DRMPipeline

        return DRMPipeline
    if name == "build_reward_model":
        from drm.model import build_reward_model

        return build_reward_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

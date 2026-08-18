"""Model and device specs. One file per model; adding a model is
adding a file and a registry line."""

from .base import ACT_BYTES_PER_HIDDEN, DeviceSpec, ModelSpec
from .h100_sxm import H100_SXM
from .qwen3_4b import QWEN3_4B_FP8

MODELS = {QWEN3_4B_FP8.name: QWEN3_4B_FP8}
DEVICES = {H100_SXM.name: H100_SXM}

__all__ = ["ACT_BYTES_PER_HIDDEN", "DeviceSpec", "ModelSpec",
           "MODELS", "DEVICES", "QWEN3_4B_FP8", "H100_SXM"]

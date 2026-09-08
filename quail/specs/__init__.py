"""Model and device specs.

One file per model; adding a model is adding a file and a registry
line.
"""

from .base import ACT_BYTES_PER_HIDDEN, DeviceSpec, ModelSpec, Precision
from .h100_sxm import H100_SXM
from .qwen3_4b import QWEN3_4B_FP8
from .qwen3_32b import QWEN3_32B_FP8
from .rtx_pro_6000_blackwell_server import RTX_PRO_6000_BLACKWELL_SERVER

MODELS = {QWEN3_4B_FP8.name: QWEN3_4B_FP8,
         QWEN3_32B_FP8.name: QWEN3_32B_FP8}
DEVICES = {device.name: device for device in (
    H100_SXM, RTX_PRO_6000_BLACKWELL_SERVER,
)}

__all__ = ["ACT_BYTES_PER_HIDDEN", "DeviceSpec", "ModelSpec", "Precision",
           "MODELS", "DEVICES", "QWEN3_4B_FP8", "QWEN3_32B_FP8",
           "H100_SXM", "RTX_PRO_6000_BLACKWELL_SERVER"]

"""Model and device specs.

One file per model; adding a model is adding a file and a registry
line.
"""

from .base import ACT_BYTES_PER_HIDDEN, DeviceSpec, ModelSpec, Precision, Role
from .diffusion_gemma_26b import (
    DIFFUSION_GEMMA_26B_FP8,
    DIFFUSION_GEMMA_26B_FP8_CANVAS8,
    DIFFUSION_GEMMA_26B_FP8_CANVAS32,
    DIFFUSION_GEMMA_26B_FP8_CANVAS256,
)
from .h100_sxm import H100_PRICE_SOURCE, H100_SXM, H100_USD_PER_HOUR
from .qwen3_4b import QWEN3_4B_FP8
from .qwen3_32b import QWEN3_32B_FP8
from .qwen3_reranker_0_6b import QWEN3_RERANKER_0_6B_BF16
from .qwen3_reranker_4b import QWEN3_RERANKER_4B_BF16
from .rtx_pro_6000_blackwell_server import RTX_PRO_6000_BLACKWELL_SERVER

MODELS = {QWEN3_4B_FP8.name: QWEN3_4B_FP8,
         QWEN3_32B_FP8.name: QWEN3_32B_FP8,
         QWEN3_RERANKER_0_6B_BF16.name: QWEN3_RERANKER_0_6B_BF16,
         QWEN3_RERANKER_4B_BF16.name: QWEN3_RERANKER_4B_BF16,
         DIFFUSION_GEMMA_26B_FP8.name: DIFFUSION_GEMMA_26B_FP8,
         DIFFUSION_GEMMA_26B_FP8_CANVAS32.name: DIFFUSION_GEMMA_26B_FP8_CANVAS32,
         DIFFUSION_GEMMA_26B_FP8_CANVAS8.name: DIFFUSION_GEMMA_26B_FP8_CANVAS8,
         DIFFUSION_GEMMA_26B_FP8_CANVAS256.name: DIFFUSION_GEMMA_26B_FP8_CANVAS256}
DEVICES = {device.name: device for device in (
    H100_SXM, RTX_PRO_6000_BLACKWELL_SERVER,
)}
# hourly rental price of one device, keyed by device name
MODAL_GPU_USD_PER_HOUR = {
    name: device.usd_per_hour for name, device in DEVICES.items()}

__all__ = ["ACT_BYTES_PER_HIDDEN", "DeviceSpec", "ModelSpec", "Precision",
           "Role", "MODELS", "DEVICES", "QWEN3_4B_FP8", "QWEN3_32B_FP8",
           "QWEN3_RERANKER_0_6B_BF16",
           "QWEN3_RERANKER_4B_BF16", "DIFFUSION_GEMMA_26B_FP8",
           "DIFFUSION_GEMMA_26B_FP8_CANVAS32",
           "DIFFUSION_GEMMA_26B_FP8_CANVAS8", "DIFFUSION_GEMMA_26B_FP8_CANVAS256",
           "H100_SXM", "H100_USD_PER_HOUR", "H100_PRICE_SOURCE",
           "RTX_PRO_6000_BLACKWELL_SERVER", "MODAL_GPU_USD_PER_HOUR"]

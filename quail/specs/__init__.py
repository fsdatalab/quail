"""Model and device specs.

One file per model; adding a model is adding a file and a registry
line. qwen3_spec builds the spec of any Qwen3 checkpoint from its
config.json.
"""

from .b200 import B200
from .base import (
    ACT_BYTES_PER_HIDDEN,
    AnswerCanvas,
    DeviceSpec,
    ModelSpec,
    Precision,
    Role,
)
from .decision_2_kai_0_6b import DECISION_2_KAI_0_6B_BF16
from .diffusion_gemma_26b import DIFFUSION_GEMMA_26B_FP8
from .h100_sxm import H100_PRICE_SOURCE, H100_SXM, H100_USD_PER_HOUR
from .l40s import L40S
from .qwen3 import qwen3_spec
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
         DECISION_2_KAI_0_6B_BF16.name: DECISION_2_KAI_0_6B_BF16}
DEVICES = {device.name: device for device in (
    H100_SXM, L40S, B200, RTX_PRO_6000_BLACKWELL_SERVER,
)}
# hourly rental price of one device, keyed by device name
MODAL_GPU_USD_PER_HOUR = {
    name: device.usd_per_hour for name, device in DEVICES.items()}


def modal_gpu_type(device: str, gpus: int = 1) -> str:
    """Return the Modal ``gpu=`` string for a registered device.

    One GPU is the spec's ``modal_gpu`` field. More than one appends
    the count, as in ``H100!:8``.

    Args:
        device: Registered device name.
        gpus: GPU count. Defaults to 1.

    Returns:
        The Modal ``gpu=`` value.
    """
    spec = DEVICES[device]
    if not spec.modal_gpu:
        raise ValueError(f"device {device!r} has no Modal GPU type")
    if gpus == 1:
        return spec.modal_gpu
    return f"{spec.modal_gpu}:{gpus}"


__all__ = ["ACT_BYTES_PER_HIDDEN", "AnswerCanvas", "DeviceSpec", "ModelSpec",
           "Precision", "Role", "MODELS", "DEVICES", "QWEN3_4B_FP8", "QWEN3_32B_FP8",
           "QWEN3_RERANKER_0_6B_BF16",
           "QWEN3_RERANKER_4B_BF16", "DIFFUSION_GEMMA_26B_FP8",
           "DECISION_2_KAI_0_6B_BF16",
           "H100_SXM", "H100_USD_PER_HOUR", "H100_PRICE_SOURCE",
           "L40S", "B200", "RTX_PRO_6000_BLACKWELL_SERVER",
           "MODAL_GPU_USD_PER_HOUR", "modal_gpu_type", "qwen3_spec"]

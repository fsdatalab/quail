"""Model and device specs.

Each built-in model and device is one TOML file under
``quail/specs/data``; adding a model is adding a file. qwen3_spec
builds the spec of any Qwen3 checkpoint from its config.json.
"""

from .base import (
    ACT_BYTES_PER_HIDDEN,
    AnswerCanvas,
    DeviceSpec,
    LayerKind,
    ModelSpec,
    Precision,
    Role,
    SaeSpec,
    SupportResult,
)
from .loading import read_devices, read_models
from .qwen3 import qwen3_spec

MODELS = read_models()
DEVICES = read_devices()
# hourly rental price of one device, keyed by device name
MODAL_GPU_USD_PER_HOUR = {
    name: device.usd_per_hour for name, device in DEVICES.items()}

QWEN3_4B_FP8 = MODELS["qwen3-4b-fp8"]
QWEN3_32B_FP8 = MODELS["qwen3-32b-fp8"]
QWEN3_5_2B_SAE_BF16 = MODELS["qwen3.5-2b-sae-bf16"]
QWEN3_5_4B_BF16 = MODELS["qwen3.5-4b-bf16"]
QWEN3_5_4B_FP8 = MODELS["qwen3.5-4b-fp8"]
QWEN3_5_27B_FP8 = MODELS["qwen3.5-27b-fp8"]
QWEN3_6_35B_A3B_FP8 = MODELS["qwen3.6-35b-a3b-fp8"]
QWEN3_8_27B_FP8 = MODELS["qwen3.8-27b-fp8"]
QWEN3_RERANKER_0_6B_BF16 = MODELS["qwen3-reranker-0.6b-bf16"]
QWEN3_RERANKER_4B_BF16 = MODELS["qwen3-reranker-4b-bf16"]
LFM2_2_6B_BF16 = MODELS["lfm2-2.6b-bf16"]
DIFFUSION_GEMMA_26B_FP8 = MODELS["diffusion-gemma-26b-a4b-fp8"]
DECISION_2_KAI_0_6B_BF16 = MODELS["decision-2.0-kai-0.6b-bf16"]
D1_2B_BF16 = MODELS["d1-2b-bf16"]
D1_2B_FP8 = MODELS["d1-2b-fp8"]

H100_SXM = DEVICES["h100-sxm"]
L40S = DEVICES["l40s"]
B200 = DEVICES["b200"]
RTX_PRO_6000_BLACKWELL_SERVER = DEVICES["rtx-pro-6000-blackwell-server"]
H100_USD_PER_HOUR = H100_SXM.usd_per_hour
H100_PRICE_SOURCE = H100_SXM.price_source


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


__all__ = ["ACT_BYTES_PER_HIDDEN", "AnswerCanvas", "DeviceSpec", "LayerKind",
           "ModelSpec",
           "Precision", "Role", "SaeSpec", "SupportResult", "MODELS", "DEVICES",
           "QWEN3_4B_FP8",
           "QWEN3_5_2B_SAE_BF16", "QWEN3_5_4B_BF16",
           "QWEN3_5_4B_FP8", "QWEN3_5_27B_FP8", "QWEN3_8_27B_FP8", "LFM2_2_6B_BF16",
           "QWEN3_32B_FP8", "QWEN3_6_35B_A3B_FP8",
           "QWEN3_RERANKER_0_6B_BF16",
           "QWEN3_RERANKER_4B_BF16", "DIFFUSION_GEMMA_26B_FP8",
           "DECISION_2_KAI_0_6B_BF16", "D1_2B_BF16", "D1_2B_FP8",
           "H100_SXM", "H100_USD_PER_HOUR", "H100_PRICE_SOURCE",
           "L40S", "B200", "RTX_PRO_6000_BLACKWELL_SERVER",
           "MODAL_GPU_USD_PER_HOUR", "modal_gpu_type",
           "qwen3_spec"]

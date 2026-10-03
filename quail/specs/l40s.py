"""NVIDIA L40S peak hardware specifications.

Dense tensor rates: https://www.nvidia.com/en-us/data-center/l40s/
(733 TFLOPS FP8 and 362.05 TFLOPS BF16 without sparsity). mem_bytes is
Modal's torch.cuda total_memory, not the 48 GB datasheet figure.
Measurement: /results/ablations/device_probe/20261003T140154Z/L40S.json.
"""

from .base import DeviceSpec

L40S = DeviceSpec(
    name="l40s",
    mem_bytes=47_695_921_152,
    hbm_bw=0.864e12,
    peak_flops=0.733e15,
    bf16_flops=0.36605e15,
    # GPU-only rate from Modal, checked 2026-10-03
    usd_per_hour=1.9512,
    price_source="https://modal.com/pricing",
)

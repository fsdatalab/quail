"""NVIDIA B200 peak hardware specifications.

Tensor rates: https://www.nvidia.com/en-us/data-center/hgx/ (HGX B200
column). That table is eight GPUs: 72 PFLOPS FP8 and 36 PFLOPS BF16,
both with sparsity. One GPU is those totals divided by eight, then
halved: 4.5 PFLOPS dense FP8, 2.25 PFLOPS dense BF16.
Bandwidth: NVIDIA B200 SXM "Up to 8 TB/s" in
https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html
mem_bytes is Modal's torch.cuda total_memory, not the 180 GB datasheet
figure. Measurement:
/results/ablations/device_probe/20261003T140154Z/B200.json.
"""

from .base import DeviceSpec

B200 = DeviceSpec(
    name="b200",
    mem_bytes=191_503_138_816,
    hbm_bw=8e12,
    peak_flops=4.5e15,
    bf16_flops=2.25e15,
    # GPU-only rate from Modal, checked 2026-10-03
    usd_per_hour=6.2496,
    price_source="https://modal.com/pricing",
)

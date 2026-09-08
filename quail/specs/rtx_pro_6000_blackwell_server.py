"""RTX PRO 6000 Blackwell Server Edition peak hardware specifications.

Source: https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/
Dense tensor rates are estimates: the server page lists 2 PFLOPS FP8
and 1 PFLOPS BF16 without labeling sparsity. We assume those are sparse
rates and halve them. NVIDIA's RTX PRO Blackwell architecture paper,
Table 4, confirms this ratio for the Workstation Edition, not exact
Server Edition dense rates. See the feature report for both sources.
"""

from .base import DeviceSpec

RTX_PRO_6000_BLACKWELL_SERVER = DeviceSpec(
    name="rtx-pro-6000-blackwell-server",
    mem_bytes=96e9,
    hbm_bw=1.597e12,
    peak_flops=1e15,
    bf16_flops=0.5e15,
)

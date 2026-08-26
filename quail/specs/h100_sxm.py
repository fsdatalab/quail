from .base import DeviceSpec

H100_SXM = DeviceSpec(
    name="h100-sxm",
    mem_bytes=80e9,
    hbm_bw=3.35e12,
    peak_flops=1.979e15,        # fp8 dense ceiling
    bf16_peak_flops=0.9895e15,  # bf16 ceiling; fp8 is 2x on Hopper
)

from .base import DeviceSpec

H100_SXM = DeviceSpec(
    name="h100-sxm",
    mem_bytes=80e9,
    hbm_bw=3.35e12,
    peak_flops=1.979e15,   # fp8 dense ceiling (3958 TFLOPS with
    #                        sparsity, halved)
    bf16_flops=0.9895e15,  # bf16 dense ceiling (1979 TFLOPS with
    #                        sparsity, halved)
)

from .base import DeviceSpec

H100_SXM = DeviceSpec(
    name="h100-sxm",
    mem_bytes=80e9,
    hbm_bw=3.35e12,
    peak_flops=1.979e15,   # fp8 dense ceiling (3958 TFLOPS with
    #                        sparsity, halved)
    bf16_flops=0.9895e15,  # bf16 dense ceiling (1979 TFLOPS with
    #                        sparsity, halved)
    fp32_flops=67e12,      # FP32 on the CUDA cores, no sparsity
    #                        (https://www.nvidia.com/en-us/data-center/h100/)
    usd_per_hour=3.9492,
    price_source="https://modal.com/pricing",
    sustained_fp8_flops=1.352e15,  # mamf-finder MSMF at 30ddd970, the
    #                                fastest shape held at 700 W:
    #                                /results/ablations/gemm_ceilings_h100_fp8.json
    sustained_hbm_bw=3.048e12,     # 4 GB copy held 2 s:
    #                                /results/ablations/attention_ceilings_h100.json
)

# The price every $/query number in reports uses: query hours times
# GPU count times this rate.
H100_USD_PER_HOUR = H100_SXM.usd_per_hour
H100_PRICE_SOURCE = H100_SXM.price_source

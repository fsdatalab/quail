from .base import AttentionRates, DeviceSpec

# Faster of FlashAttention 3 and FlashInfer at each point, each held
# 2 s; cached reads on private 16-token pages:
# /results/ablations/attention_ceilings_h100.json
ATTENTION_32_8_128 = AttentionRates(
    causal_tokens=(128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768),
    causal=(104.1e12, 191.7e12, 301.0e12, 411.2e12, 506.1e12, 583.5e12,
            628.6e12, 652.4e12, 667.6e12),
    read_rows=(16, 32, 64, 128, 256, 512, 1024),
    read_contexts=(1024, 4096, 8192, 32768),
    read=((172e12, 191e12, 195e12, 195e12),
          (317e12, 371e12, 385e12, 389e12),
          (412e12, 481e12, 496e12, 510e12),
          (439e12, 518e12, 533e12, 543e12),
          (451e12, 539e12, 549e12, 561e12),
          (489e12, 560e12, 572e12, 576e12),
          (492e12, 564e12, 577e12, 578e12)),
)

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
    attention_rates=(((32, 8, 128), ATTENTION_32_8_128),),
)

# The price every $/query number in reports uses: query hours times
# GPU count times this rate.
H100_USD_PER_HOUR = H100_SXM.usd_per_hour
H100_PRICE_SOURCE = H100_SXM.price_source

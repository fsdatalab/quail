from .base import AttentionRates, DeviceSpec

# One table per head layout. At each point: the faster of FlashAttention 3
# and FlashInfer, the highest steady rate over 3 H100s, at the call sizes
# Quail uses for the layout (calibrate/README.md). calibrate runs
# /results/calibrate/h100-sxm/attention/20261014T173307Z-<layout>-g{0,1,2}/

ATTENTION_16_8_128 = AttentionRates(
    causal_tokens=(128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768),
    causal=(98.9e12, 183.3e12, 305.1e12, 420.2e12, 509.1e12, 586.0e12,
            635.1e12, 660.4e12, 675.6e12),
    read_rows=(16, 32, 64, 128, 256, 512, 1024),
    read_contexts=(1024, 4096, 8192, 32768),
    read=((89.4e12, 92.1e12, 95.6e12, 69.6e12),
          (170.6e12, 181.3e12, 189.5e12, 139.5e12),
          (315.1e12, 352.2e12, 371.8e12, 277.6e12),
          (415.2e12, 478.6e12, 480.6e12, 439.6e12),
          (443.4e12, 526.1e12, 533.5e12, 513.5e12),
          (456.2e12, 540.5e12, 556.7e12, 566.2e12),
          (495.5e12, 562.5e12, 577.6e12, 581.2e12)),
)

ATTENTION_32_8_128 = AttentionRates(
    causal_tokens=(128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768),
    causal=(106.6e12, 193.2e12, 301.9e12, 416.8e12, 510.1e12, 588.7e12,
            635.5e12, 663.4e12, 675.7e12),
    read_rows=(16, 32, 64, 128, 256, 512, 1024),
    read_contexts=(1024, 4096, 8192, 32768),
    read=((168.0e12, 185.6e12, 188.4e12, 163.3e12),
          (309.9e12, 361.8e12, 372.5e12, 322.3e12),
          (415.3e12, 484.4e12, 497.7e12, 421.9e12),
          (441.7e12, 527.1e12, 530.0e12, 526.8e12),
          (451.6e12, 542.8e12, 561.4e12, 537.8e12),
          (492.1e12, 561.6e12, 578.7e12, 583.7e12),
          (496.2e12, 567.9e12, 580.1e12, 585.7e12)),
)

ATTENTION_64_8_128 = AttentionRates(
    causal_tokens=(128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768),
    causal=(109.5e12, 194.6e12, 308.1e12, 417.4e12, 514.3e12, 586.3e12,
            634.2e12, 662.9e12, 674.9e12),
    read_rows=(16, 32, 64, 128, 256, 512, 1024),
    read_contexts=(1024, 4096, 8192, 32768),
    read=((278.2e12, 313.9e12, 291.9e12, 315.0e12),
          (413.7e12, 430.0e12, 435.3e12, 334.8e12),
          (443.4e12, 498.1e12, 551.0e12, 461.5e12),
          (452.2e12, 530.8e12, 555.3e12, 477.0e12),
          (494.4e12, 562.2e12, 575.6e12, 586.6e12),
          (497.6e12, 570.0e12, 577.1e12, 589.4e12),
          (499.1e12, 573.5e12, 585.6e12, 588.8e12)),
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
    sustained_fp8_flops=1.431e15,  # highest mamf-finder MSMF over 7 H100s
    #                                (1,315 to 1,431 TFLOPS), at 700 W:
    #                                /results/calibrate/h100-sxm/gemm/
    #                                20261014T173306Z-g2/
    sustained_hbm_bw=3.030e12,     # highest steady 4 GB copy over 9 H100s
    attention_rates=(((16, 8, 128), ATTENTION_16_8_128),
                     ((32, 8, 128), ATTENTION_32_8_128),
                     ((64, 8, 128), ATTENTION_64_8_128)),
)

# The price every $/query number in reports uses: query hours times
# GPU count times this rate.
H100_USD_PER_HOUR = H100_SXM.usd_per_hour
H100_PRICE_SOURCE = H100_SXM.price_source

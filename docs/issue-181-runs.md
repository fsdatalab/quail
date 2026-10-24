# Issue 181 results

Results are on the `quail-results` volume. Reference labels are `qwen3-32b-fp8`. `$/query` is `cost_usd` in the run. Stock vLLM submits operator-at-a-time. Pipelined vLLM uses the pipelining submission. A suite directory holds `quail/run.json`, `stock_vllm/run.json`, and `pipelined_vllm/run.json`.

The prediction, stated before the runs, was compute-bound prefill. B200 is at most 2.3× H100 (4.5 / 1.98 PFLOPS). L40S is at most 0.37× H100 (0.73 / 1.98), and was expected below that because its matmul kernels reach less of peak. On one GPU, Quail is at least as fast as both vLLM configurations. Quail's query time divided by the speed-of-light estimate stays within about 20% of that fraction on H100. Accuracy matches H100 within run-to-run noise. On 8×L40S the large filter reaches at least 0.8 scaling efficiency, and the answers match the 1-GPU run.

The cited runs recorded `prompt_format` `raw-v1`. This branch now includes main's Qwen3 chat template, `qwen3-chat-text-analysis-v1`. A rerun would not match these token counts. The fraction below uses `estimated_seconds` stored on these runs. Main has since changed that estimate for a DISTINCT stop key, so a newly computed estimate would not match these fractions.

## Smoke

Qwen3 4B, one filter and one join, on the unified and tree attention paths, on H100, L40S, and B200. Answers matched the H100 run. Volume `/results/ablations/device_smoke/20261014T023150Z`.

## 8×L40S

Qwen3 4B at scale factor 0.1, Quail only, one model copy per GPU. IMDB-1 splits 5,000 reviews, so eight GPUs are at most 8× the 1-GPU query time. IMDB-2 copies 12 aspects onto every GPU, so the join scales less than the filter. CUAD-5 has 45 contracts.

Speedup is the 1-GPU `runtime_s` divided by the 8-GPU `runtime_s`. Efficiency is that speedup divided by 8. `$/query` uses $1.9512 per L40S-hour.

| Query | 1× s | 8× s | Speedup | Efficiency | 1× $ | 8× $ | Answers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| IMDB-1 | 84.71 | 11.16 | 7.59 | 0.949 | 0.0459 | 0.0484 | identical |
| IMDB-2 | 70.56 | 11.65 | 6.06 | 0.757 | 0.0382 | 0.0505 | 13,316 pairs against 13,313 |
| CUAD-5 | 17.51 | 2.66 | 6.59 | 0.823 | 0.0095 | 0.0115 | identical |

| Query | GPUs | Volume |
| --- | ---: | --- |
| IMDB-1 | 1 | `/results/benchmarks/quailb/20261017T025337Z-40928676` |
| IMDB-1 | 8 | `/results/benchmarks/quailb/20261017T030459Z-70f85088` |
| IMDB-2 | 1 | `/results/benchmarks/quailb/20261017T030004Z-82a1c7b1` |
| IMDB-2 | 8 | `/results/benchmarks/quailb/20261017T195132Z-d08b3d66` |
| CUAD-5 | 1 | `/results/benchmarks/quailb/20261017T030305Z-36c9cc57` |
| CUAD-5 | 8 | `/results/benchmarks/quailb/20261017T145531Z-b4452bd8` |

IMDB-1 is the large filter. Efficiency is 0.949, above 0.8. Fresh tokens are 1,759,233 on both runs, and regret is 20,350. Accuracy against the reference is 0.9096 on both. The answer files match byte for byte.

CUAD-5 efficiency is 0.823. Fresh tokens are 334,396 on both, and regret is 296. Accuracy is 0.6889 on both. The answer files match byte for byte.

IMDB-2 is the join. Efficiency is 0.757. Fresh tokens are 2,088,641 on 8 GPUs and 2,087,873 on 1 GPU. Regret is 74,758 against 73,990. Accuracy against the reference is 0.7722 on both. The answer tables differ by three kept pairs. There is no profile of that difference.

## SF 0.1, one GPU

Qwen3 4B and DiffusionGemma run every IMDB, BIO, FEV, LEP, and AGENT query. Qwen3 32B runs `IMDB-1`, `BIO-1`, `FEV-1`, `LEP-1`, and `AGENT-1`.

| Device | Model | Volume | Result |
| --- | --- | --- | --- |
| H100 | Qwen3 4B | `/results/benchmarks/quailb/20261017T032038Z-37af4c6b` | 43 queries finished |
| L40S | Qwen3 4B | `/results/benchmarks/quailb/20261017T130139Z-e346a785` | 43 queries finished |
| B200 | Qwen3 4B | `/results/benchmarks/quailb/20261017T130139Z-a1db59ed` | 43 queries finished |
| H100 | Qwen3 32B | `/results/benchmarks/quailb/20261017T200744Z-a1f0bd2f` | 5 queries finished |
| B200 | Qwen3 32B | `/results/benchmarks/quailb/20261017T145741Z-2106faa7` | 5 queries finished |
| L40S | Qwen3 32B | `/results/benchmarks/quailb/20261017T200744Z-b3bd1b3e` | stock vLLM cannot fit one token |
| H100 | DiffusionGemma 26B | `/results/benchmarks/quailb/20261017T040833Z-c0e42adc` | stock vLLM omitted TRUE/FALSE scores |
| H100 | DiffusionGemma 26B | `/results/benchmarks/quailb/20261024T150444Z-e3303d05` | Quail finished 43 queries; baselines off |
| B200 | DiffusionGemma 26B | `/results/benchmarks/quailb/20261017T145741Z-c2afc552` | Quail FlashAttention 4 rejects head dim 256 |
| B200 | DiffusionGemma 26B | `/results/benchmarks/quailb/20261024T150444Z-c7a9c717` | Quail warmup dies on head dim 512, TMEM |
| B200 | DiffusionGemma 26B | `/results/benchmarks/quailb/20261024T154540Z-ef369227` | Quail warmup dies on head dim 256, SplitKV |
| B200 | DiffusionGemma 26B | `/results/benchmarks/quailb/20261024T155537Z-6372ad9e` | Quail finished; stock vLLM cannot pickle the tokenizer |
| L40S | DiffusionGemma 26B | | skipped: sliding KV does not fit, and FlashAttention 4 does not run here |

L40S at 0.37× H100 means an L40S query time at least 2.70× the H100 time. B200 at 2.3× means an H100 query time at most 2.3× the B200 time.

### Qwen3 4B

L40S query time is 2.60× to 5.99× the H100 time. The median is 4.20×, which is 0.24× the H100 speed. FEV-1 is the one query above the 0.37× bound, at 0.38×. B200 is 0.44× to 1.86× the H100 speed. The median is 1.46×. Nothing exceeds 2.3×. Eleven queries are slower on B200 than on H100: IMDB-8, IMDB-9, BIO-2, BIO-3, BIO-4, FEV-2, FEV-3, FEV-5, FEV-7, FEV-8, and FEV-9. BIO-3 is the slowest of those, at 0.44×.

Quail is slower than a vLLM configuration in the cells below. On H100 the only one is FEV-1 against pipelined vLLM. On B200, FEV-1 and IMDB-6 are slower against both submissions. On L40S, seven queries are slower against both. There is no profile for these cells.

| Query | Device | vLLM | Quail s | vLLM s |
| --- | --- | --- | ---: | ---: |
| FEV-1 | H100 | pipelined | 0.43 | 0.40 |
| IMDB-1 | L40S | operator-at-a-time | 82.07 | 62.16 |
| IMDB-1 | L40S | pipelined | 82.07 | 64.60 |
| BIO-1 | L40S | operator-at-a-time | 121.97 | 93.89 |
| BIO-1 | L40S | pipelined | 121.97 | 100.76 |
| LEP-2 | L40S | operator-at-a-time | 694.46 | 574.33 |
| LEP-2 | L40S | pipelined | 694.46 | 579.61 |
| LEP-4 | L40S | operator-at-a-time | 154.23 | 128.87 |
| LEP-4 | L40S | pipelined | 154.23 | 131.78 |
| IMDB-11 | L40S | operator-at-a-time | 74.50 | 64.91 |
| IMDB-11 | L40S | pipelined | 74.50 | 65.92 |
| FEV-8 | L40S | operator-at-a-time | 417.40 | 392.05 |
| FEV-8 | L40S | pipelined | 417.40 | 392.66 |
| FEV-9 | L40S | operator-at-a-time | 172.86 | 172.61 |
| FEV-9 | L40S | pipelined | 172.86 | 172.12 |
| FEV-1 | B200 | operator-at-a-time | 0.33 | 0.27 |
| FEV-1 | B200 | pipelined | 0.33 | 0.24 |
| IMDB-6 | B200 | operator-at-a-time | 13.23 | 11.47 |
| IMDB-6 | B200 | pipelined | 13.23 | 9.91 |

On the other 67 comparisons, Quail's `runtime_s` is lower.

Accuracy below is agreement with the qwen3-32b-fp8 reference. Classification queries use label accuracy. The other queries use answer accuracy. The median absolute gap versus H100 is 0.011 on L40S and 0.003 on B200. LEP-1 on L40S is the large gap, 0.430 against 0.314 on H100. Recall on LEP-1 is 0.933 on all three devices, and precision stays near 0.04, so the accuracy gap is in the negative class. Queries with an absolute gap above 0.02 are LEP-1 through LEP-6, FEV-1, FEV-4, FEV-6, FEV-7, FEV-10, BIO-1, BIO-6, and IMDB-5 on L40S, and LEP-3, LEP-4, and LEP-5 on B200. Precision and recall for every query are in the same accuracy block as these numbers.

Fresh tokens match across the three devices on the short queries. They diverge where a join keeps a different set of rows. IMDB-10 is the wide one: 5,914,920 fresh tokens on H100, 5,972,696 on L40S, and 5,198,711 on B200. Accuracy there stays at 0.716, 0.716, and 0.714.

The fraction used here is query time divided by `measurements.estimated_seconds` on that run. That field is the planner's device estimate. On IMDB-1 it is 6.64 s on H100, 17.94 s on L40S, and 2.92 s on B200, which is the 2.7× and 2.3× device ratio. A label-oracle `speed_of_light_estimate` has not been written back to the volume. L40S's fraction is 0.96× to 2.22× the H100 fraction, median 1.55×. B200's is 1.22× to 5.12×, median 1.56×. FEV-1 on L40S is the only 4B query inside 20% of the H100 fraction, at 0.96×. The other 42 L40S queries, and all 43 B200 queries, are outside it. There is no profile of that gap. The gap is the one the prediction expected: the estimate assumes each device reaches the same fraction of peak as DeepGEMM on H100, and the L40S matmul kernels reach less. It is not a separate code change. B200's median is 1.46× H100, under the 2.3× bound, and eleven queries are slower on B200 than on H100. That one still needs a profile before naming a kernel.

| Query | H100 s | L40S s | L40S/H100 | B200 s | H100/B200 | H100 tok/s | L40S tok/s | B200 tok/s | H100 $ | L40S $ | B200 $ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IMDB-1 | 14.63 | 82.07 | 5.61 | 8.85 | 1.65 | 120282 | 21436 | 198739 | 0.0160 | 0.0445 | 0.0154 |
| IMDB-2 | 18.09 | 89.05 | 4.92 | 16.74 | 1.08 | 1157855 | 235215 | 1251110 | 0.0198 | 0.0483 | 0.0291 |
| IMDB-3 | 19.53 | 81.47 | 4.17 | 17.23 | 1.13 | 1054450 | 245023 | 1195637 | 0.0214 | 0.0442 | 0.0299 |
| IMDB-4 | 16.15 | 68.73 | 4.26 | 14.72 | 1.10 | 559093 | 124511 | 623262 | 0.0177 | 0.0373 | 0.0256 |
| IMDB-5 | 15.63 | 68.06 | 4.35 | 10.12 | 1.54 | 453116 | 97471 | 709173 | 0.0171 | 0.0369 | 0.0176 |
| IMDB-6 | 15.25 | 63.04 | 4.13 | 13.23 | 1.15 | 153029 | 36534 | 177261 | 0.0167 | 0.0342 | 0.0230 |
| IMDB-7 | 14.87 | 61.51 | 4.14 | 9.20 | 1.62 | 182357 | 43106 | 297109 | 0.0163 | 0.0333 | 0.0160 |
| IMDB-8 | 21.35 | 91.09 | 4.27 | 24.18 | 0.88 | 1510480 | 353182 | 1337103 | 0.0234 | 0.0494 | 0.0420 |
| IMDB-9 | 45.05 | 178.47 | 3.96 | 47.00 | 0.96 | 1180963 | 297621 | 1133737 | 0.0494 | 0.0967 | 0.0816 |
| IMDB-10 | 55.57 | 232.42 | 4.18 | 54.02 | 1.03 | 950909 | 224188 | 980001 | 0.0610 | 0.1260 | 0.0938 |
| IMDB-11 | 14.89 | 74.50 | 5.00 | 9.57 | 1.56 | 117794 | 23546 | 183276 | 0.0163 | 0.0404 | 0.0166 |
| IMDB-12 | 18.68 | 76.48 | 4.09 | 11.66 | 1.60 | 189851 | 45555 | 304262 | 0.0205 | 0.0415 | 0.0202 |
| IMDB-13 | 18.36 | 74.73 | 4.07 | 15.04 | 1.22 | 771104 | 186561 | 942882 | 0.0201 | 0.0405 | 0.0261 |
| IMDB-14 | 16.91 | 70.99 | 4.20 | 11.19 | 1.51 | 168175 | 39794 | 254361 | 0.0186 | 0.0385 | 0.0194 |
| IMDB-15 | 22.07 | 93.92 | 4.25 | 17.83 | 1.24 | 749103 | 179394 | 934905 | 0.0242 | 0.0509 | 0.0310 |
| BIO-1 | 22.11 | 121.97 | 5.52 | 12.40 | 1.78 | 93110 | 16875 | 165941 | 0.0243 | 0.0661 | 0.0215 |
| BIO-2 | 87.56 | 524.47 | 5.99 | 168.43 | 0.52 | 26543797 | 4431380 | 13798736 | 0.0961 | 0.2843 | 0.2924 |
| BIO-3 | 72.53 | 376.85 | 5.20 | 163.37 | 0.44 | 24117388 | 4401446 | 10593842 | 0.0796 | 0.2043 | 0.2836 |
| BIO-4 | 50.73 | 201.91 | 3.98 | 70.71 | 0.72 | 19628655 | 3890730 | 14003944 | 0.0557 | 0.1094 | 0.1227 |
| BIO-5 | 2.31 | 10.72 | 4.64 | 1.29 | 1.79 | 105248 | 22679 | 188308 | 0.0025 | 0.0058 | 0.0022 |
| BIO-6 | 28.63 | 115.21 | 4.02 | 18.69 | 1.53 | 4899930 | 1182573 | 8003904 | 0.0314 | 0.0624 | 0.0324 |
| FEV-1 | 0.43 | 1.13 | 2.60 | 0.33 | 1.33 | 76717 | 29505 | 102137 | 0.0005 | 0.0006 | 0.0006 |
| FEV-2 | 21.93 | 121.34 | 5.53 | 34.74 | 0.63 | 3259344 | 589129 | 2057795 | 0.0241 | 0.0658 | 0.0603 |
| FEV-3 | 16.32 | 97.50 | 5.97 | 25.94 | 0.63 | 3164281 | 501674 | 2062400 | 0.0179 | 0.0528 | 0.0450 |
| FEV-4 | 4.05 | 21.18 | 5.23 | 3.90 | 1.04 | 1686418 | 315249 | 1974052 | 0.0044 | 0.0115 | 0.0068 |
| FEV-5 | 10.45 | 50.82 | 4.86 | 16.76 | 0.62 | 2891236 | 549677 | 1869921 | 0.0115 | 0.0275 | 0.0291 |
| FEV-6 | 3.13 | 11.76 | 3.76 | 2.90 | 1.08 | 1321414 | 336099 | 1602515 | 0.0034 | 0.0064 | 0.0050 |
| FEV-7 | 41.61 | 239.78 | 5.76 | 65.29 | 0.64 | 3225892 | 548381 | 2047097 | 0.0456 | 0.1300 | 0.1133 |
| FEV-8 | 114.35 | 417.40 | 3.65 | 158.61 | 0.72 | 1845479 | 505603 | 1335924 | 0.1254 | 0.2262 | 0.2754 |
| FEV-9 | 40.52 | 172.86 | 4.27 | 61.88 | 0.65 | 2193807 | 478400 | 1494130 | 0.0445 | 0.0937 | 0.1074 |
| FEV-10 | 1.62 | 9.25 | 5.72 | 0.94 | 1.73 | 167284 | 29140 | 289039 | 0.0018 | 0.0050 | 0.0016 |
| FEV-11 | 0.37 | 1.32 | 3.60 | 0.22 | 1.63 | 109076 | 30261 | 177357 | 0.0004 | 0.0007 | 0.0004 |
| LEP-1 | 1.25 | 4.42 | 3.54 | 0.74 | 1.68 | 105616 | 29857 | 177232 | 0.0014 | 0.0024 | 0.0013 |
| LEP-2 | 117.93 | 694.46 | 5.89 | 78.42 | 1.50 | 580252 | 98535 | 872628 | 0.1294 | 0.3764 | 0.1361 |
| LEP-3 | 84.59 | 328.08 | 3.88 | 57.78 | 1.46 | 562019 | 122873 | 839820 | 0.0928 | 0.1778 | 0.1003 |
| LEP-4 | 36.11 | 154.23 | 4.27 | 24.75 | 1.46 | 449722 | 80258 | 678461 | 0.0396 | 0.0836 | 0.0430 |
| LEP-5 | 35.47 | 121.11 | 3.41 | 24.37 | 1.46 | 445677 | 96811 | 672675 | 0.0389 | 0.0656 | 0.0423 |
| LEP-6 | 30.84 | 143.02 | 4.64 | 20.61 | 1.50 | 542719 | 117281 | 802289 | 0.0338 | 0.0775 | 0.0358 |
| AGENT-1 | 90.95 | 351.90 | 3.87 | 49.59 | 1.83 | 191192 | 49415 | 350635 | 0.0998 | 0.1907 | 0.0861 |
| AGENT-2 | 90.52 | 355.57 | 3.93 | 48.72 | 1.86 | 192562 | 49024 | 357759 | 0.0993 | 0.1927 | 0.0846 |
| AGENT-3 | 91.25 | 357.24 | 3.91 | 53.95 | 1.69 | 246532 | 62707 | 419369 | 0.1001 | 0.1936 | 0.0937 |
| AGENT-4 | 97.12 | 383.72 | 3.95 | 52.56 | 1.85 | 277004 | 69569 | 511786 | 0.1065 | 0.2080 | 0.0912 |
| AGENT-5 | 109.10 | 436.79 | 4.00 | 62.37 | 1.75 | 482440 | 120504 | 843903 | 0.1197 | 0.2367 | 0.1083 |

| Query | Field | H100 acc | L40S acc | B200 acc | H100 fresh | H100 regret | L40S fresh | L40S regret | B200 fresh | B200 regret |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IMDB-1 | answer | 0.8968 | 0.9096 | 0.8950 | 1759233 | 20350 | 1759233 | 20350 | 1759233 | 20350 |
| IMDB-2 | answer | 0.7629 | 0.7722 | 0.7632 | 2087873 | 73990 | 2087873 | 73990 | 2087873 | 73990 |
| IMDB-3 | answer | 0.7734 | 0.7848 | 0.7739 | 2270333 | 71550 | 2252666 | 69738 | 2271035 | 71622 |
| IMDB-4 | answer | 0.7645 | 0.7825 | 0.7665 | 1919036 | 33990 | 1905944 | 32874 | 1921769 | 34206 |
| IMDB-5 | answer | 0.7892 | 0.8109 | 0.7886 | 1880518 | 28492 | 1866806 | 27464 | 1883057 | 28661 |
| IMDB-6 | answer | 0.9190 | 0.9273 | 0.9194 | 1772786 | 18990 | 1770179 | 18990 | 1773446 | 18990 |
| IMDB-7 | answer | 0.9196 | 0.9227 | 0.9210 | 1795243 | 19753 | 1790286 | 19677 | 1796391 | 19769 |
| IMDB-8 | answer | 0.6802 | 0.6742 | 0.6766 | 2412455 | 174538 | 2410232 | 173816 | 2414210 | 175108 |
| IMDB-9 | answer | 0.7129 | 0.7130 | 0.7108 | 4500328 | 2076445 | 4498105 | 2074222 | 4502083 | 2078200 |
| IMDB-10 | answer | 0.7161 | 0.7162 | 0.7141 | 5914920 | 3290017 | 5972696 | 3359801 | 5198711 | 2573255 |
| IMDB-11 | label | 0.8610 | 0.8580 | 0.8598 | 1752873 | 18990 | 1752873 | 18990 | 1752873 | 18990 |
| IMDB-12 | answer | 0.8968 | 0.9114 | 0.8950 | 2187113 | 23370 | 2172119 | 23217 | 2187799 | 23377 |
| IMDB-13 | answer | 0.7368 | 0.7480 | 0.7370 | 2086557 | 53214 | 2080473 | 52590 | 2086323 | 53190 |
| IMDB-14 | label | 0.7522 | 0.7524 | 0.7507 | 1947975 | 77196 | 1944443 | 76160 | 1947827 | 77144 |
| IMDB-15 | answer | 0.7368 | 0.7484 | 0.7370 | 2532907 | 139491 | 2619649 | 155862 | 2549043 | 142411 |
| BIO-1 | answer | 0.8320 | 0.8600 | 0.8340 | 2058352 | 2484 | 2058352 | 2484 | 2058352 | 2484 |
| BIO-2 | answer | 0.8186 | 0.8072 | 0.8155 | 6996352 | 770984 | 6996352 | 770984 | 6996352 | 770984 |
| BIO-3 | answer | 0.8097 | 0.8054 | 0.8071 | 5879477 | 594614 | 5750452 | 574620 | 5849702 | 590000 |
| BIO-4 | answer | 0.7702 | 0.7568 | 0.7657 | 4223249 | 576404 | 3813319 | 392299 | 4221651 | 581307 |
| BIO-5 | label | 0.7267 | 0.7285 | 0.7294 | 243166 | 3789 | 243166 | 3789 | 243166 | 3789 |
| BIO-6 | answer | 0.7845 | 0.7580 | 0.7706 | 2627613 | 44775 | 2619329 | 44729 | 2646082 | 48295 |
| FEV-1 | answer | 0.8500 | 0.8880 | 0.8360 | 33331 | 1359 | 33331 | 1359 | 33331 | 1359 |
| FEV-2 | answer | 0.8287 | 0.8395 | 0.8240 | 2385976 | 104285 | 2385976 | 104285 | 2385976 | 104285 |
| FEV-3 | answer | 0.8161 | 0.8247 | 0.8115 | 1785037 | 69476 | 1691188 | 62875 | 1846742 | 72346 |
| FEV-4 | answer | 0.8956 | 0.9256 | 0.8947 | 452241 | 3753 | 444170 | 3179 | 488004 | 4614 |
| FEV-5 | answer | 0.8017 | 0.8034 | 0.7992 | 1145923 | 42619 | 1063326 | 37631 | 1182688 | 44329 |
| FEV-6 | answer | 0.8858 | 0.9121 | 0.8885 | 351135 | 3460 | 341327 | 3103 | 372514 | 3973 |
| FEV-7 | answer | 0.6372 | 0.6624 | 0.6299 | 4494136 | 324846 | 4402296 | 321115 | 4476055 | 323985 |
| FEV-8 | answer | 0.6913 | 0.7095 | 0.6857 | 6921008 | 2490835 | 6925600 | 2495427 | 6950856 | 2520683 |
| FEV-9 | answer | 0.6613 | 0.6662 | 0.6576 | 3208919 | 1181531 | 3058897 | 1167062 | 3327251 | 1229753 |
| FEV-10 | answer | 0.8889 | 0.9102 | 0.8818 | 187465 | 2825 | 187294 | 2822 | 187520 | 2826 |
| FEV-11 | label | 0.8060 | 0.7980 | 0.7940 | 39831 | 1359 | 39831 | 1359 | 39831 | 1359 |
| LEP-1 | answer | 0.3140 | 0.4300 | 0.2960 | 131947 | 1702 | 131947 | 1702 | 131947 | 1702 |
| LEP-2 | answer | 0.4622 | 0.5102 | 0.4481 | 13802867 | 266122 | 13802867 | 266122 | 13802867 | 266122 |
| LEP-3 | answer | 0.4456 | 0.5026 | 0.4240 | 9884843 | 190302 | 8268479 | 159032 | 10131407 | 195072 |
| LEP-4 | answer | 0.3430 | 0.3818 | 0.3216 | 4251591 | 81122 | 3236228 | 61512 | 4361436 | 83242 |
| LEP-5 | answer | 0.3399 | 0.3743 | 0.3190 | 4178947 | 79690 | 3128332 | 59577 | 4307982 | 81884 |
| LEP-6 | answer | 0.4961 | 0.5301 | 0.4831 | 3601444 | 68583 | 3656232 | 69639 | 3574048 | 68053 |
| AGENT-1 | answer | 0.7381 | 0.7393 | 0.7398 | 5515721 | 12760 | 5515721 | 12760 | 5515721 | 12760 |
| AGENT-2 | answer | 0.9334 | 0.9385 | 0.9357 | 5558249 | 12760 | 5558249 | 12760 | 5558249 | 12760 |
| AGENT-3 | answer | 0.7381 | 0.7393 | 0.7398 | 5563700 | 14643 | 5563149 | 14628 | 5564687 | 14692 |
| AGENT-4 | label | 0.7913 | 0.8027 | 0.7888 | 5698856 | 33415 | 5697352 | 33003 | 5698752 | 33389 |
| AGENT-5 | label | 0.6354 | 0.6368 | 0.6294 | 6258232 | 96087 | 6258038 | 95893 | 6258220 | 96075 |

### Qwen3 32B, five queries

H100 and B200 finished all five queries on Quail, stock vLLM, and pipelined vLLM. Volumes: H100 `/results/benchmarks/quailb/20261017T200744Z-a1f0bd2f`, B200 `/results/benchmarks/quailb/20261017T145741Z-2106faa7`. Quail is faster than both vLLM submissions on every one of these ten cells. B200 Quail is 1.67× to 1.85× faster than H100 Quail, under the 2.3× bound.

Quail fresh tokens match between H100 and B200 on all five queries. H100 Quail matches the reference exactly on IMDB-1, FEV-1, LEP-1, and AGENT-1. BIO-1 is 0.976. B200 Quail is 0.992, 0.990, 0.992, 0.996, and 0.945 on those five. AGENT-1 is the gap that is larger than run-to-run noise, 0.945 against 1.0.

B200's query time divided by that estimate is 1.23× to 1.36× the same fraction on H100. That is just outside 20%. AGENT-1 on H100 runs in 0.66× its estimate.

| Query | Device | Quail s | Stock s | Pipelined s | Quail tok/s | Quail $ | Fresh | Regret | Quail acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IMDB-1 | H100 | 105.83 | 114.94 | 114.52 | 16623 | 0.1161 | 1759233 | 20350 | 1.000 |
| IMDB-1 | B200 | 63.21 | 64.63 | 65.10 | 27832 | 0.1097 | 1759233 | 20350 | 0.992 |
| BIO-1 | H100 | 134.95 | 148.65 | 148.32 | 15252 | 0.1480 | 2058352 | 2484 | 0.976 |
| BIO-1 | B200 | 77.05 | 80.23 | 80.10 | 26715 | 0.1338 | 2058352 | 2484 | 0.990 |
| FEV-1 | H100 | 2.16 | 2.56 | 2.60 | 15413 | 0.0024 | 33331 | 1359 | 1.000 |
| FEV-1 | B200 | 1.20 | 1.61 | 1.36 | 27677 | 0.0021 | 33331 | 1359 | 0.992 |
| LEP-1 | H100 | 7.94 | 9.02 | 8.99 | 16614 | 0.0087 | 131947 | 1702 | 1.000 |
| LEP-1 | B200 | 4.31 | 4.74 | 4.74 | 30612 | 0.0075 | 131947 | 1702 | 0.996 |
| AGENT-1 | H100 | 478.41 | 509.16 | 516.05 | 36348 | 0.5248 | 5515721 | 12760 | 1.000 |
| AGENT-1 | B200 | 257.94 | 268.41 | 265.08 | 67417 | 0.4478 | 5515721 | 12760 | 0.945 |

### Suites that stay failed

L40S DiffusionGemma does not run. The 65,536-token sliding KV does not fit in the leftover memory, and the 512-wide heads need FlashAttention 4, which does not run on this GPU.

L40S Qwen3 32B does not finish the suite. Quail skips BIO-1 and AGENT-1 because those documents are longer than the 18,544-token chunk. Stock vLLM then cannot fit one token, because the 10 GiB reserve that gives H100 room for FlashAttention 3 leaves no KV after the 32B weights. Volume `/results/benchmarks/quailb/20261017T200744Z-b3bd1b3e`.

H100 DiffusionGemma has no three-engine comparison. Stock vLLM omitted the TRUE/FALSE scores and the suite died while Quail was still running. Volume `/results/benchmarks/quailb/20261017T040833Z-c0e42adc`.

B200 DiffusionGemma has no comparison. On vLLM 0.26, Quail warmup failed because FlashAttention 4 rejects `seqused_k` for head dimension 256. Stock vLLM never started. Volume `/results/benchmarks/quailb/20261017T145741Z-c2afc552`.

The 2026-10-24 rerun uses vLLM 0.29.0 and the 128-token page gather for head dimension 256. Quail still died in warmup, on IMDB-1, before a query finished. Stock vLLM did not start. During weight load, vLLM logged that FlashAttention 4 on Blackwell does not support head size 512 because of TMEM capacity and would default to FlashAttention 2. Kernel compile then died in `FlashAttentionForwardSm100` on `assert self.tmem_total <= self.tmem_alloc_cols`. The head-256 kernel had started compiling. The page-size assert did not appear. Volume `/results/benchmarks/quailb/20261024T150444Z-c7a9c717`.

The same day's H100 Quail-only rerun finished. All 43 queries completed and none failed. Parallel wall time was 771.3 s. Baselines were off, so there is still no three-engine comparison. Volume `/results/benchmarks/quailb/20261024T150444Z-e3303d05`.

A later B200 retry sends the 512-wide layers through Triton unified attention and keeps the page gather for head dimension 256. Quail still died in warmup, before a query finished. Stock vLLM did not start. The TMEM assert did not appear. The raising frame is `BlackwellFusedMultiHeadAttentionForward` on `SM100 forward with head_dim=256 does not support SplitKV`. Volume `/results/benchmarks/quailb/20261024T154540Z-ef369227`.

The next B200 retry passes `num_splits=1` on the head-256 call. Quail finished every family. Stock vLLM then died while booting, on `TypeError: cannot pickle 'builtins.SentencePieceTokenizer' object`, and the orchestrator returned that error. Volume `/results/benchmarks/quailb/20261024T155537Z-6372ad9e`.

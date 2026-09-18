# QUAIL-B comparison

- All 33 queries use Qwen3's chat format with thinking disabled.
  Both methods use Qwen3 4B FP8, sf=0.1, lf=1, and one H100.
  Quail and pipelined stock vLLM share a physical GPU within each family.
  Pipelined stock vLLM advances documents through filter stages
  independently, then starts joins after filtering finishes.
- The measured run is on `quail-results`:
  `/results/benchmarks/quailb/family-runs/20260918T071546Z-all-chat/`.
  It reuses the completed BioDEX run after checking that its corpus,
  query definitions, prompt format, and reference answers match.
  BioDEX source: `/results/benchmarks/quailb/family-runs/20260918T060700Z-biodex-chat`.
  All other measurements are new. Earlier prompt formats are omitted.
- Model references use Qwen3 32B FP8 with thinking disabled.
  The collection is `gt_91df55461cea394013812087a6ca6625`.
  Reference join prompts put the first argument first. FEVER joins
  were regenerated after fixing automatic prompt reordering. Saved
  benchmark answers were rescored without changing timings.
  The existing FEVER annotation and LePaRD citation rules still apply.
  Saved-answer checks repeated 320 answers
  with 0 differences.
- Quail's planned limits are 110,376 tokens per chunk and 362,250
  resident KV tokens. Pipelined stock vLLM uses prefix caching,
  25,305 batched tokens, and
  4,096 sequences.
  Its measured KV capacity is 479,616 tokens.
  Both methods use the same planner's filter and join ordering rules.
- Quail is faster on 30 of 33 queries.
  The arithmetic mean speedup is 1.65x,
  the median is 1.50x, and the maximum is
  4.99x on BIO-2. Each query has equal weight.
  Speedup is pipelined stock vLLM time divided by Quail time.
  Query time excludes startup and result collection. Table throughput counts
  input documents for filters and evaluated pairs across stages for joins.
  GPU cost is query seconds / 3,600 * $3.9492.
- SoL means speed of light. It estimates ideal GPU time by dividing
  arithmetic and memory traffic by the hardware's peak rates. For each
  model component, it takes the larger time, then adds component times.
  It assumes ideal batching and unlimited retained KV. Matching token
  prefixes are computed once across requests, documents, and aliases.
  It excludes startup and software scheduling overhead and uses exact
  reference-label survivors. The supported join search uses left-deep
  plans. Different measured answers change the work, so the gap from
  SoL is not purely execution overhead. SoL has no measured accuracy.
- Fresh input tokens count every input position processed by a model
  forward pass. Repeated computation counts again. Recomputed KV tokens
  are fresh tokens minus the minimum for the run's actual requests with
  unlimited KV. They are included in fresh tokens, not added to them.
  The benchmark computes this minimum from saved answers after the run.
  Token throughput is total fresh input tokens divided by query seconds.
  It includes recomputation and excludes generated answer tokens.
  More recomputation can raise this rate without making a query faster.
- Answer agreement counts matching evaluated predicate answers. Output
  precision is the fraction of returned rows matching the reference.
  Output recall is the fraction of reference rows returned. Most join
  pairs can be negative, so high agreement can coexist with low recall.
  Quail's BIO-2 and BIO-3 recall is 10.70% and
  10.41%, respectively.
  Its IMDB-9 output recall is 0.0038%, and its FEV-8
  output precision is 0.0020%.
- The main PDF shows all queries with one metric per page. Each dataset
  PDF includes every query in that dataset. Input counts list each alias
  separately, before filtering. SoL is a horizontal line, not a measured
  bar. Latency labels show vLLM time divided by Quail time.
  Log scales are labeled; a dash marks zero.

[Open the main vector PDF](plots/quailb_main.pdf)

Figure: plots/quailb_main.pdf

SoL estimates on `quail-results`:
`/results/sol/2026-09-18-all-chat/sol_quailb_sf0.1.json`.

The download commands are in `reports/make_quailb_comparison_plots.py`.

## IMDB

[Open the IMDB vector PDF](plots/quailb_imdb.pdf)

Figure: plots/quailb_imdb.pdf

| Query | Input documents by alias and set |
|---|---|
| IMDB-1 | r (reviews) = 5,000 |
| IMDB-2 | r (reviews) = 5,000, a (aspects) = 12 |
| IMDB-3 | r (reviews) = 5,000, a (aspects) = 12 |
| IMDB-4 | r (reviews) = 5,000, a (aspects) = 12 |
| IMDB-5 | r (reviews) = 5,000, a (aspects) = 12 |
| IMDB-6 | r (reviews) = 5,000 |
| IMDB-7 | r (reviews) = 5,000 |
| IMDB-8 | r (reviews) = 5,000, a (aspects) = 12, a2 (aspects) = 12 |
| IMDB-9 | r1 (reviews) = 5,000, a1 (aspects) = 12, r2 (reviews) = 5,000, a2 (aspects) = 12 |
| IMDB-10 | r1 (reviews) = 5,000, a1 (aspects) = 12, r2 (reviews) = 5,000, a2 (aspects) = 12 |

| Query | Method | Seconds | Recomputed KV tokens | Fresh input tokens | Tokens/second | Throughput | Unit | $/query | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| IMDB-1 | Quail | 15.26 | 35,347 | 1,859,233 | 121,837.02 | 327.65 | docs/s | 0.01674 | 92.54 | 93.558 | 97.373 |
| IMDB-1 | Pipelined vLLM | 18.37 | 34,995 | 1,858,881 | 101,191.13 | 272.18 | docs/s | 0.02015 | 92.62 | 93.585 | 97.448 |
| IMDB-1 | SoL estimate | 6.995 | 0 (assumed) | 1,827,891 | 261,310.86 | 714.79 | docs/s | 0.00767 | Not measured | Not measured | Not measured |
| IMDB-2 | Quail | 26.39 | 35,347 | 3,014,233 | 114,218.76 | 2,273.59 | pairs/s | 0.02895 | 64.32 | 98.876 | 0.81542 |
| IMDB-2 | Pipelined vLLM | 34.89 | 61,390 | 3,040,276 | 87,138.89 | 1,719.69 | pairs/s | 0.03827 | 64.26 | 99.286 | 0.644 |
| IMDB-2 | SoL estimate | 11.483 | 0 (assumed) | 2,982,891 | 259,762.64 | 5,225.05 | pairs/s | 0.01260 | Not measured | Not measured | Not measured |
| IMDB-3 | Quail | 26.93 | 39,507 | 3,103,073 | 115,227.37 | 1,853.69 | pairs/s | 0.02954 | 66.56 | 97.605 | 0.93217 |
| IMDB-3 | Pipelined vLLM | 45.88 | 1,375,633 | 4,439,795 | 96,769.73 | 1,088.58 | pairs/s | 0.05033 | 66.46 | 96.094 | 0.70342 |
| IMDB-3 | SoL estimate | 11.644 | 0 (assumed) | 3,022,994 | 259,613.42 | 4,119.13 | pairs/s | 0.01277 | Not measured | Not measured | Not measured |
| IMDB-4 | Quail | 18.55 | 36,355 | 2,178,085 | 117,416.98 | 652.08 | pairs/s | 0.02035 | 66.21 | 91.304 | 0.64546 |
| IMDB-4 | Pipelined vLLM | 26.92 | 479,778 | 2,617,302 | 97,225.19 | 443.98 | pairs/s | 0.02953 | 66.31 | 91.667 | 0.50715 |
| IMDB-4 | SoL estimate | 8.602 | 0 (assumed) | 2,235,240 | 259,839.48 | 1,611.18 | pairs/s | 0.00944 | Not measured | Not measured | Not measured |
| IMDB-5 | Quail | 17.51 | 36,470 | 2,061,921 | 117,756.77 | 374.19 | pairs/s | 0.01921 | 70.33 | 88.235 | 0.63898 |
| IMDB-5 | Pipelined vLLM | 24.09 | 295,449 | 2,315,513 | 96,119.26 | 265.01 | pairs/s | 0.02643 | 70.56 | 89.655 | 0.55378 |
| IMDB-5 | SoL estimate | 8.377 | 0 (assumed) | 2,176,334 | 259,800.95 | 1,104.46 | pairs/s | 0.00919 | Not measured | Not measured | Not measured |
| IMDB-6 | Quail | 15.68 | 35,347 | 1,876,693 | 119,687.05 | 318.88 | docs/s | 0.01720 | 92.53 | 87.897 | 76.71 |
| IMDB-6 | Pipelined vLLM | 18.97 | 64,666 | 1,905,382 | 100,441.86 | 263.57 | docs/s | 0.02081 | 92.66 | 88.454 | 76.277 |
| IMDB-6 | SoL estimate | 7.240 | 0 (assumed) | 1,889,895 | 261,038.07 | 690.62 | docs/s | 0.00794 | Not measured | Not measured | Not measured |
| IMDB-7 | Quail | 15.97 | 35,924 | 1,898,667 | 118,889.61 | 313.09 | docs/s | 0.01752 | 91.73 | 88.462 | 62.646 |
| IMDB-7 | Pipelined vLLM | 20.36 | 162,065 | 2,023,593 | 99,390.62 | 245.58 | docs/s | 0.02233 | 91.84 | 88.91 | 61.349 |
| IMDB-7 | SoL estimate | 7.461 | 0 (assumed) | 1,945,805 | 260,787.57 | 670.13 | docs/s | 0.00819 | Not measured | Not measured | Not measured |
| IMDB-8 | Quail | 28.70 | 64,397 | 3,277,403 | 114,195.23 | 2,437.63 | pairs/s | 0.03148 | 74.21 | 70.844 | 0.33404 |
| IMDB-8 | Pipelined vLLM | 39.55 | 306,489 | 3,516,063 | 88,901.72 | 1,764.96 | pairs/s | 0.04339 | 74.22 | 72.754 | 0.29304 |
| IMDB-8 | SoL estimate | 16.449 | 0 (assumed) | 4,259,671 | 258,969.64 | 6,726.44 | pairs/s | 0.01804 | Not measured | Not measured | Not measured |
| IMDB-9 | Quail | 53.27 | 1,942,890 | 6,081,636 | 114,166.25 | 2,251.92 | pairs/s | 0.05844 | 71.98 | 68.887 | 0.0037753 |
| IMDB-9 | Pipelined vLLM | 66.65 | 2,172,515 | 6,222,872 | 93,366.42 | 1,722.49 | pairs/s | 0.07312 | 68.57 | 68.025 | 0.0025487 |
| IMDB-9 | SoL estimate | 23.066 | 0 (assumed) | 5,966,261 | 258,660.14 | 7,634.61 | pairs/s | 0.02530 | Not measured | Not measured | Not measured |
| IMDB-10 | Quail | 54.28 | 1,918,490 | 6,205,756 | 114,328.59 | 2,055.27 | pairs/s | 0.05955 | 72.94 | 68.308 | 0.0043301 |
| IMDB-10 | Pipelined vLLM | 81.20 | 3,458,744 | 7,675,663 | 94,527.87 | 1,320.96 | pairs/s | 0.08908 | 70.04 | 66.642 | 0.0027763 |
| IMDB-10 | SoL estimate | 22.415 | 0 (assumed) | 5,794,774 | 258,525.91 | 7,075.90 | pairs/s | 0.02459 | Not measured | Not measured | Not measured |

## BIO

[Open the BIO vector PDF](plots/quailb_bio.pdf)

Figure: plots/quailb_bio.pdf

| Query | Input documents by alias and set |
|---|---|
| BIO-1 | r (reports) = 500 |
| BIO-2 | r (reports) = 500, m (terms) = 1,127 |
| BIO-3 | r (reports) = 500, m (terms) = 1,127 |

BIO-3 filter survivors: 362 in the reference, 349 in Quail, and 350 in pipelined stock vLLM.

The cause of vLLM's lower BIO-2 time than its earlier raw-prompt
run remains unknown. The runs did not isolate prompt changes
from other execution changes.

| Query | Method | Seconds | Recomputed KV tokens | Fresh input tokens | Tokens/second | Throughput | Unit | $/query | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| BIO-1 | Quail | 22.34 | 3,981 | 2,068,352 | 92,585.14 | 22.38 | docs/s | 0.02451 | 88.60 | 93.696 | 90.331 |
| BIO-1 | Pipelined vLLM | 25.15 | 3,965 | 2,068,336 | 82,240.00 | 19.88 | docs/s | 0.02759 | 89.60 | 94.286 | 91.16 |
| BIO-1 | SoL estimate | 11.111 | 0 (assumed) | 2,064,371 | 185,800.65 | 45.00 | docs/s | 0.01219 | Not measured | Not measured | Not measured |
| BIO-2 | Quail | 187.01 | 3,981 | 15,451,352 | 82,623.13 | 3,013.21 | pairs/s | 0.20515 | 97.13 | 94.034 | 10.699 |
| BIO-2 | Pipelined vLLM | 933.23 | 157,341 | 15,604,712 | 16,721.19 | 603.82 | pairs/s | 1.02375 | 97.13 | 94.771 | 10.588 |
| BIO-2 | SoL estimate | 93.223 | 0 (assumed) | 15,447,371 | 165,703.12 | 6,044.63 | pairs/s | 0.10227 | Not measured | Not measured | Not measured |
| BIO-3 | Quail | 136.51 | 4,330 | 11,432,720 | 83,750.05 | 2,881.28 | pairs/s | 0.14975 | 96.58 | 90.405 | 10.408 |
| BIO-3 | Pipelined vLLM | 647.99 | 1,464,366 | 12,919,587 | 19,937.94 | 608.73 | pairs/s | 0.71085 | 96.58 | 90.062 | 10.487 |
| BIO-3 | SoL estimate | 69.241 | 0 (assumed) | 11,777,555 | 170,095.63 | 5,892.11 | pairs/s | 0.07596 | Not measured | Not measured | Not measured |

## FEV

[Open the FEV vector PDF](plots/quailb_fev.pdf)

Figure: plots/quailb_fev.pdf

| Query | Input documents by alias and set |
|---|---|
| FEV-1 | c (claims) = 500 |
| FEV-2 | c (claims) = 500, e (evidence) = 287 |
| FEV-3 | c (claims) = 500, e (evidence) = 287 |
| FEV-4 | c (claims) = 500, e (evidence) = 287 |
| FEV-5 | c (claims) = 500, e (evidence) = 287 |
| FEV-6 | c (claims) = 500, e (evidence) = 287 |
| FEV-7 | c (claims) = 500, e (evidence) = 287, e2 (evidence) = 287 |
| FEV-8 | c1 (claims) = 500, e1 (evidence) = 287, c2 (claims) = 500, e2 (evidence) = 287 |
| FEV-9 | c1 (claims) = 500, e1 (evidence) = 287, c2 (claims) = 500, e2 (evidence) = 287 |
| FEV-10 | c (claims) = 500, e (evidence) = 287 |

| Query | Method | Seconds | Recomputed KV tokens | Fresh input tokens | Tokens/second | Throughput | Unit | $/query | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| FEV-1 | Quail | 0.37 | 2,856 | 43,331 | 117,110.81 | 1,351.35 | docs/s | 0.00041 | 93.20 | 100 | 88.514 |
| FEV-1 | Pipelined vLLM | 0.57 | 2,856 | 43,331 | 76,019.30 | 877.19 | docs/s | 0.00063 | 94.20 | 100 | 90.203 |
| FEV-1 | SoL estimate | 0.155 | 0 (assumed) | 41,790 | 268,997.24 | 3,218.44 | docs/s | 0.00017 | Not measured | Not measured | Not measured |
| FEV-2 | Quail | 40.58 | 1,536 | 4,539,911 | 111,875.58 | 3,536.22 | pairs/s | 0.04452 | 99.24 | 22.727 | 50.71 |
| FEV-2 | Pipelined vLLM | 84.61 | 158,254 | 4,696,629 | 55,509.15 | 1,696.02 | pairs/s | 0.09282 | 99.23 | 22.549 | 51.318 |
| FEV-2 | SoL estimate | 18.015 | 0 (assumed) | 4,539,285 | 251,976.33 | 7,965.70 | pairs/s | 0.01976 | Not measured | Not measured | Not measured |
| FEV-3 | Quail | 22.20 | 4,963 | 2,484,698 | 111,923.33 | 3,387.12 | pairs/s | 0.02435 | 98.81 | 17.625 | 39.452 |
| FEV-3 | Pipelined vLLM | 41.77 | 96,231 | 2,628,487 | 62,927.63 | 1,834.55 | pairs/s | 0.04582 | 98.83 | 17.831 | 40.548 |
| FEV-3 | SoL estimate | 11.039 | 0 (assumed) | 2,785,890 | 252,376.35 | 7,695.88 | pairs/s | 0.01211 | Not measured | Not measured | Not measured |
| FEV-4 | Quail | 4.47 | 4,963 | 497,055 | 111,197.99 | 1,926.17 | pairs/s | 0.00490 | 99.88 | 100 | 38.462 |
| FEV-4 | Pipelined vLLM | 6.68 | 21,875 | 539,929 | 80,827.69 | 1,417.81 | pairs/s | 0.00733 | 99.54 | 27.5 | 42.308 |
| FEV-4 | SoL estimate | 2.169 | 0 (assumed) | 552,789 | 254,861.00 | 4,631.20 | pairs/s | 0.00238 | Not measured | Not measured | Not measured |
| FEV-5 | Quail | 12.93 | 5,118 | 1,443,055 | 111,605.18 | 3,140.76 | pairs/s | 0.01418 | 98.28 | 18.316 | 47.148 |
| FEV-5 | Pipelined vLLM | 25.44 | 104,214 | 1,578,761 | 62,058.22 | 1,637.26 | pairs/s | 0.02791 | 98.31 | 18.182 | 47.909 |
| FEV-5 | SoL estimate | 6.459 | 0 (assumed) | 1,632,083 | 252,664.28 | 7,240.20 | pairs/s | 0.00709 | Not measured | Not measured | Not measured |
| FEV-6 | Quail | 3.26 | 5,118 | 368,984 | 113,185.28 | 1,426.38 | pairs/s | 0.00358 | 99.80 | 100 | 52.632 |
| FEV-6 | Pipelined vLLM | 5.33 | 16,980 | 396,159 | 74,326.27 | 965.85 | pairs/s | 0.00585 | 99.30 | 29.73 | 57.895 |
| FEV-6 | SoL estimate | 1.574 | 0 (assumed) | 402,989 | 256,013.63 | 3,513.14 | pairs/s | 0.00173 | Not measured | Not measured | Not measured |
| FEV-7 | Quail | 61.95 | 139,154 | 6,914,120 | 111,608.07 | 3,511.64 | pairs/s | 0.06796 | 96.55 | 0.0092081 | 2.9167 |
| FEV-7 | Pipelined vLLM | 117.19 | 373,680 | 7,128,556 | 60,829.05 | 1,849.01 | pairs/s | 0.12856 | 96.25 | 0.008519 | 2.9167 |
| FEV-7 | SoL estimate | 21.556 | 0 (assumed) | 5,430,436 | 251,916.87 | 7,908.44 | pairs/s | 0.02365 | Not measured | Not measured | Not measured |
| FEV-8 | Quail | 110.65 | 3,822,688 | 12,349,723 | 111,610.69 | 3,555.95 | pairs/s | 0.12138 | 97.15 | 0.0019768 | 1.8248 |
| FEV-8 | Pipelined vLLM | 214.42 | 4,377,805 | 12,935,522 | 60,327.96 | 1,843.70 | pairs/s | 0.23522 | 96.90 | 0.0015845 | 1.6423 |
| FEV-8 | SoL estimate | 29.793 | 0 (assumed) | 7,506,061 | 251,944.19 | 7,987.83 | pairs/s | 0.03268 | Not measured | Not measured | Not measured |
| FEV-9 | Quail | 34.07 | 1,257,913 | 3,819,910 | 112,119.46 | 3,278.54 | pairs/s | 0.03737 | 95.01 | 0.0023412 | 5.3097 |
| FEV-9 | Pipelined vLLM | 66.45 | 1,416,631 | 4,073,706 | 61,304.83 | 1,738.33 | pairs/s | 0.07290 | 94.60 | 0.0017721 | 4.4248 |
| FEV-9 | SoL estimate | 9.164 | 0 (assumed) | 2,312,078 | 252,306.84 | 7,266.24 | pairs/s | 0.01005 | Not measured | Not measured | Not measured |
| FEV-10 | Quail | 1.77 | 5,106 | 204,361 | 115,458.19 | 88.70 | pairs/s | 0.00194 | 94.92 | 99.099 | 89.431 |
| FEV-10 | Pipelined vLLM | 3.01 | 6,203 | 278,886 | 92,653.16 | 53.16 | pairs/s | 0.00330 | 94.09 | 84.173 | 95.122 |
| FEV-10 | SoL estimate | 0.780 | 0 (assumed) | 203,171 | 260,567.09 | 215.46 | pairs/s | 0.00086 | Not measured | Not measured | Not measured |

## LEP

[Open the LEP vector PDF](plots/quailb_lep.pdf)

Figure: plots/quailb_lep.pdf

| Query | Input documents by alias and set |
|---|---|
| LEP-1 | d (citation_contexts) = 500 |
| LEP-2 | d (citation_contexts) = 500, s (citation_passages) = 433 |
| LEP-3 | d (citation_contexts) = 500, s (citation_passages) = 433 |
| LEP-4 | d (citation_contexts) = 500, s (citation_passages) = 433 |
| LEP-5 | d (citation_contexts) = 500, s (citation_passages) = 433 |
| LEP-6 | d (citation_contexts) = 500, s (citation_passages) = 433 |
| LEP-7 | d (citation_contexts) = 500, s (citation_passages) = 433 |
| LEP-8 | d (citation_contexts) = 500 |

| Query | Method | Seconds | Recomputed KV tokens | Fresh input tokens | Tokens/second | Throughput | Unit | $/query | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| LEP-1 | Quail | 1.14 | 3,199 | 141,947 | 124,514.91 | 438.60 | docs/s | 0.00125 | 80.80 | 16.071 | 90 |
| LEP-1 | Pipelined vLLM | 1.48 | 3,119 | 141,867 | 95,856.08 | 337.84 | docs/s | 0.00162 | 80.60 | 16.522 | 95 |
| LEP-1 | SoL estimate | 0.528 | 0 (assumed) | 139,593 | 264,135.64 | 946.09 | docs/s | 0.00058 | Not measured | Not measured | Not measured |
| LEP-2 | Quail | 144.14 | 3,199 | 17,052,947 | 118,308.22 | 1,502.01 | pairs/s | 0.15812 | 99.60 | 17.73 | 20 |
| LEP-2 | Pipelined vLLM | 176.26 | 64,175 | 17,113,923 | 97,094.76 | 1,228.30 | pairs/s | 0.19336 | 99.58 | 17.089 | 21.6 |
| LEP-2 | SoL estimate | 65.742 | 0 (assumed) | 17,050,593 | 259,355.94 | 3,293.17 | pairs/s | 0.07212 | Not measured | Not measured | Not measured |
| LEP-3 | Quail | 33.48 | 3,311 | 3,937,963 | 117,621.36 | 1,448.51 | pairs/s | 0.03673 | 99.47 | 2.5641 | 10 |
| LEP-3 | Pipelined vLLM | 40.83 | 54,061 | 4,090,389 | 100,180.97 | 1,219.57 | pairs/s | 0.04479 | 99.45 | 2.381 | 10 |
| LEP-3 | SoL estimate | 3.206 | 0 (assumed) | 817,453 | 254,982.55 | 2,701.25 | pairs/s | 0.00352 | Not measured | Not measured | Not measured |
| LEP-4 | Quail | 5.90 | 3,215 | 688,491 | 116,693.39 | 1,174.24 | pairs/s | 0.00647 | 97.60 | 0 | 0 |
| LEP-4 | Pipelined vLLM | 9.04 | 16,605 | 871,455 | 96,399.89 | 1,005.86 | pairs/s | 0.00992 | 98.07 | 0 | 0 |
| LEP-4 | SoL estimate | 2.412 | 0 (assumed) | 615,375 | 255,133.43 | 2,513.29 | pairs/s | 0.00265 | Not measured | Not measured | Not measured |
| LEP-5 | Quail | 2.43 | 3,203 | 279,005 | 114,816.87 | 712.76 | pairs/s | 0.00267 | 94.97 | 100 | 100 |
| LEP-5 | Pipelined vLLM | 2.39 | 6,182 | 179,917 | 75,279.08 | 181.17 | pairs/s | 0.00262 | 89.70 | 100 | 100 |
| LEP-5 | SoL estimate | 0.521 | 0 (assumed) | 137,726 | 264,148.80 | 0.00 | pairs/s | 0.00057 | Not measured | Not measured | Not measured |
| LEP-6 | Quail | 1.27 | 3,199 | 143,133 | 112,703.15 | 0.00 | pairs/s | 0.00139 | 87.44 | 100 | 100 |
| LEP-6 | Pipelined vLLM | 2.06 | 3,852 | 143,440 | 69,631.07 | 0.00 | pairs/s | 0.00226 | 87.93 | 100 | 100 |
| LEP-6 | SoL estimate | 0.522 | 0 (assumed) | 137,899 | 264,105.68 | 0.00 | pairs/s | 0.00057 | Not measured | Not measured | Not measured |
| LEP-7 | Quail | 3.31 | 6,082 | 393,169 | 118,782.18 | 754.08 | pairs/s | 0.00363 | 89.04 | 0 | 0 |
| LEP-7 | Pipelined vLLM | 5.32 | 12,710 | 480,588 | 90,336.09 | 659.21 | pairs/s | 0.00584 | 91.57 | 0 | 0 |
| LEP-7 | SoL estimate | 2.381 | 0 (assumed) | 610,766 | 256,481.75 | 2,216.41 | pairs/s | 0.00261 | Not measured | Not measured | Not measured |
| LEP-8 | Quail | 1.23 | 3,199 | 143,133 | 116,368.29 | 406.50 | docs/s | 0.00135 | 87.44 | 100 | 100 |
| LEP-8 | Pipelined vLLM | 2.04 | 3,852 | 143,440 | 70,313.73 | 245.10 | docs/s | 0.00224 | 87.93 | 100 | 100 |
| LEP-8 | SoL estimate | 0.522 | 0 (assumed) | 137,899 | 264,105.68 | 957.61 | docs/s | 0.00057 | Not measured | Not measured | Not measured |

## AGENT

[Open the AGENT vector PDF](plots/quailb_agent.pdf)

Figure: plots/quailb_agent.pdf

| Query | Input documents by alias and set |
|---|---|
| AGENT-1 | t (agent_traces) = 1,772 |
| AGENT-2 | t (agent_traces) = 1,772 |

| Query | Method | Seconds | Recomputed KV tokens | Fresh input tokens | Tokens/second | Throughput | Unit | $/query | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| AGENT-1 | Quail | 238.98 | 11,891,465 | 17,424,553 | 72,912.18 | 7.41 | docs/s | 0.26216 | 72.40 | 69.787 | 64.062 |
| AGENT-1 | Pipelined vLLM | 99.72 | 24,921 | 5,558,009 | 55,736.15 | 17.77 | docs/s | 0.10939 | 71.78 | 68.457 | 64.714 |
| AGENT-1 | SoL estimate | 47.762 | 0 (assumed) | 5,533,088 | 115,847.57 | 37.10 | docs/s | 0.05239 | Not measured | Not measured | Not measured |
| AGENT-2 | Quail | 239.73 | 11,891,465 | 17,467,081 | 72,861.47 | 7.39 | docs/s | 0.26298 | 93.06 | 83.264 | 99.5 |
| AGENT-2 | Pipelined vLLM | 100.29 | 24,921 | 5,600,537 | 55,843.42 | 17.67 | docs/s | 0.11002 | 92.66 | 82.459 | 99.5 |
| AGENT-2 | SoL estimate | 48.168 | 0 (assumed) | 5,575,616 | 115,754.62 | 36.79 | docs/s | 0.05284 | Not measured | Not measured | Not measured |

# QUAIL-B comparison from saved results

- The main PDF covers all 32 queries with grouped bars and one metric per page.
  Its final page lists input document counts. Each dataset PDF has a page
  of four bar charts and a separate input-count page. Text and marks remain
  vector content when zoomed. The PNGs below are first-page previews.
- The five dataset plots use the same
  method colors and definitions for latency, recomputed KV tokens, fresh
  input tokens, accuracy, and input document counts for every relation alias.
- The other 31 queries reuse the original measurements from September 5, 2026.
  No inference was rerun for this report. These are historical measurements,
  not a measurement of shared retention on every query.
- The setup was Qwen3 4B FP8, sf=0.1, lf=1, and one H100 per configuration.
  Quail and the vLLM configurations shared a physical GPU within each family.
  SGLang used a separate GPU. Stock vLLM used operator-at-a-time submission.
- FEV-9 now has four filters. The old suite had only one filter for FEV-9,
  so its old measurements are excluded. The current Quail measurement appears
  in both the main plot and the FEVER plot. Missing baselines are labeled.
  The [retention report](2026-09-05-shared-kv-retention.md) gives the change details.
- The prediction for this update was that scoring and plotting would need no
  inference. We reused all 124 saved configurations for the other 31 queries.
- In these saved measurements, Quail was faster than stock vLLM on 29
  of 31 comparable queries.
- A horizontal line across each query's bar group shows its SoL estimate.
  SoL models ideal computation and memory traffic with unlimited prefix KV.
  It credits matching token prefixes across requests, documents, and aliases.
  It uses exact reference-label survivors and searches supported left-deep
  join plans. Different answers can change the work done by measured runs,
  so the gap from SoL is not purely execution overhead.
- SoL uses the distinct-prefix estimate, not the per-document-only estimate.
  Its latency, fresh-token count, and zero prefix recomputation are estimates.
  No accuracy is assigned to SoL because it is not a measured model run.
  Matching document prefixes are reusable; a partner suffix after a different
  anchor context is not an identical prefix and is still computed.
- The earlier SoL file used the old FEV-9 definition. We recalculated only
  FEV-9 on the CPU from saved labels and corpus rows. The other 31 estimates
  are unchanged. No GPU inference was run.
- FEV-9 SoL is 5.821 seconds with shared-prefix reuse,
  compared with 6.337 seconds with reuse only
  within each document. These estimates use reference-label survivors.
- Answer agreement measures evaluated calls against saved Qwen3 32B labels.
  Each method can evaluate different calls after its filters and joins.
  Output precision is the fraction of returned rows matching the reference.
  Output recall is the fraction of reference rows returned. High answer
  agreement can coexist with poor final output precision.
- Query time excludes startup. Throughput counts input documents for filters
  and evaluated document pairs across all stages for joins. GPU cost is query
  seconds divided by 3,600 and multiplied by $3.9492.
- Fresh input tokens are input token positions processed by a model forward
  pass instead of read from existing KV. They include document and prompt
  suffix tokens, and any repeated computation after KV becomes unavailable.
  A repeated token counts again. This is not a count of unique text or
  generated answers. Recomputed KV tokens are part of the fresh-token total.
- Recomputed KV is the saved `regret_tokens` total for reusable prefixes
  of documents or anchors already computed earlier in the query. This uses
  per-document accounting, not the separate distinct-prefix metric.
  Token and latency plots use a log scale when positive values span more
  than one order of magnitude. Recomputed KV retains a linear region to
  include zero. A dash marks zero; x marks an unavailable measurement.
- Document counts come from the saved corpus manifest and describe inputs
  before filtering. Repeated aliases each list their full input count.
  The report tables also show throughput, GPU cost, and final output quality.
- FEV-9 agrees with the reference on 67.77% of evaluated answers. Its final
  output matches only 5 reference rows out of 149,783,486 returned rows.
  The reference has 11 rows, so output precision is approximately 0.00000334%
  and recall is 45.45%. The retention change preserved all answers.

[Open the main vector PDF](plots/quailb_main.pdf)

[![QUAIL-B latency preview](plots/quailb_main.png)](plots/quailb_main.pdf)

Figure: plots/quailb_main.png

Source manifest on `quail-results`: `/results/benchmarks/quailb/family-runs/20260905T021527Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/manifest.json`.

Current FEV-9 source on `quail-results`: `/results/ablations/shared-kv-retention-20260906T054932Z/`.

SoL estimates on `quail-results`: `/results/sol/2026-09-06-quailb-prefix-reuse.json`.

The FEV-9 recalculation is also saved separately at `/results/sol/2026-09-06-fev9-prefix-reuse.json`.

Corpus counts on `quail-results`: `/results/ground_truth/quailb/schema_v1/corpora/c_1aa2c4f0d0b6c816fd37aa5748c33341/manifest.json`.

The manifest lists all four source suite paths. The download commands are
in `reports/make_quailb_comparison_plots.py`.

## IMDB

[Open the IMDB vector PDF](plots/quailb_imdb.pdf)

[![IMDB preview](plots/quailb_imdb.png)](plots/quailb_imdb.pdf)

Figure: plots/quailb_imdb.png

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

| Query | Method | Seconds | Recomputed KV tokens | Fresh input tokens | Throughput | Unit | $/query | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| IMDB-1 | Quail | 14.25 | 0 | 1,759,232 | 350.88 | docs/s | 0.01563 | 89.68 | 89.817 | 98.252 |
| IMDB-1 | Stock vLLM | 17.11 | 0 | 1,758,944 | 292.23 | docs/s | 0.01877 | 89.52 | 89.781 | 98.077 |
| IMDB-1 | Pipelined vLLM | 17.03 | 0 | 1,758,944 | 293.60 | docs/s | 0.01868 | 89.52 | 89.781 | 98.077 |
| IMDB-1 | Pipelined SGLang | 18.58 | 0 | 1,759,232 | 269.11 | docs/s | 0.02038 | 91.08 | 92.057 | 97.253 |
| IMDB-1 | SoL estimate | 6.653 | 0 (assumed) | 1,740,485 | 751.59 | docs/s | 0.00730 | Not measured | Not measured | Not measured |
| IMDB-2 | Quail | 21.29 | 0 | 2,419,232 | 2,818.22 | pairs/s | 0.02336 | 78.50 | 72.543 | 43.494 |
| IMDB-2 | Stock vLLM | 26.68 | 0 | 2,439,624 | 2,248.88 | pairs/s | 0.02927 | 78.30 | 69.563 | 46.864 |
| IMDB-2 | Pipelined vLLM | 26.96 | 0 | 2,439,624 | 2,225.52 | pairs/s | 0.02958 | 78.30 | 69.563 | 46.864 |
| IMDB-2 | Pipelined SGLang | 54.37 | 130,384 | 2,573,144 | 1,103.55 | pairs/s | 0.05964 | 79.23 | 70.467 | 50.857 |
| IMDB-2 | SoL estimate | 9.211 | 0 (assumed) | 2,400,485 | 6,513.89 | pairs/s | 0.01010 | Not measured | Not measured | Not measured |
| IMDB-3 | Quail | 32.35 | 1,217,732 | 3,778,504 | 1,624.73 | pairs/s | 0.03549 | 79.16 | 66.549 | 44.329 |
| IMDB-3 | Stock vLLM | 39.23 | 1,326,432 | 3,933,798 | 1,337.96 | pairs/s | 0.04304 | 78.96 | 63.436 | 47.125 |
| IMDB-3 | Pipelined vLLM | 39.32 | 1,326,432 | 3,933,798 | 1,334.89 | pairs/s | 0.04313 | 78.96 | 63.436 | 47.125 |
| IMDB-3 | Pipelined SGLang | 65.79 | 1,411,856 | 3,994,150 | 771.55 | pairs/s | 0.07217 | 79.94 | 65.386 | 51.218 |
| IMDB-3 | SoL estimate | 9.495 | 0 (assumed) | 2,473,217 | 5,060.17 | pairs/s | 0.01042 | Not measured | Not measured | Not measured |
| IMDB-4 | Quail | 19.99 | 361,440 | 2,365,616 | 754.58 | pairs/s | 0.02193 | 77.76 | 59.54 | 37.707 |
| IMDB-4 | Stock vLLM | 30.74 | 1,061,328 | 3,091,846 | 496.94 | pairs/s | 0.03372 | 77.90 | 57.771 | 41.655 |
| IMDB-4 | Pipelined vLLM | 25.82 | 533,376 | 2,563,494 | 591.63 | pairs/s | 0.02832 | 77.90 | 57.771 | 41.655 |
| IMDB-4 | Pipelined SGLang | 34.13 | 502,123 | 2,495,093 | 403.63 | pairs/s | 0.03744 | 79.60 | 62.222 | 44.108 |
| IMDB-4 | SoL estimate | 7.519 | 0 (assumed) | 1,960,727 | 1,640.73 | pairs/s | 0.00825 | Not measured | Not measured | Not measured |
| IMDB-5 | Quail | 17.76 | 188,587 | 2,118,230 | 491.22 | pairs/s | 0.01948 | 80.17 | 60.501 | 35.794 |
| IMDB-5 | Stock vLLM | 29.56 | 760,512 | 2,716,776 | 302.44 | pairs/s | 0.03243 | 80.33 | 58.53 | 40.382 |
| IMDB-5 | Pipelined vLLM | 24.94 | 490,496 | 2,446,376 | 358.46 | pairs/s | 0.02736 | 80.33 | 58.522 | 40.41 |
| IMDB-5 | Pipelined SGLang | 27.52 | 270,687 | 2,186,065 | 279.07 | pairs/s | 0.03019 | 81.92 | 62.467 | 40.98 |
| IMDB-5 | SoL estimate | 7.405 | 0 (assumed) | 1,930,916 | 1,082.48 | pairs/s | 0.00812 | Not measured | Not measured | Not measured |
| IMDB-6 | Quail | 14.62 | 0 | 1,774,145 | 342.00 | docs/s | 0.01604 | 92.03 | 73.27 | 89.591 |
| IMDB-6 | Stock vLLM | 22.75 | 561,136 | 2,346,479 | 219.78 | docs/s | 0.02496 | 91.98 | 72.506 | 89.786 |
| IMDB-6 | Pipelined vLLM | 17.84 | 33,184 | 1,818,127 | 280.27 | docs/s | 0.01957 | 91.98 | 72.506 | 89.786 |
| IMDB-6 | Pipelined SGLang | 19.86 | 9,963 | 1,781,101 | 251.76 | docs/s | 0.02179 | 92.73 | 77.787 | 86.868 |
| IMDB-6 | SoL estimate | 6.780 | 0 (assumed) | 1,772,603 | 737.51 | docs/s | 0.00744 | Not measured | Not measured | Not measured |
| IMDB-7 | Quail | 14.81 | 0 | 1,796,602 | 337.61 | docs/s | 0.01625 | 92.13 | 72.352 | 78.743 |
| IMDB-7 | Stock vLLM | 25.23 | 760,512 | 2,574,289 | 198.18 | docs/s | 0.02768 | 92.18 | 71.812 | 80.09 |
| IMDB-7 | Pipelined vLLM | 20.57 | 278,000 | 2,091,393 | 243.07 | docs/s | 0.02257 | 92.18 | 71.812 | 80.09 |
| IMDB-7 | Pipelined SGLang | 20.00 | 19,471 | 1,809,965 | 250.00 | docs/s | 0.02194 | 92.13 | 76.719 | 73.503 |
| IMDB-7 | SoL estimate | 6.922 | 0 (assumed) | 1,808,672 | 722.34 | docs/s | 0.00759 | Not measured | Not measured | Not measured |
| IMDB-8 | Quail | 26.54 | 0 | 2,918,999 | 3,458.48 | pairs/s | 0.02911 | 68.47 | 20.407 | 27.098 |
| IMDB-8 | Stock vLLM | 40.15 | 860,032 | 3,773,186 | 2,293.60 | pairs/s | 0.04404 | 67.83 | 19.224 | 28.623 |
| IMDB-8 | Pipelined vLLM | 39.88 | 860,032 | 3,773,186 | 2,309.13 | pairs/s | 0.04375 | 67.83 | 19.224 | 28.623 |
| IMDB-8 | Pipelined SGLang | 80.49 | 1,043,552 | 3,953,343 | 1,135.59 | pairs/s | 0.08830 | 68.00 | 19.894 | 31.515 |
| IMDB-8 | SoL estimate | 11.903 | 0 (assumed) | 3,092,768 | 8,772.07 | pairs/s | 0.01306 | Not measured | Not measured | Not measured |
| IMDB-9 | Quail | 47.61 | 0 | 5,338,231 | 3,188.15 | pairs/s | 0.05223 | 72.43 | 17.624 | 13.171 |
| IMDB-9 | Stock vLLM | 64.55 | 860,032 | 6,212,810 | 2,356.13 | pairs/s | 0.07081 | 71.96 | 16.224 | 14.712 |
| IMDB-9 | Pipelined vLLM | 64.83 | 860,032 | 6,212,810 | 2,345.95 | pairs/s | 0.07112 | 71.96 | 16.224 | 14.712 |
| IMDB-9 | Pipelined SGLang | 138.14 | 1,173,936 | 6,526,487 | 1,096.02 | pairs/s | 0.15154 | 72.46 | 16.636 | 17.024 |
| IMDB-9 | SoL estimate | 16.376 | 0 (assumed) | 4,250,762 | 10,700.37 | pairs/s | 0.01796 | Not measured | Not measured | Not measured |
| IMDB-10 | Quail | 59.71 | 1,217,732 | 6,787,356 | 2,522.19 | pairs/s | 0.06550 | 73.93 | 16.265 | 13.507 |
| IMDB-10 | Stock vLLM | 79.64 | 2,186,464 | 7,706,984 | 1,815.37 | pairs/s | 0.08737 | 72.11 | 14.876 | 14.885 |
| IMDB-10 | Pipelined vLLM | 78.09 | 2,186,464 | 7,706,984 | 1,851.40 | pairs/s | 0.08566 | 72.11 | 14.876 | 14.885 |
| IMDB-10 | Pipelined SGLang | 148.30 | 2,455,408 | 7,947,493 | 958.62 | pairs/s | 0.16269 | 72.53 | 15.515 | 17.219 |
| IMDB-10 | SoL estimate | 15.732 | 0 (assumed) | 4,080,500 | 9,691.37 | pairs/s | 0.01726 | Not measured | Not measured | Not measured |

## BIO

[Open the BIO vector PDF](plots/quailb_bio.pdf)

[![BIO preview](plots/quailb_bio.png)](plots/quailb_bio.pdf)

Figure: plots/quailb_bio.png

| Query | Input documents by alias and set |
|---|---|
| BIO-1 | r (reports) = 500 |
| BIO-2 | r (reports) = 500, m (terms) = 1,127 |
| BIO-3 | r (reports) = 500, m (terms) = 1,127 |

| Query | Method | Seconds | Recomputed KV tokens | Fresh input tokens | Throughput | Unit | $/query | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| BIO-1 | Quail | 21.64 | 0 | 2,057,345 | 23.11 | docs/s | 0.02374 | 93.60 | 100 | 89.542 |
| BIO-1 | Stock vLLM | 25.94 | 0 | 2,057,329 | 19.28 | docs/s | 0.02846 | 94.00 | 100 | 90.196 |
| BIO-1 | Pipelined vLLM | 25.55 | 0 | 2,057,329 | 19.57 | docs/s | 0.02803 | 94.00 | 100 | 90.196 |
| BIO-1 | Pipelined SGLang | 24.87 | 0 | 2,057,329 | 20.10 | docs/s | 0.02728 | 94.20 | 100 | 90.523 |
| BIO-1 | SoL estimate | 11.049 | 0 (assumed) | 2,054,868 | 45.25 | docs/s | 0.01212 | Not measured | Not measured | Not measured |
| BIO-2 | Quail | 129.38 | 0 | 10,374,345 | 4,355.39 | pairs/s | 0.14193 | 81.83 | 14.167 | 85.959 |
| BIO-2 | Stock vLLM | 958.19 | 0 | 10,526,615 | 588.09 | pairs/s | 1.05113 | 80.95 | 13.625 | 86.283 |
| BIO-2 | Pipelined vLLM | 932.16 | 0 | 10,526,615 | 604.51 | pairs/s | 1.02258 | 80.95 | 13.625 | 86.283 |
| BIO-2 | Pipelined SGLang | 1028.46 | 385,312 | 11,241,655 | 547.91 | pairs/s | 1.12822 | 82.04 | 14.271 | 85.578 |
| BIO-2 | SoL estimate | 62.002 | 0 (assumed) | 10,371,868 | 9,088.45 | pairs/s | 0.06802 | Not measured | Not measured | Not measured |
| BIO-3 | Quail | 89.92 | 919,409 | 7,547,348 | 3,434.14 | pairs/s | 0.09864 | 82.54 | 14.733 | 79.917 |
| BIO-3 | Stock vLLM | 515.30 | 1,079,840 | 7,875,694 | 603.63 | pairs/s | 0.56528 | 81.23 | 13.912 | 81.171 |
| BIO-3 | Pipelined vLLM | 510.91 | 1,079,840 | 7,875,694 | 608.82 | pairs/s | 0.56047 | 81.23 | 13.912 | 81.171 |
| BIO-3 | Pipelined SGLang | 493.74 | 1,285,376 | 8,273,023 | 632.27 | pairs/s | 0.54163 | 82.22 | 14.536 | 81.012 |
| BIO-3 | SoL estimate | 43.087 | 0 (assumed) | 7,159,254 | 8,003.77 | pairs/s | 0.04727 | Not measured | Not measured | Not measured |

## FEV

[Open the FEV vector PDF](plots/quailb_fev.pdf)

[![FEV preview](plots/quailb_fev.png)](plots/quailb_fev.pdf)

Figure: plots/quailb_fev.png

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

| Query | Method | Seconds | Recomputed KV tokens | Fresh input tokens | Throughput | Unit | $/query | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| FEV-1 | Quail | 0.28 | 0 | 33,331 | 1,785.71 | docs/s | 0.00031 | 85.00 | 80.609 | 98.311 |
| FEV-1 | Stock vLLM | 0.45 | 0 | 33,331 | 1,111.11 | docs/s | 0.00049 | 84.00 | 78.877 | 99.662 |
| FEV-1 | Pipelined vLLM | 0.41 | 0 | 33,331 | 1,219.51 | docs/s | 0.00045 | 84.00 | 78.877 | 99.662 |
| FEV-1 | Pipelined SGLang | 0.88 | 0 | 33,331 | 568.18 | docs/s | 0.00097 | 89.20 | 85.174 | 98.986 |
| FEV-1 | SoL estimate | 0.121 | 0 (assumed) | 32,498 | 4,139.22 | docs/s | 0.00013 | Not measured | Not measured | Not measured |
| FEV-2 | Quail | 28.46 | 0 | 3,245,254 | 5,042.16 | pairs/s | 0.03122 | 82.58 | 1.1787 | 95.82 |
| FEV-2 | Stock vLLM | 71.39 | 0 | 3,350,613 | 2,010.09 | pairs/s | 0.07831 | 82.31 | 1.1645 | 96.141 |
| FEV-2 | Pipelined vLLM | 70.02 | 0 | 3,350,613 | 2,049.41 | pairs/s | 0.07681 | 82.31 | 1.1645 | 96.141 |
| FEV-2 | Pipelined SGLang | 105.30 | 18,320 | 3,438,341 | 1,362.77 | pairs/s | 0.11551 | 84.58 | 1.3339 | 96.141 |
| FEV-2 | SoL estimate | 12.845 | 0 (assumed) | 3,244,940 | 11,171.71 | pairs/s | 0.01409 | Not measured | Not measured | Not measured |
| FEV-3 | Quail | 21.06 | 0 | 2,404,957 | 4,919.61 | pairs/s | 0.02310 | 81.20 | 0.89349 | 95.135 |
| FEV-3 | Stock vLLM | 62.62 | 0 | 2,562,915 | 1,714.12 | pairs/s | 0.06869 | 80.63 | 0.85189 | 96.757 |
| FEV-3 | Pipelined vLLM | 55.74 | 0 | 2,562,915 | 1,925.69 | pairs/s | 0.06115 | 80.63 | 0.85189 | 96.757 |
| FEV-3 | Pipelined SGLang | 73.94 | 18,320 | 2,437,052 | 1,335.24 | pairs/s | 0.08111 | 82.93 | 1.0309 | 95.135 |
| FEV-3 | SoL estimate | 7.943 | 0 (assumed) | 2,010,333 | 10,694.58 | pairs/s | 0.00871 | Not measured | Not measured | Not measured |
| FEV-4 | Quail | 4.66 | 0 | 531,453 | 2,894.64 | pairs/s | 0.00511 | 89.31 | 0.73677 | 78.571 |
| FEV-4 | Stock vLLM | 8.26 | 0 | 575,830 | 1,772.03 | pairs/s | 0.00906 | 90.50 | 0.76655 | 78.571 |
| FEV-4 | Pipelined vLLM | 8.03 | 0 | 575,830 | 1,822.79 | pairs/s | 0.00881 | 90.50 | 0.76655 | 78.571 |
| FEV-4 | Pipelined SGLang | 11.02 | 17,567 | 538,441 | 1,119.87 | pairs/s | 0.01209 | 92.70 | 1.174 | 78.571 |
| FEV-4 | SoL estimate | 1.868 | 0 (assumed) | 477,123 | 5,991.95 | pairs/s | 0.00205 | Not measured | Not measured | Not measured |
| FEV-5 | Quail | 13.35 | 0 | 1,515,283 | 4,624.04 | pairs/s | 0.01464 | 79.57 | 1.0522 | 96.429 |
| FEV-5 | Stock vLLM | 32.66 | 44,000 | 1,657,580 | 1,969.63 | pairs/s | 0.03583 | 79.01 | 0.99111 | 97.143 |
| FEV-5 | Pipelined vLLM | 31.73 | 43,744 | 1,657,324 | 2,027.36 | pairs/s | 0.03481 | 79.01 | 0.99111 | 97.143 |
| FEV-5 | Pipelined SGLang | 43.43 | 18,720 | 1,519,821 | 1,338.61 | pairs/s | 0.04764 | 80.73 | 1.1812 | 96.429 |
| FEV-5 | SoL estimate | 4.743 | 0 (assumed) | 1,202,271 | 9,923.87 | pairs/s | 0.00520 | Not measured | Not measured | Not measured |
| FEV-6 | Quail | 3.56 | 0 | 398,331 | 2,257.58 | pairs/s | 0.00391 | 88.44 | 0.998 | 76.923 |
| FEV-6 | Stock vLLM | 6.62 | 0 | 426,594 | 1,325.08 | pairs/s | 0.00726 | 89.54 | 1.0215 | 76.923 |
| FEV-6 | Pipelined vLLM | 6.27 | 0 | 426,594 | 1,399.04 | pairs/s | 0.00688 | 89.54 | 1.0215 | 76.923 |
| FEV-6 | Pipelined SGLang | 8.19 | 15,071 | 407,798 | 887.30 | pairs/s | 0.00898 | 91.68 | 1.5267 | 76.923 |
| FEV-6 | SoL estimate | 1.375 | 0 (assumed) | 352,709 | 4,510.24 | pairs/s | 0.00151 | Not measured | Not measured | Not measured |
| FEV-7 | Quail | 53.46 | 0 | 6,103,058 | 5,030.28 | pairs/s | 0.05865 | 64.66 | 0.00029045 | 15.652 |
| FEV-7 | Stock vLLM | 141.44 | 0 | 6,255,630 | 1,899.26 | pairs/s | 0.15516 | 63.95 | 0.0003172 | 17.391 |
| FEV-7 | Pipelined vLLM | 135.77 | 0 | 6,255,630 | 1,978.58 | pairs/s | 0.14894 | 63.95 | 0.0003172 | 17.391 |
| FEV-7 | Pipelined SGLang | 205.33 | 24,320 | 6,165,002 | 1,285.93 | pairs/s | 0.22525 | 68.06 | 0.00033273 | 15.652 |
| FEV-7 | SoL estimate | 15.072 | 0 (assumed) | 3,807,384 | 11,082.38 | pairs/s | 0.01653 | Not measured | Not measured | Not measured |
| FEV-8 | Quail | 83.11 | 0 | 9,476,743 | 5,100.46 | pairs/s | 0.09117 | 71.36 | 3.5365e-06 | 13.986 |
| FEV-8 | Stock vLLM | 206.75 | 132,384 | 9,881,067 | 2,053.07 | pairs/s | 0.22680 | 70.80 | 4.6155e-06 | 18.881 |
| FEV-8 | Pipelined vLLM | 201.38 | 132,384 | 9,881,067 | 2,107.82 | pairs/s | 0.22091 | 70.80 | 4.6155e-06 | 18.881 |
| FEV-8 | Pipelined SGLang | 344.00 | 30,784 | 9,801,008 | 1,226.42 | pairs/s | 0.37737 | 74.28 | 5.0116e-06 | 15.385 |
| FEV-8 | SoL estimate | 19.139 | 0 (assumed) | 4,818,015 | 11,157.11 | pairs/s | 0.02100 | Not measured | Not measured | Not measured |
| FEV-9 | Quail | 39.02 | 7,309 | 4,314,219 | 4,667.22 | pairs/s | 0.04280 | 67.77 | 3.3382e-06 | 45.455 |
| FEV-9 | Stock vLLM | Not measured for this query definition | | | | | | | | |
| FEV-9 | Pipelined vLLM | Not measured for this query definition | | | | | | | | |
| FEV-9 | Pipelined SGLang | Not measured for this query definition | | | | | | | | |
| FEV-9 | SoL estimate | 5.821 | 0 (assumed) | 1,474,838 | 9,857.99 | pairs/s | 0.00639 | Not measured | Not measured | Not measured |

## LEP

[Open the LEP vector PDF](plots/quailb_lep.pdf)

[![LEP preview](plots/quailb_lep.png)](plots/quailb_lep.pdf)

Figure: plots/quailb_lep.png

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

| Query | Method | Seconds | Recomputed KV tokens | Fresh input tokens | Throughput | Unit | $/query | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| LEP-1 | Quail | 1.08 | 0 | 131,947 | 462.96 | docs/s | 0.00118 | 31.20 | 3.6517 | 92.857 |
| LEP-1 | Stock vLLM | 1.54 | 0 | 131,867 | 324.68 | docs/s | 0.00169 | 27.40 | 3.7135 | 100 |
| LEP-1 | Pipelined vLLM | 1.36 | 0 | 131,867 | 367.65 | docs/s | 0.00149 | 27.40 | 3.7135 | 100 |
| LEP-1 | Pipelined SGLang | 1.80 | 0 | 131,899 | 277.78 | docs/s | 0.00197 | 44.00 | 4.4521 | 92.857 |
| LEP-1 | SoL estimate | 0.494 | 0 (assumed) | 130,583 | 1,012.61 | docs/s | 0.00054 | Not measured | Not measured | Not measured |
| LEP-2 | Quail | 129.36 | 0 | 15,098,947 | 1,673.62 | pairs/s | 0.14191 | 46.08 | 0.41551 | 97.4 |
| LEP-2 | Stock vLLM | 157.51 | 0 | 15,184,675 | 1,374.52 | pairs/s | 0.17279 | 44.41 | 0.40554 | 98 |
| LEP-2 | Pipelined vLLM | 155.38 | 0 | 15,184,675 | 1,393.36 | pairs/s | 0.17045 | 44.41 | 0.40554 | 98 |
| LEP-2 | Pipelined SGLang | 270.91 | 33,872 | 15,393,811 | 799.16 | pairs/s | 0.29719 | 52.75 | 0.47102 | 96.8 |
| LEP-2 | SoL estimate | 58.086 | 0 (assumed) | 15,097,583 | 3,727.22 | pairs/s | 0.06372 | Not measured | Not measured | Not measured |
| LEP-3 | Quail | 92.51 | 0 | 10,807,675 | 1,666.28 | pairs/s | 0.10148 | 44.23 | 0.015075 | 92.857 |
| LEP-3 | Stock vLLM | 117.47 | 71,936 | 11,600,657 | 1,389.64 | pairs/s | 0.12886 | 42.91 | 0.014976 | 100 |
| LEP-3 | Pipelined vLLM | 117.69 | 71,936 | 11,600,657 | 1,387.04 | pairs/s | 0.12911 | 42.91 | 0.014976 | 100 |
| LEP-3 | Pipelined SGLang | 157.35 | 27,328 | 9,087,351 | 803.53 | pairs/s | 0.17261 | 50.71 | 0.020779 | 92.857 |
| LEP-3 | SoL estimate | 2.146 | 0 (assumed) | 550,415 | 2,825.44 | pairs/s | 0.00235 | Not measured | Not measured | Not measured |
| LEP-4 | Quail | 39.53 | 0 | 4,640,471 | 1,643.06 | pairs/s | 0.04336 | 34.05 | 0.011608 | 100 |
| LEP-4 | Stock vLLM | 46.78 | 19,360 | 4,748,742 | 1,406.93 | pairs/s | 0.05132 | 32.69 | 0.011225 | 100 |
| LEP-4 | Pipelined vLLM | 46.61 | 19,360 | 4,748,742 | 1,412.06 | pairs/s | 0.05113 | 32.69 | 0.011225 | 100 |
| LEP-4 | Pipelined SGLang | 54.46 | 16,564 | 3,246,285 | 810.98 | pairs/s | 0.05974 | 37.47 | 0.017967 | 100 |
| LEP-4 | SoL estimate | 1.087 | 0 (assumed) | 281,181 | 1,992.45 | pairs/s | 0.00119 | Not measured | Not measured | Not measured |
| LEP-5 | Quail | 24.83 | 0 | 2,904,378 | 1,604.35 | pairs/s | 0.02724 | 32.39 | 0 | 0 |
| LEP-5 | Stock vLLM | 34.52 | 13,664 | 3,478,978 | 1,379.78 | pairs/s | 0.03787 | 32.33 | 0 | 0 |
| LEP-5 | Pipelined vLLM | 34.37 | 13,664 | 3,478,978 | 1,385.80 | pairs/s | 0.03770 | 32.33 | 0 | 0 |
| LEP-5 | Pipelined SGLang | 33.42 | 19,628 | 2,038,764 | 803.29 | pairs/s | 0.03666 | 37.55 | 0 | 0 |
| LEP-5 | SoL estimate | 0.491 | 0 (assumed) | 129,884 | 0.00 | pairs/s | 0.00054 | Not measured | Not measured | Not measured |
| LEP-6 | Quail | 9.02 | 0 | 1,048,442 | 1,440.13 | pairs/s | 0.00989 | 26.02 | 0 | 0 |
| LEP-6 | Stock vLLM | 14.34 | 1,888 | 1,389,283 | 1,238.01 | pairs/s | 0.01573 | 23.60 | 0 | 0 |
| LEP-6 | Pipelined vLLM | 13.93 | 1,776 | 1,356,814 | 1,243.36 | pairs/s | 0.01528 | 23.33 | 0 | 0 |
| LEP-6 | Pipelined SGLang | 8.09 | 19,773 | 558,132 | 695.80 | pairs/s | 0.00887 | 18.87 | 0 | 0 |
| LEP-6 | SoL estimate | 0.483 | 0 (assumed) | 127,849 | 0.00 | pairs/s | 0.00053 | Not measured | Not measured | Not measured |
| LEP-7 | Quail | 38.79 | 0 | 4,556,127 | 1,624.13 | pairs/s | 0.04255 | 33.73 | 0.0071088 | 100 |
| LEP-7 | Stock vLLM | 46.18 | 19,552 | 4,684,952 | 1,385.71 | pairs/s | 0.05066 | 32.40 | 0.0068622 | 100 |
| LEP-7 | Pipelined vLLM | 46.08 | 19,552 | 4,684,952 | 1,388.72 | pairs/s | 0.05055 | 32.40 | 0.0068622 | 100 |
| LEP-7 | Pipelined SGLang | 52.09 | 16,564 | 3,131,339 | 793.05 | pairs/s | 0.05714 | 36.87 | 0.011324 | 100 |
| LEP-7 | SoL estimate | 1.129 | 0 (assumed) | 294,562 | 1,554.45 | pairs/s | 0.00124 | Not measured | Not measured | Not measured |
| LEP-8 | Quail | 1.34 | 0 | 148,802 | 373.13 | docs/s | 0.00147 | 48.26 | 0 | 0 |
| LEP-8 | Stock vLLM | 2.09 | 0 | 155,123 | 239.23 | docs/s | 0.00229 | 45.56 | 0 | 0 |
| LEP-8 | Pipelined vLLM | 2.25 | 0 | 155,123 | 222.22 | docs/s | 0.00247 | 45.56 | 0 | 0 |
| LEP-8 | Pipelined SGLang | 1.89 | 3,213 | 147,115 | 264.55 | docs/s | 0.00207 | 56.35 | 0 | 0 |
| LEP-8 | SoL estimate | 0.483 | 0 (assumed) | 127,849 | 1,034.40 | docs/s | 0.00053 | Not measured | Not measured | Not measured |

## AGENT

[Open the AGENT vector PDF](plots/quailb_agent.pdf)

[![AGENT preview](plots/quailb_agent.png)](plots/quailb_agent.pdf)

Figure: plots/quailb_agent.png

| Query | Input documents by alias and set |
|---|---|
| AGENT-1 | t (agent_traces) = 1,772 |
| AGENT-2 | t (agent_traces) = 1,772 |

| Query | Method | Seconds | Recomputed KV tokens | Fresh input tokens | Throughput | Unit | $/query | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| AGENT-1 | Quail | 240.49 | 0 | 17,389,113 | 7.37 | docs/s | 0.26382 | 75.00 | 68.605 | 41.331 |
| AGENT-1 | Stock vLLM | 102.35 | 0 | 5,526,889 | 17.31 | docs/s | 0.11228 | 74.15 | 65.915 | 40.981 |
| AGENT-1 | Pipelined vLLM | 99.15 | 0 | 5,526,889 | 17.87 | docs/s | 0.10877 | 74.15 | 65.915 | 40.981 |
| AGENT-1 | Pipelined SGLang | 218.13 | 0 | 13,068,441 | 8.12 | docs/s | 0.23929 | 73.93 | 67.085 | 37.478 |
| AGENT-1 | SoL estimate | 47.465 | 0 (assumed) | 5,502,961 | 37.33 | docs/s | 0.05207 | Not measured | Not measured | Not measured |
| AGENT-2 | Quail | 241.20 | 0 | 17,431,641 | 7.35 | docs/s | 0.26460 | 93.68 | 83.465 | 98.696 |
| AGENT-2 | Stock vLLM | 102.62 | 0 | 5,569,417 | 17.27 | docs/s | 0.11257 | 93.57 | 83.099 | 98.883 |
| AGENT-2 | Pipelined vLLM | 99.87 | 0 | 5,569,417 | 17.74 | docs/s | 0.10956 | 93.57 | 83.099 | 98.883 |
| AGENT-2 | Pipelined SGLang | 218.08 | 0 | 13,032,569 | 8.13 | docs/s | 0.23923 | 94.07 | 84.951 | 97.765 |
| AGENT-2 | SoL estimate | 47.870 | 0 (assumed) | 5,545,489 | 37.02 | docs/s | 0.05251 | Not measured | Not measured | Not measured |

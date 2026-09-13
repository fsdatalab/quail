# QUAIL-B comparison

- The main PDF covers all 33 queries with grouped bars and one metric per page.
  Its final page lists input document counts. Each dataset PDF has a page
  of four bar charts and a separate input-count page. Text and marks remain
  vector content when zoomed.
- The setup was Qwen3 4B FP8, sf=0.1, lf=1, and one H100 per configuration.
  Quail and the vLLM configurations shared a physical GPU within each
  family. Stock vLLM used operator-at-a-time submission; the other two
  pipeline their requests.
- Quail, stock vLLM, and pipelined vLLM were run on September 12, 2026: `/results/benchmarks/quailb/family-runs/20260912T225100Z-902686c5/`,
  function calls `fc-01M2BX2FQ0S11V5Q5WDBGC865R`, `fc-01M2BX3KM90P4D4ME4SFDXB88Z`, `fc-01M2BX3KR9EZXZ3HW43XWNJT7N`, `fc-01M2BX3KX1ENZPPV61RRCMN7TG`, `fc-01M2BX3M38FRY6FE2PAGGS2WD2`, `fc-01M2BX3M80WJDF3HDE9PYB903D`.
  That run's FEVER container failed on FEV-10 in the request backends (an
  equality join's key columns were not passed to them; fixed since) and
  was not rerun. 2 of the 99 cells come from the September 11
  FEV-10 run (`/results/benchmarks/quailb/family-runs/20260911T201441Z-d16f87d8/`): Pipelined vLLM FEV-10; Stock vLLM FEV-10.
  9 cells have no measurement and are marked missing, in the
  plots by an x below the axis and in the tables by a row that says so:
  Pipelined vLLM FEV-1, FEV-2, FEV-3, FEV-4, FEV-5, FEV-6, FEV-7, FEV-8, FEV-9.
- Quail was faster than stock vLLM on 31 of 33 queries
  where both were measured.
- A horizontal line across each query's bar group shows its SoL estimate.
  SoL models ideal computation and memory traffic with unlimited prefix KV.
  It credits matching token prefixes across requests, documents, and aliases.
  It uses exact reference-label survivors and searches supported left-deep
  join plans. Different answers can change the work done by measured runs,
  so the gap from SoL is not purely execution overhead. SoL uses the
  distinct-prefix estimate, not the per-document-only estimate. No
  accuracy is assigned to SoL because it is not a measured model run.
  SoL was recalculated for all 33 queries on the CPU on September 11,
  2026, from saved labels and corpus rows.
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
- Recomputed KV is `regret_tokens`: fresh tokens minus the fewest input
  tokens the run's requests needed with unlimited KV, where every
  distinct prefix across the requests is computed once. quail-bench
  derives it on the CPU after the run from the saved answer tables and
  the prompt token pieces the runner reports (`quail_b.minimum`); the
  engine tracks nothing. A run saved without that minimum is not measured.
  Token and latency plots use a log scale when positive values span more
  than one order of magnitude. Recomputed KV retains a linear region to
  include zero. A dash marks zero.
- Document counts come from the saved corpus manifest and describe inputs
  before filtering. Repeated aliases each list their full input count.
  The report tables also show throughput, GPU cost, and final output quality.
- FEV-9 agrees with the reference on 67.77% of evaluated answers. Its final
  output matches only 5 reference rows out of 149,783,486 returned rows.
  The reference has 11 rows, so output precision is approximately 0.00000334%
  and recall is 45.45%.

[Open the main vector PDF](plots/quailb_main.pdf)

Figure: plots/quailb_main.pdf

SoL estimates on `quail-results`: `/results/sol/2026-09-11-quailb-prefix-reuse.json`.

Corpus counts on `quail-results`: `/results/ground_truth/quailb/schema_v1/corpora/c_1aa2c4f0d0b6c816fd37aa5748c33341/manifest.json`.

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

| Query | Method | Seconds | Recomputed KV tokens | Fresh input tokens | Throughput | Unit | $/query | Answer agreement (%) | Output precision (%) | Output recall (%) | Source |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| IMDB-1 | Quail | 14.43 | 20,349 | 1,759,232 | 346.50 | docs/s | 0.01583 | 89.68 | 89.817 | 98.252 | September 12 |
| IMDB-1 | Stock vLLM | 17.52 | 20,061 | 1,758,944 | 285.39 | docs/s | 0.01922 | 89.52 | 89.781 | 98.077 | September 12 |
| IMDB-1 | Pipelined vLLM | 17.31 | 20,061 | 1,758,944 | 288.85 | docs/s | 0.01899 | 89.52 | 89.781 | 98.077 | September 12 |
| IMDB-1 | SoL estimate | 6.653 | 0 (assumed) | 1,740,485 | 751.59 | docs/s | 0.00730 | Not measured | Not measured | Not measured | estimate |
| IMDB-2 | Quail | 21.25 | 405,349 | 2,419,232 | 2,823.53 | pairs/s | 0.02331 | 78.50 | 72.543 | 43.494 | September 12 |
| IMDB-2 | Stock vLLM | 27.87 | 425,741 | 2,439,624 | 2,152.85 | pairs/s | 0.03057 | 78.30 | 69.563 | 46.864 | September 12 |
| IMDB-2 | Pipelined vLLM | 28.40 | 425,741 | 2,439,624 | 2,112.68 | pairs/s | 0.03115 | 78.30 | 69.563 | 46.864 | September 12 |
| IMDB-2 | SoL estimate | 9.211 | 0 (assumed) | 2,400,485 | 6,513.89 | pairs/s | 0.01010 | Not measured | Not measured | Not measured | estimate |
| IMDB-3 | Quail | 22.64 | 361,989 | 2,560,772 | 2,321.55 | pairs/s | 0.02484 | 79.16 | 66.549 | 44.329 | September 12 |
| IMDB-3 | Stock vLLM | 41.23 | 1,735,645 | 3,933,798 | 1,273.05 | pairs/s | 0.04523 | 78.96 | 63.436 | 47.125 | September 12 |
| IMDB-3 | Pipelined vLLM | 42.10 | 1,735,645 | 3,933,798 | 1,246.75 | pairs/s | 0.04618 | 78.96 | 63.436 | 47.125 | September 12 |
| IMDB-3 | SoL estimate | 9.495 | 0 (assumed) | 2,473,217 | 5,060.17 | pairs/s | 0.01042 | Not measured | Not measured | Not measured | estimate |
| IMDB-4 | Quail | 17.38 | 118,395 | 2,004,176 | 867.89 | pairs/s | 0.01907 | 77.76 | 59.54 | 37.707 | September 12 |
| IMDB-4 | Stock vLLM | 31.48 | 1,203,527 | 3,091,846 | 485.26 | pairs/s | 0.03453 | 77.90 | 57.771 | 41.655 | September 12 |
| IMDB-4 | Pipelined vLLM | 27.46 | 675,175 | 2,563,494 | 556.30 | pairs/s | 0.03012 | 77.90 | 57.771 | 41.655 | September 12 |
| IMDB-4 | SoL estimate | 7.519 | 0 (assumed) | 1,960,727 | 1,640.73 | pairs/s | 0.00825 | Not measured | Not measured | Not measured | estimate |
| IMDB-5 | Quail | 16.64 | 77,818 | 1,929,643 | 524.28 | pairs/s | 0.01825 | 80.17 | 60.501 | 35.794 | September 12 |
| IMDB-5 | Stock vLLM | 29.92 | 861,391 | 2,716,776 | 298.80 | pairs/s | 0.03282 | 80.33 | 58.53 | 40.382 | September 12 |
| IMDB-5 | Pipelined vLLM | 26.36 | 590,991 | 2,446,376 | 339.15 | pairs/s | 0.02892 | 80.33 | 58.522 | 40.41 | September 12 |
| IMDB-5 | SoL estimate | 7.405 | 0 (assumed) | 1,930,916 | 1,082.48 | pairs/s | 0.00812 | Not measured | Not measured | Not measured | estimate |
| IMDB-6 | Quail | 14.96 | 20,349 | 1,774,145 | 334.22 | docs/s | 0.01641 | 92.03 | 73.27 | 89.591 | September 12 |
| IMDB-6 | Stock vLLM | 23.33 | 591,825 | 2,346,479 | 214.32 | docs/s | 0.02559 | 91.98 | 72.506 | 89.786 | September 12 |
| IMDB-6 | Pipelined vLLM | 18.29 | 63,473 | 1,818,127 | 273.37 | docs/s | 0.02006 | 91.98 | 72.506 | 89.786 | September 12 |
| IMDB-6 | SoL estimate | 6.780 | 0 (assumed) | 1,772,603 | 737.51 | docs/s | 0.00744 | Not measured | Not measured | Not measured | estimate |
| IMDB-7 | Quail | 15.23 | 21,112 | 1,796,602 | 328.30 | docs/s | 0.01671 | 92.13 | 72.352 | 78.743 | September 12 |
| IMDB-7 | Stock vLLM | 26.11 | 797,129 | 2,574,289 | 191.50 | docs/s | 0.02864 | 92.18 | 71.812 | 80.09 | September 12 |
| IMDB-7 | Pipelined vLLM | 21.24 | 337,625 | 2,114,785 | 235.40 | docs/s | 0.02330 | 92.18 | 71.812 | 80.09 | September 12 |
| IMDB-7 | SoL estimate | 6.922 | 0 (assumed) | 1,808,672 | 722.34 | docs/s | 0.00759 | Not measured | Not measured | Not measured | estimate |
| IMDB-8 | Quail | 26.09 | 680,845 | 2,918,999 | 3,518.13 | pairs/s | 0.02862 | 68.47 | 20.407 | 27.098 | September 12 |
| IMDB-8 | Stock vLLM | 41.34 | 1,533,057 | 3,773,186 | 2,227.58 | pairs/s | 0.04535 | 67.83 | 19.224 | 28.623 | September 12 |
| IMDB-8 | Pipelined vLLM | 43.62 | 1,533,057 | 3,773,186 | 2,111.14 | pairs/s | 0.04785 | 67.83 | 19.224 | 28.623 | September 12 |
| IMDB-8 | SoL estimate | 11.903 | 0 (assumed) | 3,092,768 | 8,772.07 | pairs/s | 0.01306 | Not measured | Not measured | Not measured | estimate |
| IMDB-9 | Quail | 47.45 | 2,914,348 | 5,338,231 | 3,198.90 | pairs/s | 0.05205 | 72.43 | 17.624 | 13.171 | September 12 |
| IMDB-9 | Stock vLLM | 70.09 | 3,788,927 | 6,212,810 | 2,169.90 | pairs/s | 0.07689 | 71.96 | 16.224 | 14.712 | September 12 |
| IMDB-9 | Pipelined vLLM | 69.74 | 3,788,927 | 6,212,810 | 2,180.79 | pairs/s | 0.07650 | 71.96 | 16.224 | 14.712 | September 12 |
| IMDB-9 | SoL estimate | 16.376 | 0 (assumed) | 4,250,762 | 10,700.37 | pairs/s | 0.01796 | Not measured | Not measured | Not measured | estimate |
| IMDB-10 | Quail | 48.58 | 2,854,868 | 5,479,771 | 2,971.35 | pairs/s | 0.05329 | 72.59 | 16.265 | 13.507 | September 12 |
| IMDB-10 | Stock vLLM | 85.89 | 5,082,555 | 7,706,984 | 1,683.27 | pairs/s | 0.09422 | 72.11 | 14.876 | 14.885 | September 12 |
| IMDB-10 | Pipelined vLLM | 84.18 | 5,082,555 | 7,706,984 | 1,717.46 | pairs/s | 0.09235 | 72.11 | 14.876 | 14.885 | September 12 |
| IMDB-10 | SoL estimate | 15.732 | 0 (assumed) | 4,080,500 | 9,691.37 | pairs/s | 0.01726 | Not measured | Not measured | Not measured | estimate |

## BIO

[Open the BIO vector PDF](plots/quailb_bio.pdf)

Figure: plots/quailb_bio.pdf

| Query | Input documents by alias and set |
|---|---|
| BIO-1 | r (reports) = 500 |
| BIO-2 | r (reports) = 500, m (terms) = 1,127 |
| BIO-3 | r (reports) = 500, m (terms) = 1,127 |

| Query | Method | Seconds | Recomputed KV tokens | Fresh input tokens | Throughput | Unit | $/query | Answer agreement (%) | Output precision (%) | Output recall (%) | Source |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| BIO-1 | Quail | 21.52 | 2,477 | 2,057,345 | 23.23 | docs/s | 0.02361 | 93.60 | 100 | 89.542 | September 12 |
| BIO-1 | Stock vLLM | 25.75 | 2,461 | 2,057,329 | 19.42 | docs/s | 0.02825 | 94.00 | 100 | 90.196 | September 12 |
| BIO-1 | Pipelined vLLM | 25.50 | 2,461 | 2,057,329 | 19.61 | docs/s | 0.02797 | 94.00 | 100 | 90.196 | September 12 |
| BIO-1 | SoL estimate | 11.049 | 0 (assumed) | 2,054,868 | 45.25 | docs/s | 0.01212 | Not measured | Not measured | Not measured | estimate |
| BIO-2 | Quail | 127.75 | 4,148,977 | 10,374,345 | 4,410.96 | pairs/s | 0.14014 | 81.83 | 14.167 | 85.959 | September 12 |
| BIO-2 | Stock vLLM | 1074.34 | 4,301,247 | 10,526,615 | 524.51 | pairs/s | 1.17855 | 80.95 | 13.625 | 86.283 | September 12 |
| BIO-2 | Pipelined vLLM | 1069.16 | 4,301,247 | 10,526,615 | 527.05 | pairs/s | 1.17287 | 80.95 | 13.624 | 86.283 | September 12 |
| BIO-2 | SoL estimate | 62.002 | 0 (assumed) | 10,371,868 | 9,088.45 | pairs/s | 0.06802 | Not measured | Not measured | Not measured | estimate |
| BIO-3 | Quail | 79.96 | 2,275,033 | 6,627,939 | 3,861.91 | pairs/s | 0.08772 | 82.54 | 14.733 | 79.917 | September 12 |
| BIO-3 | Stock vLLM | 590.22 | 3,506,014 | 7,875,694 | 527.01 | pairs/s | 0.64747 | 81.23 | 13.912 | 81.171 | September 12 |
| BIO-3 | Pipelined vLLM | 585.63 | 3,506,014 | 7,875,694 | 531.14 | pairs/s | 0.64244 | 81.23 | 13.912 | 81.171 | September 12 |
| BIO-3 | SoL estimate | 43.087 | 0 (assumed) | 7,159,254 | 8,003.77 | pairs/s | 0.04727 | Not measured | Not measured | Not measured | estimate |

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

| Query | Method | Seconds | Recomputed KV tokens | Fresh input tokens | Throughput | Unit | $/query | Answer agreement (%) | Output precision (%) | Output recall (%) | Source |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| FEV-1 | Quail | 0.30 | 1,359 | 33,331 | 1,666.67 | docs/s | 0.00033 | 85.00 | 80.609 | 98.311 | September 12 |
| FEV-1 | Stock vLLM | 0.48 | 1,359 | 33,331 | 1,041.67 | docs/s | 0.00053 | 84.00 | 78.877 | 99.662 | September 12 |
| FEV-1 | Pipelined vLLM | missing | | | | | | | | | not run |
| FEV-1 | SoL estimate | 0.121 | 0 (assumed) | 32,498 | 4,139.22 | docs/s | 0.00013 | Not measured | Not measured | Not measured | estimate |
| FEV-2 | Quail | 28.87 | 963,563 | 3,245,254 | 4,970.56 | pairs/s | 0.03167 | 82.58 | 1.1787 | 95.82 | September 12 |
| FEV-2 | Stock vLLM | 66.26 | 1,068,922 | 3,350,613 | 2,165.71 | pairs/s | 0.07269 | 82.31 | 1.1645 | 96.141 | September 12 |
| FEV-2 | Pipelined vLLM | missing | | | | | | | | | not run |
| FEV-2 | SoL estimate | 12.845 | 0 (assumed) | 3,244,940 | 11,171.71 | pairs/s | 0.01409 | Not measured | Not measured | Not measured | estimate |
| FEV-3 | Quail | 21.40 | 689,396 | 2,404,957 | 4,841.45 | pairs/s | 0.02348 | 81.20 | 0.89349 | 95.135 | September 12 |
| FEV-3 | Stock vLLM | 45.44 | 801,721 | 2,562,915 | 2,362.19 | pairs/s | 0.04985 | 80.63 | 0.85189 | 96.757 | September 12 |
| FEV-3 | Pipelined vLLM | missing | | | | | | | | | not run |
| FEV-3 | SoL estimate | 7.943 | 0 (assumed) | 2,010,333 | 10,694.58 | pairs/s | 0.00871 | Not measured | Not measured | Not measured | estimate |
| FEV-4 | Quail | 4.72 | 82,965 | 531,453 | 2,857.84 | pairs/s | 0.00518 | 89.31 | 0.73677 | 78.571 | September 12 |
| FEV-4 | Stock vLLM | 6.43 | 110,913 | 575,830 | 2,276.36 | pairs/s | 0.00705 | 90.50 | 0.76655 | 78.571 | September 12 |
| FEV-4 | Pipelined vLLM | missing | | | | | | | | | not run |
| FEV-4 | SoL estimate | 1.868 | 0 (assumed) | 477,123 | 5,991.95 | pairs/s | 0.00205 | Not measured | Not measured | Not measured | estimate |
| FEV-5 | Quail | 13.43 | 411,979 | 1,515,283 | 4,596.50 | pairs/s | 0.01473 | 79.57 | 1.0522 | 96.429 | September 12 |
| FEV-5 | Stock vLLM | 28.35 | 521,243 | 1,657,324 | 2,269.07 | pairs/s | 0.03110 | 79.01 | 0.99111 | 97.143 | September 12 |
| FEV-5 | Pipelined vLLM | missing | | | | | | | | | not run |
| FEV-5 | SoL estimate | 4.743 | 0 (assumed) | 1,202,271 | 9,923.87 | pairs/s | 0.00520 | Not measured | Not measured | Not measured | estimate |
| FEV-6 | Quail | 3.53 | 50,656 | 398,331 | 2,276.77 | pairs/s | 0.00387 | 88.44 | 0.998 | 76.923 | September 12 |
| FEV-6 | Stock vLLM | 5.74 | 68,025 | 426,594 | 1,528.22 | pairs/s | 0.00630 | 89.54 | 1.0215 | 76.923 | September 12 |
| FEV-6 | Pipelined vLLM | missing | | | | | | | | | not run |
| FEV-6 | SoL estimate | 1.375 | 0 (assumed) | 352,709 | 4,510.24 | pairs/s | 0.00151 | Not measured | Not measured | Not measured | estimate |
| FEV-7 | Quail | 54.39 | 1,935,777 | 6,103,058 | 4,944.27 | pairs/s | 0.05967 | 64.66 | 0.00029045 | 15.652 | September 12 |
| FEV-7 | Stock vLLM | 113.21 | 2,095,237 | 6,255,630 | 2,372.86 | pairs/s | 0.12419 | 63.95 | 0.0003172 | 17.391 | September 12 |
| FEV-7 | Pipelined vLLM | missing | | | | | | | | | not run |
| FEV-7 | SoL estimate | 15.072 | 0 (assumed) | 3,807,384 | 11,082.38 | pairs/s | 0.01653 | Not measured | Not measured | Not measured | estimate |
| FEV-8 | Quail | 84.49 | 5,046,570 | 9,476,743 | 5,017.15 | pairs/s | 0.09269 | 71.36 | 3.5365e-06 | 13.986 | September 12 |
| FEV-8 | Stock vLLM | 181.54 | 5,450,894 | 9,881,067 | 2,338.18 | pairs/s | 0.19915 | 70.80 | 4.6155e-06 | 18.881 | September 12 |
| FEV-8 | Pipelined vLLM | missing | | | | | | | | | not run |
| FEV-8 | SoL estimate | 19.139 | 0 (assumed) | 4,818,015 | 11,157.11 | pairs/s | 0.02100 | Not measured | Not measured | Not measured | estimate |
| FEV-9 | Quail | 38.23 | 2,279,522 | 4,306,910 | 4,763.67 | pairs/s | 0.04194 | 67.77 | 3.3382e-06 | 45.455 | September 12 |
| FEV-9 | Stock vLLM | 82.41 | 2,499,983 | 4,592,900 | 2,304.19 | pairs/s | 0.09040 | 67.05 | 2.9055e-06 | 45.455 | September 12 |
| FEV-9 | Pipelined vLLM | missing | | | | | | | | | not run |
| FEV-9 | SoL estimate | 5.821 | 0 (assumed) | 1,474,838 | 9,857.99 | pairs/s | 0.00639 | Not measured | Not measured | Not measured | estimate |
| FEV-10 | Quail | 1.65 | 2,927 | 187,567 | 112.12 | pairs/s | 0.00181 | 89.09 | 82.759 | 96.774 | September 12 |
| FEV-10 | Stock vLLM | 2.97 | 4,308 | 270,220 | 63.30 | pairs/s | 0.00326 | 86.67 | 73.78 | 97.581 | FEV-10 run |
| FEV-10 | Pipelined vLLM | 2.91 | 4,308 | 270,220 | 64.60 | pairs/s | 0.00319 | 86.67 | 73.78 | 97.581 | FEV-10 run |
| FEV-10 | SoL estimate | 0.712 | 0 (assumed) | 185,703 | 235.79 | pairs/s | 0.00078 | Not measured | Not measured | Not measured | estimate |

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

| Query | Method | Seconds | Recomputed KV tokens | Fresh input tokens | Throughput | Unit | $/query | Answer agreement (%) | Output precision (%) | Output recall (%) | Source |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| LEP-1 | Quail | 1.10 | 1,702 | 131,947 | 454.55 | docs/s | 0.00121 | 31.20 | 3.6517 | 92.857 | September 12 |
| LEP-1 | Stock vLLM | 1.43 | 1,622 | 131,867 | 349.65 | docs/s | 0.00157 | 27.40 | 3.7135 | 100 | September 12 |
| LEP-1 | Pipelined vLLM | 1.40 | 1,622 | 131,867 | 357.14 | docs/s | 0.00154 | 27.40 | 3.7135 | 100 | September 12 |
| LEP-1 | SoL estimate | 0.494 | 0 (assumed) | 130,583 | 1,012.61 | docs/s | 0.00054 | Not measured | Not measured | Not measured | estimate |
| LEP-2 | Quail | 130.39 | 1,562,202 | 15,098,947 | 1,660.40 | pairs/s | 0.14304 | 46.08 | 0.41551 | 97.4 | September 12 |
| LEP-2 | Stock vLLM | 157.27 | 1,647,930 | 15,184,675 | 1,376.61 | pairs/s | 0.17253 | 44.41 | 0.40554 | 98 | September 12 |
| LEP-2 | Pipelined vLLM | 157.01 | 1,647,930 | 15,184,675 | 1,378.89 | pairs/s | 0.17224 | 44.41 | 0.40554 | 98 | September 12 |
| LEP-2 | SoL estimate | 58.086 | 0 (assumed) | 15,097,583 | 3,727.22 | pairs/s | 0.06372 | Not measured | Not measured | Not measured | estimate |
| LEP-3 | Quail | 93.40 | 1,113,134 | 10,807,675 | 1,650.41 | pairs/s | 0.10246 | 44.23 | 0.015075 | 92.857 | September 12 |
| LEP-3 | Stock vLLM | 119.10 | 1,341,930 | 11,600,657 | 1,370.62 | pairs/s | 0.13065 | 42.91 | 0.014976 | 100 | September 12 |
| LEP-3 | Pipelined vLLM | 118.73 | 1,341,930 | 11,600,657 | 1,374.89 | pairs/s | 0.13025 | 42.91 | 0.014976 | 100 | September 12 |
| LEP-3 | SoL estimate | 2.146 | 0 (assumed) | 550,415 | 2,825.44 | pairs/s | 0.00235 | Not measured | Not measured | Not measured | estimate |
| LEP-4 | Quail | 39.92 | 470,002 | 4,640,471 | 1,627.00 | pairs/s | 0.04379 | 34.05 | 0.011608 | 100 | September 12 |
| LEP-4 | Stock vLLM | 47.42 | 523,932 | 4,748,742 | 1,387.94 | pairs/s | 0.05202 | 32.69 | 0.011225 | 100 | September 12 |
| LEP-4 | Pipelined vLLM | 47.45 | 523,932 | 4,748,742 | 1,387.06 | pairs/s | 0.05205 | 32.69 | 0.011225 | 100 | September 12 |
| LEP-4 | SoL estimate | 1.087 | 0 (assumed) | 281,181 | 1,992.45 | pairs/s | 0.00119 | Not measured | Not measured | Not measured | estimate |
| LEP-5 | Quail | 25.76 | 288,926 | 2,904,378 | 1,546.43 | pairs/s | 0.02826 | 32.39 | 0 | 0 | September 12 |
| LEP-5 | Stock vLLM | 34.87 | 378,637 | 3,478,978 | 1,365.93 | pairs/s | 0.03825 | 32.33 | 0 | 0 | September 12 |
| LEP-5 | Pipelined vLLM | 34.69 | 378,637 | 3,478,978 | 1,373.02 | pairs/s | 0.03805 | 32.33 | 0 | 0 | September 12 |
| LEP-5 | SoL estimate | 0.491 | 0 (assumed) | 129,884 | 0.00 | pairs/s | 0.00054 | Not measured | Not measured | Not measured | estimate |
| LEP-6 | Quail | 9.39 | 95,362 | 1,048,442 | 1,383.39 | pairs/s | 0.01030 | 26.02 | 0 | 0 | September 12 |
| LEP-6 | Stock vLLM | 14.38 | 138,719 | 1,389,283 | 1,234.56 | pairs/s | 0.01577 | 23.60 | 0 | 0 | September 12 |
| LEP-6 | Pipelined vLLM | 14.00 | 133,145 | 1,356,814 | 1,237.14 | pairs/s | 0.01536 | 23.33 | 0 | 0 | September 12 |
| LEP-6 | SoL estimate | 0.483 | 0 (assumed) | 127,849 | 0.00 | pairs/s | 0.00053 | Not measured | Not measured | Not measured | estimate |
| LEP-7 | Quail | 39.14 | 456,870 | 4,556,127 | 1,609.61 | pairs/s | 0.04294 | 33.73 | 0.0071088 | 100 | September 12 |
| LEP-7 | Stock vLLM | 47.21 | 511,564 | 4,684,760 | 1,355.48 | pairs/s | 0.05179 | 32.40 | 0.0068622 | 100 | September 12 |
| LEP-7 | Pipelined vLLM | 46.62 | 511,564 | 4,684,760 | 1,372.63 | pairs/s | 0.05114 | 32.40 | 0.0068622 | 100 | September 12 |
| LEP-7 | SoL estimate | 1.129 | 0 (assumed) | 294,562 | 1,554.45 | pairs/s | 0.00124 | Not measured | Not measured | Not measured | estimate |
| LEP-8 | Quail | 1.34 | 1,702 | 148,802 | 373.13 | docs/s | 0.00147 | 48.26 | 0 | 0 | September 12 |
| LEP-8 | Stock vLLM | 2.09 | 6,065 | 155,123 | 239.23 | docs/s | 0.00229 | 45.56 | 0 | 0 | September 12 |
| LEP-8 | Pipelined vLLM | 2.29 | 6,065 | 155,123 | 218.34 | docs/s | 0.00251 | 45.56 | 0 | 0 | September 12 |
| LEP-8 | SoL estimate | 0.483 | 0 (assumed) | 127,849 | 1,034.40 | docs/s | 0.00053 | Not measured | Not measured | Not measured | estimate |

## AGENT

[Open the AGENT vector PDF](plots/quailb_agent.pdf)

Figure: plots/quailb_agent.pdf

| Query | Input documents by alias and set |
|---|---|
| AGENT-1 | t (agent_traces) = 1,772 |
| AGENT-2 | t (agent_traces) = 1,772 |

| Query | Method | Seconds | Recomputed KV tokens | Fresh input tokens | Throughput | Unit | $/query | Answer agreement (%) | Output precision (%) | Output recall (%) | Source |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| AGENT-1 | Quail | 237.65 | 11,886,152 | 17,389,113 | 7.46 | docs/s | 0.26070 | 75.00 | 68.605 | 41.331 | September 12 |
| AGENT-1 | Stock vLLM | 104.00 | 23,928 | 5,526,889 | 17.04 | docs/s | 0.11409 | 74.15 | 65.915 | 40.981 | September 12 |
| AGENT-1 | Pipelined vLLM | 99.29 | 23,928 | 5,526,889 | 17.85 | docs/s | 0.10892 | 74.15 | 65.915 | 40.981 | September 12 |
| AGENT-1 | SoL estimate | 47.465 | 0 (assumed) | 5,502,961 | 37.33 | docs/s | 0.05207 | Not measured | Not measured | Not measured | estimate |
| AGENT-2 | Quail | 239.15 | 11,886,152 | 17,431,641 | 7.41 | docs/s | 0.26235 | 93.68 | 83.465 | 98.696 | September 12 |
| AGENT-2 | Stock vLLM | 103.35 | 23,928 | 5,569,417 | 17.15 | docs/s | 0.11337 | 93.57 | 83.099 | 98.883 | September 12 |
| AGENT-2 | Pipelined vLLM | 100.05 | 23,928 | 5,569,417 | 17.71 | docs/s | 0.10975 | 93.57 | 83.099 | 98.883 | September 12 |
| AGENT-2 | SoL estimate | 47.870 | 0 (assumed) | 5,545,489 | 37.02 | docs/s | 0.05251 | Not measured | Not measured | Not measured | estimate |

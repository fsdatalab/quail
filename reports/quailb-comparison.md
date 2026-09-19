# QUAIL-B comparison

- All 30 queries use raw document/question prompts ending in `ANSWER:`.
  LePaRD has five queries. Original LEP-7 is now LEP-5.
  Both methods use Qwen3 4B FP8, sf=0.1, and one H100.
- Measurements are reused from saved raw runs. No inference was repeated.
  Source on `quail-results`: `/results/benchmarks/quailb/20260914T070913Z-f7beefb6`.
  Saved plans and prompt token pieces match the restored definitions.
  Scores and token counts were recalculated from the saved answers.
- BIO-1 and BIO-3 are not measured for the serious-adverse-event filter.
  The saved demographic-filter results do not match those queries.
  All plots include both queries, with missing measurements marked x.
  Comparisons below use the 28 queries with both measurements.
- Original LEP-5, LEP-6, and LEP-8 are excluded because their reference
  outputs are empty at sf=0.1 with raw prompts too. Historical IDs are
  matched by query-definition hash before measurements are reused.
- References use Qwen3 32B FP8 and the benchmark's dataset annotations.
  Raw reference collection: `gt_be81cb241d74555dc2da79b5b0662554`.
  Answer agreement counts matching predicate answers. Output precision
  and recall compare final rows with the reference output.
- Quail uses pipelining, token-based admission, and KV rewind.
  Stock vLLM uses pipelining and prefix caching. Filter stages advance
  independently; joins begin after filtering finishes.
  vLLM batched-token limits: 25,305.
  Sequence limits: 4,096.
  Measured vLLM KV capacity: 479,616 tokens.
- Quail is faster on 26 of 28 queries.
  Mean speedup: 1.77x; median: 1.45x; maximum: 10.04x (BIO-2).
  Each query has equal weight. Speedup is vLLM time divided by Quail time.
- Latency excludes startup and result collection. GPU cost is query
  seconds / 3,600 * $3.9492.
- Tokens/second counts the full input prompt for every evaluated answer,
  including input tokens served from KV. Generated answers are excluded.
  Cost per million tokens uses that same total input count.
  Different survivors can change which requests each method evaluates.
  Counts use the named HuggingFace tokenizer. The old runs used bpe-qwen
  for documents; recounts differ from vLLM counters by at most 0.000339%.
- KV regret is recomputed tokens / fresh computed tokens * 100%.
  The minimum computes each distinct input prefix once with unlimited KV.
  A pair's partner suffix is counted after its anchor. Regret is
  recalculated with the current benchmark rule from saved prompt pieces.
- SoL estimates ideal compute and memory time with unlimited retained KV
  and exact raw-reference survivors. Matching prefixes are reused across
  requests, documents, and aliases. The join search uses left-deep plans.
  It excludes software overhead. Different answers change the work, so
  the gap from SoL is not solely execution overhead. It has no accuracy.
  These estimates were recalculated on CPU for the restored queries.
  Saved estimates: `/results/reports/quailb-raw-2026-09-19/sol_quailb_sf0.1.json`.
- PDFs show latency, total input tokens/second, cost/query, cost per
  million input tokens, and KV regret percentage. Each PDF lists input
  counts separately for every alias. SoL uses lines; measurements use bars.
  A dash marks zero. An x marks a missing measurement.

[Main comparison PDF](plots/quailb_main.pdf)

CPU-derived summary on `quail-results`:
`/results/reports/quailb-raw-2026-09-19/comparison.json`.

Rebuild commands: `reports/make_quailb_comparison_plots.py`.

## IMDB

[IMDB comparison PDF](plots/quailb_imdb.pdf)

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

| Query | Method | Seconds | Tokens/second | $/query | $/million input tokens | KV regret (%) |
|---|---|---:|---:|---:|---:|---:|
| IMDB-1 | Quail | 14.44 | 121,830.54 | 0.01584 | 0.009004 | 1.16 |
| IMDB-1 | Pipelined vLLM | 17.33 | 101,513.73 | 0.01901 | 0.010806 | 1.14 |
| IMDB-1 | SoL estimate | 6.653 | 264,443.38 | 0.00730 | 0.004148 | 0 (assumed) |
| IMDB-2 | Quail | 21.20 | 988,009.25 | 0.02326 | 0.001110 | 0.84 |
| IMDB-2 | Pipelined vLLM | 25.32 | 827,243.13 | 0.02778 | 0.001326 | 1.67 |
| IMDB-2 | SoL estimate | 9.211 | 2,273,975.42 | 0.01010 | 0.000482 | 0 (assumed) |
| IMDB-3 | Quail | 22.39 | 919,632.92 | 0.02456 | 0.001193 | 0.97 |
| IMDB-3 | Pipelined vLLM | 40.01 | 513,777.63 | 0.04389 | 0.002135 | 35.56 |
| IMDB-3 | SoL estimate | 9.495 | 2,003,008.14 | 0.01042 | 0.000548 | 0 (assumed) |
| IMDB-4 | Quail | 17.31 | 524,522.13 | 0.01899 | 0.002091 | 1.08 |
| IMDB-4 | Pipelined vLLM | 26.26 | 350,212.60 | 0.02881 | 0.003132 | 22.51 |
| IMDB-4 | SoL estimate | 7.521 | 1,026,113.86 | 0.00825 | 0.001069 | 0 (assumed) |
| IMDB-5 | Quail | 16.55 | 427,405.08 | 0.01816 | 0.002567 | 1.13 |
| IMDB-5 | Pipelined vLLM | 25.22 | 287,819.55 | 0.02767 | 0.003811 | 21.76 |
| IMDB-5 | SoL estimate | 7.408 | 879,626.62 | 0.00813 | 0.001247 | 0 (assumed) |
| IMDB-6 | Quail | 14.93 | 156,267.58 | 0.01638 | 0.007020 | 1.15 |
| IMDB-6 | Pipelined vLLM | 18.25 | 128,625.97 | 0.02002 | 0.008529 | 3.49 |
| IMDB-6 | SoL estimate | 6.779 | 333,850.55 | 0.00744 | 0.003286 | 0 (assumed) |
| IMDB-7 | Quail | 15.47 | 175,238.07 | 0.01697 | 0.006260 | 1.18 |
| IMDB-7 | Pipelined vLLM | 21.29 | 128,818.51 | 0.02336 | 0.008516 | 15.96 |
| IMDB-7 | SoL estimate | 6.922 | 379,526.20 | 0.00759 | 0.002890 | 0 (assumed) |
| IMDB-8 | Quail | 26.06 | 1,238,001.34 | 0.02859 | 0.000886 | 3.15 |
| IMDB-8 | Pipelined vLLM | 39.49 | 819,743.58 | 0.04332 | 0.001338 | 24.97 |
| IMDB-8 | SoL estimate | 12.036 | 3,174,099.85 | 0.01320 | 0.000346 | 0 (assumed) |
| IMDB-9 | Quail | 47.77 | 1,113,839.46 | 0.05240 | 0.000985 | 40.17 |
| IMDB-9 | Pipelined vLLM | 65.42 | 815,002.60 | 0.07177 | 0.001346 | 48.59 |
| IMDB-9 | SoL estimate | 16.431 | 3,738,700.60 | 0.01802 | 0.000293 | 0 (assumed) |
| IMDB-10 | Quail | 48.40 | 1,092,001.98 | 0.05309 | 0.001005 | 38.92 |
| IMDB-10 | Pipelined vLLM | 80.04 | 661,268.33 | 0.08780 | 0.001659 | 56.58 |
| IMDB-10 | SoL estimate | 15.865 | 3,606,897.01 | 0.01740 | 0.000304 | 0 (assumed) |

| Query | Method | Reference rows | Returned rows | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|---:|---:|
| IMDB-1 | Quail | 4,004 | 4,380 | 89.68 | 89.817 | 98.252 |
| IMDB-1 | Pipelined vLLM | 4,004 | 4,374 | 89.60 | 89.826 | 98.127 |
| IMDB-2 | Quail | 19,709 | 10,602 | 76.41 | 76.203 | 40.991 |
| IMDB-2 | Pipelined vLLM | 19,709 | 11,914 | 76.44 | 73.376 | 44.355 |
| IMDB-3 | Quail | 15,973 | 9,650 | 77.48 | 69.72 | 42.121 |
| IMDB-3 | Pipelined vLLM | 15,973 | 10,762 | 77.47 | 66.856 | 45.045 |
| IMDB-4 | Quail | 5,422 | 3,176 | 76.49 | 62.311 | 36.499 |
| IMDB-4 | Pipelined vLLM | 5,422 | 3,616 | 76.66 | 60.343 | 40.243 |
| IMDB-5 | Quail | 3,800 | 2,076 | 78.89 | 62.428 | 34.105 |
| IMDB-5 | Pipelined vLLM | 3,800 | 2,423 | 79.06 | 60.462 | 38.553 |
| IMDB-6 | Quail | 1,032 | 1,257 | 91.90 | 73.27 | 89.244 |
| IMDB-6 | Pipelined vLLM | 1,032 | 1,273 | 91.83 | 72.427 | 89.341 |
| IMDB-7 | Quail | 672 | 727 | 91.96 | 72.352 | 78.274 |
| IMDB-7 | Pipelined vLLM | 672 | 745 | 91.97 | 71.678 | 79.464 |
| IMDB-8 | Quail | 58,550 | 62,777 | 68.06 | 23.512 | 25.209 |
| IMDB-8 | Pipelined vLLM | 58,550 | 70,390 | 67.48 | 22.265 | 26.767 |
| IMDB-9 | Quail | 127,710,136 | 71,374,780 | 71.36 | 21.096 | 11.79 |
| IMDB-9 | Pipelined vLLM | 127,710,136 | 86,603,976 | 71.01 | 19.59 | 13.284 |
| IMDB-10 | Quail | 103,372,135 | 64,840,220 | 71.69 | 19.47 | 12.213 |
| IMDB-10 | Pipelined vLLM | 103,372,135 | 78,126,680 | 71.32 | 17.986 | 13.593 |

## BIO

[BIO comparison PDF](plots/quailb_bio.pdf)

| Query | Input documents by alias and set |
|---|---|
| BIO-1 | r (reports) = 500 |
| BIO-2 | r (reports) = 500, m (terms) = 1,127 |
| BIO-3 | r (reports) = 500, m (terms) = 1,127 |

| Query | Method | Seconds | Tokens/second | $/query | $/million input tokens | KV regret (%) |
|---|---|---:|---:|---:|---:|---:|
| BIO-1 | Quail | Not measured | | | | |
| BIO-1 | Pipelined vLLM | Not measured | | | | |
| BIO-1 | SoL estimate | 11.055 | 186,194.32 | 0.01213 | 0.005892 | 0 (assumed) |
| BIO-2 | Quail | 127.14 | 18,280,145.54 | 0.13947 | 0.000060 | 0.02 |
| BIO-2 | Pipelined vLLM | 1276.94 | 1,820,083.72 | 1.40080 | 0.000603 | 1.47 |
| BIO-2 | SoL estimate | 62.002 | 37,485,021.13 | 0.06802 | 0.000029 | 0 (assumed) |
| BIO-3 | Quail | Not measured | | | | |
| BIO-3 | Pipelined vLLM | Not measured | | | | |
| BIO-3 | SoL estimate | 42.784 | 32,399,507.32 | 0.04693 | 0.000034 | 0 (assumed) |

| Query | Method | Reference rows | Returned rows | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|---:|---:|
| BIO-1 | Quail | | Not measured | | | |
| BIO-1 | Pipelined vLLM | | Not measured | | | |
| BIO-2 | Quail | 22,582 | 116,156 | 81.86 | 15.726 | 80.892 |
| BIO-2 | Pipelined vLLM | 22,582 | 121,240 | 81.02 | 15.2 | 81.605 |
| BIO-3 | Quail | | Not measured | | | |
| BIO-3 | Pipelined vLLM | | Not measured | | | |

## FEV

[FEV comparison PDF](plots/quailb_fev.pdf)

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

| Query | Method | Seconds | Tokens/second | $/query | $/million input tokens | KV regret (%) |
|---|---|---:|---:|---:|---:|---:|
| FEV-1 | Quail | 0.29 | 114,934.48 | 0.00032 | 0.009545 | 4.08 |
| FEV-1 | Pipelined vLLM | 0.41 | 81,295.12 | 0.00045 | 0.013494 | 4.08 |
| FEV-1 | SoL estimate | 0.121 | 275,929.00 | 0.00013 | 0.003976 | 0 (assumed) |
| FEV-2 | Quail | 29.43 | 2,429,051.89 | 0.03228 | 0.000452 | 0.02 |
| FEV-2 | Pipelined vLLM | 60.20 | 1,187,491.64 | 0.06604 | 0.000924 | 3.16 |
| FEV-2 | SoL estimate | 12.845 | 5,565,379.99 | 0.01409 | 0.000197 | 0 (assumed) |
| FEV-3 | Quail | 21.83 | 2,365,436.42 | 0.02395 | 0.000464 | 0.11 |
| FEV-3 | Pipelined vLLM | 46.44 | 1,151,702.39 | 0.05094 | 0.000953 | 3.54 |
| FEV-3 | SoL estimate | 7.898 | 5,326,652.62 | 0.00866 | 0.000206 | 0 (assumed) |
| FEV-4 | Quail | 4.83 | 1,412,976.60 | 0.00530 | 0.000776 | 0.49 |
| FEV-4 | Pipelined vLLM | 7.17 | 1,031,351.46 | 0.00787 | 0.001064 | 4.16 |
| FEV-4 | SoL estimate | 1.816 | 2,967,229.86 | 0.00199 | 0.000370 | 0 (assumed) |
| FEV-5 | Quail | 13.70 | 2,204,798.47 | 0.01503 | 0.000498 | 0.18 |
| FEV-5 | Pipelined vLLM | 28.12 | 1,114,560.88 | 0.03085 | 0.000984 | 5.76 |
| FEV-5 | SoL estimate | 4.668 | 5,018,814.70 | 0.00512 | 0.000219 | 0 (assumed) |
| FEV-6 | Quail | 4.05 | 1,019,926.17 | 0.00444 | 0.001076 | 0.70 |
| FEV-6 | Pipelined vLLM | 5.63 | 794,140.32 | 0.00618 | 0.001381 | 3.73 |
| FEV-6 | SoL estimate | 1.338 | 2,347,116.17 | 0.00147 | 0.000467 | 0 (assumed) |
| FEV-7 | Quail | 55.64 | 2,412,313.39 | 0.06104 | 0.000455 | 2.21 |
| FEV-7 | Pipelined vLLM | 110.56 | 1,212,675.96 | 0.12128 | 0.000905 | 4.76 |
| FEV-7 | SoL estimate | 21.254 | 5,691,842.82 | 0.02332 | 0.000193 | 0 (assumed) |
| FEV-8 | Quail | 86.36 | 2,448,696.29 | 0.09474 | 0.000448 | 32.93 |
| FEV-8 | Pipelined vLLM | 171.47 | 1,234,927.22 | 0.18810 | 0.000888 | 35.68 |
| FEV-8 | SoL estimate | 33.614 | 5,725,697.73 | 0.03687 | 0.000192 | 0 (assumed) |
| FEV-9 | Quail | 39.00 | 2,283,860.05 | 0.04278 | 0.000480 | 33.92 |
| FEV-9 | Pipelined vLLM | 76.41 | 1,210,305.72 | 0.08382 | 0.000906 | 35.88 |
| FEV-9 | SoL estimate | 11.387 | 5,452,523.60 | 0.01249 | 0.000201 | 0 (assumed) |
| FEV-10 | Quail | 1.68 | 161,098.21 | 0.00184 | 0.006810 | 1.48 |
| FEV-10 | Pipelined vLLM | 2.92 | 92,886.30 | 0.00320 | 0.011810 | 1.59 |
| FEV-10 | SoL estimate | 0.712 | 371,434.56 | 0.00078 | 0.002953 | 0 (assumed) |

| Query | Method | Reference rows | Returned rows | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|---:|---:|
| FEV-1 | Quail | 294 | 361 | 85.00 | 80.332 | 98.639 |
| FEV-1 | Pipelined vLLM | 294 | 374 | 83.60 | 78.342 | 99.66 |
| FEV-2 | Quail | 7,598 | 25,283 | 82.78 | 16.145 | 53.725 |
| FEV-2 | Pipelined vLLM | 7,598 | 25,676 | 82.51 | 15.929 | 53.83 |
| FEV-3 | Quail | 5,173 | 19,698 | 81.52 | 14.448 | 55.016 |
| FEV-3 | Pipelined vLLM | 5,173 | 21,012 | 80.89 | 13.507 | 54.862 |
| FEV-4 | Quail | 115 | 1,493 | 89.51 | 2.0094 | 26.087 |
| FEV-4 | Pipelined vLLM | 115 | 1,435 | 90.60 | 2.0906 | 26.087 |
| FEV-5 | Quail | 2,951 | 12,830 | 80.10 | 13.18 | 57.303 |
| FEV-5 | Pipelined vLLM | 2,951 | 13,722 | 79.44 | 12.178 | 56.625 |
| FEV-6 | Quail | 70 | 1,002 | 88.70 | 1.6966 | 24.286 |
| FEV-6 | Pipelined vLLM | 70 | 979 | 89.73 | 1.7365 | 24.286 |
| FEV-7 | Quail | 355,264 | 6,197,246 | 63.84 | 2.4349 | 42.474 |
| FEV-7 | Pipelined vLLM | 355,264 | 6,305,092 | 63.12 | 2.4099 | 42.769 |
| FEV-8 | Quail | 11,254,492 | 565,523,363 | 69.17 | 0.45158 | 22.692 |
| FEV-8 | Pipelined vLLM | 11,254,492 | 584,989,951 | 68.72 | 0.43787 | 22.76 |
| FEV-9 | Quail | 2,103,099 | 149,783,486 | 66.18 | 0.43483 | 30.969 |
| FEV-9 | Pipelined vLLM | 2,103,099 | 172,090,043 | 65.41 | 0.37066 | 30.33 |
| FEV-10 | Quail | 122 | 145 | 88.89 | 81.379 | 96.721 |
| FEV-10 | Pipelined vLLM | 122 | 164 | 86.26 | 72.561 | 97.541 |

## LEP

[LEP comparison PDF](plots/quailb_lep.pdf)

| Query | Input documents by alias and set |
|---|---|
| LEP-1 | d (citation_contexts) = 500 |
| LEP-2 | d (citation_contexts) = 500, s (citation_passages) = 433 |
| LEP-3 | d (citation_contexts) = 500, s (citation_passages) = 433 |
| LEP-4 | d (citation_contexts) = 500, s (citation_passages) = 433 |
| LEP-5 | d (citation_contexts) = 500, s (citation_passages) = 433 |

| Query | Method | Seconds | Tokens/second | $/query | $/million input tokens | KV regret (%) |
|---|---|---:|---:|---:|---:|---:|
| LEP-1 | Quail | 1.10 | 119,951.82 | 0.00121 | 0.009145 | 1.29 |
| LEP-1 | Pipelined vLLM | 1.38 | 95,613.77 | 0.00151 | 0.011473 | 1.23 |
| LEP-1 | SoL estimate | 0.494 | 267,221.28 | 0.00054 | 0.004105 | 0 (assumed) |
| LEP-2 | Quail | 131.53 | 520,246.72 | 0.14429 | 0.002109 | 0.01 |
| LEP-2 | Pipelined vLLM | 159.21 | 429,797.44 | 0.17465 | 0.002552 | 0.58 |
| LEP-2 | SoL estimate | 58.086 | 1,178,044.06 | 0.06372 | 0.000931 | 0 (assumed) |
| LEP-3 | Quail | 94.22 | 504,584.10 | 0.10336 | 0.002174 | 0.02 |
| LEP-3 | Pipelined vLLM | 120.83 | 419,350.83 | 0.13255 | 0.002616 | 1.43 |
| LEP-3 | SoL estimate | 2.260 | 1,348,878.62 | 0.00248 | 0.000813 | 0 (assumed) |
| LEP-4 | Quail | 40.29 | 403,043.39 | 0.04420 | 0.002722 | 0.04 |
| LEP-4 | Pipelined vLLM | 47.86 | 347,763.77 | 0.05250 | 0.003154 | 1.04 |
| LEP-4 | SoL estimate | 1.202 | 1,054,502.31 | 0.00132 | 0.001040 | 0 (assumed) |
| LEP-5 | Quail | 39.47 | 400,512.79 | 0.04330 | 0.002739 | 0.08 |
| LEP-5 | Pipelined vLLM | 47.62 | 341,269.89 | 0.05224 | 0.003214 | 1.09 |
| LEP-5 | SoL estimate | 1.236 | 900,460.88 | 0.00136 | 0.001218 | 0 (assumed) |

| Query | Method | Reference rows | Returned rows | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|---:|---:|
| LEP-1 | Quail | 15 | 356 | 31.40 | 3.9326 | 93.333 |
| LEP-1 | Pipelined vLLM | 15 | 377 | 27.60 | 3.9788 | 100 |
| LEP-2 | Quail | 500 | 117,205 | 46.08 | 0.41551 | 97.4 |
| LEP-2 | Pipelined vLLM | 500 | 120,827 | 44.41 | 0.40554 | 98 |
| LEP-3 | Quail | 15 | 86,234 | 44.24 | 0.016235 | 93.333 |
| LEP-3 | Pipelined vLLM | 15 | 93,482 | 42.91 | 0.016046 | 100 |
| LEP-4 | Quail | 6 | 43,073 | 34.04 | 0.01393 | 100 |
| LEP-4 | Pipelined vLLM | 6 | 44,543 | 32.69 | 0.01347 | 100 |
| LEP-5 | Quail | 4 | 42,201 | 33.74 | 0.0094784 | 100 |
| LEP-5 | Pipelined vLLM | 4 | 43,718 | 32.41 | 0.0091495 | 100 |

## AGENT

[AGENT comparison PDF](plots/quailb_agent.pdf)

| Query | Input documents by alias and set |
|---|---|
| AGENT-1 | t (agent_traces) = 1,772 |
| AGENT-2 | t (agent_traces) = 1,772 |

| Query | Method | Seconds | Tokens/second | $/query | $/million input tokens | KV regret (%) |
|---|---|---:|---:|---:|---:|---:|
| AGENT-1 | Quail | 237.80 | 73,124.95 | 0.26087 | 0.015002 | 68.35 |
| AGENT-1 | Pipelined vLLM | 98.45 | 176,628.88 | 0.10800 | 0.006211 | 0.43 |
| AGENT-1 | SoL estimate | 47.465 | 366,357.25 | 0.05207 | 0.002994 | 0 (assumed) |
| AGENT-2 | Quail | 238.89 | 72,969.32 | 0.26206 | 0.015034 | 68.19 |
| AGENT-2 | Pipelined vLLM | 99.53 | 175,139.57 | 0.10918 | 0.006264 | 0.43 |
| AGENT-2 | SoL estimate | 47.870 | 364,144.27 | 0.05251 | 0.003013 | 0 (assumed) |

| Query | Method | Reference rows | Returned rows | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|---:|---:|
| AGENT-1 | Quail | 608 | 344 | 73.81 | 70.93 | 40.132 |
| AGENT-1 | Pipelined vLLM | 608 | 355 | 73.65 | 69.859 | 40.789 |
| AGENT-2 | Quail | 527 | 635 | 93.34 | 82.205 | 99.051 |
| AGENT-2 | Pipelined vLLM | 527 | 639 | 93.12 | 81.69 | 99.051 |

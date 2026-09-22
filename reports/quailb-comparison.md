# QUAIL-B comparison

- All 31 queries use raw document/question prompts ending in `ANSWER:`.
  LePaRD has five queries. Original LEP-7 is now LEP-5.
  Both methods use Qwen3 4B FP8, sf=0.1, and one H100.
- Measurements are reused from saved raw runs. No inference was repeated.
  Base source on `quail-results`: `/results/benchmarks/quailb/20260914T070913Z-f7beefb6`.
  BIO-4 source: `/results/benchmarks/quailb/family-runs/20260920T062701Z-bio4-4b`.
  Base saved plans and prompt token pieces match the restored definitions.
  Base scores and token counts were recalculated from saved answers.
  BIO-4 uses the saved current query hash, plan hash, and runner scores.
- BIO-1 and BIO-3 are not measured for the serious-adverse-event filter.
  The saved demographic-filter results do not match those queries.
  All plots include both queries, with missing measurements marked x.
  Comparisons below use the 29 queries with both measurements.
- Original LEP-5, LEP-6, and LEP-8 are excluded because their reference
  outputs are empty at sf=0.1 with raw prompts too. Historical IDs are
  matched by query-definition hash before measurements are reused.
- References use Qwen3 32B FP8 and the benchmark's dataset annotations.
  Base reference collection: `gt_be81cb241d74555dc2da79b5b0662554`.
  BIO-4 sf=0.1 reference collection: `gt_cd3ebdb784f64b9e028e50ea73cdedd0`.
  Answer agreement counts matching predicate answers. Output precision
  and recall compare final rows with the reference output.
- Quail uses pipelining, token-based admission, and KV rewind.
  Pipelined vLLM uses pipelining and prefix caching. Filter stages advance
  independently; joins begin after filtering finishes.
  vLLM batched-token limits: 25,305.
  Sequence limits: 4,096.
  Measured vLLM KV capacity: 479,616 tokens.
- Quail is faster on 27 of 29 queries.
  Mean speedup: 1.91x; median: 1.48x; maximum: 10.04x (BIO-2).
  Each query has equal weight. Speedup is vLLM time divided by Quail time.
- Latency excludes startup and result collection. GPU cost is query
  seconds / 3,600 * $3.9492.
  Throughput is input documents/second for filter-only queries. For queries
  with joins, it is evaluated document pairs across all join stages divided
  by query seconds. Different answers can change the work each method does.
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
  Base estimates: `/results/reports/quailb-raw-2026-09-19/sol_quailb_sf0.1.json`.
  BIO-4 estimate: `/results/sol/2026-09-20-bio4-qwen3-4b-sf0.1.json`.
- PDFs show latency, fresh input tokens, recomputed KV tokens, and answer
  agreement. Each PDF lists input counts separately for every alias.
  SoL uses lines for latency and token totals. Measurements use bars.
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

| Query | Method | Seconds | Throughput | Unit | $/query | Fresh tokens | Recomputed KV tokens | KV regret (%) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| IMDB-1 | Quail | 14.44 | 346.26 | documents/s | 0.01584 | 1,759,232 | 20,349 | 1.16 |
| IMDB-1 | Pipelined vLLM | 17.33 | 288.52 | documents/s | 0.01901 | 1,758,944 | 20,061 | 1.14 |
| IMDB-1 | SoL estimate | 6.653 | 751.59 | documents/s | 0.00730 | 1,740,485 | 0 (assumed) | 0 (assumed) |
| IMDB-2 | Quail | 21.20 | 2,830.19 | document pairs/s | 0.02326 | 2,419,232 | 20,349 | 0.84 |
| IMDB-2 | Pipelined vLLM | 25.32 | 2,369.67 | document pairs/s | 0.02778 | 2,439,624 | 40,741 | 1.67 |
| IMDB-2 | SoL estimate | 9.211 | 6,513.89 | document pairs/s | 0.01010 | 2,400,485 | 0 (assumed) | 0 (assumed) |
| IMDB-3 | Quail | 22.39 | 2,347.48 | document pairs/s | 0.02456 | 2,560,772 | 24,729 | 0.97 |
| IMDB-3 | Pipelined vLLM | 40.01 | 1,311.87 | document pairs/s | 0.04389 | 3,933,798 | 1,398,847 | 35.56 |
| IMDB-3 | SoL estimate | 9.495 | 5,060.10 | document pairs/s | 0.01042 | 2,473,217 | 0 (assumed) | 0 (assumed) |
| IMDB-4 | Quail | 17.31 | 871.40 | document pairs/s | 0.01899 | 2,004,176 | 21,606 | 1.08 |
| IMDB-4 | Pipelined vLLM | 26.26 | 581.72 | document pairs/s | 0.02881 | 2,563,494 | 577,154 | 22.51 |
| IMDB-4 | SoL estimate | 7.521 | 1,646.61 | document pairs/s | 0.00825 | 1,961,306 | 0 (assumed) | 0 (assumed) |
| IMDB-5 | Quail | 16.55 | 527.13 | document pairs/s | 0.01816 | 1,929,643 | 21,839 | 1.13 |
| IMDB-5 | Pipelined vLLM | 25.22 | 354.48 | document pairs/s | 0.02767 | 2,444,744 | 531,994 | 21.76 |
| IMDB-5 | SoL estimate | 7.408 | 1,088.54 | document pairs/s | 0.00813 | 1,931,654 | 0 (assumed) | 0 (assumed) |
| IMDB-6 | Quail | 14.93 | 334.90 | documents/s | 0.01638 | 1,774,145 | 20,349 | 1.15 |
| IMDB-6 | Pipelined vLLM | 18.25 | 273.97 | documents/s | 0.02002 | 1,818,127 | 63,473 | 3.49 |
| IMDB-6 | SoL estimate | 6.779 | 737.57 | documents/s | 0.00744 | 1,772,450 | 0 (assumed) | 0 (assumed) |
| IMDB-7 | Quail | 15.47 | 323.21 | documents/s | 0.01697 | 1,796,602 | 21,112 | 1.18 |
| IMDB-7 | Pipelined vLLM | 21.29 | 234.85 | documents/s | 0.02336 | 2,114,785 | 337,625 | 15.96 |
| IMDB-7 | SoL estimate | 6.922 | 722.34 | documents/s | 0.00759 | 1,808,678 | 0 (assumed) | 0 (assumed) |
| IMDB-8 | Quail | 26.06 | 3,522.18 | document pairs/s | 0.02859 | 2,918,999 | 91,872 | 3.15 |
| IMDB-8 | Pipelined vLLM | 39.49 | 2,331.93 | document pairs/s | 0.04332 | 3,773,186 | 942,159 | 24.97 |
| IMDB-8 | SoL estimate | 12.036 | 8,864.47 | document pairs/s | 0.01320 | 3,127,538 | 0 (assumed) | 0 (assumed) |
| IMDB-9 | Quail | 47.77 | 3,177.48 | document pairs/s | 0.05240 | 5,338,231 | 2,144,348 | 40.17 |
| IMDB-9 | Pipelined vLLM | 65.42 | 2,324.79 | document pairs/s | 0.07177 | 6,212,810 | 3,018,927 | 48.59 |
| IMDB-9 | SoL estimate | 16.431 | 10,614.78 | document pairs/s | 0.01802 | 4,265,419 | 0 (assumed) | 0 (assumed) |
| IMDB-10 | Quail | 48.40 | 2,982.40 | document pairs/s | 0.05309 | 5,479,771 | 2,132,608 | 38.92 |
| IMDB-10 | Pipelined vLLM | 80.04 | 1,806.30 | document pairs/s | 0.08780 | 7,706,984 | 4,360,757 | 56.58 |
| IMDB-10 | SoL estimate | 15.865 | 9,753.67 | document pairs/s | 0.01740 | 4,115,270 | 0 (assumed) | 0 (assumed) |

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
| BIO-4 | r (reports) = 500, n (terms) = 1,127, c (terms) = 1,127 |

| Query | Method | Seconds | Throughput | Unit | $/query | Fresh tokens | Recomputed KV tokens | KV regret (%) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| BIO-1 | Quail | Not measured | | | | | | |
| BIO-1 | Pipelined vLLM | Not measured | | | | | | |
| BIO-1 | SoL estimate | 11.055 | 45.23 | documents/s | 0.01213 | 2,055,868 | 0 (assumed) | 0 (assumed) |
| BIO-2 | Quail | 127.14 | 4,432.12 | document pairs/s | 0.13947 | 10,374,345 | 2,477 | 0.02 |
| BIO-2 | Pipelined vLLM | 1276.94 | 441.29 | document pairs/s | 1.40080 | 10,526,615 | 154,747 | 1.47 |
| BIO-2 | SoL estimate | 62.002 | 9,088.45 | document pairs/s | 0.06802 | 10,371,868 | 0 (assumed) | 0 (assumed) |
| BIO-3 | Quail | Not measured | | | | | | |
| BIO-3 | Pipelined vLLM | Not measured | | | | | | |
| BIO-3 | SoL estimate | 42.784 | 8,403.00 | document pairs/s | 0.04693 | 7,377,107 | 0 (assumed) | 0 (assumed) |
| BIO-4 | Quail | 72.48 | 3,404.87 | document pairs/s | 0.07951 | 5,717,819 | 624,144 | 10.92 |
| BIO-4 | Pipelined vLLM | 430.05 | 577.37 | document pairs/s | 0.47176 | 9,001,636 | 3,907,993 | 43.41 |
| BIO-4 | SoL estimate | 34.845 | 7,983.84 | document pairs/s | 0.03822 | 6,113,069 | 0 (assumed) | 0 (assumed) |

| Query | Method | Reference rows | Returned rows | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|---:|---:|
| BIO-1 | Quail | | Not measured | | | |
| BIO-1 | Pipelined vLLM | | Not measured | | | |
| BIO-2 | Quail | 22,582 | 116,156 | 81.86 | 15.726 | 80.892 |
| BIO-2 | Pipelined vLLM | 22,582 | 121,240 | 81.02 | 15.2 | 81.605 |
| BIO-3 | Quail | | Not measured | | | |
| BIO-3 | Pipelined vLLM | | Not measured | | | |
| BIO-4 | Quail | 321,216 | 3,850,311 | 77.00 | 2.427 | 29.092 |
| BIO-4 | Pipelined vLLM | 321,216 | 4,147,303 | 76.06 | 2.3399 | 30.211 |

### BIO-4 at sf=1.0

The prediction was that Quail would beat its 5.93x speedup at sf=0.1. The working target was 10x. At sf=1.0, Quail was 14.04x faster.

Quail took 29.26 minutes. Pipelined vLLM took 6.84 hours. Quail recomputed 17,972,915 KV tokens, compared with 50,305,674 for pipelined vLLM.

The methods evaluated different numbers of document pairs because
their answers changed which rows reached the joins. The throughput
for each method uses its own evaluated pair count.

Both methods returned many incorrect final rows. Quail's output
precision was 1.57%, compared with 1.41% for pipelined vLLM.
Output recall was 22.02% for Quail and 22.57% for pipelined vLLM.

| Method | Seconds | Document pairs/s | $/query | Fresh tokens | Recomputed KV tokens | KV regret (%) | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Quail | 1755.31 | 4,572.90 | 1.92558 | 140,583,122 | 17,972,915 | 12.78 | 80.37 | 1.5747 | 22.024 |
| Pipelined vLLM | 24640.51 | 330.44 | 27.03064 | 174,555,566 | 50,305,674 | 28.82 | 79.04 | 1.4127 | 22.572 |
| SoL estimate | 894.368 | 9,894.16 | 0.98112 | 150,837,663 | 0 (assumed) | 0 (assumed) | Not measured | Not measured | Not measured |

Input documents: r (reports) = 5,000, n (terms) = 4,144, and c (terms) = 4,144.

Measured source on `quail-results`: `/results/benchmarks/quailb/family-runs/20260920T064415Z-bio4-4b-sf1.0`.
SoL source: `/results/sol/2026-09-20-bio4-qwen3-4b-sf1.0.json`.
Reference collection: `gt_e87691add604b02c4e43f0ff5bf0cc4f`.

Quail result function call: `fc-01M2YRXV1TAVKDDPPBKNHM90XH`.
Stock vLLM result function call: `fc-01M2YXPZA8E6EJYJMSGDTR0X69`.

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

| Query | Method | Seconds | Throughput | Unit | $/query | Fresh tokens | Recomputed KV tokens | KV regret (%) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| FEV-1 | Quail | 0.29 | 1,724.14 | documents/s | 0.00032 | 33,331 | 1,359 | 4.08 |
| FEV-1 | Pipelined vLLM | 0.41 | 1,219.51 | documents/s | 0.00045 | 33,331 | 1,359 | 4.08 |
| FEV-1 | SoL estimate | 0.121 | 4,139.22 | documents/s | 0.00013 | 32,498 | 0 (assumed) | 0 (assumed) |
| FEV-2 | Quail | 29.43 | 4,875.98 | document pairs/s | 0.03228 | 3,245,254 | 678 | 0.02 |
| FEV-2 | Pipelined vLLM | 60.20 | 2,383.72 | document pairs/s | 0.06604 | 3,350,613 | 106,037 | 3.16 |
| FEV-2 | SoL estimate | 12.845 | 11,171.71 | document pairs/s | 0.01409 | 3,244,940 | 0 (assumed) | 0 (assumed) |
| FEV-3 | Quail | 21.83 | 4,746.08 | document pairs/s | 0.02395 | 2,404,957 | 2,605 | 0.11 |
| FEV-3 | Pipelined vLLM | 46.44 | 2,311.33 | document pairs/s | 0.05094 | 2,562,915 | 90,822 | 3.54 |
| FEV-3 | SoL estimate | 7.898 | 10,683.49 | document pairs/s | 0.00866 | 1,998,853 | 0 (assumed) | 0 (assumed) |
| FEV-4 | Quail | 4.83 | 2,792.75 | document pairs/s | 0.00530 | 531,453 | 2,605 | 0.49 |
| FEV-4 | Pipelined vLLM | 7.17 | 2,041.42 | document pairs/s | 0.00787 | 575,830 | 23,952 | 4.16 |
| FEV-4 | SoL estimate | 1.816 | 5,846.08 | document pairs/s | 0.00199 | 464,102 | 0 (assumed) | 0 (assumed) |
| FEV-5 | Quail | 13.70 | 4,505.91 | document pairs/s | 0.01503 | 1,515,283 | 2,776 | 0.18 |
| FEV-5 | Pipelined vLLM | 28.12 | 2,287.62 | document pairs/s | 0.03085 | 1,657,580 | 95,455 | 5.76 |
| FEV-5 | SoL estimate | 4.668 | 9,887.39 | document pairs/s | 0.00512 | 1,183,087 | 0 (assumed) | 0 (assumed) |
| FEV-6 | Quail | 4.05 | 1,984.44 | document pairs/s | 0.00444 | 398,331 | 2,776 | 0.70 |
| FEV-6 | Pipelined vLLM | 5.63 | 1,558.08 | document pairs/s | 0.00618 | 426,594 | 15,909 | 3.73 |
| FEV-6 | SoL estimate | 1.338 | 4,341.63 | document pairs/s | 0.00147 | 343,316 | 0 (assumed) | 0 (assumed) |
| FEV-7 | Quail | 55.64 | 4,833.20 | document pairs/s | 0.06104 | 6,103,058 | 135,139 | 2.21 |
| FEV-7 | Pipelined vLLM | 110.56 | 2,429.74 | document pairs/s | 0.12128 | 6,255,630 | 298,043 | 4.76 |
| FEV-7 | SoL estimate | 21.254 | 11,410.10 | document pairs/s | 0.02332 | 5,366,444 | 0 (assumed) | 0 (assumed) |
| FEV-8 | Quail | 86.36 | 4,908.51 | document pairs/s | 0.09474 | 9,476,743 | 3,120,800 | 32.93 |
| FEV-8 | Pipelined vLLM | 171.47 | 2,475.49 | document pairs/s | 0.18810 | 9,881,067 | 3,525,124 | 35.68 |
| FEV-8 | SoL estimate | 33.614 | 11,483.75 | document pairs/s | 0.03687 | 8,485,273 | 0 (assumed) | 0 (assumed) |
| FEV-9 | Quail | 39.00 | 4,669.62 | document pairs/s | 0.04278 | 4,306,910 | 1,461,116 | 33.92 |
| FEV-9 | Pipelined vLLM | 76.41 | 2,485.12 | document pairs/s | 0.08382 | 4,593,156 | 1,648,151 | 35.88 |
| FEV-9 | SoL estimate | 11.387 | 10,754.04 | document pairs/s | 0.01249 | 2,878,864 | 0 (assumed) | 0 (assumed) |
| FEV-10 | Quail | 1.68 | 110.12 | document pairs/s | 0.00184 | 187,567 | 2,773 | 1.48 |
| FEV-10 | Pipelined vLLM | 2.92 | 64.38 | document pairs/s | 0.00320 | 270,220 | 4,308 | 1.59 |
| FEV-10 | SoL estimate | 0.712 | 234.49 | document pairs/s | 0.00078 | 185,622 | 0 (assumed) | 0 (assumed) |

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

| Query | Method | Seconds | Throughput | Unit | $/query | Fresh tokens | Recomputed KV tokens | KV regret (%) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| LEP-1 | Quail | 1.10 | 454.55 | documents/s | 0.00121 | 131,947 | 1,702 | 1.29 |
| LEP-1 | Pipelined vLLM | 1.38 | 362.32 | documents/s | 0.00151 | 131,867 | 1,622 | 1.23 |
| LEP-1 | SoL estimate | 0.494 | 1,012.61 | documents/s | 0.00054 | 130,583 | 0 (assumed) | 0 (assumed) |
| LEP-2 | Quail | 131.53 | 1,646.01 | document pairs/s | 0.14429 | 15,098,947 | 1,702 | 0.01 |
| LEP-2 | Pipelined vLLM | 159.21 | 1,359.84 | document pairs/s | 0.17465 | 15,184,675 | 87,430 | 0.58 |
| LEP-2 | SoL estimate | 58.086 | 3,727.22 | document pairs/s | 0.06372 | 15,097,583 | 0 (assumed) | 0 (assumed) |
| LEP-3 | Quail | 94.22 | 1,636.04 | document pairs/s | 0.10336 | 10,807,675 | 2,058 | 0.02 |
| LEP-3 | Pipelined vLLM | 120.83 | 1,351.00 | document pairs/s | 0.13255 | 11,600,657 | 165,313 | 1.43 |
| LEP-3 | SoL estimate | 2.260 | 2,873.50 | document pairs/s | 0.00248 | 580,403 | 0 (assumed) | 0 (assumed) |
| LEP-4 | Quail | 40.29 | 1,612.06 | document pairs/s | 0.04420 | 4,640,471 | 1,852 | 0.04 |
| LEP-4 | Pipelined vLLM | 47.86 | 1,375.18 | document pairs/s | 0.05250 | 4,748,742 | 49,540 | 1.04 |
| LEP-4 | SoL estimate | 1.202 | 2,162.15 | document pairs/s | 0.00132 | 311,216 | 0 (assumed) | 0 (assumed) |
| LEP-5 | Quail | 39.47 | 1,596.15 | document pairs/s | 0.04330 | 4,556,127 | 3,420 | 0.08 |
| LEP-5 | Pipelined vLLM | 47.62 | 1,343.81 | document pairs/s | 0.05224 | 4,684,760 | 51,004 | 1.09 |
| LEP-5 | SoL estimate | 1.236 | 1,733.57 | document pairs/s | 0.00136 | 322,319 | 0 (assumed) | 0 (assumed) |

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

| Query | Method | Seconds | Throughput | Unit | $/query | Fresh tokens | Recomputed KV tokens | KV regret (%) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| AGENT-1 | Quail | 237.80 | 7.45 | documents/s | 0.26087 | 17,389,113 | 11,886,152 | 68.35 |
| AGENT-1 | Pipelined vLLM | 98.45 | 18.00 | documents/s | 0.10800 | 5,526,889 | 23,928 | 0.43 |
| AGENT-1 | SoL estimate | 47.465 | 37.33 | documents/s | 0.05207 | 5,502,961 | 0 (assumed) | 0 (assumed) |
| AGENT-2 | Quail | 238.89 | 7.42 | documents/s | 0.26206 | 17,431,641 | 11,886,152 | 68.19 |
| AGENT-2 | Pipelined vLLM | 99.53 | 17.80 | documents/s | 0.10918 | 5,569,417 | 23,928 | 0.43 |
| AGENT-2 | SoL estimate | 47.870 | 37.02 | documents/s | 0.05251 | 5,545,489 | 0 (assumed) | 0 (assumed) |

| Query | Method | Reference rows | Returned rows | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|---:|---:|
| AGENT-1 | Quail | 608 | 344 | 73.81 | 70.93 | 40.132 |
| AGENT-1 | Pipelined vLLM | 608 | 355 | 73.65 | 69.859 | 40.789 |
| AGENT-2 | Quail | 527 | 635 | 93.34 | 82.205 | 99.051 |
| AGENT-2 | Pipelined vLLM | 527 | 639 | 93.12 | 81.69 | 99.051 |

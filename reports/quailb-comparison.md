# QUAIL-B comparison

- All 30 retained queries use Qwen3's chat format with thinking disabled.
  Both methods use Qwen3 4B FP8, sf=0.1, lf=1, and one H100.
  Quail and pipelined stock vLLM share a physical GPU within each family.
  Pipelined stock vLLM advances documents through filter stages
  independently, then starts joins after filtering finishes.
- Previous LEP-5, LEP-6, and LEP-8 were removed because their reference
  filters leave no rows at sf=0.1. Previous LEP-7 is now LEP-5.
  This report selects 60 measurements from the saved 66-measurement run.
  It checks query definitions before mapping historical IDs.
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
- Quail is faster on 28 of 30 queries.
  The arithmetic mean speedup is 1.68x,
  the median is 1.50x, and the maximum is
  4.99x on BIO-2. Each query has equal weight.
  Speedup is pipelined stock vLLM time divided by Quail time.
  Query time excludes startup and result collection.
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
  unlimited KV: each document once, each question and join frame once
  per document, and each pair's partner label, partner document, and
  answer cue once per pair. They are included in fresh tokens, not added
  to them. The benchmark computes this minimum from saved answers after
  the run. KV regret is recomputed tokens / fresh tokens * 100%.
  A missing minimum or zero computed tokens leaves regret unreported.
- Token throughput is total requested input tokens divided by query seconds.
  Count each complete prompt once per evaluated filter or join pair,
  including tokens served from KV. Exclude generated answer tokens.
  Counts come from saved answers, prompt pieces, and document tokens.
  All retained vLLM totals match its recorded fresh plus cached token counts.
  Cost per million input tokens is query dollars / requested tokens * 1e6.
  It uses the same complete-prompt count as tokens per second.
  Different survivors can change which prompts a method evaluates.
  The SoL line counts complete prompts under its reference survivors.
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

| Query | Method | Seconds | Tokens/second | $/query | $/million input tokens | KV regret (%) |
|---|---|---:|---:|---:|---:|---:|
| IMDB-1 | Quail | 15.26 | 121,837.02 | 0.01674 | 0.009004 | 1.90 |
| IMDB-1 | Pipelined vLLM | 18.37 | 101,210.29 | 0.02015 | 0.010839 | 1.88 |
| IMDB-1 | SoL estimate | 6.995 | 265,791.44 | 0.00767 | 0.004127 | 0 (assumed) |
| IMDB-2 | Quail | 26.39 | 839,173.78 | 0.02895 | 0.001307 | 1.17 |
| IMDB-2 | Pipelined vLLM | 34.89 | 634,731.90 | 0.03827 | 0.001728 | 2.02 |
| IMDB-2 | SoL estimate | 11.483 | 1,928,548.66 | 0.01260 | 0.000569 | 0 (assumed) |
| IMDB-3 | Quail | 26.93 | 775,578.35 | 0.02954 | 0.001414 | 1.27 |
| IMDB-3 | Pipelined vLLM | 45.88 | 456,119.68 | 0.05033 | 0.002405 | 30.98 |
| IMDB-3 | SoL estimate | 11.644 | 1,729,525.86 | 0.01277 | 0.000634 | 0 (assumed) |
| IMDB-4 | Quail | 18.55 | 440,392.29 | 0.02035 | 0.002491 | 1.67 |
| IMDB-4 | Pipelined vLLM | 26.92 | 300,073.92 | 0.02953 | 0.003656 | 18.33 |
| IMDB-4 | SoL estimate | 8.602 | 1,049,019.42 | 0.00944 | 0.001046 | 0 (assumed) |
| IMDB-5 | Quail | 17.51 | 346,578.01 | 0.01921 | 0.003165 | 1.77 |
| IMDB-5 | Pipelined vLLM | 24.09 | 247,009.42 | 0.02643 | 0.004441 | 12.76 |
| IMDB-5 | SoL estimate | 8.377 | 919,000.93 | 0.00919 | 0.001194 | 0 (assumed) |
| IMDB-6 | Quail | 15.68 | 150,707.97 | 0.01720 | 0.007279 | 1.88 |
| IMDB-6 | Pipelined vLLM | 18.97 | 124,069.90 | 0.02081 | 0.008842 | 3.39 |
| IMDB-6 | SoL estimate | 7.240 | 339,928.98 | 0.00794 | 0.003227 | 0 (assumed) |
| IMDB-7 | Quail | 15.97 | 166,566.75 | 0.01752 | 0.006586 | 1.89 |
| IMDB-7 | Pipelined vLLM | 20.36 | 129,715.18 | 0.02233 | 0.008457 | 8.01 |
| IMDB-7 | SoL estimate | 7.461 | 388,906.21 | 0.00819 | 0.002821 | 0 (assumed) |
| IMDB-8 | Quail | 28.70 | 897,864.88 | 0.03148 | 0.001222 | 1.96 |
| IMDB-8 | Pipelined vLLM | 39.55 | 649,532.01 | 0.04339 | 0.001689 | 8.72 |
| IMDB-8 | SoL estimate | 16.449 | 2,515,879.32 | 0.01804 | 0.000436 | 0 (assumed) |
| IMDB-9 | Quail | 53.27 | 830,224.37 | 0.05844 | 0.001321 | 31.95 |
| IMDB-9 | Pipelined vLLM | 66.65 | 634,689.99 | 0.07312 | 0.001728 | 34.91 |
| IMDB-9 | SoL estimate | 23.066 | 2,835,627.74 | 0.02530 | 0.000387 | 0 (assumed) |
| IMDB-10 | Quail | 54.28 | 801,141.21 | 0.05955 | 0.001369 | 30.91 |
| IMDB-10 | Pipelined vLLM | 81.20 | 515,418.71 | 0.08908 | 0.002128 | 45.06 |
| IMDB-10 | SoL estimate | 22.415 | 2,744,697.38 | 0.02459 | 0.000400 | 0 (assumed) |

Correctness against saved reference labels:

| Query | Method | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|
| IMDB-1 | Quail | 92.54 | 93.558 | 97.373 |
| IMDB-1 | Pipelined vLLM | 92.62 | 93.585 | 97.448 |
| IMDB-2 | Quail | 64.32 | 98.876 | 0.81542 |
| IMDB-2 | Pipelined vLLM | 64.26 | 99.286 | 0.644 |
| IMDB-3 | Quail | 66.56 | 97.605 | 0.93217 |
| IMDB-3 | Pipelined vLLM | 66.46 | 96.094 | 0.70342 |
| IMDB-4 | Quail | 66.21 | 91.304 | 0.64546 |
| IMDB-4 | Pipelined vLLM | 66.31 | 91.667 | 0.50715 |
| IMDB-5 | Quail | 70.33 | 88.235 | 0.63898 |
| IMDB-5 | Pipelined vLLM | 70.56 | 89.655 | 0.55378 |
| IMDB-6 | Quail | 92.53 | 87.897 | 76.71 |
| IMDB-6 | Pipelined vLLM | 92.66 | 88.454 | 76.277 |
| IMDB-7 | Quail | 91.73 | 88.462 | 62.646 |
| IMDB-7 | Pipelined vLLM | 91.84 | 88.91 | 61.349 |
| IMDB-8 | Quail | 74.21 | 70.844 | 0.33404 |
| IMDB-8 | Pipelined vLLM | 74.22 | 72.754 | 0.29304 |
| IMDB-9 | Quail | 71.98 | 68.887 | 0.0037753 |
| IMDB-9 | Pipelined vLLM | 68.57 | 68.025 | 0.0025487 |
| IMDB-10 | Quail | 72.94 | 68.308 | 0.0043301 |
| IMDB-10 | Pipelined vLLM | 70.04 | 66.642 | 0.0027763 |

## BIO

[Open the BIO vector PDF](plots/quailb_bio.pdf)

Figure: plots/quailb_bio.pdf

| Query | Input documents by alias and set |
|---|---|
| BIO-1 | r (reports) = 500 |
| BIO-2 | r (reports) = 500, m (terms) = 1,127 |
| BIO-3 | r (reports) = 500, m (terms) = 1,127 |

BIO-3 filter survivors: 362 in the reference, 349 in Quail, and 350 in pipelined stock vLLM.

vLLM's BIO-2 time is 13% lower than in the September 12 run with
raw prompts, saved at
`/results/benchmarks/quailb/family-runs/20260912T225100Z-902686c5/`.
Its time per pair is 13% lower on BIO-3 as well. The cause is the
host, not the prompt layout and not KV reuse. The two runs were on
different physical GPUs. On one container, after this report's
runs, the same join took 2.152 and 2.115 ms per pair with raw
prompts and 2.141 and 2.098 ms with chat prompts: two rounds in
alternating order, each on a fresh engine with BIO-1 run first,
and the chat runs' fresh tokens matched this run's exactly. The
result is
`/results/ablations/bio2-prompt-layout-20260918T155056Z/result.json`
(`fc-01M2TKCMJQVM0HV104NVCHTKY5`; the cell is
`experiments/bio2_prompt_layout.py`). Across the three machines
the raw layout has run on, the join took 1069, 1163, and 1213
seconds, a 13% spread with nothing but the host changing. The
September 12 recomputed KV figures came from an earlier benchmark
rule that counted each pair's partner label once per anchor and
shared partner document prefixes across pairs. On BIO-2 that
understates the minimum by 8,293 tokens per anchor, 4,146,500 in
total, which is all of Quail's 4,148,977 recomputed tokens in that
run beyond its 2,477 filter-stage tokens. Under the current rule,
vLLM recomputed 154,747 tokens there and 157,341 here.

| Query | Method | Seconds | Tokens/second | $/query | $/million input tokens | KV regret (%) |
|---|---|---:|---:|---:|---:|---:|
| BIO-1 | Quail | 22.34 | 92,585.14 | 0.02451 | 0.011849 | 0.19 |
| BIO-1 | Pipelined vLLM | 25.15 | 82,240.64 | 0.02759 | 0.013339 | 0.19 |
| BIO-1 | SoL estimate | 11.111 | 186,158.96 | 0.01219 | 0.005893 | 0 (assumed) |
| BIO-2 | Quail | 187.01 | 12,488,143.44 | 0.20515 | 0.000088 | 0.03 |
| BIO-2 | Pipelined vLLM | 933.23 | 2,502,499.60 | 1.02375 | 0.000438 | 1.01 |
| BIO-2 | SoL estimate | 93.223 | 25,051,793.33 | 0.10227 | 0.000044 | 0 (assumed) |
| BIO-3 | Quail | 136.51 | 11,490,998.83 | 0.14975 | 0.000095 | 0.04 |
| BIO-3 | Pipelined vLLM | 647.99 | 2,385,829.14 | 0.71085 | 0.000460 | 11.33 |
| BIO-3 | SoL estimate | 69.241 | 22,958,114.26 | 0.07596 | 0.000048 | 0 (assumed) |

Correctness against saved reference labels:

| Query | Method | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|
| BIO-1 | Quail | 88.60 | 93.696 | 90.331 |
| BIO-1 | Pipelined vLLM | 89.60 | 94.286 | 91.16 |
| BIO-2 | Quail | 97.13 | 94.034 | 10.699 |
| BIO-2 | Pipelined vLLM | 97.13 | 94.771 | 10.588 |
| BIO-3 | Quail | 96.58 | 90.405 | 10.408 |
| BIO-3 | Pipelined vLLM | 96.58 | 90.062 | 10.487 |

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

| Query | Method | Seconds | Tokens/second | $/query | $/million input tokens | KV regret (%) |
|---|---|---:|---:|---:|---:|---:|
| FEV-1 | Quail | 0.37 | 117,110.81 | 0.00041 | 0.009367 | 6.59 |
| FEV-1 | Pipelined vLLM | 0.57 | 76,019.30 | 0.00063 | 0.014431 | 6.59 |
| FEV-1 | SoL estimate | 0.155 | 278,916.47 | 0.00017 | 0.003933 | 0 (assumed) |
| FEV-2 | Quail | 40.58 | 1,832,355.77 | 0.04452 | 0.000599 | 0.03 |
| FEV-2 | Pipelined vLLM | 84.61 | 878,820.43 | 0.09282 | 0.001248 | 3.37 |
| FEV-2 | SoL estimate | 18.015 | 4,127,567.03 | 0.01976 | 0.000266 | 0 (assumed) |
| FEV-3 | Quail | 22.20 | 1,756,858.20 | 0.02435 | 0.000624 | 0.20 |
| FEV-3 | Pipelined vLLM | 41.77 | 951,743.91 | 0.04582 | 0.001153 | 3.66 |
| FEV-3 | SoL estimate | 11.039 | 3,991,679.53 | 0.01211 | 0.000275 | 0 (assumed) |
| FEV-4 | Quail | 4.47 | 1,019,569.35 | 0.00490 | 0.001076 | 1.00 |
| FEV-4 | Pipelined vLLM | 6.68 | 748,997.75 | 0.00733 | 0.001465 | 4.05 |
| FEV-4 | SoL estimate | 2.169 | 2,449,754.89 | 0.00238 | 0.000448 | 0 (assumed) |
| FEV-5 | Quail | 12.93 | 1,671,689.95 | 0.01418 | 0.000656 | 0.35 |
| FEV-5 | Pipelined vLLM | 25.44 | 868,685.26 | 0.02791 | 0.001263 | 6.60 |
| FEV-5 | SoL estimate | 6.459 | 3,815,750.31 | 0.00709 | 0.000287 | 0 (assumed) |
| FEV-6 | Quail | 3.26 | 820,741.72 | 0.00358 | 0.001337 | 1.39 |
| FEV-6 | Pipelined vLLM | 5.33 | 549,685.37 | 0.00585 | 0.001996 | 4.29 |
| FEV-6 | SoL estimate | 1.574 | 1,983,154.15 | 0.00173 | 0.000553 | 0 (assumed) |
| FEV-7 | Quail | 61.95 | 1,821,414.40 | 0.06796 | 0.000602 | 2.01 |
| FEV-7 | Pipelined vLLM | 117.19 | 959,084.12 | 0.12856 | 0.001144 | 5.24 |
| FEV-7 | SoL estimate | 21.556 | 4,112,617.24 | 0.02365 | 0.000267 | 0 (assumed) |
| FEV-8 | Quail | 110.65 | 1,851,852.57 | 0.12138 | 0.000592 | 30.95 |
| FEV-8 | Pipelined vLLM | 214.42 | 959,712.20 | 0.23522 | 0.001143 | 33.84 |
| FEV-8 | SoL estimate | 29.793 | 4,134,326.72 | 0.03268 | 0.000265 | 0 (assumed) |
| FEV-9 | Quail | 34.07 | 1,750,376.05 | 0.03737 | 0.000627 | 32.93 |
| FEV-9 | Pipelined vLLM | 66.45 | 922,258.69 | 0.07290 | 0.001189 | 34.77 |
| FEV-9 | SoL estimate | 9.164 | 3,909,699.32 | 0.01005 | 0.000281 | 0 (assumed) |
| FEV-10 | Quail | 1.77 | 157,416.38 | 0.00194 | 0.006969 | 2.50 |
| FEV-10 | Pipelined vLLM | 3.01 | 93,168.77 | 0.00330 | 0.011774 | 2.22 |
| FEV-10 | SoL estimate | 0.780 | 364,140.62 | 0.00086 | 0.003013 | 0 (assumed) |

Correctness against saved reference labels:

| Query | Method | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|
| FEV-1 | Quail | 93.20 | 100 | 88.514 |
| FEV-1 | Pipelined vLLM | 94.20 | 100 | 90.203 |
| FEV-2 | Quail | 99.24 | 22.727 | 50.71 |
| FEV-2 | Pipelined vLLM | 99.23 | 22.549 | 51.318 |
| FEV-3 | Quail | 98.81 | 17.625 | 39.452 |
| FEV-3 | Pipelined vLLM | 98.83 | 17.831 | 40.548 |
| FEV-4 | Quail | 99.88 | 100 | 38.462 |
| FEV-4 | Pipelined vLLM | 99.54 | 27.5 | 42.308 |
| FEV-5 | Quail | 98.28 | 18.316 | 47.148 |
| FEV-5 | Pipelined vLLM | 98.31 | 18.182 | 47.909 |
| FEV-6 | Quail | 99.80 | 100 | 52.632 |
| FEV-6 | Pipelined vLLM | 99.30 | 29.73 | 57.895 |
| FEV-7 | Quail | 96.55 | 0.0092081 | 2.9167 |
| FEV-7 | Pipelined vLLM | 96.25 | 0.008519 | 2.9167 |
| FEV-8 | Quail | 97.15 | 0.0019768 | 1.8248 |
| FEV-8 | Pipelined vLLM | 96.90 | 0.0015845 | 1.6423 |
| FEV-9 | Quail | 95.01 | 0.0023412 | 5.3097 |
| FEV-9 | Pipelined vLLM | 94.60 | 0.0017721 | 4.4248 |
| FEV-10 | Quail | 94.92 | 99.099 | 89.431 |
| FEV-10 | Pipelined vLLM | 94.09 | 84.173 | 95.122 |

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

| Query | Method | Seconds | Tokens/second | $/query | $/million input tokens | KV regret (%) |
|---|---|---:|---:|---:|---:|---:|
| LEP-1 | Quail | 1.14 | 124,514.91 | 0.00125 | 0.008810 | 2.25 |
| LEP-1 | Pipelined vLLM | 1.48 | 95,910.14 | 0.00162 | 0.011438 | 2.20 |
| LEP-1 | SoL estimate | 0.528 | 268,589.84 | 0.00058 | 0.004084 | 0 (assumed) |
| LEP-2 | Quail | 144.14 | 504,773.49 | 0.15812 | 0.002173 | 0.02 |
| LEP-2 | Pipelined vLLM | 176.26 | 412,788.22 | 0.19336 | 0.002658 | 0.37 |
| LEP-2 | SoL estimate | 65.742 | 1,106,720.03 | 0.07212 | 0.000991 | 0 (assumed) |
| LEP-3 | Quail | 33.48 | 573,064.93 | 0.03673 | 0.001914 | 0.08 |
| LEP-3 | Pipelined vLLM | 40.83 | 493,506.27 | 0.04479 | 0.002223 | 1.32 |
| LEP-3 | SoL estimate | 3.206 | 1,376,321.79 | 0.00352 | 0.000797 | 0 (assumed) |
| LEP-4 | Quail | 5.90 | 526,265.76 | 0.00647 | 0.002084 | 0.47 |
| LEP-4 | Pipelined vLLM | 9.04 | 435,941.48 | 0.00992 | 0.002516 | 1.91 |
| LEP-4 | SoL estimate | 2.412 | 1,339,812.16 | 0.00265 | 0.000819 | 0 (assumed) |
| LEP-5 | Quail | 3.31 | 388,061.63 | 0.00363 | 0.002827 | 1.55 |
| LEP-5 | Pipelined vLLM | 5.32 | 316,522.56 | 0.00584 | 0.003466 | 2.64 |
| LEP-5 | SoL estimate | 2.381 | 1,214,356.27 | 0.00261 | 0.000903 | 0 (assumed) |

Correctness against saved reference labels:

| Query | Method | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|
| LEP-1 | Quail | 80.80 | 16.071 | 90 |
| LEP-1 | Pipelined vLLM | 80.60 | 16.522 | 95 |
| LEP-2 | Quail | 99.60 | 17.73 | 20 |
| LEP-2 | Pipelined vLLM | 99.58 | 17.089 | 21.6 |
| LEP-3 | Quail | 99.47 | 2.5641 | 10 |
| LEP-3 | Pipelined vLLM | 99.45 | 2.381 | 10 |
| LEP-4 | Quail | 97.60 | 0 | 0 |
| LEP-4 | Pipelined vLLM | 98.07 | 0 | 0 |
| LEP-5 | Quail | 89.04 | 0 | 0 |
| LEP-5 | Pipelined vLLM | 91.57 | 0 | 0 |

## AGENT

[Open the AGENT vector PDF](plots/quailb_agent.pdf)

Figure: plots/quailb_agent.pdf

| Query | Input documents by alias and set |
|---|---|
| AGENT-1 | t (agent_traces) = 1,772 |
| AGENT-2 | t (agent_traces) = 1,772 |

| Query | Method | Seconds | Tokens/second | $/query | $/million input tokens | KV regret (%) |
|---|---|---:|---:|---:|---:|---:|
| AGENT-1 | Quail | 238.98 | 72,912.18 | 0.26216 | 0.015045 | 68.25 |
| AGENT-1 | Pipelined vLLM | 99.72 | 174,734.79 | 0.10939 | 0.006278 | 0.45 |
| AGENT-1 | SoL estimate | 47.762 | 364,821.97 | 0.05239 | 0.003007 | 0 (assumed) |
| AGENT-2 | Quail | 239.73 | 72,861.47 | 0.26298 | 0.015056 | 68.08 |
| AGENT-2 | Pipelined vLLM | 100.29 | 174,165.73 | 0.11002 | 0.006299 | 0.44 |
| AGENT-2 | SoL estimate | 48.168 | 362,631.73 | 0.05284 | 0.003025 | 0 (assumed) |

Correctness against saved reference labels:

| Query | Method | Answer agreement (%) | Output precision (%) | Output recall (%) |
|---|---|---:|---:|---:|
| AGENT-1 | Quail | 72.40 | 69.787 | 64.062 |
| AGENT-1 | Pipelined vLLM | 71.78 | 68.457 | 64.714 |
| AGENT-2 | Quail | 93.06 | 83.264 | 99.5 |
| AGENT-2 | Pipelined vLLM | 92.66 | 82.459 | 99.5 |

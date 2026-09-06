# QUAIL-B on the request backends: two SoL estimates and two KV regrets

Date: 2026-09-05. One H100! per query family on Modal, Qwen3 4B fp8, all
32 QUAIL-B queries at scale factor 0.1.

## Setup

- Four methods ran every query. Quail uses pipelining, token-based
  admission, and KV rewind. Stock vLLM uses operator-at-a-time filter
  execution. Pipelined vLLM submits a document's next filter as
  soon as its current filter answers. Pipelined SGLang does the same
  through SGLang, in waves of blocking generate calls, and submits join
  pairs suffix-major in anchor tiles. All three request backends are
  `RequestBackend` instances under `quail/backends/`; the standalone
  runners they replaced were removed in the same change.
- One Modal container per query family for Quail and the two vLLM
  backends, on one H100!. Quail runs in a fresh process; the two vLLM
  backends share one loaded model. SGLang runs in a second container per
  family on its own H100!, at the same time. Within the shared container
  the order is Quail, then stock vLLM, then pipelined vLLM, each finishing
  the family's queries before the next starts.
- vLLM 0.26.0 with `gpu_memory_utilization=0.91`, 25,305 batched tokens,
  4,096 sequences, 479,616 KV tokens in 16-token blocks, and one CUDA
  graph capture size of 8,192. SGLang 0.5.18 with 16-token pages, 4,096
  sequences, and 403,744 KV tokens. Quail clears KV before each query;
  vLLM and SGLang reset their prefix caches before each configuration.
- Query time and cost exclude model startup. Cost uses $3.9492 per H100!
  hour from `quail.bench.evaluate.H100_USD_PER_HOUR`.
- Accuracy is against ground truth collection
  `gt_77bb8b128743a79aedddaa24c808c3f8`, answered by Qwen3 32B fp8.
- The speed of light (SoL) file `/results/sol/sol_quailb_sf0.1.json`
  carries two estimates per query, described next.

## Two SoL estimates and two KV regrets

- **Per document.** Each alias's document is computed once and reused
  only across its own questions. The matching **per document KV regret**
  counts a document's own prefix recomputed after an earlier request of
  the same query had computed it. First use is never regret.
- **Distinct prefix.** Each distinct token prefix in the query's scanned
  documents is computed once, across documents and across aliases of one
  column. The credit is the size of the corpus prefix trie
  (`quail.runtime.tokens.shared_prefix_lengths`), plus every copy of a
  column scanned under a second alias. The matching **distinct prefix KV
  regret** is the per document regret plus those shared prefix tokens the
  engine recomputed, minus the cached tokens its requests received from
  other documents' requests (`cross_row_cached_tokens`).
- Cross-row cached tokens count only hits inside a document's own tokens
  that the document's own earlier request cannot explain. Hits on the
  preamble, the label tokens after a join prefix, the question, or the KV
  block that straddles the end of a prefix count for neither regret nor
  cross-row. Quail never reuses KV across rows, so its cross-row count is
  zero by construction.
- The evaluator that ran with this benchmark charged shared prefixes per
  alias. `reports/make_quailb_two_regrets_plots.py` derives the distinct
  prefix regret from the recorded regret, the recorded cross-row tokens,
  and the SoL file's corpus statistics, so the figures use the definition
  above. This changes 20 rows, all on the five self join queries IMDB-9,
  IMDB-10, FEV-7, FEV-8, and FEV-9. The SoL file's corpus statistics come
  from the transformers tokenizer while the run's token store uses
  bpe-qwen; the two differ by 3 shared tokens on the 5,000 reviews, which
  is why the other rows differ by that much.

## Prediction

Stated when the run was launched: totals within 10 percent of the same
day's first run (Quail about 1,630 s over 32 queries, stock vLLM about
3,620 s, pipelined vLLM about 3,670 s, pipelined SGLang about 4,370 s),
answer accuracy 69 to 73 percent for every method, per document regrets
unchanged, no negative distinct prefix regret, the request backends'
distinct prefix regret on join queries within a few thousand tokens of
their per document regret, and AGENT-1 cross-row cached tokens still
about 11.86 million so that vLLM's distinct prefix regret stays about
20,000 tokens against Quail's 11.9 million.

## Result

All 32 queries completed for all four methods with no failed query.

- Quail took 1,611.71 seconds. Stock vLLM took 3,223.84, pipelined vLLM
  3,153.17, and pipelined SGLang 4,255.02. Quail was 2.00 times faster
  than stock vLLM, 1.96 times faster than pipelined vLLM, and 2.64 times
  faster than pipelined SGLang by total query time.
- Quail had the lowest time on 29 of 32 queries. Pipelined vLLM was
  fastest on AGENT-1 and AGENT-2; pipelined SGLang was fastest on LEP-6.
- Quail cost $1.7680 for all 32 queries; stock vLLM $3.5366, pipelined
  vLLM $3.4590, pipelined SGLang $4.6678.
- Answer accuracy: Quail 70.07%, both vLLM backends 69.03%, SGLang
  72.91%.
- Quail is 3.55 times its distinct prefix SoL and 2.55 times its per
  document SoL over the 32 queries. Stock vLLM is 7.11 and 5.10 times,
  pipelined vLLM 6.95 and 4.98, pipelined SGLang 9.38 and 6.73.
- Per document KV regret: Quail 3,904,900 tokens, stock vLLM 9,891,456,
  pipelined vLLM 8,082,656, pipelined SGLang 8,998,896. Quail's is 2.48%
  of its fresh tokens.
- Distinct prefix KV regret: Quail 31,153,292 tokens, stock vLLM
  13,411,626, pipelined vLLM 11,601,014, pipelined SGLang 27,148,275.
  Quail's number is the largest because it does not reuse KV across rows
  at all: 23.8 million of it is the two agent queries and 4.2 million the
  two self join queries IMDB-9 and IMDB-10.

Against the prediction:

- Quail's total (1,611.71 s) is within 1.1% of the first run. The request
  backends came in lower than predicted: stock vLLM by 11.0%, pipelined
  vLLM by 14.0%, pipelined SGLang by 2.7%. The difference is in the two
  BioDEX joins: stock vLLM took 958 s on BIO-2 against 1,063 s in the
  first run, and 515 s on BIO-3 against 580 s. This experiment does not
  isolate the cause; the same code and settings ran both times.
- Accuracy, per document regrets, and the AGENT-1 numbers (11,862,224
  cross-row tokens; 20,386 distinct prefix regret for both vLLM backends
  against Quail's 11,882,610) matched.
- No distinct prefix regret is negative. On single-alias join queries the
  request backends' distinct regret exceeds their per document regret by
  at most the column's shared prefix tokens (10,352 on the reviews), as
  predicted. On the five self join queries it is not:
  with cross-alias reuse in the definition, the second alias's whole
  prefix set is chargeable, and only SGLang served it (next section).

## Aggregate metrics

Figure: plots/quailb_two_regrets_metrics.png

| Method | Total time (s) | Total cost | Filter throughput | Join throughput | KV regret, per document | Regret / fresh tokens | KV regret, distinct prefix | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Quail | 1,611.71 | $1.7680 | 37.4 docs/s | 3,165.2 pairs/s | 3,904,900 | 2.48% | 31,153,292 | 70.07% |
| Stock vLLM | 3,223.84 | $3.5366 | 68.5 docs/s | 1,161.2 pairs/s | 9,891,456 | 6.90% | 13,411,626 | 69.03% |
| Pipelined vLLM | 3,153.17 | $3.4590 | 72.3 docs/s | 1,183.1 pairs/s | 8,082,656 | 5.71% | 11,601,014 | 69.03% |
| Pipelined SGLang | 4,255.02 | $4.6678 | 39.2 docs/s | 867.6 pairs/s | 8,998,896 | 6.02% | 27,148,275 | 72.91% |

The distinct prefix regret column covers all 32 queries for every method.

## Time relative to SoL

| Method | Time / SoL, distinct prefix | Median | Time / SoL, per document | Median | Best query | Worst query |
|---|---:|---:|---:|---:|---|---|
| Quail | 3.55x | 2.65x | 2.55x | 2.43x | BIO-1 (1.96x) | LEP-5 (50.55x) |
| Stock vLLM | 7.11x | 4.37x | 5.10x | 4.19x | AGENT-2 (2.14x) | LEP-5 (70.28x) |
| Pipelined vLLM | 6.95x | 4.43x | 4.98x | 4.20x | AGENT-2 (2.09x) | LEP-5 (69.97x) |
| Pipelined SGLang | 9.38x | 7.11x | 6.73x | 6.80x | BIO-1 (2.25x) | LEP-3 (73.34x) |

The worst ratios are LePaRD's dependent joins (LEP-3 through LEP-7). The
SoL uses the ground truth survivors of the LEP1 filter, 14 of 500, while
Qwen3 4B passes 356, so every method does about 25 times the modeled join
work. That gap is in the model's answers, not the scheduler.

## Results by family

| Family | Queries | Quail (s) | Stock vLLM (s) | Pipelined vLLM (s) | Pipelined SGLang (s) |
|---|---:|---:|---:|---:|---:|
| IMDB | 10 | 268.93 | 375.64 | 355.28 | 607.18 |
| BioDEX | 3 | 240.94 | 1,499.43 | 1,468.62 | 1,547.07 |
| FEVER | 9 | 283.69 | 723.37 | 712.58 | 1,084.55 |
| LePaRD | 8 | 336.46 | 420.43 | 417.67 | 580.01 |
| Agent | 2 | 481.69 | 204.97 | 199.02 | 436.21 |

Quail wins every family except Agent. On the two agent queries pipelined
vLLM is 2.4 times faster than Quail: the 1,772 trace rows are prefixes of
one another, vLLM's prefix cache serves 11.86 million of their tokens
from KV, and Quail computes every row from scratch. That is the whole
gap: Quail's distinct prefix regret on AGENT-1 is 11,882,610 tokens,
vLLM's 20,386. SGLang's radix cache served 4.32 million tokens there, a
third of what vLLM served, and finished at 218 s against vLLM's 99 s.

## Per document KV regret

Queries with a nonzero value for at least one method:

| Query | Quail | Stock vLLM | Pipelined vLLM | Pipelined SGLang |
|---|---:|---:|---:|---:|
| IMDB-2 | 0 | 0 | 0 | 130,384 |
| IMDB-3 | 1,217,732 | 1,326,432 | 1,326,432 | 1,411,856 |
| IMDB-4 | 361,440 | 1,061,328 | 533,376 | 502,123 |
| IMDB-5 | 188,587 | 760,512 | 490,496 | 270,687 |
| IMDB-6 | 0 | 561,136 | 33,184 | 9,963 |
| IMDB-7 | 0 | 760,512 | 278,000 | 19,471 |
| IMDB-8 | 0 | 860,032 | 860,032 | 1,043,552 |
| IMDB-9 | 0 | 860,032 | 860,032 | 1,173,936 |
| IMDB-10 | 1,217,732 | 2,186,464 | 2,186,464 | 2,455,408 |
| BIO-2 | 0 | 0 | 0 | 385,312 |
| BIO-3 | 919,409 | 1,079,840 | 1,079,840 | 1,285,376 |
| FEV-2 | 0 | 0 | 0 | 18,320 |
| FEV-3 | 0 | 0 | 0 | 18,320 |
| FEV-4 | 0 | 0 | 0 | 17,567 |
| FEV-5 | 0 | 44,000 | 43,744 | 18,720 |
| FEV-6 | 0 | 0 | 0 | 15,071 |
| FEV-7 | 0 | 0 | 0 | 24,320 |
| FEV-8 | 0 | 132,384 | 132,384 | 30,784 |
| FEV-9 | 0 | 132,384 | 132,384 | 30,784 |
| LEP-2 | 0 | 0 | 0 | 33,872 |
| LEP-3 | 0 | 71,936 | 71,936 | 27,328 |
| LEP-4 | 0 | 19,360 | 19,360 | 16,564 |
| LEP-5 | 0 | 13,664 | 13,664 | 19,628 |
| LEP-6 | 0 | 1,888 | 1,776 | 19,773 |
| LEP-7 | 0 | 19,552 | 19,552 | 16,564 |
| LEP-8 | 0 | 0 | 0 | 3,213 |

Quail has nonzero per document regret on five queries, all filter-then-join
chains on IMDB and BioDEX where the filter's survivors did not fit in the
retained KV pool. Both vLLM backends have it on 17 queries. SGLang has it
on 26, including every join-only query: its radix cache did not serve the
whole anchor prefix to every later pair of the same anchor.

## Distinct prefix KV regret and its parts

| Query | Shared prefix tokens | Quail cross row | Quail distinct | Stock vLLM cross row | Stock vLLM distinct | Pipelined vLLM cross row | Pipelined vLLM distinct | Pipelined SGLang cross row | Pipelined SGLang distinct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IMDB-1 | 10,352 | 0 | 10,352 | 252 | 10,100 | 252 | 10,100 | 0 | 10,352 |
| IMDB-2 | 10,352 | 0 | 10,352 | 250 | 10,102 | 250 | 10,102 | 0 | 140,736 |
| IMDB-3 | 10,352 | 0 | 1,228,084 | 252 | 1,336,532 | 252 | 1,336,532 | 0 | 1,422,208 |
| IMDB-4 | 10,352 | 0 | 371,792 | 265 | 1,071,415 | 718 | 543,010 | 337 | 512,138 |
| IMDB-5 | 10,352 | 0 | 198,939 | 265 | 770,599 | 718 | 500,130 | 306 | 280,733 |
| IMDB-6 | 10,352 | 0 | 10,352 | 252 | 571,236 | 705 | 42,831 | 324 | 19,991 |
| IMDB-7 | 10,352 | 0 | 10,352 | 252 | 770,612 | 705 | 287,647 | 293 | 29,530 |
| IMDB-8 | 10,352 | 0 | 10,352 | 250 | 870,134 | 250 | 870,134 | 0 | 1,053,904 |
| IMDB-9 | 1,504,585 | 0 | 1,504,585 | 500 | 2,364,117 | 500 | 2,364,117 | 0 | 2,678,521 |
| IMDB-10 | 1,504,585 | 0 | 2,722,317 | 502 | 3,690,547 | 502 | 3,690,547 | 0 | 3,959,993 |
| BIO-1 | 1,486 | 0 | 1,486 | 16 | 1,470 | 16 | 1,470 | 16 | 1,470 |
| BIO-2 | 1,486 | 0 | 1,486 | 14 | 1,472 | 14 | 1,472 | 14 | 386,784 |
| BIO-3 | 1,486 | 0 | 920,895 | 16 | 1,081,310 | 16 | 1,081,310 | 16 | 1,286,846 |
| FEV-1 | 361 | 0 | 361 | 0 | 361 | 0 | 361 | 0 | 361 |
| FEV-2 | 106 | 0 | 106 | 0 | 106 | 0 | 106 | 0 | 18,426 |
| FEV-3 | 467 | 0 | 467 | 0 | 467 | 0 | 467 | 0 | 18,787 |
| FEV-4 | 467 | 0 | 467 | 0 | 467 | 0 | 467 | 0 | 18,034 |
| FEV-5 | 467 | 0 | 467 | 0 | 44,467 | 0 | 44,211 | 0 | 19,187 |
| FEV-6 | 467 | 0 | 467 | 0 | 467 | 0 | 467 | 0 | 15,538 |
| FEV-7 | 125,957 | 0 | 125,957 | 0 | 125,957 | 0 | 125,957 | 125,851 | 24,426 |
| FEV-8 | 125,957 | 0 | 125,957 | 0 | 258,341 | 0 | 258,341 | 125,851 | 30,890 |
| FEV-9 | 126,318 | 0 | 126,318 | 0 | 258,702 | 0 | 258,702 | 125,851 | 31,251 |
| LEP-1 | 704 | 0 | 704 | 80 | 624 | 80 | 624 | 48 | 656 |
| LEP-2 | 704 | 0 | 704 | 0 | 704 | 0 | 704 | 42 | 34,534 |
| LEP-3 | 704 | 0 | 704 | 80 | 72,560 | 80 | 72,560 | 48 | 27,984 |
| LEP-4 | 704 | 0 | 704 | 80 | 19,984 | 80 | 19,984 | 48 | 17,220 |
| LEP-5 | 704 | 0 | 704 | 80 | 14,288 | 80 | 14,288 | 48 | 20,284 |
| LEP-6 | 704 | 0 | 704 | 80 | 2,512 | 80 | 2,400 | 48 | 20,429 |
| LEP-7 | 1,233 | 0 | 1,233 | 208 | 20,577 | 208 | 20,577 | 80 | 17,717 |
| LEP-8 | 704 | 0 | 704 | 80 | 624 | 80 | 624 | 48 | 3,869 |
| AGENT-1 | 11,882,610 | 0 | 11,882,610 | 11,862,224 | 20,386 | 11,862,224 | 20,386 | 4,320,672 | 7,561,938 |
| AGENT-2 | 11,882,610 | 0 | 11,882,610 | 11,862,224 | 20,386 | 11,862,224 | 20,386 | 4,399,072 | 7,483,538 |

Three things stand out.

- **The agent queries.** The shared prefix tokens are 11,882,610, 68.9% of
  the corpus. vLLM's prefix cache served 11,862,224 of them. Quail served
  none.
- **The self joins.** IMDB-9, IMDB-10, FEV-7, FEV-8 and FEV-9 scan one
  column under two aliases (reviews as `r1` and `r2`, evidence as `e` and
  `e2`). The second alias's prefixes are the same token sequences as the
  first's. SGLang's radix cache served them on the FEVER queries: 125,851
  cross-row tokens, the whole evidence corpus, which the earlier per-alias
  definition could not charge and which made its distinct regret negative
  in the first run. vLLM served none of them. The likely reason is
  eviction: the first join's 143,500 pair suffixes pass through vLLM's
  479,616-token block cache, while SGLang's radix cache evicts leaf
  suffixes before shared internal prefixes. This run does not test that.
  On IMDB-9 and IMDB-10 no method reused the reviews under the
  second alias; Quail's distinct regret there is the 1,494,233 tokens of
  the reviews corpus computed twice.
- **Block rounding is not regret.** The request backends' `cached_other_tokens`
  (in the run records, per step) hold the straddling-block hits that the
  first run counted as cross-row: 937,504 tokens for SGLang on FEV-2
  alone.

## Per query time, throughput, cost, and regret

Figure: plots/quailb_two_regrets_per_query.png

| Query | Unit | SoL, distinct prefix | SoL, per document | Quail | Stock vLLM | Pipelined vLLM | Pipelined SGLang |
|---|---|---:|---:|---:|---:|---:|---:|
| IMDB-1 | docs/s | 6.65 s; 751.6 docs/s; $0.0073 | 6.72 s | 14.25 s; 350.9 docs/s; $0.0156; regret 0 per doc, 10,352 distinct; 2.14x SoL | 17.11 s; 292.2 docs/s; $0.0188; regret 0 per doc, 10,100 distinct; 2.57x SoL | 17.03 s; 293.6 docs/s; $0.0187; regret 0 per doc, 10,100 distinct; 2.56x SoL | 18.58 s; 269.1 docs/s; $0.0204; regret 0 per doc, 10,352 distinct; 2.79x SoL |
| IMDB-2 | pairs/s | 9.21 s; 6,513.9 pairs/s; $0.0101 | 9.28 s | 21.29 s; 2,818.2 pairs/s; $0.0234; regret 0 per doc, 10,352 distinct; 2.31x SoL | 26.68 s; 2,248.9 pairs/s; $0.0293; regret 0 per doc, 10,102 distinct; 2.90x SoL | 26.96 s; 2,225.5 pairs/s; $0.0296; regret 0 per doc, 10,102 distinct; 2.93x SoL | 54.37 s; 1,103.5 pairs/s; $0.0596; regret 130,384 per doc, 140,736 distinct; 5.90x SoL |
| IMDB-3 | pairs/s | 9.50 s; 5,060.2 pairs/s; $0.0104 | 9.56 s | 32.35 s; 1,624.7 pairs/s; $0.0355; regret 1,217,732 per doc, 1,228,084 distinct; 3.41x SoL | 39.23 s; 1,338.0 pairs/s; $0.0430; regret 1,326,432 per doc, 1,336,532 distinct; 4.13x SoL | 39.32 s; 1,334.9 pairs/s; $0.0431; regret 1,326,432 per doc, 1,336,532 distinct; 4.14x SoL | 65.79 s; 771.5 pairs/s; $0.0722; regret 1,411,856 per doc, 1,422,208 distinct; 6.93x SoL |
| IMDB-4 | pairs/s | 7.52 s; 1,640.7 pairs/s; $0.0082 | 7.59 s | 19.99 s; 754.6 pairs/s; $0.0219; regret 361,440 per doc, 371,792 distinct; 2.66x SoL | 30.74 s; 496.9 pairs/s; $0.0337; regret 1,061,328 per doc, 1,071,415 distinct; 4.09x SoL | 25.82 s; 591.6 pairs/s; $0.0283; regret 533,376 per doc, 543,010 distinct; 3.43x SoL | 34.13 s; 403.6 pairs/s; $0.0374; regret 502,123 per doc, 512,138 distinct; 4.54x SoL |
| IMDB-5 | pairs/s | 7.41 s; 1,082.5 pairs/s; $0.0081 | 7.47 s | 17.76 s; 491.2 pairs/s; $0.0195; regret 188,587 per doc, 198,939 distinct; 2.40x SoL | 29.56 s; 302.4 pairs/s; $0.0324; regret 760,512 per doc, 770,599 distinct; 3.99x SoL | 24.94 s; 358.5 pairs/s; $0.0274; regret 490,496 per doc, 500,130 distinct; 3.37x SoL | 27.52 s; 279.1 pairs/s; $0.0302; regret 270,687 per doc, 280,733 distinct; 3.72x SoL |
| IMDB-6 | docs/s | 6.78 s; 737.5 docs/s; $0.0074 | 6.85 s | 14.62 s; 342.0 docs/s; $0.0160; regret 0 per doc, 10,352 distinct; 2.16x SoL | 22.75 s; 219.8 docs/s; $0.0250; regret 561,136 per doc, 571,236 distinct; 3.36x SoL | 17.84 s; 280.3 docs/s; $0.0196; regret 33,184 per doc, 42,831 distinct; 2.63x SoL | 19.86 s; 251.8 docs/s; $0.0218; regret 9,963 per doc, 19,991 distinct; 2.93x SoL |
| IMDB-7 | docs/s | 6.92 s; 722.3 docs/s; $0.0076 | 6.99 s | 14.81 s; 337.6 docs/s; $0.0162; regret 0 per doc, 10,352 distinct; 2.14x SoL | 25.23 s; 198.2 docs/s; $0.0277; regret 760,512 per doc, 770,612 distinct; 3.64x SoL | 20.57 s; 243.1 docs/s; $0.0226; regret 278,000 per doc, 287,647 distinct; 2.97x SoL | 20.00 s; 250.0 docs/s; $0.0219; regret 19,471 per doc, 29,530 distinct; 2.89x SoL |
| IMDB-8 | pairs/s | 11.90 s; 8,772.1 pairs/s; $0.0131 | 11.97 s | 26.54 s; 3,458.5 pairs/s; $0.0291; regret 0 per doc, 10,352 distinct; 2.23x SoL | 40.15 s; 2,293.6 pairs/s; $0.0440; regret 860,032 per doc, 870,134 distinct; 3.37x SoL | 39.88 s; 2,309.1 pairs/s; $0.0437; regret 860,032 per doc, 870,134 distinct; 3.35x SoL | 80.49 s; 1,135.6 pairs/s; $0.0883; regret 1,043,552 per doc, 1,053,904 distinct; 6.76x SoL |
| IMDB-9 | pairs/s | 16.38 s; 10,700.4 pairs/s; $0.0180 | 21.25 s | 47.61 s; 3,188.2 pairs/s; $0.0522; regret 0 per doc, 1,504,585 distinct; 2.91x SoL | 64.55 s; 2,356.1 pairs/s; $0.0708; regret 860,032 per doc, 2,364,117 distinct; 3.94x SoL | 64.83 s; 2,346.0 pairs/s; $0.0711; regret 860,032 per doc, 2,364,117 distinct; 3.96x SoL | 138.14 s; 1,096.0 pairs/s; $0.1515; regret 1,173,936 per doc, 2,678,521 distinct; 8.44x SoL |
| IMDB-10 | pairs/s | 15.73 s; 9,691.4 pairs/s; $0.0173 | 21.54 s | 59.71 s; 2,522.2 pairs/s; $0.0655; regret 1,217,732 per doc, 2,722,317 distinct; 3.80x SoL | 79.64 s; 1,815.4 pairs/s; $0.0874; regret 2,186,464 per doc, 3,690,547 distinct; 5.06x SoL | 78.09 s; 1,851.4 pairs/s; $0.0857; regret 2,186,464 per doc, 3,690,547 distinct; 4.96x SoL | 148.30 s; 958.6 pairs/s; $0.1627; regret 2,455,408 per doc, 3,959,993 distinct; 9.43x SoL |
| BIO-1 | docs/s | 11.05 s; 45.3 docs/s; $0.0121 | 11.06 s | 21.64 s; 23.1 docs/s; $0.0237; regret 0 per doc, 1,486 distinct; 1.96x SoL | 25.94 s; 19.3 docs/s; $0.0285; regret 0 per doc, 1,470 distinct; 2.35x SoL | 25.55 s; 19.6 docs/s; $0.0280; regret 0 per doc, 1,470 distinct; 2.31x SoL | 24.87 s; 20.1 docs/s; $0.0273; regret 0 per doc, 1,470 distinct; 2.25x SoL |
| BIO-2 | pairs/s | 62.00 s; 9,088.5 pairs/s; $0.0680 | 62.01 s | 129.38 s; 4,355.4 pairs/s; $0.1419; regret 0 per doc, 1,486 distinct; 2.09x SoL | 958.19 s; 588.1 pairs/s; $1.0511; regret 0 per doc, 1,472 distinct; 15.45x SoL | 932.16 s; 604.5 pairs/s; $1.0226; regret 0 per doc, 1,472 distinct; 15.03x SoL | 1028.46 s; 547.9 pairs/s; $1.1282; regret 385,312 per doc, 386,784 distinct; 16.59x SoL |
| BIO-3 | pairs/s | 43.09 s; 8,003.8 pairs/s; $0.0473 | 43.10 s | 89.92 s; 3,434.1 pairs/s; $0.0986; regret 919,409 per doc, 920,895 distinct; 2.09x SoL | 515.30 s; 603.6 pairs/s; $0.5653; regret 1,079,840 per doc, 1,081,310 distinct; 11.96x SoL | 510.91 s; 608.8 pairs/s; $0.5605; regret 1,079,840 per doc, 1,081,310 distinct; 11.86x SoL | 493.74 s; 632.3 pairs/s; $0.5416; regret 1,285,376 per doc, 1,286,846 distinct; 11.46x SoL |
| FEV-1 | docs/s | 0.12 s; 4,139.2 docs/s; $0.0001 | 0.12 s | 0.28 s; 1,785.7 docs/s; $0.0003; regret 0 per doc, 361 distinct; 2.32x SoL | 0.45 s; 1,111.1 docs/s; $0.0005; regret 0 per doc, 361 distinct; 3.73x SoL | 0.41 s; 1,219.5 docs/s; $0.0004; regret 0 per doc, 361 distinct; 3.39x SoL | 0.88 s; 568.2 docs/s; $0.0010; regret 0 per doc, 361 distinct; 7.29x SoL |
| FEV-2 | pairs/s | 12.84 s; 11,171.7 pairs/s; $0.0141 | 12.85 s | 28.46 s; 5,042.2 pairs/s; $0.0312; regret 0 per doc, 106 distinct; 2.22x SoL | 71.39 s; 2,010.1 pairs/s; $0.0783; regret 0 per doc, 106 distinct; 5.56x SoL | 70.02 s; 2,049.4 pairs/s; $0.0768; regret 0 per doc, 106 distinct; 5.45x SoL | 105.30 s; 1,362.8 pairs/s; $0.1155; regret 18,320 per doc, 18,426 distinct; 8.20x SoL |
| FEV-3 | pairs/s | 7.94 s; 10,694.6 pairs/s; $0.0087 | 7.95 s | 21.06 s; 4,919.6 pairs/s; $0.0231; regret 0 per doc, 467 distinct; 2.65x SoL | 62.62 s; 1,714.1 pairs/s; $0.0687; regret 0 per doc, 467 distinct; 7.88x SoL | 55.74 s; 1,925.7 pairs/s; $0.0611; regret 0 per doc, 467 distinct; 7.02x SoL | 73.94 s; 1,335.2 pairs/s; $0.0811; regret 18,320 per doc, 18,787 distinct; 9.31x SoL |
| FEV-4 | pairs/s | 1.87 s; 5,991.9 pairs/s; $0.0020 | 1.87 s | 4.66 s; 2,894.6 pairs/s; $0.0051; regret 0 per doc, 467 distinct; 2.49x SoL | 8.26 s; 1,772.0 pairs/s; $0.0091; regret 0 per doc, 467 distinct; 4.42x SoL | 8.03 s; 1,822.8 pairs/s; $0.0088; regret 0 per doc, 467 distinct; 4.30x SoL | 11.02 s; 1,119.9 pairs/s; $0.0121; regret 17,567 per doc, 18,034 distinct; 5.90x SoL |
| FEV-5 | pairs/s | 4.74 s; 9,923.9 pairs/s; $0.0052 | 4.75 s | 13.35 s; 4,624.0 pairs/s; $0.0146; regret 0 per doc, 467 distinct; 2.81x SoL | 32.66 s; 1,969.6 pairs/s; $0.0358; regret 44,000 per doc, 44,467 distinct; 6.89x SoL | 31.73 s; 2,027.4 pairs/s; $0.0348; regret 43,744 per doc, 44,211 distinct; 6.69x SoL | 43.43 s; 1,338.6 pairs/s; $0.0476; regret 18,720 per doc, 19,187 distinct; 9.16x SoL |
| FEV-6 | pairs/s | 1.37 s; 4,510.2 pairs/s; $0.0015 | 1.38 s | 3.56 s; 2,257.6 pairs/s; $0.0039; regret 0 per doc, 467 distinct; 2.59x SoL | 6.62 s; 1,325.1 pairs/s; $0.0073; regret 0 per doc, 467 distinct; 4.81x SoL | 6.27 s; 1,399.0 pairs/s; $0.0069; regret 0 per doc, 467 distinct; 4.56x SoL | 8.19 s; 887.3 pairs/s; $0.0090; regret 15,071 per doc, 15,538 distinct; 5.96x SoL |
| FEV-7 | pairs/s | 15.07 s; 11,082.4 pairs/s; $0.0165 | 15.44 s | 53.46 s; 5,030.3 pairs/s; $0.0586; regret 0 per doc, 125,957 distinct; 3.55x SoL | 141.44 s; 1,899.3 pairs/s; $0.1552; regret 0 per doc, 125,957 distinct; 9.38x SoL | 135.77 s; 1,978.6 pairs/s; $0.1489; regret 0 per doc, 125,957 distinct; 9.01x SoL | 205.33 s; 1,285.9 pairs/s; $0.2252; regret 24,320 per doc, 24,426 distinct; 13.62x SoL |
| FEV-8 | pairs/s | 19.14 s; 11,157.1 pairs/s; $0.0210 | 19.50 s | 83.11 s; 5,100.5 pairs/s; $0.0912; regret 0 per doc, 125,957 distinct; 4.34x SoL | 206.75 s; 2,053.1 pairs/s; $0.2268; regret 132,384 per doc, 258,341 distinct; 10.80x SoL | 201.38 s; 2,107.8 pairs/s; $0.2209; regret 132,384 per doc, 258,341 distinct; 10.52x SoL | 344.00 s; 1,226.4 pairs/s; $0.3774; regret 30,784 per doc, 30,890 distinct; 17.97x SoL |
| FEV-9 | pairs/s | 16.45 s; 10,915.4 pairs/s; $0.0180 | 16.65 s | 75.75 s; 5,069.4 pairs/s; $0.0831; regret 0 per doc, 126,318 distinct; 4.60x SoL | 193.18 s; 2,010.1 pairs/s; $0.2119; regret 132,384 per doc, 258,702 distinct; 11.74x SoL | 203.23 s; 1,910.7 pairs/s; $0.2229; regret 132,384 per doc, 258,702 distinct; 12.35x SoL | 292.46 s; 1,289.5 pairs/s; $0.3208; regret 30,784 per doc, 31,251 distinct; 17.78x SoL |
| LEP-1 | docs/s | 0.49 s; 1,012.6 docs/s; $0.0005 | 0.50 s | 1.08 s; 463.0 docs/s; $0.0012; regret 0 per doc, 704 distinct; 2.19x SoL | 1.54 s; 324.7 docs/s; $0.0017; regret 0 per doc, 624 distinct; 3.12x SoL | 1.36 s; 367.6 docs/s; $0.0015; regret 0 per doc, 624 distinct; 2.75x SoL | 1.80 s; 277.8 docs/s; $0.0020; regret 0 per doc, 656 distinct; 3.65x SoL |
| LEP-2 | pairs/s | 58.09 s; 3,727.2 pairs/s; $0.0637 | 58.09 s | 129.36 s; 1,673.6 pairs/s; $0.1419; regret 0 per doc, 704 distinct; 2.23x SoL | 157.51 s; 1,374.5 pairs/s; $0.1728; regret 0 per doc, 704 distinct; 2.71x SoL | 155.38 s; 1,393.4 pairs/s; $0.1705; regret 0 per doc, 704 distinct; 2.67x SoL | 270.91 s; 799.2 pairs/s; $0.2972; regret 33,872 per doc, 34,534 distinct; 4.66x SoL |
| LEP-3 | pairs/s | 2.15 s; 2,825.4 pairs/s; $0.0024 | 2.15 s | 92.51 s; 1,666.3 pairs/s; $0.1015; regret 0 per doc, 704 distinct; 43.12x SoL | 117.47 s; 1,389.6 pairs/s; $0.1289; regret 71,936 per doc, 72,560 distinct; 54.75x SoL | 117.69 s; 1,387.0 pairs/s; $0.1291; regret 71,936 per doc, 72,560 distinct; 54.85x SoL | 157.35 s; 803.5 pairs/s; $0.1726; regret 27,328 per doc, 27,984 distinct; 73.34x SoL |
| LEP-4 | pairs/s | 1.09 s; 1,992.5 pairs/s; $0.0012 | 1.09 s | 39.53 s; 1,643.1 pairs/s; $0.0434; regret 0 per doc, 704 distinct; 36.38x SoL | 46.78 s; 1,406.9 pairs/s; $0.0513; regret 19,360 per doc, 19,984 distinct; 43.05x SoL | 46.61 s; 1,412.1 pairs/s; $0.0511; regret 19,360 per doc, 19,984 distinct; 42.90x SoL | 54.46 s; 811.0 pairs/s; $0.0597; regret 16,564 per doc, 17,220 distinct; 50.12x SoL |
| LEP-5 | pairs/s | 0.49 s; 1,899.4 pairs/s; $0.0005 | 0.50 s | 24.83 s; 1,604.3 pairs/s; $0.0272; regret 0 per doc, 704 distinct; 50.55x SoL | 34.52 s; 1,379.8 pairs/s; $0.0379; regret 13,664 per doc, 14,288 distinct; 70.28x SoL | 34.37 s; 1,385.8 pairs/s; $0.0377; regret 13,664 per doc, 14,288 distinct; 69.97x SoL | 33.42 s; 803.3 pairs/s; $0.0367; regret 19,628 per doc, 20,284 distinct; 68.04x SoL |
| LEP-6 | pairs/s | 0.48 s; 1,930.2 pairs/s; $0.0005 | 0.49 s | 9.02 s; 1,440.1 pairs/s; $0.0099; regret 0 per doc, 704 distinct; 18.66x SoL | 14.34 s; 1,238.0 pairs/s; $0.0157; regret 1,888 per doc, 2,512 distinct; 29.67x SoL | 13.93 s; 1,243.4 pairs/s; $0.0153; regret 1,776 per doc, 2,400 distinct; 28.82x SoL | 8.09 s; 695.8 pairs/s; $0.0089; regret 19,773 per doc, 20,429 distinct; 16.74x SoL |
| LEP-7 | pairs/s | 1.13 s; 1,554.5 pairs/s; $0.0012 | 1.14 s | 38.79 s; 1,624.1 pairs/s; $0.0426; regret 0 per doc, 1,233 distinct; 34.36x SoL | 46.18 s; 1,385.7 pairs/s; $0.0507; regret 19,552 per doc, 20,577 distinct; 40.90x SoL | 46.08 s; 1,388.7 pairs/s; $0.0505; regret 19,552 per doc, 20,577 distinct; 40.81x SoL | 52.09 s; 793.1 pairs/s; $0.0571; regret 16,564 per doc, 17,717 distinct; 46.14x SoL |
| LEP-8 | docs/s | 0.48 s; 1,034.4 docs/s; $0.0005 | 0.49 s | 1.34 s; 373.1 docs/s; $0.0015; regret 0 per doc, 704 distinct; 2.77x SoL | 2.09 s; 239.2 docs/s; $0.0023; regret 0 per doc, 624 distinct; 4.32x SoL | 2.25 s; 222.2 docs/s; $0.0025; regret 0 per doc, 624 distinct; 4.65x SoL | 1.89 s; 264.6 docs/s; $0.0021; regret 3,213 per doc, 3,869 distinct; 3.91x SoL |
| AGENT-1 | docs/s | 47.46 s; 37.3 docs/s; $0.0521 | 130.96 s | 240.49 s; 7.4 docs/s; $0.2638; regret 0 per doc, 11,882,610 distinct; 5.07x SoL | 102.35 s; 17.3 docs/s; $0.1123; regret 0 per doc, 20,386 distinct; 2.16x SoL | 99.15 s; 17.9 docs/s; $0.1088; regret 0 per doc, 20,386 distinct; 2.09x SoL | 218.13 s; 8.1 docs/s; $0.2393; regret 0 per doc, 7,561,938 distinct; 4.60x SoL |
| AGENT-2 | docs/s | 47.87 s; 37.0 docs/s; $0.0525 | 131.36 s | 241.20 s; 7.3 docs/s; $0.2646; regret 0 per doc, 11,882,610 distinct; 5.04x SoL | 102.62 s; 17.3 docs/s; $0.1126; regret 0 per doc, 20,386 distinct; 2.14x SoL | 99.87 s; 17.7 docs/s; $0.1096; regret 0 per doc, 20,386 distinct; 2.09x SoL | 218.08 s; 8.1 docs/s; $0.2392; regret 0 per doc, 7,483,538 distinct; 4.56x SoL |

## Source data

Primary run, `20260905T021527Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families`,
on the `quail-results` volume:

- Manifest:
  `/results/benchmarks/quailb/family-runs/20260905T021527Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/manifest.json`
- Quail:
  `/results/benchmarks/quailb/runs/qb_20260905T021527Z_21224cc4/20260905T021527Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families-quail.json`
- Stock vLLM:
  `/results/benchmarks/quailb/runs/qb_20260905T021527Z_9376111f/20260905T021527Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families-stock_vllm.json`
- Pipelined vLLM:
  `/results/benchmarks/quailb/runs/qb_20260905T021527Z_2d21f5a6/20260905T021527Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families-pipelined_vllm.json`
- Pipelined SGLang:
  `/results/benchmarks/quailb/runs/qb_20260905T021527Z_8887dacd/20260905T021527Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families-pipelined_sglang.json`
- Per family files:
  `/results/benchmarks/quailb/families/20260905T021527Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/<family>-<method>.json`
- SoL: `/results/sol/sol_quailb_sf0.1.json`, regenerated on 2026-09-05
  with the cross-alias credit (`reports/2026-08-29-sol-quailb.md`).

Family function calls (Quail and vLLM container, SGLang container):

- IMDB: `fc-01M1QNK2J7YWGZ4D71ZBSAF1Q7`, `fc-01M1QNK2PP1EVS1QPRFV1ENA8Y`
- BioDEX: `fc-01M1QNK2TV9HRQZ3KEP986QH9P`, `fc-01M1QNK2YZ2KQFCZYNHMXTKQ6K`
- FEVER: `fc-01M1QNK33JQQ5ZAKCDG8QXTABQ`, `fc-01M1QNK37ZP57JGEQHTSNTPDPN`
- LePaRD: `fc-01M1QNK3CBEMA86FKFF5YT1RE4`, `fc-01M1QNK3H5QBH43WZQJVGZM29M`
- Agent: `fc-01M1QNK3PCP4SW5BE7JJP3S6H0`, `fc-01M1QNK3V0HCZ19C2QTWSBTD8D`

Two earlier runs the same day are on the volume and superseded:

- `20260905T001658Z-...-families` (coordinator `fc-01M1QESRZADQRFQXY0FN15M8R2`)
  ran on the accounting that credited straddling-block hits as cross-row
  hits; its distinct prefix regrets are wrong for every join query.
- `20260905T012736Z-...-families` (coordinator `fc-01M1QJV1WTQCWWHV0ZG7MHJNF2`)
  ran the fixed accounting, but Modal preempted the coordinator while the
  BioDEX family was on its last method. All 20 family files finished; the
  restarted coordinator launched the primary run above with a new id, so
  this run has no manifest or merged files.

## Rebuild

Run `reports/make_quailb_two_regrets_plots.py` from the repository root.
Its docstring has every `modal volume get` command; it prints the tables
above and writes both figures.

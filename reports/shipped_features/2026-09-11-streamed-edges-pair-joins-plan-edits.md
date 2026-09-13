# 2026-09-11: streamed edges, equality joins, and editable plans

This PR extends one physical-plan path from planning through execution:

- An `AiFilter` can stream survivors into an `AiJoin` while their KV stays
  pinned.
- A `HashJoin` applies ordinary equality conditions before an AI predicate.
- A `Foreign` node runs a user function on a stream or behind a barrier.
- Physical nodes have readable ids, per-node estimates, and `insert`, `remove`,
  and `move` operations.
- quail-bench computes KV regret after a run from the requests that were made.

All GPU measurements used one H100 and Qwen3 4B FP8 at `sf=0.1`. Query time
excludes model startup. Cost uses $3.9492 per H100 hour.

## Stream filter survivors into joins

Previously, a filter completed before its join started. Survivors that did not
fit in the retention pool lost their KV and recomputed their document prefix at
the join. The new `SurvivorStream` lets the join drive the filter one batch at a
time. A passing document keeps its KV pinned until its join tuples finish.

The prediction was that the five queries with filtered join anchors would stop
recomputing those anchors and save 10 to 30 percent of query time. FEV-9 was
expected to save its smaller 7,309-token recomputation with little time change.

The first five-query ablation is at
`/results/ablations/streamed-filter-join-20260911T055517Z/`, function call
`fc-01M27GHD7K71XA6RNX9XEY6XP8`. The planner-order confirmation for IMDB-10 and
FEV-9 is at `/results/ablations/streamed-filter-join-20260911T072243Z/`,
function call `fc-01M27NHND7E2RVB4VKD4XZ1JQ0`.

| Query | Configuration | Query time, s | Document pairs/s | $/query | Fresh tokens |
|---|---|---:|---:|---:|---:|
| IMDB-3 | Materialized | 32.58 | 1,613.3 | 0.03574 | 3,777,943 |
| IMDB-3 | Streamed | 22.47 | 2,339.1 | 0.02465 | 2,560,772 |
| IMDB-4 | Materialized | 20.32 | 742.3 | 0.02229 | 2,365,565 |
| IMDB-4 | Streamed | 17.35 | 869.4 | 0.01903 | 2,004,176 |
| IMDB-5 | Materialized | 18.16 | 480.4 | 0.01992 | 2,118,038 |
| IMDB-5 | Streamed | 16.60 | 525.5 | 0.01821 | 1,929,643 |
| IMDB-10 | Materialized | 57.03 | 2,531.1 | 0.06256 | 6,696,942 |
| IMDB-10 | Streamed | 47.16 | 3,060.8 | 0.05173 | 5,479,771 |
| BIO-3 | Materialized | 89.97 | 3,432.2 | 0.09870 | 7,547,851 |
| BIO-3 | Streamed | 80.65 | 3,828.9 | 0.08847 | 6,627,939 |
| FEV-9 | Materialized | 38.11 | 4,778.7 | 0.04181 | 4,314,219 |
| FEV-9 | Streamed | 37.83 | 4,814.0 | 0.04150 | 4,306,910 |

The five filtered-anchor queries saved 7.0 to 31.0 percent in the paired
ablations. FEV-9 saved 0.28 seconds. Every answer table was identical between
the two configurations. IMDB-10 required both streaming and the planner change:
the planner now prices reuse of an already-computed prefix as free.

## Equality joins and user functions

An ordinary equality condition now produces a `HashJoin` node. It creates a
pair table before the AI predicate runs, so the model sees only matching pairs.
FEV-10 adds `c.evidence_wiki_url = e.id` to FEV-5.

The prediction was 200 to 300 evaluated pairs, about 160,000 fresh tokens, and
1 to 2.5 seconds. The focused run evaluated 185 pairs and finished in 1.66
seconds, compared with 61,731 pairs and 13.47 seconds for FEV-5.

Run: `/results/benchmarks/quailb/20260911T160843Z-pair-join/`, function call
`fc-01M28KRVDWC1PN2J7WRAS0F2YN`.

| Query | Query time, s | Document pairs/s | $/query | Fresh tokens | Evaluated pairs | Output precision, % | Output recall, % |
|---|---:|---:|---:|---:|---:|---:|---:|
| FEV-5 | 13.47 | 4,582.9 | 0.01478 | 1,515,283 | 61,731 | 1.05 | 96.43 |
| FEV-10 | 1.66 | 111.4 | 0.00182 | 187,567 | 185 | 82.76 | 96.77 |

`.apply()` and `.apply_table()` add a `Foreign` node. A per-batch function keeps
the stream. A barrier function materializes its complete input. The prediction
was that an equality condition and equivalent functions would return the same
185 pairs, with the barrier adding at most 0.4 seconds.

All three versions took 1.69 seconds and returned identical answer tables. The
barrier added no measurable time because all 171 evidence survivors fit in the
retention pool.

Run: `/results/ablations/foreign-pairs-20260911T163940Z/`, function call
`fc-01M28NDEB2GN84ZJKNTTKM1Y3R`.

## Plan estimates and edits

Physical node ids now describe their operation and alias, such as
`ai_filter:c` and `ai_join:e`. Each model node has its own estimated seconds.
For a pinned filter, the estimate also states the recomputation expected if an
edit releases its KV.

`PhysicalPlan.insert`, `remove`, and `move` return a new validated plan. They
recompute streaming pins and estimates. `demos/plan_walkthrough.py` demonstrates
the API on the CPU. No GPU measurement was needed for this API.

## KV regret

`regret_tokens` is now:

`fresh_tokens - minimum_tokens`

`minimum_tokens` is the number of input token positions needed when every
distinct request prefix is computed once with unlimited KV. quail-bench derives
it after the run from saved answer tables and `prompt_pieces`. The engine loop
does not track regret. `quail.bench.restate` adds the required prompt pieces to
older runs.

This replaces the old per-document regret, cache-hit adjustments, and
`regret_distinct_tokens`.

## Full QUAIL-B result

The final run covered all 33 queries and four methods. The prediction was that
the six affected Quail queries would reduce recomputed KV while unchanged
queries kept the same work and rows. It also predicted similar fresh-token
counts for Quail and the vLLM configurations on filters, and lower counts for
Quail on joins.

Run directory:
`/results/benchmarks/quailb/family-runs/20260912T225100Z-902686c5/`.
Parent function call: `fc-01M2BX28NBM9T7W4HDJTCBM3XQ`. The 11 family and
method function calls are listed in
[`reports/quailb-comparison.md`](../quailb-comparison.md).

Fourteen of the 132 method-query cells use earlier compatible runs. Each is
marked in the comparison report. Nine are pipelined vLLM FEV-1 through FEV-9,
two are pipelined SGLang BIO-2 and BIO-3, and three are the FEV-10 baselines.

| Query | Time before, s | Time after, s | Change | Recomputed KV before | Recomputed KV after |
|---|---:|---:|---:|---:|---:|
| IMDB-3 | 32.35 | 22.64 | -30.0% | 1,579,160 | 361,989 |
| IMDB-4 | 19.99 | 17.38 | -13.1% | 479,784 | 118,395 |
| IMDB-5 | 17.76 | 16.64 | -6.3% | 266,213 | 77,818 |
| IMDB-10 | 59.71 | 48.58 | -18.6% | 4,072,039 | 2,854,868 |
| BIO-3 | 89.92 | 79.96 | -11.1% | 3,194,945 | 2,275,033 |
| FEV-9 | 41.14 | 38.23 | -7.1% | 2,286,831 | 2,279,522 |

- Output rows were unchanged on all 33 queries.
- Quail was faster than stock vLLM on 31 of 33 queries. The exceptions were
  AGENT-1 and AGENT-2, where Quail recomputed the shared trace prefix once per
  document.
- IMDB-10 agreement changed from 73.93 to 72.59 percent despite unchanged
  output rows. A streamed and recomputed FP8 prefix can decode differently;
  comparing the saved pair answers would confirm the cause.
- The affected queries improved by 6.3 to 30.0 percent. This matches the
  prediction except that IMDB-5 was below the predicted 10-percent minimum.

[Open the complete 33-query tables and vector plots](../quailb-comparison.md).

## Confirming run on the merge head

The full run measured commit `c45e378`; the engine changed after it
(`2863aef` consolidated the streamed and pair execution contracts, and
the request backends gained the equality join's key columns). The
seven queries the branch changes were rerun with Quail on head
`82bcb83`: `/results/benchmarks/quailb/family-runs/20260913T005721Z-a7356962/`,
function calls `fc-01M2C49M1MPKYN0JKYNK6ZVHE8` (parent),
`fc-01M2C4E3MQP0R85S98PNZGKG0K` (imdb), `fc-01M2C4E3QCBKM6ND0Y114TJX8Q`
(biodex), `fc-01M2C4E3TFDSXA12PCZ31RKTVX` (fever); log
`results/benchmark/20260913T005716Z-quail-head-seven-sf0.1.log`.

Prediction, stated before the run: times and recomputed KV within
noise of the September 12 run, rows identical.

Measured: fresh tokens, recomputed KV, output rows, and agreement are
identical to the September 12 run on all seven queries; times are
within 0.6 s (IMDB-3 22.29 s against 22.64, IMDB-4 17.29 against
17.38, IMDB-5 16.59 against 16.64, IMDB-10 48.74 against 48.58, BIO-3
79.37 against 79.96, FEV-9 38.29 against 38.23, FEV-10 1.68 against
1.65). The report keeps the September 12 rows.

[Open the main vector PDF](../plots/quailb_main.pdf)

Figure: ../plots/quailb_main.pdf

# 2026-09-11: streamed edges, joins over pairs, the Foreign operator, plan edits

One branch, seven changes, in the order they landed. Every measured
run is on the `quail-results` volume at the path given with it, with
its Modal function call id. All GPU runs: one H100, Qwen3 4B fp8,
sf=0.1, model startup excluded from query time. Query time, document
pairs per second, and $/query follow the definitions in CLAUDE.md;
$/query uses $3.9492 per H100 hour. The recomputed KV columns in
sections 1 to 5 are the engine's per-document `regret_tokens`, a
document's own prefix computed again. Section 9 and the benchmark
report use `regret_distinct_tokens`, which also counts the prompt
prefix the documents share; section 10 says why.

## 1. Filter survivors stream into the join anchored on them

What changed: the first join group's anchor filter no longer runs to
completion before the join starts. `FilterStream` in
`quail/executor/loop.py` runs the chain one chunk per `next()` call; a
document that passes its last stage keeps its arena pages pinned and
is handed to `run_join`, which pulls anchors until its own chunk can
fill, runs one join chunk, and pulls again. A pinned survivor's pages
cover the join's largest frame, so the join never claims a page for a
streamed anchor. The planner marks the edge (`AiFilter.pin_survivors`,
`hold_tokens`) and `explain()` says "survivors stream into the join
with KV pinned". On several GPUs the anchor's chain runs inside the
join round on each GPU over that GPU's shard.

Why: under operator-at-a-time execution every survivor's KV waited in
the retention pool (about 470 reviews on IMDB) and the rest was
recomputed at the join: 1,217,732 recomputed KV tokens on IMDB-3, 32.2%
of its fresh tokens. Streaming bounds the pinned set to what the join
has not answered yet.

Prediction, stated before the run: recomputed KV tokens go to 0 on all
five queries and fresh tokens fall by exactly those counts; at 8.4 to
11.9 microseconds per fresh token, about 10.4, 3.1, 1.6, 10.7, and 11.0
seconds saved.

Run: `experiments/cells/join_continuous_batching.py` with the 8338d92
baseline (materialized survivors) and this branch (streamed) in the
same container, one warmup and one measured run per query. Function
call `fc-01M27GHD7K71XA6RNX9XEY6XP8`, data at
`/results/ablations/streamed-filter-join-20260911T055517Z/`.

| Query | Configuration | Query time, s | Document pairs/s | $/query | Fresh tokens | Per-document recomputed KV | Saved, s (predicted) |
|---|---|---:|---:|---:|---:|---:|---:|
| IMDB-3 | Materialized | 32.58 | 1,613.3 | 0.03574 | 3,777,943 | 1,217,171 | |
| IMDB-3 | Streamed | 22.47 | 2,339.1 | 0.02465 | 2,560,772 | 0 | 10.11 (10.4) |
| IMDB-4 | Materialized | 20.32 | 742.3 | 0.02229 | 2,365,565 | 361,389 | |
| IMDB-4 | Streamed | 17.35 | 869.4 | 0.01903 | 2,004,176 | 0 | 2.97 (3.1) |
| IMDB-5 | Materialized | 18.16 | 480.4 | 0.01992 | 2,118,038 | 188,395 | |
| IMDB-5 | Streamed | 16.60 | 525.5 | 0.01821 | 1,929,643 | 0 | 1.56 (1.6) |
| IMDB-10 | Materialized | 59.00 | 2,446.6 | 0.06472 | 6,696,942 | 1,217,171 | |
| IMDB-10 | Streamed | 58.97 | 2,447.8 | 0.06469 | 6,696,942 | 1,217,171 | 0.03 (10.7) |
| BIO-3 | Materialized | 89.97 | 3,432.2 | 0.09870 | 7,547,851 | 919,912 | |
| BIO-3 | Streamed | 80.65 | 3,828.9 | 0.08847 | 6,627,939 | 0 | 9.32 (11.0) |

Evaluated pairs, returned rows, and every answer table were identical
between the two configurations on all five queries. IMDB-10 did not
move in this run: its first join group anchors on an unfiltered alias
and the filtered alias anchors a later group, which change 2 fixed.

## 2. The join search prices KV reuse as unlimited

What changed: the join search (`quail/planner/joins.py`) assumes a
document prefix computed once is free at every later anchor use, the
same assumption the speed-of-light estimate makes. The capacity-based
credit (`retention.allocate`, the `resident_*` fields of `AliasStats`,
`keep_min_doc_tokens`, `keep_resident_fraction`) is gone. Plan
emission places a filtered alias's chain right before the first group
anchored on it, after any barrier, when no earlier group uses the
alias as a partner, and streams it into that group. The executor is
unchanged: it keeps as much KV as the arena holds and `regret_tokens`
reports what it could not keep.

Why: the old credit priced about 11.6% of IMDB-10's filtered reviews as
resident wherever their join sat, so the search picked an order that
saved a few thousand tiny tuples and lost 1.2 million prefix tokens to
recompute. With streaming, residency at a first anchor use is 100%.

Prediction, stated before the run: IMDB-10 loses all 1,217,171
recomputed tokens and about 10.7 seconds; FEV-9's 7,309 recomputed
tokens go to 0 with its time within noise.

Run: same cell and baseline, function call
`fc-01M27NHND7E2RVB4VKD4XZ1JQ0`, data at
`/results/ablations/streamed-filter-join-20260911T072243Z/`.

| Query | Configuration | Query time, s | Document pairs/s | $/query | Fresh tokens | Per-document recomputed KV | Saved, s (predicted) |
|---|---|---:|---:|---:|---:|---:|---:|
| IMDB-10 | Materialized | 57.03 | 2,531.1 | 0.06256 | 6,696,942 | 1,217,171 | |
| IMDB-10 | Streamed | 47.16 | 3,060.8 | 0.05173 | 5,479,771 | 0 | 9.87 (10.7) |
| FEV-9 | Materialized | 38.11 | 4,778.7 | 0.04181 | 4,314,219 | 7,309 | |
| FEV-9 | Streamed | 37.83 | 4,814.0 | 0.04150 | 4,306,910 | 0 | 0.28 (0.1) |

On IMDB-10 the search still puts the `r2` group first (16.64 estimated
seconds against 16.82 for `r1` first). Answer tables identical.

## 3. Survivor streams are the normal edge between GPU operators

What changed: the physical operators are `Scan`, `AiFilter`, `AiJoin`,
`Barrier`, `Exchange`, `Recombine`, `Project`, and `Limit`
(`DocumentInput`, `PackedFilter`, and `AnchoredJoin` were renamed; the
old materializing `Exchange` is `Barrier`; the new `Exchange` appears
only in multi-GPU plans, once before each join group). An `AiFilter`
that feeds the join anchored on it returns its survivor port as a
`SurvivorStream` and a `finalize` callable; the consuming `AiJoin`
drives the chain and fills the stream's holder; the runner calls
`finalize` after the graph has run. The runner's streamed-edge special
case (`stream_anchor`, `StreamedInput`, `NodeResult.produced`) is gone.

Confirming run, same 8338d92 baseline in the same container. Function
call `fc-01M27PPPG5PSWTMR1H3WX549BM`, data at
`/results/ablations/streamed-filter-join-20260911T074256Z/`.
Prediction: IMDB-3 and FEV-9 reproduce the earlier streamed runs within
noise, zero recomputed KV tokens, identical answer tables. Measured:
IMDB-3 22.66 seconds against the baseline's 32.88 (22.47 before the
refactor), FEV-9 39.21 against 39.48 (37.83 before, on a different
card); recomputed KV tokens 0 on both; fresh tokens 2,560,772 and
4,306,910 exactly as before; all 9 answer tables identical.

## 4. Joins over pairs, and FEV-10

What changed: a logical `Join(left, right, on)` holds ordinary column
equalities under the `SemanticJoin`; the AI predicate is asked only of
the pairs the equalities allow. Builder:
`.join(other, on=col("c.url") == col("e.id"))` then `.ai_filter(...)`
over both tables. SQL: `JOIN evidence e ON c.evidence_wiki_url = e.id
AND AI_FILTER(...)`. In the physical plan the equality is a `HashJoin`
node (`hash_join:c-e`) that reads the two scans and the key columns
the request carries, pairs the rows whose keys are equal with an Arrow
hash join (`quail/runtime/pairs.py`), and feeds the pairs to the
`AiJoin` on its `pairs:<written position>` port, the same port a
pairs-returning `Foreign` uses. The planner prices the join by its pair
count; `JoinAdmission` streams a partner list per anchor per stage so
an anchor packs only its own pairs; the request backends submit only
the listed pairs. QUAIL-B gains FEV-10: FEV-5 with
`ON c.evidence_wiki_url = e.id` (evidence ids are FEVER page names),
merged in `fsdatalab/quail-bench` as commit `ec4682f2`; no labeling run
was needed at any scale factor.

Prediction, stated before the run: FEV-10 asks SUPPORT of at most one
pair per surviving claim, about 200 to 300 pairs, so about 160,000
fresh tokens and 1 to 2.5 seconds against FEV-5's 13.35; precision far
above FEV-5's 1.05% with recall near 96%; FEV-9 unchanged.

Run: `experiments/cells/pair_join.py`, function call
`fc-01M28KRVDWC1PN2J7WRAS0F2YN`, run directory
`/results/benchmarks/quailb/20260911T160843Z-pair-join/` (a first
attempt, `fc-01M28JYT36H7N9E21KJNE1ME3N`, ran FEV-5 first after the
cold boot and measured it at 24.0 seconds; its FEV-10 and FEV-9 agree
with the second run). SoL estimates for the three queries are at
`/results/sol/2026-09-11-fev10-prefix-reuse.json`.

| Query | Query time, s | Document pairs/s | $/query | Fresh tokens | Per-document recomputed KV | Evaluated pairs | Answer agreement, % | Output precision, % | Output recall, % | Rows (expected) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FEV-5 | 13.47 | 4,582.9 | 0.01478 | 1,515,283 | 0 | 61,731 | 79.57 | 1.05 | 96.43 | 12,830 (140) |
| FEV-10 | 1.66 | 111.4 | 0.00182 | 187,567 | 0 | 185 | 89.09 | 82.76 | 96.77 | 145 (124) |
| FEV-9 | 38.44 | 4,737.6 | 0.04217 | 4,306,910 | 0 | 182,115 | 67.77 | 0.00 | 45.45 | 149,783,486 (11) |
| FEV-10, SoL estimate | 0.712 | 236 | 0.00078 | 185,703 | 0 (assumed) | 168 | | | | |

FEV-10's pairs per second is low because the query is almost all
filter work: the two filters are 177,550 of its 187,567 fresh tokens.
Agreement is with the Qwen3 32B fp8 labels (collection
`gt_77bb8b128743a79aedddaa24c808c3f8`).

## 5. The Foreign operator: a user function between two GPU operators

What changed: `.apply(fn, columns=[...])` and `.apply_table(fn,
columns=[...])` put a Python function in the plan (logical `Apply`,
physical `Foreign`, node id `apply:<name>`). The function gets one
Arrow table per alias and returns the ids to keep, every id
(`ids="preserve"`), or, after `join(other)`, the pairs the next AI
predicate is asked about; it never invents an id. `per_batch` keeps
the survivor stream and runs on each batch before admission; `barrier`
makes the planner materialize the chain and runs once. The planner
refuses per-batch functions on several GPUs and the request backends
refuse `apply`.

Prediction, stated before the run: per_batch matches the equality run
(185 pairs, 145 rows, 187,567 fresh tokens, time within noise of 1.66
seconds); barrier costs the lost overlap only, 0.1 to 0.4 seconds.

Run: `experiments/cells/foreign_pairs.py`, function call
`fc-01M28NDEB2GN84ZJKNTTKM1Y3R`, data at
`/results/ablations/foreign-pairs-20260911T163940Z/` (a first attempt,
`fc-01M28MSCPZR6Y3ZGWXJTXKEF9V`, failed because Arrow's default join is
a left outer join and returned null ids; the runtime now refuses a
null id and the function uses an inner join).

| Variant | Query time, s | Document pairs/s | $/query | Fresh tokens | Per-document recomputed KV | Pairs | Rows | Evidence chain pinned | Function calls |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| equality (`on=`) | 1.69 | 109.5 | 0.00185 | 187,567 | 0 | 185 | 145 | yes | 0 |
| per_batch (`apply`) | 1.69 | 109.5 | 0.00185 | 187,567 | 0 | 185 | 145 | yes | 2 |
| barrier (`apply_table`) | 1.69 | 109.5 | 0.00185 | 187,567 | 0 | 185 | 145 | no | 1 |

Answer tables identical row for row across the three variants. The
barrier variant's chain took 1.27 seconds and its join 0.13; at sf=0.1
the 171 evidence survivors fit the retention pool, so the lost overlap
cost nothing measurable.

## 6. Per-node estimates, readable node ids, and plan edits

What changed: node ids name the operator and its alias (`scan:c1`,
`ai_filter:c1`, `ai_join:e1`, `barrier:e2`, `apply:same_page`,
`project`). `PhysicalPlan.estimates` prices each model node's own work
in seconds; a pinned chain also shows the recompute it would pay if its
KV were released. `plan.insert(node, between=(producer, consumer))`,
`remove(node_id)`, and `move(node_id, between=...)` edit the DAG and
return a new plan; an illegal edit raises `PlanEditError` with the
rule it broke; after an edit the plan re-derives `pin_survivors`,
`keep_kv`, and `hold_tokens` and re-estimates every node.
`query.run(plan=edited)` executes it. `demos/plan_walkthrough.py` runs
the whole thing on the CPU.

No GPU run. Prediction, stated before the CPU script ran: IMDB-3's
release-recompute figure on the pinned reviews chain is close to the
1,217,171 tokens measured before streaming; a Barrier on that edge
turns the pin off and raises the estimate by its seconds; FEV-9's
chains fit the pool so its figure is zero.

| Query | Plan estimate, s | Sum of node seconds | Pinned chain | Release recompute, tokens (s) | Estimate with a Barrier on that edge, s |
|---|---:|---:|---|---:|---:|
| IMDB-3 | 9.503 | 9.504 | ai_filter:r | 1,087,633 (4.092) | 13.523 |
| FEV-9 | 8.260 | 8.261 | ai_filter:e1, ai_filter:e2 | 0 (0.000) | 8.260 |

## 7. FEV-10 as a benchmark query, with two runner fixes

FEV-10 has all four measurements through the standard family runner
(`quail.bench.quailb_parallel`, `--query FEV-10`). Run directory
`/results/benchmarks/quailb/family-runs/20260911T201441Z-d16f87d8/`,
function call `fc-01M291QA9AAV9JSMYN5KCRJSM4`. The main and FEVER plots
and the saved-results report include it, and the SoL file for all 33
queries is `/results/sol/2026-09-11-quailb-prefix-reuse.json` (the 32
earlier estimates unchanged).

Prediction, stated before the run: stock and pipelined vLLM in 3 to 5
seconds and pipelined SGLang in 4 to 7, about 190,000 fresh tokens on
every method.

| Configuration | FEV-10 time, s | FEV-5 time, s | Document pairs/s | $/query | Fresh tokens | Answer agreement, % | Output precision, % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Quail | 1.68 | 13.47 | 110.1 | 0.00184 | 187,567 | 89.09 | 82.76 |
| Stock vLLM (operator-at-a-time) | 2.97 | 32.66 | 63.3 | 0.00326 | 270,220 | 86.67 | 73.78 |
| Pipelined vLLM | 2.91 | 31.73 | 64.6 | 0.00319 | 270,220 | 86.67 | 73.78 |
| Pipelined SGLang | 3.32 | 43.43 | 54.2 | 0.00364 | 267,414 | 89.56 | 75.62 |

The baselines compute the shared prompt preamble again for every
document, which is why their fresh tokens exceed the prediction.

Two attempts failed before this one, on bugs fixed here:

- `run_all` in `quail/bench/quailb_parallel.py` resolved the run
  directory before checking that it lies under `/results`; inside the
  container that mount resolves elsewhere, so every run directory was
  refused (`fc-01M28Z7H6F8X634WWVQHMX1DGC`).
- `gpu_problem()` in `quail/runtime/execute.py` used
  `torch.cuda.is_available()`, which starts the CUDA runtime and marks
  every later fork as bad; vLLM forks its engine process after the
  probe (`fc-01M29181E51A1T4E2840AGD2XT`). It now uses PyTorch's NVML
  based check. Quail's backend never forks, so Quail-only runs never
  hit it.

## 8. Two ponytail passes and the test fold

The `ponytail-review` checklist asks of each piece of code whether it
needs to exist and whether it can be shorter. Nothing in either pass
changes a result, a plan shape, a node id, a report key, or a public
call.

- First pass, over the pair-join, Foreign, and plan-edit lines under
  `quail/`: 20,422 to 20,338 lines. Deleted dead helpers and fields
  (`possible_anchor_aliases`, `_length_stats`, `AliasStats.histogram`,
  `FilterAdmission.sync_free_pages`, `FilterStream.budget`, the
  `StreamedPairs` fields nothing read, `_REMOVABLE`); one
  `pair_partner` instead of two; the join search records each stage's
  work so the planner stops walking the chosen sequence twice; one
  `retention.group_sequence` call instead of three copies of the merge
  rule.
- Second pass, over everything the first skipped: 255 fewer lines,
  20,338 to 20,276 under `quail/`. Three copies of an unreachable "no
  join consumed its stream" guard (`validate_streams` refuses that
  shape first), the arena-writes check on a pinned chain, the root
  check in `remove`, fields written and never read (`stage_tokens`,
  `held`, the `full` flag of a join group, the `alias=` parameter of
  `apply()`), one pair of member-list helpers in
  `quail/runtime/pairs.py` instead of two copies, `triangle()` for two
  hand-rolled n(n+1)/2, and the streamed filter-join cell folded into
  `experiments/cells/join_continuous_batching.py` as arguments.
- The five test files the branch added went into the files that test
  the same modules, with the CPU fakes three of them copied
  (`tests/fakes.py`) and one `fever_executor` instead of three: the
  same 19 tests, 251 fewer lines under `tests/`.
- Found and kept: the argument checks in `Query.apply` that repeat
  `Apply.validate` (their messages are the ones a user sees first),
  `FilterStream.chunks` and `attention_mode`, `plan.move` (documented
  API), the int-or-list `suffix_count` in the request scheduler.
- A third pass over the HashJoin commit and the plot script: the
  request backend's `over_pairs` wire field went (a stage runs over
  pairs when a pair table reached it on a port), the hash join
  runtime's unreachable table-or-list branch went, and the plot
  script fails on a rerun directory without a manifest instead of
  skipping it.

## 9. Quail-only rerun of all 33 queries against the saved Quail rows

The whole benchmark was rerun with this branch's engine, Quail only,
to check the saved Quail rows in `reports/2026-09-05-quailb-saved-results.md`
and the figures under `reports/plots/`. Two run directories, because
the FEVER family of the first was cancelled while scoring FEV-8 with
the id-code scorer (quail-bench PR #3) and rerun once quail-bench
scored a traced run from its answers (PR #4) and sampled its rows
chunk by chunk (PR #5):

- IMDB, BIO, LEP, AGENT:
  `/results/benchmarks/quailb/family-runs/20260912T023323Z-e689d27e/`,
  function calls `fc-01M29QF6N38ZTPKESAJSA53EP4` (imdb),
  `fc-01M29QF6Q0WM29JPQHA1XY98T4` (biodex),
  `fc-01M29QF6VJ1SW7YAMM1SHQ93FN` (lepard),
  `fc-01M29QF6XNTEG0ZBTYWVRXPBD9` (agent).
- FEV: `/results/benchmarks/quailb/family-runs/20260912T032335Z-609d6410/`,
  function call `fc-01M29TCRAG0JW20ME880WJXXTS`.

The saved-results report now carries a "Quail rerun against the saved
Quail rows" table with every query, and the Quail bars in the main and
dataset figures come from these runs. The baseline rows and bars are
unchanged.

Prediction, stated before the runs: the five queries whose filtered
alias anchors a join (IMDB-3, IMDB-4, IMDB-5, IMDB-10, BIO-3) drop to
0 recomputed KV tokens and lose 10 to 30 percent of their time; FEV-9
drops from 7,309 recomputed tokens to 0 and from 41.14 s to 38 or 39;
FEV-10 stays near 1.7 s; every other query stays within a few percent
with the same fresh tokens; answers and rows are identical everywhere.

| Query | Saved time, s | Rerun time, s | Change | Saved recomputed KV | Rerun recomputed KV | Agreement saved / rerun, % |
|---|---:|---:|---:|---:|---:|---:|
| IMDB-3 | 32.35 | 22.20 | -31.4% | 1,228,087 | 10,355 | 79.16 / 79.16 |
| IMDB-4 | 19.99 | 17.34 | -13.3% | 371,795 | 10,355 | 77.76 / 77.76 |
| IMDB-5 | 17.76 | 16.52 | -7.0% | 198,942 | 10,355 | 80.17 / 80.17 |
| IMDB-10 | 59.71 | 47.65 | -20.2% | 1,238,442 | 1,504,587 | 73.93 / 72.59 |
| BIO-3 | 89.92 | 79.31 | -11.8% | 920,895 | 1,486 | 82.54 / 82.54 |
| FEV-9 | 41.14 | 39.59 | -3.8% | 139,458 | 132,149 | 67.77 / 67.77 |
| FEV-10 | 1.68 | 1.69 | +0.6% | 467 | 467 | 89.09 / 89.09 |

Recomputed KV here is `regret_distinct_tokens` (section 10). The
per-document `regret_tokens` the prediction was stated in went from
1,217,732, 361,440, 188,587, 1,217,732, 919,409, and 7,309 to 0 on the
six queries. What remains is the prompt prefix the documents share:
10,355 tokens across the reviews, 1,486 across the reports, and on
FEV-9 132,149 across its two claim and two evidence aliases. IMDB-10's
rerun figure also holds the 1,494,232-token second copy of the reviews
that its `r2` anchor computes; both runs computed it (identical fresh
tokens), but the saved suite's build credited only each alias's
within-set prefix (2 x 10,355), while the current accounting credits a
second alias's full copy.

What happened against the prediction:

- The six queries with per-document recomputed KV all went to 0 on
  that count, and the five with a filtered anchor lost 7.0 to 31.4
  percent of their time. IMDB-5's 7.0 percent is at the low end of the
  predicted range because its recomputed share was the smallest
  (188,587 of 2,118,230 fresh tokens).
- FEV-9 came in at 39.59 s, just above the predicted 38 to 39, with
  0 per-document recomputed tokens as predicted.
- The 26 queries with no per-document recomputed KV have the same fresh
  tokens and
  the same rows as before. Their times moved between -3.4 and +11.5
  percent. The FEVER family ran 2.5 to 7.1 percent slower across all
  eight of its unchanged queries, and LEP-6 11.5 percent slower, which
  is more than the few percent predicted; the fresh-token counts are
  identical, so the difference is GPU-side timing between containers
  rather than the engine doing different work.
- Output rows are identical on all 33 queries. Agreement is identical
  on 32; IMDB-10 moved from 73.93 to 72.59 percent while its output
  rows did not change (64,840,220). The likely cause is that its join
  prompts now sit on the streamed survivor's KV instead of a
  recomputed copy, and with fp8 numerics the same prompt can decode
  differently when its prefix was computed in a different batch. That
  is not verified. Comparing the per-pair answers of the two runs on
  the volume would settle it.

Command, teed to `results/benchmark/20260912T023319Z-quail-only-sf0.1.log`
and `results/benchmark/20260912T032332Z-quail-only-fever-sf0.1.log`
(the FEVER run passed `--query FEV-1,...,FEV-10`):

    modal run --detach -m quail.bench.quailb_parallel \
      --output-dir /results/benchmarks/quailb/family-runs \
      --model qwen3-4b-fp8 --sf 0.1 \
      --ground-truth-collection gt_77bb8b128743a79aedddaa24c808c3f8 \
      --no-include-baselines --no-include-sglang

## 10. Recomputed KV counts the shared prompt prefix

The benchmark report and its figures showed the engine's per-document
`regret_tokens`, which is 0 on both AGENT queries although Quail
computes the 11,882,610 tokens the 1,772 agent traces share as a
prefix once per document. They now show `regret_distinct_tokens`: the
per-document count plus the prefix tokens the scanned documents share
(a set scanned under two aliases counts its second copy in full) minus
the cross-row cache hits the engine reported. `quail/runtime/prefixes.py`
derives it on the CPU after the run from the saved counts and the
corpus tokens; every saved row on the volume already carries it, and
nothing is tracked in the engine loop. The rule in CLAUDE.md names
this figure now.

AGENT-1 under the new figure, all from the saved rows:

| Configuration | Query time, s | Fresh tokens | Recomputed KV tokens |
|---|---:|---:|---:|
| Quail | 234.86 | 17,389,113 | 11,882,610 |
| Stock vLLM (operator-at-a-time) | 102.35 | 5,526,889 | 20,386 |
| Pipelined vLLM | 99.15 | 5,526,889 | 20,386 |
| Pipelined SGLang | 218.13 | 13,068,441 | 7,561,938 |

vLLM's prefix cache reuses the shared prompt across documents and
Quail's arena does not, which is the whole gap between them on AGENT.
Two saved rows have no figure: the earlier SGLang adapter's cross-row
count on FEV-7 and FEV-8 is 125,851 tokens, the whole evidence set,
more than any prefix the trie credits, so the derived value is
negative and the report marks it not measured rather than zero.


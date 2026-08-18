# Quail engine: a declarative query engine for AI_FILTER and AI_JOIN

Status: design proposal, nothing implemented. This is the "actual
system" the README promised once the exploration settled. It settles
now: the exploration measured three mechanisms (pipelining,
token-based admission, KV rewind), a packed join executor, a KV
store for cross-query reuse, and a calibrated cost model. This
document proposes the engine that puts them behind one declarative
interface.

Scope is unchanged: AI_FILTER and AI_JOIN only, Qwen3 4B fp8 only,
H100 workers on Modal only. No maps, no classification, no
aggregation, no cascades. If a design decision below seems to need
one of those, the decision is wrong, not the scope.

What the numbers below rest on (all committed in this repo):

| result | where |
|---|---|
| 5-filter chain, 10k docs: KV rewind 39.8 s vs stock vLLM (separate requests per stage, matched token budget) 42.9 s | `results/engine/filter_cells.json` |
| 2-way join, 256k pairs: packed executor 103.6 s vs stock vLLM (grouped requests per pair) 429 s — 4.1x | `results/engine/join2way.json` |
| 3-way join, gated and deduped between stages: 95.6 s, replay-consistent | `results/engine/join_nway3.json` |
| Cross-query KV restore from a pinned CPU store: 1.9–2.1x faster than recompute | `results/engine/persist_split7_quail_waves_chain_10k.json` |
| Cost model: makespan predictions within ~2–3% on filters, +7% on the packed join | `results/engine/makespan_check.json`, `join_findings.md` |

---

## 1. Shape of the system

Four layers, in the style of DataFusion: a session that owns a
catalog, a SQL front end that compiles to a logical plan, an
optimizer that prices physical alternatives with the calibrated cost
model, and executors that run the physical plan on Modal GPUs.

    SQL text or builder calls
            |
            v
    +------------------+
    |  compiler        |  parse (sqlglot), bind names against the
    |                  |  catalog, emit a LogicalPlan tree
    +------------------+
            |
            v
    +------------------+
    |  optimizer       |  enumerate filter orders, filter placement,
    |                  |  join stage orders, anchor choices; price
    |                  |  each with quail/plan/cost.py; emit a
    |                  |  PhysicalPlan or a Refusal
    +------------------+
            |
            v
    +------------------+
    |  coordinator     |  local process: ships per-worker plans,
    |                  |  collects answers, runs the CPU relational
    |                  |  steps (gate, dedup, assemble)
    +------------------+
            |
            v
    +------------------+
    |  Modal workers   |  one container per GPU. Each hosts BOTH
    |  (N x H100)      |  executors: the vLLM engine with
    |                  |  QuailScheduler (filter chains), and the
    |                  |  packed forward executor (join stages).
    |                  |  Plus the pinned CPU KV store.
    +------------------+

Everything above the workers is CPU-only Python and unit-testable
without a GPU, which keeps the repo's testing discipline: the 45
existing CPU tests grow, and GPU runs stay confirmation cells.

---

## 2. The user interface

### 2.1 Configuration

```python
@dataclass(frozen=True)
class EngineConfig:
    gpus: int = 1               # data-parallel H100 workers. tp=1
                                # always: 4B fp8 weights fit one card
                                # (plan_query already refuses if not)
    cpu_memory_gb: int = 64     # per-worker cap on host memory. The
                                # pinned KV store gets this minus a
                                # fixed working headroom; 0 disables
                                # the store entirely (access="read")
    model: str = "qwen3-4b-fp8" # the only accepted value today; any
                                # other string is a Refusal, not a
                                # fallback
```

These three knobs map one-to-one onto Modal resources: `gpus` is the
worker container count (`gpu="H100!"` each), `cpu_memory_gb` is the
container `memory=` request, `model` selects the weights baked into
the image volume.

### 2.2 Session and catalog

```python
sess = quail.Session(EngineConfig(gpus=1, cpu_memory_gb=256))

sess.register("reviews",  DocumentSet.from_parquet("imdb.parquet", text_col="review"))
sess.register("products", DocumentSet.from_hf("...", text_col="description"))
```

A `DocumentSet` is the catalog entry — the analogue of a DataFusion
`TableProvider`. Registration is cheap and lazy: tokenization runs
on first scan and is cached (token ids plus the `CorpusStats` the
planner needs: count, total, mean, max, sum of squared lengths).
Each set gets a content hash; the KV store and the tokenization
cache key on it, so re-registering identical data reuses both.

### 2.3 Two equivalent query entry points

SQL, the AISQL subset (Snowflake Cortex AISQL syntax, arXiv
2511.07663 — filters and joins only):

```python
q = sess.sql("""
    SELECT r.id, p.id
    FROM reviews r
    JOIN products p
      ON AI_FILTER(PROMPT('Review {0} discusses product {1}', r.text, p.text))
    WHERE AI_FILTER(PROMPT('This review is negative: {0}', r.text))
""")
```

Builder, for programmatic use (the DataFrame-style API):

```python
q = (sess.docs("reviews")
         .ai_filter("This review is negative: {doc}")
         .ai_join(sess.docs("products"),
                  "Review {left} discusses product {right}"))
```

Both produce the same `LogicalPlan`. Then:

```python
q.explain()   # logical tree, chosen physical plan, per-operator
              # predicted wall and token counts — the prediction is
              # printed BEFORE any run, per house rule
res = q.run() # executes; result carries measured walls next to the
              # predictions, plus tokens computed, KV bytes restored,
              # pairs evaluated per stage
```

`run()` on an infeasible configuration raises the `Refusal` the
planner produced (existing behavior: named constraint, needed vs
available, unit). The engine never silently degrades.

### 2.4 What a result is

Filters return surviving document ids. Joins return id tuples.
Projection is only ever id columns plus pass-through text — there is
no expression evaluation, no aggregation. Results also carry the
per-stage answer matrices, so the nested-loop replay check from the
join work (`joinlogic.brute_force_triples`) stays available as a
correctness gate on every run.

---

## 3. The SQL subset and the compiler

### 3.1 Grammar

Accepted, and nothing else:

- `SELECT <id/text columns> FROM <set> [alias]`
- zero or more `JOIN <set> [alias] ON AI_FILTER(PROMPT('...', a.col, b.col))`
- `WHERE` as a conjunction (`AND`) of `AI_FILTER(PROMPT('...', x.col))`
  terms, each referencing exactly one table
- `WHERE EXISTS (SELECT 1 FROM <set> s WHERE AI_FILTER(PROMPT('...', outer.col, s.col)))`
  — the exists-semantics join (keep a document if some partner
  matches; the executor stops each partner stream at the first YES)

Rejected with a named error, not worked around: OR between AI
predicates, non-AI predicates, AI_FILTER over more than two tables,
subqueries other than the EXISTS form, any other AI_* function.
Rejection at parse time is the same honesty as `Refusal` at plan
time.

### 3.2 Compile pipeline

1. **Parse.** `sqlglot` (CPU-only dependency) parses the text; a
   walker checks the tree against the subset above.
2. **Bind.** Table names resolve against the session catalog;
   PROMPT placeholders resolve to document sets. A prompt that
   references one set is a filter predicate; two sets, a join
   predicate. The prompt template is tokenized once here, split
   into the shared preamble and the per-document tail (the 33-token
   shared-preamble split that rewind and pricing both use).
3. **Logical plan.** Three logical operators only:

   | operator | fields |
   |---|---|
   | `Scan(set)` | document set id |
   | `SemanticFilter(input, predicates)` | ordered list of (prompt, selectivity estimate) — conjunctions collapse into one node |
   | `SemanticJoin(left, right, predicate, semantics)` | semantics: `full` (all matching pairs) or `exists` |

   N-way joins appear as a left-deep tree of `SemanticJoin` nodes;
   the optimizer flattens them into the join graph before pricing,
   so the SQL join order carries no meaning — same as the filter
   order in WHERE.

Selectivity estimates come from a sample pass, exactly as
`join_plan.md` decided: ~2k sampled documents or pairs per
predicate, labeled by the model itself during planning, only when
the plan actually depends on survivors (gated filter chains, n ≥ 3
joins, exists semantics). A 2-way full join samples nothing.

---

## 4. The optimizer

The optimizer turns one logical plan into one physical plan by
enumerating alternatives and pricing each with
`quail/plan/cost.py`. Query graphs here are tiny — at most ~5
filters per set and 2–4 relations — so it enumerates exhaustively.
No heuristics, per the join plan's decision; what the cost model
prices best, runs.

Decisions, in the order they are taken:

1. **Filter placement against joins.** A filter that reads one side
   of a join can run before it (shrinking the pair list) or after
   it (on surviving tuples). Both placements are priced: pushdown
   costs (docs x filter tokens) and saves (killed docs x partner
   count x pair tokens); pull-up is the reverse. This is the
   AI-aware placement result from the Cortex AISQL paper, done here
   with a calibrated token cost model instead of call counts. At
   our costs pushdown nearly always wins — a pair costs a whole
   partner-suffix prefill and a filter costs one question — but the
   pricing keeps it honest for explosive cases.
2. **Filter order within a chain.** Per-stage cost is
   survivors x (question tail tokens + cached re-read of the
   resident document), so cheap-and-selective filters run first.
   All orders of ≤5 filters are priced (≤120 permutations of pure
   arithmetic).
3. **Join stage order and per-stage anchor.** For the join graph,
   every stage order and anchor choice is enumerated and priced
   (the `join_plan.md` formulas: prefix computations x f + |pairs|
   x s, survivor-thinned). Anchoring the longer side wins almost
   always; the enumeration is cheap enough that "almost" never has
   to be trusted.
4. **Executor assignment.** A fixed mapping today, chosen by the
   measurements, revisable when a measurement says otherwise:

   | logical operator | executor | why |
   |---|---|---|
   | SemanticFilter chain | vLLM engine, chain mode (KV rewind + token admission + in-engine pipelining) | measured 39.8 s vs 42.9 s stock; gating needs per-document decisions the engine makes at zero client latency |
   | SemanticJoin stages | packed forward executor (brim packing, shared-prefix attention merge, kept anchor KV) | measured 4.1x over grouped stock; no pool, no admission, chunk cap = min(B*, kernel cap) derived at load |
   | gate / dedup / assemble | CPU (`quail/joinlogic.py`) | pair-list construction is bookkeeping, measured negligible |

   The engine chain path remains the join fallback (it exists and
   was priced at 6.1 h vs packed 4.4 h at BioDEX scale) — used only
   if a packed-path parity gate fails on some future shape.
5. **Budgets and caps** — all reused, not redesigned:
   - admission budget: `SATURATION_SLACK x n_filters x step budget`,
     clamped to the pool (`plan_query`, unchanged)
   - engine boot: `engine_max_seqs`, `engine_step_tokens` as today
   - packed chunk budget: `min(B*_memory, kernel index cap)` — the
     cap computed from the loaded weights (110,375 tokens today),
     the lesson of the join2way crash
   - keep-vs-recompute per anchor: the size rule (m, f/S ~8%,
     feeds-later-stage), from `join_plan.md`
6. **Access per scan: read, restore, or spill.** Decided per
   document set from store state and the measured break-even
   (channel > kappa x prefill rate = 7.2 GB/s at 4B; pinned host
   memory measured 55 GB/s, so restore wins whenever the KV is
   warm). `store_length_threshold` decides which documents' KV is
   worth the capped capacity, as today.
7. **Sharding across workers.** Filters: balanced document shards
   by token count (`_balanced_shards`, unchanged). Joins: shard by
   anchor document — pair lists partition cleanly because every
   pair belongs to exactly one anchor. Workers never talk to each
   other; the coordinator merges answers.

The output is a `PhysicalPlan`: a JSON-serializable tree of
physical operators, each carrying its settings and its predicted
tokens and wall. `explain()` prints it. The per-operator prediction
is stored in the result next to the measurement, so every
production run doubles as a cost-model check — the
prediction-before-run rule enforced by the tool rather than by
discipline.

---

## 5. Physical operators and executors

### 5.1 Operator set

| operator | runs on | what it does |
|---|---|---|
| `DocScan` | worker CPU + store | resolve the document set to token arrays (cached tokenization); execute the planned access: read (compute KV fresh), restore (batched loads from the pinned store via the offloading connector), spill (store absorbs pool overflow) |
| `FilterChain` | vLLM engine + QuailScheduler | one living request per document; token-budget admission; rewind to the document boundary between stages; gated stage advance inside the scheduler |
| `JoinStage` | packed executor | brim-packed chunks over the stage's pair list; anchor prefix computed once ever (kept KV on cut); shared-prefix attention merged by softmax state; YES/NO logits read at suffix ends; exists semantics stop a partner stream at the first YES |
| `Gate` / `Dedup` / `Assemble` | coordinator CPU | between join stages: survivors, distinct anchors, tuple reassembly from recorded answers |
| `Sink` | coordinator CPU | ids/tuples out, plus the run report (predicted vs measured, token and byte counters) |

`DocScan` is explicit in the plan tree — it is where tokenization,
corpus statistics, and the store decision live, and it is the unit
the end-to-end workload reuses across queries.

### 5.2 One worker, two executors

Each worker container hosts both executors, serialized (never
concurrent):

- the vLLM 0.26 engine booted with `QuailScheduler`, fp8 KV, the
  planned `max_num_seqs` / `max_num_batched_tokens`, prefix caching
  on — exactly the `modal_filters.py` boot;
- the packed executor with its own weight copy through vLLM's
  loader (as `modal_join_forward.py` does), bf16 KV for kept
  anchors.

Two weight copies cost ~8 GB of the 80 GB card. That is the price
of not sharing mutable GPU state between an engine that owns its
memory pool and a loop that owns plain tensors. The engine boots
with `gpu_memory_utilization` lowered to leave the packed
executor's ceiling (activations at the chunk cap ~9 GB, kept
anchor KV bounded by the ring buffer): 0.92 today becomes ~0.75,
shrinking the filter pool from ~947k to ~770k tokens — the corpus
goes from 3.4x to 4.2x pool, which the admission budget already
handles by design. If a measured filter regression says the shared
boot is too expensive, the fallback is per-query executor boot,
priced as boot time in the plan.

### 5.3 The KV store

One pinned-host-memory store per worker, capacity =
`cpu_memory_gb` minus headroom, keyed by
(model hash, document set hash, document id).

- **Writers/readers, phase 1: the engine path only.** Filter
  queries write document KV at first scan through the
  `QuailOffloadingConnector` and restore it in planned waves on
  later queries — the measured 1.9–2.1x restore path, unchanged.
- **Phase 2: join anchor restore.** The packed executor reads
  anchor prefix KV directly from the pinned store (no paged pool
  in the way, so this is a plain batched H2D copy at the measured
  55 GB/s, far above the 7.2 GB/s break-even). Phase 2 because it
  needs a dtype decision — the store holds fp8 (engine format);
  the packed path prefers bf16 for kept anchors — and the honest
  resolution is a measured parity cell, not a design assertion.
- Eviction is the length threshold, not LRU, for the reason in
  `planner.py`: a scanning query thrashes LRU and cannot thrash a
  length cutoff.

The store is what makes the 15-query suite an end-to-end story:
query 1 pays the write, queries 2..15 restore whatever documents
they share with earlier queries.

### 5.4 Modal wiring

- One `modal.App`; the existing image (CUDA 13.0.1, pinned
  `vllm==0.26.0`, weights in the `quail-hf-cache` volume).
- Workers are a `modal.Cls` with `gpu="H100!"`,
  `memory=cpu_memory_gb x 1024`, kept warm for the session's
  lifetime (`min_containers=gpus` while a session is open); a
  session maps to a set of warm containers, which is what lets the
  store persist across the suite's queries.
- The coordinator is the user's local process (or any CPU box):
  compile and plan are pure Python; per-worker physical plans ship
  as JSON; answers come back as compact answer matrices.
- Every worker run is teed to a file in the results volume, per
  house rule; the coordinator collects the logs next to the run
  report.

---

## 6. One query, end to end

The Q6-shaped query from §2.3: filter reviews, join survivors to
products.

1. `sess.sql(...)` parses and binds: one `SemanticFilter` on
   `reviews`, one `SemanticJoin(reviews, products, full)`.
2. Planning samples the filter's selectivity (~2k reviews labeled
   by the model, seconds of GPU) because the join's pair count
   depends on it; measures nothing for the join predicate.
3. The optimizer prices both placements. Pushdown: 1,000 reviews x
   ~50 question tokens plus the killed documents' pair savings.
   Pull-up: 1,000 x 100 pairs x ~160 suffix tokens, filtered
   after. Pushdown wins by ~20x here; the plan records both prices.
4. Anchor choice: reviews (mean 320 tokens) anchor, products
   (~160-token suffix) stream — enumerated, not assumed.
5. `explain()` prints the tree with per-operator predicted tokens
   and wall; `run()` ships the plan.
6. The worker runs `DocScan(reviews)` (restore if warm),
   `FilterChain` on the engine, then hands survivor ids back; the
   coordinator builds the pair list; the worker runs one
   `JoinStage` on the packed executor; `Sink` returns pairs with
   predicted-vs-measured attached.

---

## 7. The benchmark: QUAIL-B

Fifteen queries over five document sets, shaped like TPC-H in the
only sense that transfers: a big fact-like set most queries touch,
smaller dimension-like sets, and a fixed query list that mixes
selective scans, joins of different shapes, and semi-joins, run
both per-query and as a suite. Only filters and joins, per scope.

### 7.1 Document sets (scale factor 1)

| set | source | docs | mean tokens | total tokens | role |
|---|---|---|---|---|---|
| `reviews` | IMDB (the existing corpus, same seed) | 10,000 | 320 | 3.20M | the fact set; filter chains run here |
| `rsample` | seeded 1,000-review subset of `reviews` | 1,000 | 320 | 0.32M | join-sized review side |
| `reports` | BioDEX patient reports (existing sample protocol) | 100 | ~2,977 | 0.30M | long documents; anchor and keep-KV stress |
| `terms` | BioDEX reaction terms | 2,560 | ~32 | 0.08M | short partner stream |
| `products` | ABT-BUY product descriptions | 100 | ~120 | 0.01M | mid-length dimension set |

A scale factor multiplies document counts; SF=1 is sized so the
whole suite fits one Modal session comfortably (see 7.4).

### 7.2 Predicates the instrument can read

The join findings showed the 4B checkpoint answers YES to almost
everything when a predicate requires comparing two planted keys
across a long context (9,763 of 10,000 stage-1 answers wrong), while
single-flag lookup in the filter workload worked. So every
benchmark predicate with a designed selectivity is a single lookup:

- **Filters**: the existing planted `[FLAGS]` line; filter j asks
  what flag j says. Selectivity = the planted rate.
- **Joins**: the anchor document carries a planted `[KEYS] X=<k>`
  line; the partner's key is printed in the question tail
  ("Does the KEYS line above contain X=k7?"), so the model
  compares a value in context against a value in the question —
  the same single-lookup difficulty as the filters. Pair
  selectivity = the key collision rate, set per query.

Ground truth is known by construction; each run reports answer
accuracy as an instrument check, and the replay check (recorded
answers through the nested-loop reference) gates every join query.
Timing claims never depend on the model answering correctly;
gating claims depend only on it answering consistently with the
planted rates, which this predicate design is built to give.

### 7.3 The queries

F = filter stage, J = join stage. Selectivities are the planted
rates. Predicted walls are cost-model arithmetic at the packed
82k tok/s and engine 96k tok/s rates, stated now per house rule
and re-derived by `explain()` when the planner lands; treat them
as targets with the model's demonstrated ±10% band.

| id | shape | sets | selectivity design | what it isolates | TPC-H analogue | predicted wall |
|---|---|---|---|---|---|---|
| Q1 | 1F | reviews | 0.5 | the degenerate chain; per-query floor (c0, boot amortization) | Q6 selective scan | ~35 s |
| Q2 | 5F | reviews | 0.9/0.9/0.9/0.8/0.8 | the headline rewind workload, unchanged as the regression anchor | Q1 heavy scan | ~40 s |
| Q3 | 5F | reviews | 0.9/0.9/0.2/0.9/0.9 with the 0.2 written LAST in SQL | filter reordering: planner must move the selective filter early | Q1 + optimizer twist | ~34 s |
| Q4 | 2F | reports | 0.8/0.5 | long documents: quadratic surcharge, admission with fat docs | Q1 on wide rows | ~10 s |
| Q5 | 1J | reports x terms | pair 0.08 | the measured BioDEX join shape; packed executor anchor orientation | Q12 two-table join | ~104 s |
| Q6 | 1F + 1J | rsample x products | F 0.25, pair 0.05 | filter pushdown below a join (paper's placement result, priced not asserted) | Q3 filtered join | ~55 s |
| Q7 | 1J | rsample(400) x products | pair 0.05, predicate reads both texts | pure pair predicate: nothing can push; upper bound on pair cost | Q19 predicate join | ~195 s |
| Q8 | 1J exists | rsample x products | match 0.05 | exists semantics: early-stop streams; expected-scan pricing | Q4 / Q21 semi-join | ~45 s |
| Q9 | 2J chain | rsample(100) x reports(100 planted keys) x terms(100) | stage sels 0.2 / 0.1 | 3-way gating and dedup; stage-2 pair count must equal survivors x partners exactly | Q3 three-table | ~96 s |
| Q10 | 2J chain | rsample(200) x reports x terms | stage sels 0.3 / 0.1 | middle anchor is long (reports): the keep-KV branch of the size rule across stages | Q9 deep join | ~120 s |
| Q11 | 2F + 1J | rsample, products | F 0.3 and 0.5, pair 0.05 | pushdown on BOTH sides; pair list shrinks multiplicatively | Q16 two-sided filters | ~35 s |
| Q12 | 5F | reviews (new flags) | 0.9…0.8 | warm-store restore for a filter query: R1 of the persist result inside the engine proper | repeated Q1 | ~22 s warm |
| Q13 | 1J | reports x products (new keys) | pair 0.1 | warm anchor restore (phase 2): prefix share 16%, so restore is visible in the wall | repeated Q12-analogue | ~23 s cold, ~19 s warm |
| Q14 | 2J star | reports x terms, reports x products | pair 0.08 / 0.1 | star shape: one anchor set feeds two predicates; anchor KV computed once, read twice | Q9 star | ~130 s |
| Q15 | 2F + 1J + 1J exists | rsample, reports, terms | F 0.3/0.5, pairs 0.1, exists 0.05 | everything at once: placement + ordering + gating + exists; the optimizer's full search space | Q21 kitchen sink | ~90 s |

Suite total, predicted: ~17 GPU-minutes warm, plus one engine boot.
Q2 and Q5 reproduce committed results inside the new engine — if
either moves more than the known band, the engine added overhead
the exploration didn't have, and that is a finding.

### 7.4 Protocol and metrics

Two suite runs, same session, same container, fixed query order
Q1→Q15 (the order above deliberately re-touches `reviews`,
`reports`, and `products` so later queries can restore):

- **cold**: store disabled (`cpu_memory_gb` headroom only), engine
  prefix cache reset between queries. Measures each query alone.
- **warm**: store enabled, no resets. Measures the suite as a
  workload: per-query walls again, plus the end-to-end wall.

Reported per query: predicted wall, measured wall, fresh tokens,
cached-read tokens, KV bytes restored, pairs per stage, answer
accuracy vs planted truth, replay-check verdict. Reported per
suite: end-to-end wall (cold and warm), warm/cold ratio per query
and total, store hit tokens.

Baseline, per house rule (a baseline gets the analytically
equivalent configuration, and the submission strategy is named):
stock vLLM 0.26 in the same container, prefix caching on, no
store. Filters run as separate requests per stage under the same
token-budget admission (the committed `run_stock` protocol). Joins
run as grouped requests per pair, one request per pair in anchor
order, admission matched from the same pool arithmetic (the
committed grouped-stock protocol). Stock runs both cold and warm
suites; its warm benefit is whatever vLLM's own prefix cache holds
across queries, which is the honest comparison for the store.

Predictions to state before the first full run, from the measured
constants: cold suite ~19 min Quail vs ~45–60 min stock (the join
queries dominate the gap at the measured 4.1x; filter queries at
1.08x); warm suite ~15 min Quail, stock nearly unchanged from cold
(its pool cannot hold 3.5M tokens of corpus KV across queries, so
its prefix cache re-serves little).

---

## 8. What gets built, in order

Each step lands with CPU tests; GPU cells only confirm.

1. **`quail/logical.py` + `quail/sqlfront.py`** — logical operators,
   the sqlglot subset parser, the builder API, binding against a
   catalog. Pure Python. (New code; nothing carries over.)
2. **`quail/plan/optimizer.py`** — the enumerating optimizer over
   logical plans, wrapping the existing `plan_query` budget
   derivations and the `join_plan.md` stage pricing into one
   `PhysicalPlan`. (Mostly rearrangement: `cost.py` and
   `planner.py` carry over intact; the join pricing moves from
   `plans/join_estimates.py` prose into code.)
3. **`quail/runtime/worker.py` + Modal app** — the worker class
   hosting both executors behind one RPC surface; `Session` and
   the coordinator. (The executors exist: `engine_client.py`,
   `QuailScheduler`, the `run_join` loop from
   `modal_join_forward.py` promoted out of experiment code. The
   shared-boot memory split is the one new GPU-facing decision.)
4. **`quail/bench/quailb.py`** — document set builders (planted
   flags and keys per 7.2), the 15 queries as code, the two-suite
   runner, the report. (The corpus builders and planted-flag
   machinery carry over from `experiments/workload.py`.)
5. **Phase 2, after the suite runs end to end**: packed-path anchor
   restore from the store (the dtype parity cell first), and the
   stock host-ingestion term the join findings showed missing from
   `cost.py`.

Risks, named:

- **The shared boot.** Engine pool shrinks to make room for the
  packed executor. The admission budget is designed for corpora
  bigger than the pool, so the prediction is a small filter
  slowdown at most; Q2's regression gate catches it if the
  prediction is wrong.
- **Serialized executors idle the GPU between operators.** Within
  one query the handoff is one pair-list construction (CPU,
  milliseconds at these sizes). Across the suite it is real only
  if boot-per-executor were chosen; the shared boot avoids that.
- **Sampling cost is on the critical path** for gated plans. ~2k
  labels at ~350 tokens each is ~7 s of GPU per sampled predicate;
  the planner charges it in the predicted wall rather than hiding
  it.
- **The optimizer trusts sampled selectivities.** A wrong sample
  mis-orders filters or mis-orders join stages; the run report
  prints planned vs observed selectivity per stage so the miss is
  visible. Runtime re-ordering (the paper's adaptive path) is out
  of scope until a measured miss shows it is needed.
- **The 27% flag-misread rate** of this checkpoint bounds how
  precisely planted selectivities control gating. Same instrument
  caveat as every result in this repo; comparisons hold, accuracy
  claims stay off the table.

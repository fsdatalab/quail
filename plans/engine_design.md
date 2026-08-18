# Quail engine: a declarative query engine for AI_FILTER and AI_JOIN

Status: design for the new repo. This exploration repo stays as the
evidence base; the engine gets its own repository (working name:
`quail`), built clean from this design. Nothing in the new repo may
depend on experiment scripts here — the constants, the executor
loop, and the pure-logic modules move over; the rest is history.

Scope: AI_FILTER and AI_JOIN only. Qwen3 4B fp8 is the first model,
H100 workers on Modal the first device — but the optimizer is built
against model/device structs from day one (§8), not against our
profiled numbers. No maps, no classification, no aggregation, no
cascades. The same discipline applies to relational algebra: the
only relational operator is projection, because a result has to
name its columns. No non-AI predicates, no equality joins, no
GROUP BY, no ORDER BY, no LIMIT, no DISTINCT, no expressions. A
query that needs those runs Quail for the semantic part and does
the relational part in whatever database the ids came from.

What the design rests on (all committed in this repo):

| result | where |
|---|---|
| Packed executor, 2-way join, 256k pairs: 103.6 s vs 429 s for stock vLLM submitting grouped requests per pair — 4.1x | `results/engine/join2way.json` |
| Packed executor, 3-way join, gated and deduped between stages, replay-consistent | `results/engine/join_nway3.json` |
| Packed single filter: 121,045 tok/s vs 96,946 on the engine — the packed loop is the faster substrate for filters too | `plans/packed_forward.md`, `single_filter_forward_vllm_kernels.json` |
| Chain semantics (keep document KV, attach question suffixes) beat per-stage requests: 39.8 s vs 42.9 s at 10k docs | `results/engine/filter_cells.json` |
| Cross-query KV restore from a pinned CPU store: 1.9–2.1x over recompute | `results/engine/persist_split7_quail_waves_chain_10k.json` |
| Attention-merge parity: 0 disagreements across all gates; rate 82.1k tok/s at the large-chunk geometry | `results/engine/join_probe.json` |
| Cost model: predictions within ~2–3% on filters, +7% on the packed join | `results/engine/makespan_check.json`, `join_findings.md` |

---

## 1. Shape of the system

Four layers, in the style of DataFusion: a session that owns a
catalog of document providers, a SQL front end that compiles the
Snowflake AISQL syntax to a logical plan, a planner that prices the
plan with a spec-derived cost model, and one executor that runs it
on Modal GPUs.

    AI SQL text or builder calls
            |
            v
    +------------------+
    |  compiler        |  sqlglot (Snowflake dialect), bind columns
    |                  |  against the catalog, emit a LogicalPlan
    +------------------+
            |
            v
    +------------------+
    |  planner         |  pushdown (always), ordering (user-given or
    |                  |  from provided selectivities), anchor
    |                  |  choice, budgets and caps from the model/
    |                  |  device specs; emit PhysicalPlan or Refusal
    +------------------+
            |
            v
    +------------------+
    |  coordinator     |  local process, deliberately thin:
    |                  |  compile, plan, ship one shard plan per
    |                  |  GPU, re-shard between join stages only
    |                  |  when the anchor relation changes, merge
    |                  |  final answers, apply the projection
    +------------------+
            |
            v
    +------------------+
    |  Modal workers   |  containers of up to 8 H100s, restored
    |                  |  from a GPU memory snapshot. ONE executor
    |                  |  per GPU: the packed forward loop, for
    |                  |  filters and joins alike. Gating, dedup,
    |                  |  pair-list construction, and chunk packing
    |                  |  run HERE, next to the GPU. One pinned CPU
    |                  |  KV store per container, shared by its GPUs.
    +------------------+

The coordinator/worker split is a latency rule, not a layering
preference: anything consulted per chunk or per stage — gate the
answers that just landed, dedup survivors, build the next pair
list, pack the next chunk — runs on the worker, in the same
process as the GPU loop, exactly where the exploration's `run_join`
ran it. Put any of that a network hop away and the GPU idles at
every stage boundary. The coordinator exists only for what one
worker cannot see: splitting work across GPUs, re-sharding answers
between join stages when the next stage anchors on a different
relation, merging shard results, and the projection.

There is exactly one execution substrate (§6). vLLM appears nowhere
in the execution path: it remains a library for weight loading and
kernels (public surface only — the version pin existed for
scheduler internals, which are gone), and as the stock baseline in
the benchmark. The engine-scheduler fork (`QuailScheduler`, chain
mode) stays in this exploration repo as evidence and fallback; it
does not move.

Everything above the workers is CPU-only Python and unit-testable
without a GPU.

---

## 2. The user interface

### 2.1 Configuration

```python
@dataclass(frozen=True)
class EngineConfig:
    gpus: int = 1               # total H100 count. Packed into as
                                # few containers as possible, up to
                                # 8 GPUs each (gpu="H100!:8"):
                                # fewer containers means one
                                # snapshot restore, one shared KV
                                # store, and stage re-sharding that
                                # stays on one host.
                                # tensor_parallel derives from the
                                # ModelSpec (1 at 4B)
    cpu_memory_gb: int = 64     # per-container host memory cap.
                                # The pinned KV store gets this
                                # minus a fixed working headroom
                                # and is shared by the container's
                                # GPUs; 0 disables the store
    model: str = "qwen3-4b-fp8" # must name a registered ModelSpec
                                # (§8); an unknown name is a Refusal
```

The three knobs map directly onto Modal resources: `gpus` becomes
`ceil(gpus/8)` containers with `gpu="H100!:k"`, `cpu_memory_gb` is
the container `memory=` request, `model` selects the weights volume
and the spec struct.

### 2.2 DocumentProvider and the catalog

`DocumentProvider` is the analogue of DataFusion's `TableProvider`:
a named source of rows, where each row has an id and one or more
text columns.

```python
sess = quail.Session(EngineConfig(gpus=1, cpu_memory_gb=256))

sess.register("reviews",  DocumentProvider.from_parquet("imdb.parquet", id_col="id"))
sess.register("products", DocumentProvider.from_hf("...", id_col="asin"))
```

There is no `text_col` at registration, because the provider does
not decide which column is "the document" — the query does. A
provider exposes a schema (its column names); the PROMPT in the
query references a column (`r.review`, `p.description`), and the
binder resolves that reference to the provider's column. One
provider can serve different queries through different columns.

`register` does two things, both cheap: binds the name into the
session catalog, and reads the schema (parquet metadata / dataset
features — no data scan). Tokenization is not registration work; it
happens at the first `DocScan` of a column and is cached under a
content hash of (provider data, column, tokenizer), so
re-registering identical data reuses the cache, and so does the KV
store.

### 2.3 The SQL entry point

The syntax is Snowflake's Cortex AISQL (arXiv 2511.07663), filters
and joins only:

```python
q = sess.sql("""
    SELECT r.id, p.id
    FROM reviews r
    JOIN products p
      ON AI_FILTER(PROMPT('Review {0} discusses product {1}',
                          r.review, p.description),
                   {'selectivity': 0.05})
    WHERE AI_FILTER(PROMPT('This review is negative: {0}', r.review),
                    {'selectivity': 0.3})
""")
```

The second argument to AI_FILTER is Snowflake's option object (the
paper uses it for `{'model': ...}`); we use it for per-predicate
options. Options attach to one predicate, not to the query, so
there are never lists to line up: each AI_FILTER call carries its
own object. A filter predicate takes `selectivity` (one rough
number — the fraction of documents that pass; used only for
ordering, §2.5). A join predicate takes `selectivity` (the
fraction of pairs that pass) and `anchor`, which names the table
alias whose documents anchor that stage — one value, because one
predicate is one join stage. Omitted options mean: no selectivity
(that predicate keeps its written position) and planner-chosen
anchor (the longer side).

An n-way join is written the way Snowflake writes it: one JOIN
clause per edge of the join graph, each edge with its own
predicate and its own options. A 3-way chain:

```sql
SELECT a.id, b.id, c.id
FROM reviews a
JOIN threads b
  ON AI_FILTER(PROMPT('Does review {0} praise this thread? {1}',
                      a.review, b.thread),
               {'selectivity': 0.2, 'anchor': 'b'})
JOIN products c
  ON AI_FILTER(PROMPT('Does thread {0} recommend product {1}?',
                      b.thread, c.description),
               {'selectivity': 0.1, 'anchor': 'b'})
```

Two predicates give two edges (a–b, b–c), which the planner runs
as two stages. Here `b` anchors both, so each surviving thread's
document KV is computed in stage 1 and read again in stage 2 —
the cross-stage reuse the executor's kept KV exists for. Writing
`{'anchor': 'a'}` on the first edge instead would force reviews to
anchor stage 1; the planner would then warn in `explain()` if that
choice prices worse.

### 2.4 The builder entry point

The builder mirrors AI SQL construct for construct — same PROMPT
semantics, same column references, same options — so a query
translates line by line between the two:

```python
from quail import col, prompt

q = (sess.docs("reviews").alias("r")
     .ai_filter(prompt("This review is negative: {0}", col("r.review")),
                selectivity=0.3)
     .ai_join(sess.docs("products").alias("p"),
              prompt("Review {0} discusses product {1}",
                     col("r.review"), col("p.description")),
              selectivity=0.05)
     .select("r.id", "p.id"))
```

Exists and anti semantics are the same call with a flag —
`ai_join(..., semantics="exists")` / `semantics="anti"` — matching
the SQL `WHERE [NOT] EXISTS` form (§3.1).

Both entry points produce the same `LogicalPlan`. Then:

```python
q.explain()   # logical tree, physical plan, per-operator predicted
              # wall and token counts — printed BEFORE any run
res = q.run() # measured walls next to the predictions, plus tokens
              # computed, KV bytes restored, pairs per stage
```

Failures come in three kinds, at three times, and only one of them
is a `Refusal`:

- **Compile errors, at `sess.sql()` / builder time.** The query
  itself is malformed or outside the language: unsupported syntax
  (a GROUP BY, an OR between predicates — rejected by node class,
  §3.2), an unknown provider or column, a PROMPT whose
  placeholders do not match its arguments, an unknown option key.
  Raised immediately; nothing is planned, nothing runs.
- **`Refusal`, at plan time.** The query is valid but this
  configuration cannot execute it, and the planner says which
  constraint failed, what was needed, and what was available,
  instead of running something degraded. Concrete cases: the
  model's weights need more cards than `gpus` provides; one
  document plus its question tail exceeds the chunk budget
  (suffixes are atomic — no chunk can ever hold it); the working
  set needs the store to spill but `cpu_memory_gb` is 0; `model`
  names no registered spec.
- **Runtime errors.** Not a category the design plans for:
  admission and the chunk budget guarantee memory fits by
  construction, so an OOM at run time is a bug to fix, not a
  condition to handle.

### 2.5 Controlling filter order and join order

Every physical plan has a filter order and a join stage order.
Each comes from exactly one of two places:

- **You set it.** In the builder, the order you chain calls is the
  order that runs — nothing is ever reordered behind your back. In
  SQL, `sess.sql(text, order="as_written")` runs the WHERE
  conjuncts in written order and the JOIN clauses in written
  order.
- **The planner sets it from your selectivities.**
  `sess.sql(text, order="by_cost")` sorts filters by cost per
  killed document and picks the join stage order by the priced
  enumeration (§4). Both use only the selectivity numbers you
  provided in the option objects — there is no other input.

When `order` is not passed, the default is `by_cost` if every
gated predicate carries a selectivity, `as_written` otherwise, and
`explain()` prints the chosen order and which rule chose it.

A concrete example of the difference: benchmark query B3 writes a
0.2-selectivity filter third among five. Under `as_written` it
runs third; under `by_cost` it runs first, so the other four
filters only see the 20% of documents it passes. Same answers
either way — order changes cost, never results.

There is no third source of ordering: no sampling at plan time, no
reordering at run time (both deferred, §4). For comparison,
DataFusion offers the same two sources — its DataFrame/plan-builder
API runs what you build — but no SQL-level order flag.

### 2.6 What a result is

Filters return surviving document ids; joins return id tuples.
Projection — the SELECT list, the one relational operator — is
column selection only: ids and pass-through text, nothing computed.
Results carry the per-stage answer matrices, so the nested-loop
replay check (`joinlogic.brute_force_triples`) stays available as a
correctness gate on every run.

---

## 3. Parsing AI SQL

### 3.1 The accepted grammar

Snowflake AISQL syntax, and only this subset of it:

- `SELECT <plain columns> FROM <provider> [alias]` — the SELECT
  list is the projection: bare column references, optionally
  aliased. `*` means every column of every named provider.
- zero or more
  `JOIN <provider> [alias] ON AI_FILTER(PROMPT('...', a.col, b.col) [, {options}])`
- `WHERE` as a conjunction (`AND`) of
  `AI_FILTER(PROMPT('...', x.col) [, {options}])` terms, each
  referencing exactly one provider
- `WHERE [NOT] EXISTS (SELECT 1 FROM <provider> s WHERE AI_FILTER(PROMPT('...', outer.col, s.col)))`
  — exists semantics (keep a document if some partner matches; the
  executor stops the partner stream at the first YES) and anti
  semantics (keep it if none does; the executor stops at the first
  YES and drops the document — same early stop, inverted keep)

Rejected with a named error, not worked around: every relational
operator except the projection above — non-AI predicates (including
equality join conditions), GROUP BY, ORDER BY, LIMIT, DISTINCT,
expressions in the SELECT list, set operations, subqueries other
than the EXISTS form. Also OR between AI predicates (a disjunction
belongs inside one prompt's text, where the model evaluates it),
AI_FILTER over more than two providers, and any other AI_*
function.

### 3.2 How the parse works

`sqlglot` with `dialect="snowflake"`, not a hand-rolled grammar.
What that buys: Snowflake quoting, aliasing, precedence, and the
option-object literal all parse correctly for free, and sqlglot is
pure Python with no native dependencies. What we add is a
validator and a binder over its tree:

1. sqlglot parses `AI_FILTER(...)` and `PROMPT(...)` as generic
   function nodes (`exp.Anonymous`). A walker matches them by name
   in the two legal positions (WHERE conjuncts, JOIN ON) and
   rejects everything else in the tree by node type — the
   rejection list in §3.1 is literally a list of forbidden sqlglot
   node classes, so new SQL surface cannot creep in silently.
2. The PROMPT template's `{0}`, `{1}` placeholders must match its
   column arguments in count and order. Column references resolve
   through the alias table to (provider, column); a predicate whose
   columns span one provider is a filter, two providers a join,
   three or more an error.
3. The option object parses as a Snowflake object literal; unknown
   keys are errors (again: no silent surface).
4. The prompt text is tokenized once at bind time and split into
   the shared preamble and the per-document tail — the split that
   both the executor's KV reuse and the cost model price.

Output: a `LogicalPlan` of four operators — three semantic, one
relational:

| operator | fields |
|---|---|
| `Scan(provider, column)` | which column of which provider supplies the document text |
| `SemanticFilter(input, predicates)` | ordered list of (prompt, selectivity or None) |
| `SemanticJoin(left, right, predicate, semantics, selectivity, anchor)` | semantics: `full` / `exists` / `anti` |
| `Project(columns)` | always the root, never anywhere else |

`Project` never participates in planning: it costs nothing the
model runs; the coordinator applies it at the sink.

---

## 4. The planner

Simpler than a database optimizer, on purpose. Three decisions are
not decisions at all:

- **Pushdown is unconditional.** A filter on one side of a join
  always runs before that join. Under true pairwise join semantics
  this cannot lose: evaluating the filter costs one question per
  document either way (the join neither duplicates nor drops the
  documents the filter must judge — it only pairs them), and
  running it first deletes every pair the failed documents would
  have generated. There is no Cortex-style pull-up case to price,
  because we never rewrite a join into anything else — AI_JOIN is
  evaluated as the pairwise predicate it declares.
- **No selectivity estimation.** Selectivities are provided by the
  user (rough numbers; only ordering consumes them) or absent.
  Absent means as-written order. A sampling pass at plan time is a
  named future extension, not a phase-1 behavior — it buys ordering
  quality at the cost of GPU time and plan-time complexity, and the
  interface already gives the user the cheaper path.
- **No runtime re-ordering.** The run report prints provided vs
  observed selectivity per stage, so a wrong estimate is visible;
  reacting to it mid-query is future work.

What the planner actually decides:

1. **Order** (only under `by_cost`): filters ordered by cost per
   killed document — (question tokens) / (1 − selectivity) — pure
   arithmetic on provided numbers. Join stage order over the join
   graph: enumerate (2–4 relations, trivial), price stages with the
   pair-count formulas from `join_plan.md`, survivor-thinned by the
   provided selectivities.
2. **Anchor per join stage**: the longer side anchors (anchor
   tokens are paid once per document, partner tokens once per
   pair); enumeration confirms it since it costs nothing;
   `{'anchor': ...}` overrides it.
3. **Admission budget**: how many tokens of document KV may be
   resident at once, from the spec-derived arena arithmetic (§8).
   Token-based, never a document count, for the measured reason (a
   count cannot see length; the 4,096-seq default thrashed at
   2.40x reads). How the budget is consumed at run time —
   continuous bin packing with survivor priority — is executor
   behavior, §6.
4. **Chunk budget**: `min(memory bound, kernel index cap)`, both
   derived from the specs (§8), never typed in.
5. **Access per scan**: read (compute KV fresh), restore (the
   pinned store is warm for this content hash and its bandwidth
   beats kappa x prefill rate — the measured 7.2 GB/s break-even
   at 4B, and pinned host memory measured 55 GB/s), or spill (the
   working set exceeds the pool and the store absorbs overflow).
6. **Sharding**: filters split documents by token count across
   GPUs (`_balanced_shards`); joins split by anchor document.
   Every pair belongs to exactly one anchor, so gating, dedup, and
   the next stage's pair list for that anchor are local to the GPU
   that holds it — workers never talk to each other. The
   coordinator steps in between stages only when the next stage
   anchors on a different relation (answers must regroup under the
   new anchor set), and to merge final answers across shards.

Infeasible configurations — weights don't fit the cards, pool
cannot hold one working set, a suffix exceeds the chunk budget —
return the existing `Refusal` with the violated constraint named.

The output `PhysicalPlan` is a JSON tree; each node carries its
settings and its predicted tokens and wall. `explain()` prints it;
the result stores prediction next to measurement, so every run
doubles as a cost-model check.

---

## 5. Physical operators

| operator | runs on | what it does |
|---|---|---|
| `DocScan` | worker CPU + store | resolve (provider, column) to token arrays (cached tokenization); execute the planned access: read, restore (batched pinned-memory loads), spill |
| `FilterChain` | the executor | gated stages over continuously admitted documents: documents are anchors, each stage's question tail is a one-suffix stream, survivors advance (§6) |
| `JoinStage` | the executor | brim-packed chunks over the stage's pair list; exists/anti streams stop early |
| `Gate` / `Dedup` | worker CPU, inside the chunk loop | survivors and distinct anchors, decided the moment a chunk's answers land — the next chunk's contents depend on them, so they cannot live a network hop away |
| `Assemble` | worker CPU per shard, coordinator merges | tuple reassembly from recorded answers (`joinlogic.py`, moved over as is); the coordinator only concatenates shard outputs |
| `Sink` | coordinator CPU | applies the `Project`; returns ids/tuples plus the run report |

`DocScan` is an explicit plan node because it is where
tokenization, corpus statistics, and the store decision live — and
it is the unit the benchmark's warm suite reuses across queries.

---

## 6. One executor

The exploration ran filters on a modified vLLM engine and joins on
a packed forward loop. The engine is gone from this design. One
substrate — the packed executor — runs both operators, because a
filter chain is the degenerate join:

- **Join stage**: anchor = one side's document; suffixes = partner
  document + question tail, one per pair.
- **Filter stage**: anchor = the document; suffix = the question
  tail, exactly one per stage. Stage j+1 attaches a fresh suffix to
  the same kept anchor KV.

This is not a workaround; it is the same computation KV rewind
performed, minus the machinery. Rewind existed because an engine
sequence accumulates question KV that must be erased back to the
document boundary. The packed loop never writes suffix KV, so
there is nothing to erase — keeping the immutable document KV and
attaching the next question computes bit-for-bit what rewind
computed (`join_plan.md` §4; the parity gates in
`join_probe.json`). The measured support for moving filters onto
this substrate: the packed loop ran a single filter at 121k tok/s
against the engine's 96.9k.

What this deletes from the system:

- the second weight copy and the shared-boot memory split;
- the "one worker, two executors" arrangement and its serialization;
- the `QuailScheduler` fork and with it every private-API reach
  into vLLM — the `vllm==0.26.0` pin dies here;
- the keep-vs-recompute size rule (the "f/S below ~8%" threshold).
  That rule chose between recomputing an anchor's prefix per chunk
  and keeping its KV. The rewritten executor already removed the
  recompute path — an anchor's prefix is computed exactly once,
  ever, kept in plain per-anchor tensors, and freed at its last
  use. With one executor and always-keep, the threshold has
  nothing left to decide.

To be precise about what is and is not kept, because it is the
heart of the design: **document KV is kept, suffix KV never
exists.** A document's KV is written once, stays resident across
every stage that still needs it — all five filters, or both join
stages it anchors — and is freed the moment nothing later can read
it. What is never stored, by anyone including stock vLLM, is the
per-pair, per-question suffix KV: a question tail or a streamed
partner lives only inside its chunk's forward pass.

**Two token budgets, and why neither makes the GPU faster.** The
chunk budget is the batch size: how many tokens enter one forward
pass (the analogue of `max_num_batched_tokens`). The admission
budget is KV residency: how many tokens of document KV may sit in
the arena. They cap different resources — chunk tokens cost
activation memory and kernel index range; resident documents cost
KV bytes — and neither is a speed knob, because the workload is
compute bound: wall = fresh tokens / rate, and the rate is flat in
chunk size past the ~400-token knee. What each budget actually
protects:

- The chunk budget keeps the per-chunk fixed cost amortized. Any
  value well past the knee does that; we run at the kernel cap
  because bigger is free (the measured rate was the same at 84k
  and 110k-token chunks), not because it is needed. It is not
  limited by KV at all.
- The admission budget guarantees no document token is ever
  computed twice: a document's KV must live from its first stage
  to its last, and a preallocated, never-oversubscribed arena
  turns "computed once" from a cache policy into a certainty.
  Memory here is a rework-avoidance resource, not a performance
  resource. The measured cost of getting it wrong: unbounded
  admission at 3.4x pool pressure recomputed 2.40x the corpus and
  took 80.1 s against 42.9 s.

The arena takes all the HBM left after weights and activations
not because the executor needs it to run fast — it would run at
the same rate with a quarter of it — but because idle HBM buys
nothing, and more resident documents means the packer always has
work and the pending queue drains sooner.

Two pieces replace the engine's paged pool:

**The KV arena.** One preallocated GPU buffer per layer, sized to
the admission budget, divided into fixed 16-token pages with a
free list. A resident document owns a list of pages; the pages
return to the free list the instant the document fails a stage or
answers its last one. FlashInfer's paged attention kernels read
this layout natively, so there are no copies and no compaction,
and since admission never lets page-rounded resident tokens exceed
the arena, allocation cannot fail. (The exploration's join
executor got away with plain per-anchor tensors because it held
one or two anchors at a time; a filter scan holds a thousand
documents of varying lengths, and fixed pages are what makes
thousands of small allocations and frees boring.) At 4B with bf16
KV the arena holds ~400k tokens — roughly 1,000 mean-length
documents resident at once.

**KV dtype: the planner picks the latency argmin.** One dtype per
session, identical in the arena and the store. The planner prices
the wall once per dtype and takes the minimum — no default, no
constraint, no taste. Every place the dtype enters time is a
priced term:

- KV quantization on write: fp8 pays a quantize per fresh
  document token as its KV lands in the arena (and the store
  path pays nothing extra — bytes are copied as stored). Small —
  the measured kernel mix puts all fp8 quant/scale work at 19% of
  step time, and KV quant is a sliver of that — but priced, not
  waved away;
- attention reads of kept KV: fp8 halves the bytes read and adds
  the dequant; bounded either way by attention's measured 4%
  share of step time;
- store transfers: kappa(dtype) x tokens / channel bandwidth —
  bf16 doubles the bytes of every restore and spill;
- store overflow: documents past cpu_memory_gb / kappa(dtype)
  lose their store slot and are recomputed on later queries at
  the serving rate.

`explain()` prints both walls and the pick. For a cold query the
terms nearly cancel and the argmin lands on fp8 by the byte
counts; a warm suite with store pressure widens fp8's win through
the transfer and overflow terms. bf16 wins only if a measurement
ever shows a real fp8 read/quant penalty at some shape — which is
exactly what the pricing would then reflect.

Answer quality is reported, not enforced: the measured flip count
(bf16 fixed 765 of the 2,990 answers fp8 KV got wrong on the
filter ladder) lives in the calibration overlay and appears in
the run report next to the dtype pick, so the cost of the fast
choice stays visible. `kv_dtype` in the config forces either
dtype for users who want the call made differently.

**Continuous admission — bin packing by tokens, twice.** There is
a pending queue of documents and a resident set; nothing runs in
lockstep and there are no barriers. Each chunk is packed to the
brim (the chunk budget is one bin) from two sources, in priority
order:

1. next-stage suffixes of resident survivors — small (a question
   tail, or the continuation of a partner stream), and every one
   completed moves a document toward freeing its pages;
2. fresh documents from the pending queue, admitted whenever
   their page-rounded tokens fit the free list (the admission
   budget is the other bin).

Survivor priority makes residency drain monotonically, which is
the answer to the straggler question: a long pending document that
does not fit right now waits in the queue while chunks stay full
of survivor suffixes and smaller admissions — the GPU never idles
waiting for memory, the document is admitted as soon as enough
pages free, and the wait is bounded because pages only ever flow
back. A document that could not fit even into an empty arena was
refused at plan time, so there is no deadlock case. Chunks mixing
fresh document prefixes with kept-KV continuations are the
measured normal, not an edge case — the probe's mixed-group parity
gate covers exactly that shape. Chunk construction overlaps the
previous chunk's forward pass (the existing loop), so this
scheduling runs on CPU time the GPU never sees.

**What a multi-filter run does to one document.** Document D, 400
tokens, three filters:

1. D reaches the head of the pending queue and 400 tokens of pages
   are free: some chunk packs the group
   `[D's 400 tokens | q1's 25 tokens]`. Self-attention runs within
   each segment; D's KV is written to its pages; q1's KV is
   written nowhere. The YES/NO logits at q1's last position are
   stage 1's answer.
2. On YES, a later chunk packs just the 25-token group `q2`, no
   prefix segment: its cross-attention reads D's pages, its
   positions start at 400. This computes exactly what the engine's
   KV rewind computed — with nothing to rewind, because no
   question KV was ever written. On NO, D's pages free
   immediately and D never appears again.
3. After q3 (or the first NO), D's pages free — unless the plan
   marked D for the store (a later query over this set, or a later
   join stage anchored on it), in which case its KV is copied out
   to pinned host memory first, overlapped with the next chunk.

The engine chain path is not deleted from the world: it exists in
this exploration repo, measured, as the documented fallback if a
packed-path parity gate ever fails on a new shape. It just is not
part of the new system.

Risk, stated: a full gated multi-stage filter chain on the packed
executor is unmeasured (single filter and gated join stages are).
Milestone 1 in §10 runs exactly the committed 10k-document
five-filter workload on the new executor; prediction from the
measured constants: ~32 s against the engine's 39.8 s, tolerance
the cost model's demonstrated ±10%. A miss past the band reopens
this section.

---

## 7. The Modal runtime

- One `modal.App`; image with CUDA, torch, FlashInfer/FlashAttention
  and vLLM as a kernel-and-loader library; weights in a volume.
- Workers are a `modal.Cls` with `gpu="H100!:k"`, k up to 8 GPUs
  per container before a second container starts, and
  `memory=cpu_memory_gb x 1024`, kept warm while a session is
  open. Packing GPUs into one container is not just cost hygiene:
  one container means one snapshot restore, one shared pinned KV
  store serving all k GPUs, and stage re-sharding that is a
  host-memory shuffle instead of a network transfer. The
  coordinator is the user's local process; per-shard plans ship as
  JSON, answers return as compact matrices.
- **GPU memory snapshots** (Modal's cuda-checkpoint based
  snapshot/restore) are a first-class part of the design, not an
  optimization bolted on later:
  - The worker image is snapshotted once per model after weight
    load, allocator warmup, and kernel compilation. Session start
    restores the snapshot in seconds instead of a minute-scale
    boot; every container restores from the same per-model
    snapshot, so scale-out costs no extra boots.
  - The single-executor design is what makes the snapshot surface
    small and safe: no engine state, no scheduler queues — just
    weights, the empty KV arena, and compiled
    kernels. Snapshots are taken only in this quiescent state,
    never mid-query.
  - The benchmark's cold protocol (§9.4) restores the snapshot
    before every query, making "cold" mean something exact and
    cheap to measure.
- Every worker run is teed to a file in the results volume, per
  house rule; the coordinator collects logs next to the run report.

The pinned CPU KV store rides in the container, one per container
shared by its GPUs: capacity =
`cpu_memory_gb` minus headroom, keyed by (model, content hash,
document id), holding document-prefix KV only (suffix KV never
exists anywhere). The executor reads it with plain batched H2D
copies — no connector indirection, which is where stock vLLM lost
five sixths of the link (10.2 of 55.4 GB/s). Store dtype matches
the executor's KV dtype, so the phase-2 dtype question from the
earlier draft disappears with the second executor. Eviction is the
length threshold, not LRU (a scanning query thrashes LRU; it
cannot thrash a length cutoff).

---

## 8. Supporting other models: specs in, budgets out

Every planner input derives from two structs. One file per model
under `quail/specs/`; adding a model is adding a file.

```python
@dataclass(frozen=True)
class ModelSpec:
    name: str            # "qwen3-4b-fp8"
    params: float        # P, dense parameter count
    layers: int          # L
    hidden: int          # h
    n_q: int             # query heads
    n_kv: int            # KV heads
    d_head: int          # head dim
    ffn_width: int       # widest projection's output columns
                         # (gate_up: 2 x intermediate at Qwen3)
    w_bytes: float       # weight bytes/param (fp8 = 1)
    kv_bytes: float      # KV bytes/element (bf16 = 2 default)
    # derived properties, never typed in:
    # kappa   = 2 * layers * n_kv * d_head * kv_bytes   (KV bytes/token)
    # W_mem   = params * w_bytes
    # act_per_token ≈ ACT_BYTES_PER_HIDDEN * hidden

@dataclass(frozen=True)
class DeviceSpec:
    name: str            # "h100-sxm"
    mem_bytes: float     # M
    hbm_bw: float        # BW
    peak_flops: float    # R_D at the executor's compute dtype
```

What the planner computes from the structs alone:

| quantity | formula | at 4B/H100 |
|---|---|---|
| tensor parallel | smallest tp with W_mem < M x pool fraction | 1 |
| KV arena tokens (admission budget) | (M x fraction − W_mem − act reservation) / kappa | ~400k at bf16 |
| chunk memory bound | free memory / act_per_token | ~420k |
| kernel index cap | (2^31 − 1) // ffn_width | 110,375 |
| chunk budget | min(memory bound, index cap), floored at the knee | 110,375 |
| compute knee | per-chunk fixed cost / per-token cost; roofline ridge R_D/BW says the same (~400 tokens) | ~300–400 |
| attention crossover | document length where pair work overtakes the dense projections | ~12,300 |
| store break-even | kappa x serving rate | 7.2 GB/s |
| serving rate | 2P/R_D x 1/efficiency | 96–121k tok/s |

The serving rate is the one row that is not pure spec: it needs an
efficiency factor (measured 0.35 on 4B/H100 — 91–93% of peak inside
the GEMMs, the rest lost to the memory-bound work between them).
Default for an uncalibrated model: carry the efficiency over via
spec-ratio scaling (the existing `_scale`), and say so in
`explain()`. An optional calibration overlay per (model, device) —
the batch sweep plus the parity probe, ~10 GPU-minutes — replaces
the assumption with a measurement. The overlay is also where
measured quality constants live, because they have no formula: the
fp8-KV answer-flip count reported next to the dtype pick (§6), and
the answer-accuracy caveats of the checkpoint. For an uncalibrated
model the report says the flip count is unmeasured, and the pick
stays what the pricing says. Plans never require calibration; they
only get sharper predictions from it.

On "make B* as big as possible because everything is prefill
dominated": right in substance, with two caps and one caveat. Right
because answers are constrained to one token — zero decode by
design — and because the measured rate is flat in chunk size past
the knee, so a big chunk budget costs nothing and buys per-chunk
fixed-cost amortization. The caps: activation memory, and the
32-bit kernel indexing bound — which binds far earlier (110,375 vs
~420k at 4B) and is model-dependent through `ffn_width`, which is
why it lives in the spec and is never hardcoded (the join2way crash
is the tuition already paid). The caveat: "prefill dominated"
means the dense projections dominate, which holds while documents
sit far below the attention crossover; the benchmark's document-
length scale factor (§9) deliberately pushes toward that regime,
and the quadratic term in the cost model is what prices it.

---

## 9. The benchmark: QUAIL-B

### 9.1 Scale factors

Two knobs, both explicit in every reported result:

- **SF** scales document counts (linearly, per the table below).
- **LF** scales mean document length: documents are built by
  concatenating source texts to the target length, so LF=4 means
  4x the tokens per document with real text throughout.

Filter cost scales as SF x LF. Join cost scales as SF x LF_partner
per fixed partner list, and as SF^2 where both sides scale — true
pairwise joins are quadratic, and the benchmark does not hide it:
each join query's table row states its pair-count formula. The
reference grid: (SF, LF) in {(0.1, 1), (1, 1), (1, 4)}; SF=0.1 is
the development scale, (1, 4) pushes long documents toward the
attention crossover on purpose.

### 9.2 Document sets (at SF=1, LF=1)

| set | source text | docs | mean tokens | total | role |
|---|---|---|---|---|---|
| `reviews` | IMDB | 50,000 | 400 | 20.0M | the fact set (lineitem) |
| `threads` | stacked IMDB (multi-review threads) | 10,000 | 1,200 | 12.0M | mid-size, mid-length (orders) |
| `reports` | BioDEX patient reports | 2,000 | 3,000 | 6.0M | long documents (the wide table) |
| `products` | ABT-BUY descriptions | 1,000 | 150 | 0.15M | small dimension (part) |
| `terms` | BioDEX reaction terms | 2,560 fixed | 32 | 0.08M | tiny fixed dimension (nation/region — does not scale, like TPC-H's fixed tables) |

~38M corpus tokens at (1, 1); ~150M at (1, 4).

### 9.3 Predicates and provided selectivities

Same instrument discipline as the exploration. Every predicate with
a designed selectivity is a single lookup, because the join
findings showed the 4B checkpoint cannot compare two planted keys
across a long context but reads a single planted line reliably:

- **Filters**: the planted `[FLAGS]` line; filter j asks what flag
  j says. Selectivity = the planted rate.
- **Joins**: the anchor carries `[KEYS] X=<k>`; the partner's key
  is printed in the question tail, so the model checks one value in
  context against one value in the question. Pair selectivity = the
  key collision rate.

Because rates are planted, the benchmark supplies exact
selectivities through the interface — which is the interface
contract (§2.5) exercised as designed, and it makes provided-vs-
observed selectivity a per-query instrument check. Ground truth is
known by construction; the replay check gates every join query.
Timing claims never depend on the model answering correctly.

### 9.4 The queries

Building the query set: take each TPC-H query, strip everything
Quail refuses (aggregation, grouping, ordering, arithmetic, outer
joins), and keep the filter/join skeleton. The 22 queries collapse
into eight skeleton families — filter-only scans (Q1, Q6), plain
2-way joins (Q12, Q14, Q17, Q19), filtered 2-way joins (Q3's core,
Q16), 3-way-and-deeper chains (Q3, Q10, Q18), stars (Q2, Q5, Q8,
Q9, Q11), exists (Q4, Q20), anti (Q16, Q22), exists+anti combined
(Q21) — plus Q13, which is an outer join and has no analogue here.
Fifteen queries cover all eight families with variants that isolate
one engine mechanism each. F = filter stage, J = join stage.

| id | TPC-H skeleton | shape | sets | pair/doc volume at SF=1 | what it isolates | predicted wall (SF=1, LF=1) |
|---|---|---|---|---|---|---|
| B1 | Q6 | 1F | reviews | 50k docs | the degenerate chain; per-query floor and snapshot-restore overhead | ~3.8 min |
| B2 | Q1 | 5F | reviews | 50k docs, sels .9/.9/.9/.8/.8 | the filter-chain regression anchor (the committed 10k result is this at SF=0.2) | ~4.4 min |
| B3 | Q1 + ordering | 5F | reviews | sels .9/.9/.2/.9/.9, selective one written third | run twice: `as_written` vs `by_cost` — the ordering interface measured, not asserted | ~4.2 / ~3.7 min |
| B4 | Q1 on wide rows | 2F | reports | 2k long docs | quadratic surcharge; long-doc admission | ~1.3 min |
| B5 | Q14 | 1J full | reports x terms | 5.12M pairs | the plain 2-way join; anchor orientation (the BioDEX shape, scaled 20x) | ~55 min |
| B6 | Q3 core | 1F + 1J | threads x products | F .05 -> 500 x 1,000 = 0.5M pairs | pushdown: the filter deletes 95% of the pair list | ~19 min |
| B7 | Q19 | 1J full | reviews(5k slice) x products | 5M pairs | pure pair predicate, no gating anywhere (Q19's OR lives inside the one prompt) | ~29 min |
| B8 | Q4 | 1J exists | threads x products | 10k anchors, early-stop streams | exists semantics; expected-scan pricing vs measured | ~7 min |
| B9 | Q21 | 1J exists + 1J anti | threads x products, threads x terms | two early-stop passes | anti semantics; the combined semi/anti family | ~11 min |
| B10 | Q3 | 2J chain | reviews(2k) x threads(2k) x products | stage sels .2/.1; stage-2 pairs must equal survivors x partners exactly | 3-way gating and dedup at scale | ~16 min |
| B11 | Q9 | 2J star | reports x terms, reports x products | anchor KV computed once, read by both stages | star shape; kept-anchor reuse across stages | ~60 min |
| B12 | Q16 | 2F + 1J | threads, products | both sides filtered (.3, .5) before joining | two-sided pushdown; multiplicative pair shrink | ~9 min |
| B13 | Q1 rerun | 5F | reviews, new flags | 50k docs, warm store | cross-query restore for a scan (the persist result inside the engine) | ~2.5 min warm |
| B14 | Q14 rerun | 1J | reports x products, new keys | 2M pairs, warm anchors | cross-query restore for join anchors (prefix share ~19%) | ~13 min cold / ~11 warm |
| B15 | Q21 extended | 2F + 2J (one exists) | threads, reports, terms | the full planner path in one query | everything at once | ~25 min |

Suite totals, predicted (cost-model arithmetic at the measured
rates, stated now per house rule; the planner re-derives them in
`explain()` when it lands): ~4.5 h at (1, 1); ~30 min at (0.1, 1) —
the development scale, where B2 reproduces the committed filter
result and B5 reproduces the committed join result as regression
gates.

### 9.5 Protocol and metrics

Two suite runs, fixed order B1→B15 (the order re-touches `reviews`,
`reports`, `products` so later queries can restore):

- **cold**: snapshot restore before every query, store disabled.
  Each query measured alone, from an exact, cheap-to-reproduce
  state (§7).
- **warm**: one session, store enabled, no restores between
  queries. Per-query walls again, plus the end-to-end wall — the
  number that shows what cross-query KV reuse is worth.

Per query: predicted vs measured wall, fresh tokens, KV bytes
restored, pairs per stage, provided vs observed selectivity, answer
accuracy vs planted truth, replay verdict. Per suite: end-to-end
wall cold and warm, warm/cold per query and total, store hit
tokens.

Baseline, per house rule (the baseline gets the analytically
equivalent configuration, and the submission strategy is named):
stock vLLM, same container class, prefix caching on, no store.
Filters submit separate requests per stage under the same
token-budget admission; joins submit grouped requests per pair in
anchor order with admission matched from the same pool arithmetic.
Both baseline protocols are the committed ones from this repo.
Predictions to beat, stated now: at (1, 1) the join queries carry
the gap (measured 4.1x on the B5 shape), the filter queries are
modest (measured 1.08x for chain semantics, plus whatever the
packed substrate's 121k-vs-97k rate adds); warm-suite gains accrue
almost entirely to Quail, because stock's pool cannot hold 38M
corpus tokens across queries and its per-document connector reads
run at a fifth of the link.

---

## 10. The new repo

```
quail/                        (new repository)
  quail/
    specs/                    # ModelSpec per file + DeviceSpec; qwen3_4b.py first
    catalog.py                # DocumentProvider, Session catalog
    sqlfront/                 # sqlglot subset validator + binder
    logical.py                # the four logical operators
    planner/
      plan.py                 # PhysicalPlan, Refusal
      budgets.py              # spec-derived arena/chunk arithmetic
      cost.py                 # the estimator; calibration overlay loading
    executor/
      pack.py                 # brim packing, admission queue (from joinlogic.py)
      arena.py                # the paged KV arena: pages, free list, per-doc page lists
      attention.py            # varlen self + ragged paged cross + LSE merge
      loop.py                 # the overlapped chunk loop with gate/dedup inline (from run_join)
      kvstore.py              # pinned host store, batched H2D
    runtime/
      session.py              # Session, EngineConfig, explain/run
      coordinator.py          # shard dispatch, stage re-sharding, final merge, projection, reports
      modal_app.py            # worker Cls, snapshots, volumes
    bench/
      quailb.py               # sets, planted predicates, B1-B15, both suites
  baselines/                  # stock vLLM runners (dev dependency only)
  tests/                      # CPU tests; GPU cells are confirmation only
```

Build order, each step landing with CPU tests:

1. **specs + budgets** — the structs and every derived quantity in
   the §8 table, tested against the numbers this repo measured.
2. **executor** — port `pack_stream`, the attention merge, and the
   overlapped loop out of `modal_join_forward.py`; add the paged
   KV arena, continuous admission, and the
   filter-as-degenerate-join stage form.
   **Gate: milestone 1** — the committed 10k-doc five-filter
   workload and the committed 256k-pair join, reproduced on the new
   executor within the cost model's band. This gate is what retires
   the engine fork.
3. **sqlfront + logical + planner** — the parse/bind/validate
   pipeline, ordering policy, pushdown, refusals.
4. **runtime** — Session, coordinator, Modal app, snapshot
   lifecycle, the KV store.
5. **bench** — QUAIL-B; first full grid run publishes the
   prediction-vs-measurement table.

Deferred, named: selectivity estimation by sampling (the interface
already reserves the option object key); runtime re-ordering;
larger models beyond spec-scaled predictions until their ~10-minute
calibration is run; multi-GPU tensor parallel (the spec arithmetic
already computes tp, but nothing above 1 is exercised until a model
needs it).

Risks, named:

- **The unified executor's filter path is unmeasured** end to end;
  milestone 1 exists to measure it before anything is built on top.
- **Continuous admission is new scheduling code.** The engine's
  scheduler solved the same problem (mixed-stage work, admission,
  no starvation) with far more machinery; the packed version is a
  priority rule and a page free list, validated off-GPU against a
  brute-force simulator the way the executor rewrite was — but it
  is new. Milestone 1 gates it.
- **GPU snapshots are young.** If restore proves flaky, the
  fallback is the boring one — cold boots and warm containers —
  and only the cold-protocol convenience is lost.
- **Provided selectivities can be wrong.** By design the planner
  believes them; the report prints provided vs observed so misses
  are visible, and `as_written` is always available.
- **The 27% flag-misread rate** of the 4B checkpoint bounds how
  precisely planted rates control gating; comparisons hold,
  accuracy claims stay off the table.

# Quail engine wiki

This document describes every algorithm and technique in the Quail
engine, for use as a reference when writing the paper. It covers the
full path from a user's query to GPU execution and back.

All file references are relative to `quail/quail/` unless otherwise
noted.

## Module map

The table below lists every module, what it does, and what it
depends on. The data flow is top to bottom: the user calls the
session, which calls the front end, then the planner, then ships a
payload to the worker, which calls the executor.

| Module | What it does | Depends on |
|---|---|---|
| `__init__.py` | Public API surface | builder, catalog, planner, runtime |
| `catalog.py` | Document providers (parquet, HF) and the registry | logical |
| `logical.py` | Logical operators (Scan, Filter, Join, Project) and plan assembly | nothing |
| `builder.py` | Builder API entry point | catalog, logical |
| `sqlfront/compile.py` | AI SQL entry point (sqlglot parser and binder) | catalog, logical |
| `specs/base.py` | ModelSpec and DeviceSpec structs | nothing |
| `specs/qwen3_4b.py`, `specs/h100_sxm.py` | Concrete spec instances | specs/base |
| `planner/budgets.py` | Derived quantities (chunk budget, arena budget, roofline) | specs |
| `planner/calibration.py` | Measured constants (a, a2) and scaling | specs |
| `planner/calibrate.py` | Length-sweep measure of a and a2 | calibration, executor |
| `planner/plan.py` | PhysicalPlan and Refusal structs, EngineConfig | specs |
| `planner/decide.py` | All planner decisions (order, anchor, dtype, sharding) | logical, budgets, calibration, plan |
| `executor/arena.py` | Paged KV arena (PageArena accounting + KVArena tensors) | nothing (torch lazy) |
| `executor/attention.py` | Pipeline: forward pass, attention, Triton kernels | arena (torch, vLLM, triton lazy) |
| `executor/pack.py` | Chunk packing (pack_stream, FilterAdmission) | nothing |
| `executor/loop.py` | Execution loops (run_filter, run_join, warm_kernels) | arena, attention, pack |
| `executor/model.py` | Weight loading through vLLM | nothing (vLLM lazy) |
| `executor/kvstore.py` | Pinned CPU KV store (save, load, extent allocation) | nothing (torch lazy) |
| `runtime/session.py` | Session, Query, tokenization, payload assembly | catalog, logical, planner, sqlfront, builder |
| `runtime/coordinator.py` | Multi-GPU payload splitting and answer merging | nothing |
| `runtime/worker.py` | Modal worker (boot, execute, multi-GPU dispatch) | executor, planner, coordinator |
| `runtime/calibrate.py` | Modal entry for measure, on quail-engine | planner, worker |
| `bench/quailb.py` | QUAIL-B benchmark (data, queries, driver) | runtime |

### Data flow

```
User
  |
  v
Session (runtime/session.py)
  |--- sql() ---> sqlfront/compile.py ---> logical.py (LogicalPlan)
  |--- docs() --> builder.py ------------> logical.py (LogicalPlan)
  |
  v
plan_query (planner/decide.py)
  reads: specs, calibration, budgets
  produces: PhysicalPlan
  |
  v
_payload (runtime/session.py)
  tokenizes documents, builds token-id payload
  |
  v
worker.execute (runtime/worker.py, on Modal GPU)
  boots: model.py -> attention.Pipeline -> arena.KVArena
  runs: loop.run_filter / loop.run_join
  returns: raw answer rows
  |
  v
_assemble (runtime/session.py)
  gates, assembles tuples, projects -> Result
```

## 1. System overview

Quail (QUery-Aware Inference Layer) is a query engine for two
operators over document collections: `AI_FILTER` (does this document
satisfy a true/false predicate?) and `AI_JOIN` (does this tuple of
documents - two or more, all in one prompt - satisfy a true/false
predicate?). The model answers each predicate in a single token
(TRUE or FALSE), constrained at decode time so no autoregressive
generation ever runs.

The current scope is filter queries and joins, Qwen3 4B fp8 weights,
bf16 KV, on one or more H100 GPUs hosted on Modal.

### End-to-end flow

1. The user registers document providers (parquet files or HF
   datasets) with a `Session`.
2. The user writes a query, either as AI SQL or through the builder
   API.
3. The front end compiles the query into a `LogicalPlan` (a tree of
   Scan, SemanticFilter, SemanticJoin, and Project operators).
4. The `Session` tokenizes each scanned column (cached per session)
   and calls the planner with the token counts.
5. The planner produces a `PhysicalPlan`: stage order, anchor
   choices, chunk budget, admission budget, sharding, and a store
   length threshold. KV is always bf16. No wall-time prediction is
   produced.
6. The session builds a payload (token id lists and planned settings)
   and ships it to a Modal worker over RPC.
7. The worker runs the packed executor on the GPU: filter chains,
   exists/anti gates, and the one full join (a single cross-product
   stage, however many tables it spans).
8. The worker returns raw answer rows. The session assembles output
   tuples from the full join's TRUE rows (each member checked against
   its table's final survivor set), applies the projection, and
   returns a `Result`.

## 2. Query compilation

Both entry points (AI SQL and the builder) produce the same
intermediate representation: a `QueryDesc` containing the tables,
document columns, filter predicates, join specs, and projection
columns. The `assemble_plan` function builds a `LogicalPlan` tree
from the description, so a query written in SQL and the same query
written with the builder produce identical plans.

### The logical operators

There are four operators, defined in `logical.py`:

- **Scan**: reads one column of one registered provider (e.g.,
  `reviews.body`).
- **SemanticFilter**: a conjunction of true/false predicates over a
  single scanned column. Each predicate has a prompt template, column
  references, and an optional selectivity (the fraction of documents
  expected to pass).
- **SemanticJoin**: one true/false predicate over a whole tuple of
  documents, one per table - the cross product of its tables
  filtered by a single prompt that holds every document at once
  (the BigQuery/Snowflake AI-join shape; never a chain of pairwise
  stages). A query has at most one `full` join, plus any number of
  gates:
  - `full`: produce every matching tuple.
  - `exists`: keep outer documents that match at least one inner
    document (a semi-join; two tables).
  - `anti`: keep outer documents that match no inner document (an
    anti-join; two tables).
- **Project**: column selection at the root. No computed columns.

### Prompt layout

Every operator's prompt starts from the same anchor block. A filter
runs as:

```
[SHARED_PRE] [document] [frame] [question suffix]
```

and a join renders each tuple as the anchor block, its naming line,
then one labeled block per partner and the question:

```
[SHARED_PRE] [anchor document] [naming line: (The document above is {0}.)]
[DOCUMENT {1}:] [partner document] ... [instruction + question with {0},{1},... markers kept]
```

`SHARED_PRE` is `"DOCUMENT:\n"`, defined once in `logical.py`. It is
engine-owned and identical for every operator, every query, and
every document. Because it is one fixed string, the KV of
`[SHARED_PRE + document]` is the same wherever that document
appears. The KV store keys extents by (content hash, document
index) and folds `SHARED_PRE` into the hash, so a filter-scanned
document can be restored as a join anchor and the other way around.
Changing the preamble text invalidates every stored extent.

The wording is a formatting label, not an instruction. Measured on
B1 (short reviews): "Evaluate whether the following is true or
false." dropped observed selectivity from 0.398 to 0.074, and "You
will read a document and answer a yes-or-no question about it."
dropped it to 0.0014. `"DOCUMENT:\n"` left it at 0.397. Long
documents (B4) were immune to all three. Task text therefore lives
after the document, in the frame and the suffix.

A filter's `PROMPT('template {0} ...', col)` is canonicalized at
bind time (`split_frame` in `logical.py`). Any user text before the
first placeholder is stripped out and becomes the frame. The
canonical template is `SHARED_PRE`, then `{0}` (the document the
engine owns), then the frame, then the rest. The stored KV extent
is document-only: `[SHARED_PRE + document]`. The frame is not
stored.

A filter carries the frame at the head of each stage's question
suffix. A join prompt is bound differently (`bind_join_prompt`):
the template is never inlined - each tuple renders as labeled
document blocks (the anchor first, under the bare `SHARED_PRE`
label) followed by the template as the question, appended verbatim
with its `{0}`, `{1}`, ... placeholder markers kept as written.
Each partner block is labeled with its own marker
(`DOCUMENT {1}:`, `DOCUMENT {2}:`, etc.), so a marker in the
question resolves to its block. The anchor's naming line -
"(The document above is {0}.)" - is written into its kept KV
once per anchor (`write_suffix_tokens`, the same mechanism the
old frame used), mapping the top block to its marker so the
question can reference it without paying those tokens per tuple.
Partner block labels and the question ride in every tuple's
suffix.

The planner prices the preamble once per document, the naming line
once per join anchor, and labels + question once per tuple. See
`logical.py` (`SHARED_PRE`, `split_frame`, `bind_prompt`,
`bind_join_prompt`) and the suite writeup
`2026-08-19-shared-preamble-join-store.md`.

### AI SQL front end

The SQL front end (`sqlfront/compile.py`) parses AI SQL using sqlglot
in the Snowflake dialect. `AI_FILTER(PROMPT(...))` appears in WHERE
conjuncts; join predicates appear in `JOIN ... ON` clauses; `EXISTS`
and `NOT EXISTS` subqueries map to exists and anti semantics.

The front end rejects every relational operator except projection
and LIMIT: GROUP BY, ORDER BY, DISTINCT, HAVING, UNION, INTERSECT,
EXCEPT, window functions, OR between AI predicates, and subqueries
other than the EXISTS form. LIMIT N stops the filter loop once N
survivors are found (early termination); the builder equivalent is
`.limit(n)` before `.select()`. The rejection list is explicit
(`compile.py:22-33`), so new SQL surface cannot enter silently.

### Builder API

The builder (`builder.py`) mirrors the SQL constructs: `docs()`,
`.alias()`, `.ai_filter()`, `.ai_join()`, `.limit()`, `.select()`. The builder
always uses `as_written` order (the chain order is the execution
order). Both entry points collect the same `QueryDesc` and call the
same `assemble_plan`, so the plans are structurally identical.

### Key functions: query compilation

| Function | File | What it does |
|---|---|---|
| `compile_sql` | `sqlfront/compile.py:212` | AI SQL text -> LogicalPlan |
| `assemble_plan` | `logical.py:117` | QueryDesc -> LogicalPlan tree |
| `bind_prompt` | `logical.py:207` | Canonicalize, bind column refs, count tokens |
| `split_template` | `logical.py:168` | Split at the first placeholder into preamble and tail |
| `split_frame` | `logical.py:177` | Relocate user pre-document text; emit canonical template |
| `canonicalize_template` | `logical.py:203` | `split_frame` without returning the frame |
| `DocumentProvider.from_parquet` | `catalog.py:29` | Register a parquet file (metadata only) |
| `DocumentProvider.read_column` | `catalog.py:53` | Read one column's ids and texts (at scan time) |

### Key functions: session and runtime

| Function | File | What it does |
|---|---|---|
| `Session.sql` | `session.py:170` | Compile AI SQL and return a runnable Query |
| `Session.docs` | `session.py:175` | Start the builder API |
| `Session.scan` | `session.py:210` | Tokenize a column (cached per session) |
| `Query.plan` | `session.py:310` | Run the planner (cached per Query) |
| `Query.run` | `session.py:335` | Plan, execute on Modal, replay-check, project |
| `Query.explain` | `session.py:330` | Print the logical tree and physical plan |

### Pushdown

Pushdown is unconditional, not a decision. Filters attach directly
above their scans in the logical tree, so a filter always runs before
the joins its provider feeds. There is no cost-based pushdown
decision and no runtime re-ordering of filters vs joins.

## 3. Physical planning

The planner (`planner/decide.py`) takes a logical plan and per-alias
token counts and produces a `PhysicalPlan` or a `Refusal`. A refusal
is a named constraint violation (e.g., "this document is too long for
the chunk budget") rather than a degraded execution.

### Decisions from token arithmetic alone

These decisions compare token counts and selectivities. Because they
compare things running at the same rate, the serving rate cancels out,
so they survive any miscalibration.

**Filter order** (`decide.py:81`): when every filter carries a
selectivity, `by_cost` sorts by cost per killed document. The cost of
a filter stage is its question tokens divided by the fraction of
documents it kills (1 minus selectivity). A selectivity of 1 (kills
nothing) goes last. When any filter lacks a selectivity, `as_written`
is used.

**Join order** (`order_joins` in `decide.py`): when several join
specs exist (exists/anti gates plus at most one full join) and all
carry selectivities, the planner enumerates the permutations
(typically 2 to 4 specs, so the enumeration is small) and picks the
one with the smallest survivor-thinned tuple-token total. Every
spec is self-contained (the cross product over its own tables), so
every order runs; order changes cost, never results. The survivor
thinning uses `n * (1 - (1-s)^partner_tuples)` for the expected
documents in some passing tuple (`_surviving_docs`).

**Anchor selection** (`choose_anchor` in `decide.py`): for each
join, the planner picks the table whose tuple-token total is
smallest when the other tables stream. In practice, the side with
the most document tokens anchors (anchor tokens are paid once per
document, partner tokens once per tuple). A user override wins,
with a remark when it prices worse; an exists/anti gate always
anchors on the outer table, because the gate applies to it.

**Sharding** (`balanced_shards` in `decide.py`): greedy balance by
token count across workers. Filters split documents; joins split
anchor documents (every tuple belongs to exactly one anchor, so
gating and each anchor's tuple stream stay local to the GPU holding
the anchor).

### Pseudocode: filter order decision

```
for each filter predicate p:
    killed = 1 - p.selectivity
    if killed <= 0:
        cost = infinity
    else:
        cost = p.question_tokens / killed
sort predicates by cost (stable, so ties keep written order)
```

### Pseudocode: anchor selection

```
for each table of the join:
    compute tuple_tokens(table as anchor) =
        n_anchor_docs                                (prefix + naming
        * (mean_anchor_tokens + pre + note_tokens)    line, once each)
      + product of every table's n_docs              (number of tuples)
        * (sum over partners of
             (label_tokens + mean_partner_tokens)
           + question_tokens)                        (suffix, once each)
pick the table with the fewest tuple_tokens as anchor
```

### Settings from the spec structs

The spec structs (`specs/base.py`) hold model and device parameters.
All derived quantities are in `planner/budgets.py`:

**Chunk budget** (`budgets.py:62`): the number of tokens per forward
pass (the batch size). It is the minimum of two bounds:
- Memory bound: `(M * fraction - weights) / act_bytes_per_token`,
  divided by a slack factor of 2.
- Kernel index cap: `INT32_MAX / ffn_width`, because the fused
  kernels compute element offsets in 32-bit integers.
At Qwen3 4B on H100, the index cap binds at 110,376 tokens.

**Admission budget** (`budgets.py:69`): the number of document tokens
that can be resident in the KV arena at once. It is what remains of
device memory after weights and activation reservation, divided by
kappa (the KV bytes per cached token: `2 * layers * n_kv * d_head *
kv_bytes`). At Qwen3 4B bf16 KV on H100, this is about 346,000
tokens. The admission budget is a token count, never a document
count, because a document count cannot account for varying document
lengths.

**Compute knee** (`budgets.py:95`): the chunk size where the dense
projections cross the roofline ridge and become compute-bound rather
than memory-bound. About 416 tokens at 4B/H100. Chunks below this
size waste hardware; the chunk budget is floored at the knee.

**Attention crossover** (`budgets.py:138`): the document length where
attention pair work overtakes the dense projections at the same chunk
size. About 12,200 tokens at 4B/H100. Below this, the chunk is a
GEMM problem (we say "prefill dominated"); above it, the quadratic
attention term dominates.

### Decisions that use calibration constants

One decision compares compute cost against byte-transfer cost and
therefore needs measured constants.

**Access per scan** (`decide.py:314`): `read` (compute the
document's KV from scratch) or `restore` (load KV from the pinned
host store). Restore wins when the store's bandwidth beats
`kappa / a` (the serving rate expressed as KV bytes per second), or
when the corpus's documents are past the a2-refined crossover.

The crossover is: solve `kappa / bw = a + a2 * h` for `h`, where
`h` is document length in tokens. At the measured pinned-memory
bandwidth of 55 GB/s, restore wins at every document length for the
4B model.

KV is always bf16. The planner does not choose a KV dtype and does
not model a conversion tax.

## 4. Calibration

The planner's restore break-even uses two measured constants per
(model, device) pair, stored as JSON in `calibration/`:

| Constant | Meaning |
|---|---|
| `a` (s/token) | Wall seconds per fresh token in the packed loop, efficiency included |
| `a2` (s/token^2) | The quadratic attention coefficient; bends the restore break-even for long documents |

A model/device pair without a calibration file gets defaults scaled
from the anchor measurement (Qwen3 4B on H100) using spec ratios: a
model with more parameters costs proportionally more per token, a
device with a higher FLOP ceiling costs proportionally less
(`calibration.py:70-77`).

`quail.planner.calibrate.measure` (Modal entry:
`quail/runtime/calibrate.py`) sweeps document length through the
packed filter and fits `t(h) = a + a2 * h` by least squares. It
also probes the host copy channels (pinned and unpinned, both
directions) for the store break-even.

Additionally, `calibration/channels.json` stores host-memory
bandwidth measurements (pinned device-to-host, host-to-device, etc.)
that the store break-even arithmetic consumes.

## 5. The packed executor

The packed executor is Quail's core contribution. Instead of sending
one request per document or per pair through a serving engine (the
way stock vLLM works), Quail packs multiple documents or pairs into a
single forward pass, sharing KV across them through a paged arena.

### Key functions: planner

| Function | File | What it does |
|---|---|---|
| `plan_query` | `decide.py` | Top-level: logical plan + token counts -> physical plan or refusal |
| `order_filters_indexed` | `decide.py` | Sort filter stages by cost-per-killed-document |
| `order_joins` | `decide.py` | Order the join specs (gates + the one full join) by total tuple tokens |
| `choose_anchor` | `decide.py` | Pick the cheapest anchor table for a join |
| `balanced_shards` | `decide.py` | Greedy-balance documents across workers by token count |
| `access_for_scan` | `decide.py` | Decide read vs restore for a scanned corpus |
| `store_length_threshold` | `decide.py` | Length cutoff for which documents to store |
| `chunk_budget` | `budgets.py` | Tokens per forward pass (min of memory and kernel bounds) |
| `arena_tokens` | `budgets.py` | KV residency budget (device memory minus weights and activations) |
| `load_calibration` | `calibration.py` | Load or spec-scale the calibration constants |
| `measure` | `calibrate.py` | Length sweep + affine fit of a, a2 (GPU) |
| `commit_calibration` | `calibration.py` | Write the two constants to the pair file |

### 5.1 Chunk packing

There are two packing strategies, one for each query shape:

**For joins: `pack_stream`** (`pack.py:57`). Given a list of anchors
with their suffix lists (one suffix per partner document), brim-pack
them into chunks. Each chunk is a list of groups; each group is one
anchor with a contiguous slice of its partner suffixes. An anchor's
prefix is packed at most once: if the budget cuts an anchor's stream
mid-chunk, the anchor continues in the next chunk reading its prefix
KV from the arena instead of recomputing it. The function returns the
chunks and the set of anchors whose KV must be written to the arena
(because their stream was cut or a later stage needs them).

**For filters: `FilterAdmission`** (`pack.py:184`). Continuous
admission with two bins: tokens (the chunk budget) and pages (the
arena). Each chunk fills from two sources in priority order:
1. Survivor suffixes (documents that passed the previous stage and
   need their next question). Survivors pack first so that residency
   drains monotonically.
2. Fresh admissions, in queue order, whenever their page-rounded
   tokens fit the free list.

Pages are claimed at admission and returned immediately when a
document fails a stage or answers its last one. A document that
cannot fit even an empty arena or an empty chunk is refused at plan
time, so there is no deadlock.

### Pseudocode: FilterAdmission scheduling

```
while not done:
    room = chunk_budget
    groups = []

    # priority 1: survivors from the previous stage
    for each (doc, next_stage) in the ready queue (oldest first):
        cost = stage_question_tokens[next_stage]
        if cost > room: break
        remove from ready queue
        add (doc, next_stage, fresh=false) to groups
        room -= cost

    # priority 2: fresh admissions
    for each doc in the pending queue (FIFO):
        pages_needed = ceil(doc_tokens / page_size)
        if pages_needed > free_pages:
            stop claiming pages (queue-order guarantee)
            break
        cost = doc_tokens + stage_question_tokens[0]
        if cost > room:
            skip this doc (try next, chunk-room only)
            continue
        claim pages, add (doc, 0, fresh=true) to groups
        room -= cost

    return groups
```

When answers arrive:
```
for each (doc, stage, answer):
    if answer = FALSE or stage is the last stage:
        free the document's pages
    else:
        add (doc, stage+1) to the ready queue
```

### Pseudocode: pack_stream (join packing)

```
for each anchor a in order:
    placed = (a's KV is already in the arena)
    for each group of suffixes that fits the chunk budget:
        if not placed:
            include a's prefix tokens in the group
            placed = true
        else:
            group reads a's KV from arena (no prefix tokens)
        add the group to the current chunk
        if chunk is full:
            emit chunk, start a new one
    if a's stream was cut (suffixes remain):
        mark a's KV for arena write
    if a later stage needs a:
        mark a's KV for arena write
```

### 5.2 The paged KV arena

The KV arena (`executor/arena.py`) is a preallocated buffer on the
GPU, sized to the admission budget, divided into fixed 16-token pages
with a free list.

**PageArena** (CPU, `arena.py:16`): the accounting. A free list
(a stack of page ids) and per-document page ownership. Allocation
pops pages from the free list; freeing pushes them back. Because the
admission scheduler never lets page-rounded resident tokens exceed
the arena, allocation cannot fail at runtime.

**KVArena** (GPU, `arena.py:71`): the tensor backing. Per-layer K
and V pools of shape `(n_pages * page_tokens, n_kv, d_head)`. A
document's tokens are scattered across its pages (non-contiguous in
the pool), and the paged attention kernels read them through a block
table (a 2D int32 tensor mapping `(document, page_index)` to
physical page id).

Key operations:
- `alloc(key, tokens)`: claim pages for a document.
- `free_key(key)`: return pages instantly.
- `block_table(keys)`: build the block table for a set of
  documents, used by the paged cross-attention kernel.
- `paged_kv(layer)`: reshape the flat pool into `(n_pages,
  page_tokens, n_kv, d_head)` for FlashAttention's block-table
  input.

### Key functions: arena

| Function | File | What it does |
|---|---|---|
| `PageArena.alloc` | `arena.py:29` | Claim pages for a document from the free list |
| `PageArena.free_key` | `arena.py:42` | Return a document's pages to the free list |
| `PageArena.row_indices` | `arena.py:49` | Flat row positions of a document's tokens in the pool |
| `KVArena.alloc` | `arena.py:97` | Claim pages and record the row indices (host-side; the device copy is built lazily) |
| `KVArena.rows_gpu` | `arena.py:110` | The document's row indices on device, cached per residency |
| `KVArena.block_table` | `arena.py:125` | Build the block table for paged attention (flat on the host, one staged copy) |
| `KVArena.paged_kv` | `arena.py:118` | Reshape the flat pool for FlashAttention's block input |
| `KVArena.gather` | `arena.py:149` | Extract a document's contiguous K, V rows (fallback path) |

### 5.3 The attention paths and the workload assignment

Each chunk is a sequence of groups: `[group_1 | group_2 | ...]`,
where each group is `[prefix? | suffix_1 ... suffix_k]`. The prefix
is the anchor document's tokens (present only if this is the first
time the anchor is packed). The suffixes are the partner documents
(for joins) or the question texts (for filters).

The Pipeline has three attention implementations, selected per
workload by `attention_mode` (issue #24):

- **`unified`** - one causal FlashAttention-3 paged call per layer.
  Current KV (prefix and suffix) is scattered into the document's
  arena pages first (the arena reserves capacity pages for the
  suffix beyond the document's logical length), then a single
  `causal=True` call with a block table covers retained plus current
  KV. No second call, no merge. Requires exactly one suffix per
  group, which the filter shape always satisfies.
- **`merge_quant`** - the two-call pattern below, with the LSE merge
  and the FP8 quantization for o_proj fused into one Triton kernel
  (`merge_attn_quant`), skipping the intermediate BF16 tensor.
- **`split`** - the two-call pattern with the merge as ~9 PyTorch
  kernels and a separate FP8 quantization. The reference
  implementation; also the only mode with the gather fallback
  (`meta["paged"]=False`) if the paged kernel ever fails a parity
  gate.

The assignment, fixed in `attention.py` as `FILTER_ATTENTION =
"unified"` and `JOIN_ATTENTION = "merge_quant"` and read by the
worker:

- **Filters run `unified`.** Fastest measured on the 10k-document
  five-filter workload (8.45 us/token vs 8.68 merge_quant and 8.99
  split on the TRUE/FALSE corpus; `results/attention_paths.json` -
  where all three paths return identical, 100%-correct answers on
  all 40,052 planted-flag questions), and bit-identical to a
  contiguous causal FlashAttention call
  (`results/attention_parity.json`, max_abs 0.0), so filter answers
  match full-prompt recompute exactly
  (`results/attention_end_to_end_parity.json`).
- **Joins run `merge_quant`.** The unified path cannot share one
  anchor's KV across the many partner suffixes of a chunk - each
  pair needs its own causal view, so one causal call per pair would
  read other pairs' scattered KV. Forcing correctness (one partner
  per anchor per chunk) re-reads the anchor's KV once per pair and
  is several-fold slower (`results/join_attention_paths_*.json`).
  Between the two two-call modes, the fused merge wins (11.28 vs
  11.86 us/token at 10x256).
- **Every path is validated against stock vLLM on real queries**
  (`results/accuracy_vs_stock.json`): identical token streams
  answered by standard vLLM serving and by the packed executor.
  Planted-flag accuracy is 100% for every path; disagreements with
  stock are 0.22% on filters and confined to near-zero
  TRUE/FALSE-logprob margins (unified has none above 0.875, against
  a corpus median margin of 3.25); on joins, zero disagreements at
  decisive margins. The residual is the kernel stack (fp8 GEMMs,
  fused norms), not the attention path.
- **FlashInfer (0.6.14, in the image) was evaluated and not
  adopted**: its paged causal kernel is 27% slower than the FA3
  unified call on the fresh filter chunk (2.4x on the cached
  shape), its two-call-plus-merge stack is ~35% slower than
  merge_quant on joins, and its cascade wrapper (the shared-prefix
  decomposition, single-anchor shapes only) is 21% slower. Nothing
  came within the 5% adoption threshold
  (`results/flashinfer_bench.json`).

The two-call pattern (`split` and `merge_quant`) runs per layer as
follows (`attention.py`):

**Call A** (self-attention): causal attention over the segment
boundaries. Each prefix attends to itself; each suffix attends to
itself. This is a standard FlashAttention-3 varlen call with
cumulative sequence lengths (`cu_seqlens`).

**Call B** (cross-attention): every suffix token attends to its
group's kept context in the arena. The kept context is the anchor's
KV, stored in the arena's pages. Call B uses FlashAttention-3's paged
attention variant, reading KV through the block table. Call B is
non-causal (the suffix needs to see the full prefix, not just
earlier tokens).

**Merge**: the two calls produce partial outputs and log-sum-exp
(LSE) values. The merge combines them using the identity:

    (wa * A + wb * B) / (wa + wb) = A + (B - A) * sigmoid(lse_B - lse_A)

This avoids materializing fp32 copies of the full output tensors.
See `attention.py:339-342`.

### Pseudocode: two-call attention with LSE merge

```
for each layer:
    # Step 1: scatter fresh KV into arena pages
    gather source rows from the chunk's K, V tensors (all fresh groups)
    scatter into destination rows in the arena's K, V pools
    (one gather + one scatter per K and V pool = 4 kernel launches)

    # Step 2: call A (self-attention, causal)
    out_a, lse_a = flash_attn_varlen(
        q, k, v,
        cu_seqlens = segment boundaries (each prefix, each suffix),
        causal = true)

    # Step 3: call B (cross-attention against arena pages, non-causal)
    # only for suffix tokens that have kept context
    q_suffix = q[suffix_rows]
    k_paged, v_paged = arena.paged_kv(layer)
    out_b, lse_b = flash_attn_varlen(
        q_suffix, k_paged, v_paged,
        block_table = arena.block_table(group_keys),
        seqused_k = kept_context_lengths,
        causal = false)

    # Step 4: merge by softmax state
    weight = sigmoid(lse_b - lse_a[suffix_rows])
    merged = lerp(out_a[suffix_rows], out_b, weight)
    out_a[suffix_rows] = merged
```

**KV scatter**: fresh prefix KV is written into the arena's pages
during call A's layer pass, so call B can read it in the same layer.
After the batched KV-write fix, all of a chunk's writes are
concatenated into one gather and one scatter per layer (4 kernel
launches per layer instead of the previous ~20,000).

### 5.4 The forward pass

`Pipeline.forward_chunk` (`attention.py:348`) runs the full
transformer forward pass for one chunk:

Per layer:
1. Fused RMSNorm + residual add + fp8 quantization (custom Triton
   kernel `add_rms_norm_quant`, `attention.py:135`).
2. QKV projection: DeepGEMM fp8 GEMM.
3. Fused QK-norm + RoPE (custom Triton kernel `qk_norm_rope`,
   `attention.py:170`).
4. Two-call attention (call A + call B + merge).
5. Output projection: DeepGEMM fp8 GEMM.
6. Fused RMSNorm + residual add + fp8 quantization.
7. Gate-up projection: DeepGEMM fp8 GEMM.
8. Fused SiLU + mul + fp8 quantization (custom Triton kernel
   `silu_mul_quant`, `attention.py:112`).
9. Down projection: DeepGEMM fp8 GEMM.

After the last layer, only the final-position hidden states (one per
suffix, at the last token of each suffix) are extracted and
RMS-normalized. These are the inputs to the answer readout.

### 5.5 Answer readout

The `Answerer` (`loop.py:60`) scores the final hidden states against
only the TRUE and FALSE token embeddings (not the full vocabulary). It
projects the normed hidden state through a sub-selected `lm_head`
weight matrix (only the rows for TRUE/FALSE token ids), takes the argmax
within the TRUE set and within the FALSE set, and compares.

`AsyncAnswers` (`loop.py:92`) makes the readout non-blocking: it
computes the answer bits on GPU, copies them to pinned host memory
with a non-blocking copy, and records a CUDA event. The next chunk's
forward pass can begin while the CPU waits on the event to read the
answers. This overlaps GPU compute with answer readback.

### Key functions: executor

| Function | File | What it does |
|---|---|---|
| `Pipeline.forward_chunk` | `attention.py:348` | Full transformer forward pass for one chunk |
| `Pipeline.attention` | `attention.py:284` | Two-call attention with LSE merge for one layer (`split`) |
| `Pipeline.attention_merge_quant` | `attention.py` | Two calls + fused merge/FP8-quant Triton kernel (`merge_quant`, the join path) |
| `Pipeline.attention_unified` | `attention.py` | Scatter current KV, one causal paged call (`unified`, the filter path) |
| `Pipeline.gemm` | `attention.py:68` | DeepGEMM fp8 matrix multiply |
| `Pipeline.quant` | `attention.py:78` | Per-token-group fp8 quantization |
| `Pipeline.custom_silu_quant` | `attention.py:223` | Fused SiLU + multiply + fp8 quant (Triton) |
| `Pipeline.custom_norm_quant` | `attention.py:235` | Fused residual-add + RMSNorm + fp8 quant (Triton) |
| `Pipeline.custom_qk_norm_rope` | `attention.py:248` | Fused QK-norm + RoPE (Triton) |
| `pack_chunk` | `loop.py:125` | Build GPU tensors for one chunk from group specs (all index tensors staged through pinned memory) |
| `pack_stream` | `pack.py:57` | Brim-pack the join's tuple list into chunks (join path) |
| `FilterAdmission` | `pack.py:184` | Continuous admission scheduler (filter path) |
| `run_filter` | `loop.py:462` | The filter chain execution loop |
| `run_join` | `loop.py:241` | The join execution loop (one cross-product stage per join; multi-stage gating stays available to GPU cells) |
| `warm_kernels` | `loop.py:387` | Pre-compile all DeepGEMM and Triton kernel configs |
| `Answerer` | `loop.py:60` | TRUE/FALSE scoring from final hidden states |
| `AsyncAnswers` | `loop.py:92` | Non-blocking answer readout with pinned-memory copy |

### 5.6 The overlapped execution loop

There are two loop drivers, one for each query shape:

**`run_filter`** (`loop.py:437`): the filter chain. A while loop
that runs until `FilterAdmission.done()`:
1. Build a chunk from the scheduler's `next_chunk()`.
2. Allocate arena pages for fresh documents. If a document is in the
   store, load its KV asynchronously.
3. Pack the chunk (`pack_chunk`), run the forward pass, submit the
   answers asynchronously.
4. While the GPU runs the current chunk, read the previous chunk's
   answers and gate: documents that answered NO have their pages
   freed immediately; survivors advance to their next stage.
5. Documents leaving their last stage (or failing) are saved to the
   KV store if they meet the length threshold.

**`run_join`** (`loop.py:216`): the join driver. The pair list is
pre-planned by `pack_stream`, then chunks are launched in order.
Between stages, answers are gated: anchors with no surviving pairs
are dropped, and their pages are freed. The driver also prefetches
the next group's stage-0 chunk while waiting on the current group's
gate (which cannot be planned past until answers arrive), keeping the
GPU fed across gate boundaries. Anchors already in the KV store
restore instead of recomputing. Anchors leaving their last stage
are saved if they meet the length threshold. A per-stage frame, if
present, is written into the anchor's kept KV once after the
document rows.

### 5.7 KV rewind (chain mode)

In a multi-stage filter chain, each stage asks a different question
about the same document. KV rewind means the document's KV is
computed once (at stage 1) and stays in the arena for all subsequent
stages. Later stages add only the question suffix's tokens to the
chunk; the document's KV is read from the arena through
cross-attention (call B).

The implementation detail: the shared question preamble (the longest
common token prefix across all stage questions) is written into the
arena alongside the document's KV after stage 1. Later stages'
question suffixes start after the preamble, so their position
embeddings are correct. See `loop.py:425` (`_shared_preamble_tokens`)
and the `write_suffix_tokens` field in `pack_chunk`.

### Pseudocode: the filter execution loop

```
build scheduler with document lengths, stage question lengths,
    chunk budget, arena page count

while scheduler is not done:
    groups = scheduler.next_chunk()
    if no groups:
        wait for the oldest in-flight chunk's answers
        gate those answers (free pages for FALSE, enqueue next stage for TRUE)
        continue

    for each fresh document in groups:
        allocate arena pages
        if document is in the store:
            start async KV load from host memory

    pack the chunk (build GPU tensors from group specs)
    wait for any pending KV loads to finish
    record start event
    run forward pass (pipeline.forward_chunk)
    record end event
    submit answer readout (async, non-blocking)

    # overlap: read the PREVIOUS chunk's answers while GPU runs
    while more than one chunk is in flight:
        wait for the oldest chunk's answer event
        gate those answers
        for documents leaving (NO or last stage):
            if store enabled and document is long enough:
                start async save to host memory (pages held until done)
            else:
                free pages immediately

drain remaining in-flight chunks
wait for pending store saves
```

### Pseudocode: KV rewind across filter stages

```
compute shared_preamble = longest common token prefix across all
    stage questions

at stage 1 (fresh admission):
    pack: [document_tokens | full_question_1_tokens]
    arena pages hold: document tokens + shared_preamble tokens
    (the shared_preamble portion of the question KV is written
     into the arena alongside the document's KV)

at stage j > 1 (survivor suffix):
    pack: [question_j_suffix_tokens]   (just the new part)
    suffix positions start at (document_length + preamble_length)
    call B reads the full kept context from arena:
        document KV + shared_preamble KV
    no document tokens are recomputed
```

## 6. Joins

### Packed joins

In a packed join, the anchor side's KV is computed once and shared
across all pairs in the same chunk. Multiple partner suffixes attend
to one anchor's KV pages through the paged cross-attention call.
Compare this with stock vLLM, where each pair is a separate request:
even with prefix caching, the engine re-reads the anchor's KV for
every pair, and pays per-request scheduling overhead.

The packing works through `pack_stream` (`pack.py:57`): given
anchors and their suffix lists, brim-pack into chunks. An anchor
whose stream is cut mid-chunk has its KV written to the arena; the
continuation chunk reads the KV from the arena instead of
recomputing it.

### Anchor KV sharing

The anchor is the side chosen (by token-count comparison) to have
its KV stay resident. In a chunk, one anchor's prefix appears once,
and all of its partner suffixes attend to that same set of arena
pages. The number of partners per chunk is limited by the chunk
budget and the arena; at 4B/H100, about 30 to 50 suffixes can share
one anchor's KV in a single forward pass.

### Gating

Between join stages, gating drops anchors that had no surviving
pairs. `gate()` (`pack.py:132`) returns anchor indices where any
answer was TRUE. Dropped anchors' pages are freed immediately.

### Dedup

In a chain join (e.g., A-B-C with B as anchor), an anchor B that
matched multiple A partners in stage 1 still appears only once in
stage 2's pair list with each C partner. The dedup is structural:
the `run_join` driver gates on anchors, not on individual pairs, so
the unique anchor set is what enters the next stage.

### Replay

In a chain or star join, the anchor's KV from stage 1 is reused in
stage 2. The `pack_stream` `keep` parameter tells the packer which
anchors a later stage needs; their KV stays in the arena across the
stage boundary. The `already_kept` parameter tells the packer which
anchors' KV is already resident, so their groups do not pack fresh
prefix tokens.

### Pseudocode: the n-way join as one cross-product stage

```
# the join: anchor B, partners A and C, one prompt per tuple
tuples = cross product of surviving A indices x surviving C indices
suffixes = for each tuple:
    label_A + doc_A + label_C + doc_C + question
plan = pack_stream(anchors, suffixes, budget)
for each chunk in plan:
    # each anchor's first group writes the naming line into its
    # kept KV (write_suffix_tokens), then its tuples stream
    build, launch, collect answers

# an exists/anti gate is the two-table case of the same stage,
# with the keep rule applied to the anchor's answers

# assemble output: for each anchor B with TRUE rows,
#   for each TRUE tuple (A, C) whose members survive their tables'
#   final gates, emit (B, A, C)
```

## 7. The KV store

The KV store (`executor/kvstore.py`) is a pinned CPU memory pool that
saves document KV across queries within a session. When a document's
KV is in the store and the planner says `access=restore`, the
document's KV is loaded from host memory into the arena instead of
being recomputed.

### Layout

The store is a list of 8 GiB pinned slabs (a single large
`cudaHostAlloc` failed at 280 GB; chunked slabs pin fine). Each slab
is a flat tensor of token rows, where each row is every layer's K and
V for one token (row width = `n_layers * 2 * n_kv * d_head`). A
document is a contiguous extent of rows within one slab.

### Save and load protocol

**Save** (`kvstore.py`): runs on a CUDA side stream after the
compute event that wrote the document's arena pages. For each layer,
it gathers the document's K and V rows from the arena into a GPU
staging buffer, then copies the staging buffer to the pinned host
slab. The event the caller waits on fires after the gather, so
arena pages can be freed while the host copy continues. Staging
buffers grow on demand; large documents get 2 slots so the buffers
do not claim tens of gigabytes next to the arena.

**Load** (`kvstore.py:205`): copies from the host slab to the GPU
staging buffer, then scatters into the document's arena pages. The
calling chunk must wait on the load's completion event before
launching its forward pass.

### Extent allocation

`ExtentAllocator` (`kvstore.py:61`): first-fit contiguous allocation
with free-list coalescing. When the pool is full, `alloc_with_reclaim`
evicts other datasets' extents (shortest documents first, because
they have the least recompute value), but never evicts the running
query's own dataset (that is where LRU-style thrashing comes from).

### Length threshold

The planner decides which documents to store using a length
threshold (`decide.py:222`): keep the longest documents whose KV
fits the capacity. Recompute cost per stored byte rises with document
length, so the longest documents are worth the most per byte. A
length threshold cannot be thrashed by a scanning query the way LRU
can.

### Break-even

The store break-even bandwidth (`budgets.py:160`) is `kappa / a`:
the KV bytes per token divided by the serving rate. At 4B/H100, this
is 7 to 9 GB/s. Pinned host memory achieves 55 GB/s (measured), so
the store clears the break-even. Disk and volumes do not.

### Pseudocode: store save and load

```
# save (runs on side CUDA stream, after the forward's compute event):
for each layer:
    gather document's K rows from arena pages into staging buffer
    gather document's V rows from arena pages into staging buffer
copy staging buffer to pinned host slab (non-blocking)
record completion event
# caller frees arena pages after the gather; the host copy continues

# load (runs on side CUDA stream, before the chunk's forward):
copy from pinned host slab to staging buffer (non-blocking)
for each layer:
    scatter staging buffer rows into document's arena pages
record completion event
# the chunk's forward waits on this event before launching
```

### Pseudocode: store length threshold

```
sort all document lengths in descending order
budget = capacity_bytes / kappa    (capacity in tokens)
taken = 0
threshold = 0
for each length h (longest first):
    if taken + h > budget:
        stop
    taken += h
    threshold = h
# documents with length >= threshold are stored
# shorter documents are recomputed
```

### Key functions: KV store

| Function | File | What it does |
|---|---|---|
| `PinnedStore.save` | `kvstore.py:165` | Async copy of document KV from arena to host |
| `PinnedStore.load` | `kvstore.py:205` | Async copy of document KV from host to arena |
| `PinnedStore.flush` | `kvstore.py:231` | Drop all stored KV (cold pass) |
| `ExtentAllocator.alloc` | `kvstore.py:69` | First-fit contiguous extent allocation |
| `ExtentAllocator.free` | `kvstore.py:83` | Free extent with coalescing |
| `alloc_with_reclaim` | `kvstore.py:29` | Allocate with cross-dataset eviction |

## 8. Multi-GPU dispatch

The coordinator (`runtime/coordinator.py`) splits work across GPU
workers and merges answers. Each worker is a child process with its
own CUDA context, arena, and store slice. The parent process (inside
the same Modal container) sends payloads over pipes, so there is no
network hop between rounds.

### Two rounds per query

**Round 1 (filters)**: every worker filters its shard of every alias.
Sharding is by token count (the planner's `balanced_shards`). After
this round, the parent merges survivors.

**Round 2 (joins)**: anchors follow their filter shard (their KV is
on that GPU). Every worker sees every surviving partner (replicated).
The partner index space is the same on every worker, so the merged
answer rows are consistent.

### Sharding contract

Filters split documents. Joins split anchors. Every pair belongs to
exactly one anchor, so gating, dedup, and the next stage's pair list
stay local to the GPU holding the anchor. All join stages must share
one anchor alias (the same-anchor chain and star shapes); a stage
anchored on a different alias would need a re-shard between stages,
which is not built yet.

### Pseudocode: multi-GPU coordinator

```
# round 1: filters
for each worker w:
    build sub-payload with worker w's shard of each filtered alias
    send to child process w
collect all filter answers
merge: union the per-alias answer dicts and survivor lists

# between rounds: compute survivors per alias
for each alias with filters:
    survivors[alias] = documents that passed all filter stages

# round 2: joins (if any)
for each worker w:
    anchors = documents in worker w's shard that survived filters
    partners = ALL surviving partner documents (replicated)
    send (anchors, partners, join specs) to child process w
collect all join answers
merge: concatenate per-stage answer rows
    (anchors are disjoint across workers;
     partner indices are the same on every worker)
```

### Key functions: coordinator and worker

| Function | File | What it does |
|---|---|---|
| `filter_round_payloads` | `coordinator.py:25` | Build per-worker filter sub-payloads |
| `merge_filter_round` | `coordinator.py:53` | Merge workers' filter answers |
| `join_round_payloads` | `coordinator.py:73` | Build per-worker join sub-payloads |
| `merge_join_round` | `coordinator.py:120` | Concatenate workers' join answer rows |
| `execute` | `worker.py:64` | Single-GPU Modal worker entry point |
| `execute_2/4/8` | `worker.py:495-519` | Multi-GPU Modal worker entry points |
| `_execute_single` | `worker.py:144` | Single-GPU execution core (shared by all paths) |
| `_execute_multi` | `worker.py:463` | Multi-GPU orchestration (split, dispatch, merge) |

### Scaling

The measured scaling on two GPUs: filter 1.99x, join 2.02x
(`dispatch_gate.json`). Modal functions are defined for 2, 4, and 8
GPUs (`worker.py:495-519`).

## 9. The benchmark (QUAIL-B)

QUAIL-B (`bench/quailb.py`) is a benchmark of 15 queries over five
document sets, built from real text (IMDB reviews, BioDEX patient
reports, ABT-BUY product descriptions, and derived sets).

### Planted-flag design

Each document gets a `[FLAGS]` line with planted yes/no values at
known rates (e.g., FLAG_1=YES at rate 0.9 means 90% of documents
have FLAG_1=YES). The filter predicates ask the model to read these
flags. This gives known ground truth for both speed and correctness
measurement.

The rates per set:
- Reviews: F1-F8 at 0.9, 0.9, 0.9, 0.8, 0.8, 0.2, 0.9, 0.8
- Threads: T1-T3 at 0.05, 0.3, 0.9
- Reports: R1-R2 at 0.5, 0.4

### Scale factors

SF (scale factor) controls document counts. At SF=1, there are
50,000 reviews, 10,000 threads, 2,000 reports, 1,000 products, and
2,560 terms. LF (length factor) controls document length by
concatenating real text.

### The 15 queries

| Query | Shape | What it tests |
|---|---|---|
| B1 | 1 filter | The per-query floor |
| B2 | 5 filters | The filter-chain workload |
| B3w/B3c | 5 filters | Written order vs cost order |
| B4 | 2 filters | Long documents (reports) |
| B5 | 1 join | The BioDEX shape (reports x terms) |
| B6 | 1 filter + 1 join | Filter pushdown reduces the pair list |
| B7 | 1 join | Pure pair predicate |
| B8 | 1 exists-join | Semi-join semantics |
| B9 | exists + anti | Exists then anti on the same anchor |
| B10 | 2 joins (chain) | Gating, dedup, and replay |
| B11 | 2 joins (star) | Kept anchors read by two stages |
| B12 | 1 filter + 1 join | Two-sided pushdown |
| B13 | 5 filters (rerun) | Warm-pass scan restore (new questions) |
| B14 | 1 join (rerun) | Warm-pass anchor restore |
| B15 | 1 filter + 2 joins | Everything at once |

### Cold/warm protocol

The benchmark runs two passes in one session:
- **Cold pass**: the KV store is flushed and disabled. Every
  document's KV is computed from scratch.
- **Warm pass**: the KV store is enabled. Documents whose KV was
  saved during the cold pass (or the warm pass itself) can be
  restored instead of recomputed.

Each query reports both the provided selectivity (what the planner
was told) and the observed selectivity (what the model actually
returned), which serves as an instrument check for selectivity
drift.

## 10. Weight loading and kernel infrastructure

**Model loading** (`executor/model.py`): Quail loads the model
checkpoint through vLLM's `get_model` function, which gives the
merged QKV and gate-up projections, fp8 weights, and block scales
laid out for DeepGEMM. No vLLM engine, scheduler, or KV pool is
created. vLLM is used as a library for its loader and kernels.

**DeepGEMM**: all linear projections run as fp8 GEMMs through
DeepGEMM (`attention.py:68`), which JIT-compiles a kernel
configuration per token count. Compiled artifacts persist on a Modal
volume, so each configuration compiles once per software stack.

**Triton kernels**: three custom fused kernels
(`attention.py:103-221`):
- `silu_mul_quant`: fused SiLU activation, element-wise multiply,
  and fp8 quantization for the MLP.
- `add_rms_norm_quant`: fused residual add, RMSNorm, and fp8
  quantization.
- `qk_norm_rope`: fused QK-norm and rotary position embedding.

**Kernel warmup** (`loop.py:347`): before any measured run, the
warmup function sweeps DeepGEMM across a dense set of token counts
(every 256 tokens up to 4,096, every 1,024 up to 32,768, every
2,048 up to the budget) for each of the four linear projections.
After the GEMM sweep, one budget-sized filter chunk warms the Triton
kernels and attention path. This ensures no JIT compilation occurs
during measured walls.

# Speed-of-light, docs/second, and $/query: the math, before we build it

This is a research write-up for issue #26. It does not change any
code. It answers: what math do we need, what already exists in the
repo, what's missing, and — the part that matters most — how do we
make sure Quail can never report a number better than speed of light
without that being caught as a bug.

## 1. What "speed of light" (SOL) means here

SOL for a query is the minimum possible wall time on the hardware:
the time if every kernel ran at its peak rate (compute or memory,
whichever binds) and nothing else happened — no scheduling, no
Python, no kernel launch gaps, no idle time waiting on the CPU.

It's a **lower bound**, not a target. Real runs are always slower
than SOL by some overhead multiplier. The old exploration measured
that multiplier at about 0.35 (Quail's serving loop hit 35% of the
275k tok/s ceiling for Qwen3-4B on an H100) with individual GEMMs
hitting 92-93% while resident. So "efficiency" (measured vs. SOL)
in the 30-90% range is expected and healthy. Efficiency over 100% is
not physically possible and always means a bug — either in how wall
time was measured, or in how SOL was computed. Section 5 covers this
in detail, since it's the part you specifically asked about.

## 2. The roofline idea (one paragraph)

A GPU does two things at once: move bytes and do math. Every kernel
takes `max(bytes / memory bandwidth, FLOPs / peak math rate)`.
Divide peak FLOPs by peak bandwidth and you get the **ridge**: the
FLOPs-per-byte point where the two take equally long. Work below the
ridge is memory-bound (moving data is the limit); work above it is
compute-bound (math is the limit). At fp8 on an H100 SXM, the ridge
is 1,979 TFLOP/s / 3.35 TB/s ≈ 591 FLOP/byte.

## 3. What already exists in the repo

`quail/quail/planner/budgets.py` already has a roofline section
(ported from the old exploration), lines 102-176:

- `_projection_time(model, device, chunk)` — ideal seconds for all
  dense projections (QKV, output, MLP gate/up/down), all layers, for
  `chunk` tokens in one forward pass.
- `_attention_time(model, device, chunk, context)` — ideal seconds
  for the attention kernels, all layers, for `chunk` query tokens
  reading a KV cache of length `context`.
- `compute_knee` and `attention_crossover` — batch-size and
  context-length thresholds, used today to size the chunk budget,
  not to estimate query wall time.

`old_exploration/quail/roofline.py` has three things the issue asks
us to port that aren't in `budgets.py` yet:

- `spec_ceiling_tokens_per_s(model, device)` = `peak_flops / (2 *
  params)`. The headline number: tokens/second if literally every
  FLOP went into the forward pass. ~275k tok/s for Qwen3-4B on an
  H100.
- `elementwise_time(model, device, chunk)` — ideal seconds for
  RMSNorm (x2), fp8 quantization before each GEMM, the SwiGLU
  activation, and the residual adds. All of this is memory-bound
  at every batch size — a fixed per-token tax with no compute
  knee. Leaving it out understates SOL (makes the floor too low),
  which is the safe direction for a floor, but it's real work the
  GPU has to do, so a query-level SOL should include it.
- `ideal_step_time` — just sums the three components. Trivial to
  port once `elementwise_time` exists.

Model and device constants (`quail/quail/specs/`):
`QWEN3_4B_FP8` (params=3.6e9, layers=36, hidden=2560, ...) and
`H100_SXM` (mem=80GB, hbm_bw=3.35 TB/s, peak_flops=1.979e15 fp8).
Same structs `budgets.py` already uses, so no new plumbing needed to
call the roofline functions for a real query.

## 4. The part that needs real thought: chunk vs. context isn't one shape

`_attention_time(model, device, chunk, context)` models **one**
thing: `chunk` query tokens all reading a **shared** KV cache of
length `context`. That's exactly the join case. It is *not* what
happens during a filter's self-attention, and using it naively for
filters will make SOL wrong — probably wrong in the direction that
falsely triggers "efficiency > 100%, must be a bug" on a perfectly
healthy run. This is worth getting right up front.

### Filters: causal self-attention, quadratic in document length

A filter prefills each document once, causally: token at position
`p` in a document attends to the `p` tokens before it, not to a
fixed external context. Total attention FLOPs for one document of
length `L` is roughly `2 * n_q * d_head * L^2 / 2` (the `/2` is the
causal mask cutting the full `L x L` pair count in half).

If you instead plug `chunk = L, context = L` into the existing
`_attention_time` formula, you get `4 * L * L * n_q * d_head` FLOPs
for that document — **2x too many**, because the formula assumes
every one of the `L` query tokens reads the *entire* `L`-length
context (no causal mask). Use that number as part of SOL and SOL
comes out inflated. An inflated SOL is a floor that's too high,
which is exactly what produces a false "measured beat speed of
light" alarm on a normal run.

Two ways to fix this for filters:
- **Exact**: sum per-document, using `context = L_i / 2` (the mean
  attended-context length under a causal mask) instead of `L_i`.
  This halves the FLOPs term back to the real causal count.
- **Approximate but simpler**: keep `chunk = context = L_i` but
  divide the attention FLOPs term by 2 before taking the max with
  the memory-bound term. Equivalent result, less code to touch if
  we don't want two attention-time variants.

Either way: **don't call `_attention_time` on a filter's own
document length without correcting for the causal mask.**

### Joins: exactly matches the existing formula, because of KV rewind

Joins in this codebase use KV rewind (`chain mode`): an anchor's
prefix KV is written to the arena once and reused across every
partner it's compared against — that's the whole point of the
mechanism, it's what makes the join not recompute the anchor's KV
per pair (`quail/quail/executor/loop.py:248`, `run_join`). So for a
join stage, `chunk` = partner suffix tokens packed into one forward,
`context` = the anchor's prefix length (already-cached, not
recomputed). This is precisely the shape `_attention_time` already
models — no correction needed here, provided we don't accidentally
also charge SOL for recomputing the anchor's own prefill (that
prefill already happened, its cost belongs to the operator that
built the anchor's KV, not to the join stage — usually a preceding
filter, or the anchor's own doc-scan prefill).

One more detail specific to joins: `run_join` batches multiple
anchors' partners into one chunk when `group_size` spans more than
one anchor. If a real forward pass mixes partners from two anchors
with different prefix lengths, `context` isn't a single number
anymore. For a query-level SOL estimate this is fine to approximate
with the *anchors' mean prefix length* — the issue's proposed "chunk
size and average context length" already covers this — but it's
worth writing down as an approximation, not treating it as exact.

### Why this section matters for the "never above 100%" goal

There are two independent ways a query could look faster than SOL:

1. **A real bug** in the engine or in how `wall_s` / `fresh_tokens`
   are measured (Section 5).
2. **A wrong SOL formula** — specifically, applying the
   shared-context attention formula to causal self-attention without
   the 2x correction would make SOL *too small* in the direction that
   hides bugs, and applying it *without* dividing by 2 the other way
   (if the correction is applied backwards) would make SOL too large
   and manufacture false alarms.

Getting the filter-vs-join distinction right isn't a nice-to-have —
it's the difference between the invariant in Section 5 being
trustworthy or being noise.

## 5. The invariant: efficiency must never exceed 100%

Define, per query:

```
efficiency = SOL_seconds / measured_seconds
```

SOL is a lower bound on physically possible time. So mathematically,
`efficiency <= 1.0` (100%) always, for any bug-free measurement.
**If `efficiency > 1.0`, something is wrong** — either the measured
time is wrong, or the SOL estimate is wrong (see Section 4). It is
never evidence of a real speedup past the hardware limit.

Concrete ways this invariant can be violated, worth checking for
explicitly once this is built:

- **`wall_s` measured wrong**: the timer starts after some setup
  work, or stops before some work finishes (e.g., an async copy not
  awaited before the clock stops).
- **`fresh_tokens` double-discounted**: KV rewind means restored/
  reused tokens shouldn't be charged fresh compute — SOL must use
  the same "fresh" definition as the timer does. If `fresh_tokens`
  undercounts (e.g., a store restore is marked free when it still
  cost a fresh forward), the SOL built from it will be too small,
  and a real bug (not doing the work it claims not to be doing)
  could hide inside a falsely-comfortable efficiency number instead
  of tripping the alarm. This cuts the other way from the filter
  formula issue in Section 4 — same invariant, opposite failure
  direction.
- **Wrong model/device spec used**: computing SOL with the 4B specs
  for a 32B run, or fp8 peak FLOPs applied where the executor
  actually ran a slower path.
- **SOL formula error described in Section 4**: using the
  shared-context attention formula for filter self-attention without
  the causal correction.

Recommendation: once `sol_seconds()` exists, add a check right where
the report row is built (`quailb.py`, around line 816) —

```
assert measured_wall_s >= sol_seconds * (1 - tolerance)
```

— and fail loudly (raise, don't just print) if it doesn't hold,
naming which side (measured or SOL) looks suspicious. `tolerance`
should be small (a percent or two) to absorb the join
multi-anchor-context averaging from Section 4, not a blank check.
An efficiency print that just shows ">100%" next to a query in a
results table is much easier to miss than a hard failure during the
benchmark run.

## 6. Docs/second and tokens/second — no new math

Both are already computable from data the report has today
(`quail/quail/runtime/session.py:441`, the `report` dict):

```
docs_per_s   = rows_processed / wall_s      # wall_s excludes boot
tokens_per_s = fresh_tokens / wall_s
```

`res.report["fresh_tokens"]` and `res.report["wall_s"]` already
exist. `rows` (row count) is already captured in the benchmark
driver at `quailb.py:822`. This is a formatting change to the report
row, not new instrumentation.

## 7. $/query — one new constant, one new multiply

```
cost_dollars = gpu_count * (wall_s + boot_s) * rate_per_gpu_second
```

- `gpu_count` = `EngineConfig.gpus` (`quail/quail/planner/plan.py:103`).
  Scope note: one model copy per GPU, no tensor-parallel sharing
  across cards (see project scope), so this is a flat multiply, not
  a TP-aware split.
- `rate_per_gpu_second`: Modal's Python SDK (v1.5.4, already in
  `quail/.venv`) exposes `modal.Workspace.from_context().billing
  .rates()` — an async call returning a `Mapping[str, Decimal]` of
  current per-resource rates for the workspace. This accounts for
  region multipliers and preemptibility. The worker requests
  `gpu="H100!"` (non-preemptible — the `!` pins it), so whichever key
  in that mapping corresponds to non-preemptible H100 SXM is the
  right one to read.
  - **Open item**: the exact key name in that mapping (something
    like `"h100"` or `"gpu-h100"`) isn't knowable without an actual
    authenticated call against the workspace — it's server-provided,
    not documented in the SDK source. First implementation step:
    call `rates()` once, print the dict, and note which key is used.
  - Fallback constant if the call fails or for offline estimates:
    ~$3.95/hour ≈ $0.001097/second, per the issue.
- Report the **cold pass cost separately from warm** (the issue's
  point): cold includes `boot_s`, which is a real dollar cost the
  user pays once per container, not a per-query cost that
  amortizes the same way every time.

## 8. What data the current report is missing

Two gaps, worth naming so the next implementation step doesn't
discover them mid-way:

- **`report["stages"]` has selectivity and counts, not token
  counts.** (`session.py:472` for filters, `session.py:494` for
  joins.) To split SOL by operator (filter stage vs. join stage,
  needed because their attention math differs per Section 4) we
  need fresh-token counts per stage, not just the query-level total
  currently in `report["fresh_tokens"]`.
- **`spans` (returned by `run_filter`/`run_join`,
  `quail/quail/executor/loop.py:291`) carry `(stage, start_event,
  end_event)` — GPU-time markers only, no chunk size or context
  length.** They're used today only for GPU-time sums, not exposed
  to the report. A telemetry-exact SOL (computed from what actually
  ran, forward by forward) would need `spans` extended to also carry
  `(chunk_tokens, context_tokens)`, or a parallel list alongside it.

## 9. Two ways to build `sol_seconds()` — pick one

**Option A — plan-time estimate (what the issue describes).**
Before or after a run, from `CorpusStats` (n_docs, total_tokens,
mean_doc_tokens — already computed at plan time,
`quail/quail/planner/plan.py:16`) and the query's operator list
(filter stage count, join pair count), estimate chunk/context per
operator type and sum `_projection_time + _attention_time +
elementwise_time`. Cheap, no new instrumentation, matches the issue
spec directly. Downside: uses averaged context length, so it's an
approximation of the true floor, not an exact one (Section 4's
join-batching caveat, plus filters using mean document length
instead of the real per-document distribution).

**Option B — telemetry-exact.** Extend `spans` to record the real
`(chunk_tokens, context_tokens)` of every forward pass, and compute
SOL from the exact sequence of forwards that ran, not an estimate of
what a "typical" one looks like. This is a strictly tighter floor —
it would catch a bug the plan-time estimate might paper over (e.g.,
a store restore that quietly does more work than it should, but
still averages out fine at the corpus level). Costs more: new
instrumentation in `loop.py`, and the report format changes.

**Recommendation**: build Option A first — it's what's needed for
the report table in the issue, and it's most of the way to Option B
already (same underlying `_projection_time`/`_attention_time`
calls, just fed estimates instead of exact per-forward data). Keep
Option B in mind as the natural follow-up if Option A's efficiency
numbers ever look suspicious without an obvious cause — at that
point the coarser estimate becomes the thing you can't fully trust,
and the exact version answers whether the SOL model or the engine is
at fault.

## 10. Summary: what actually needs to change (unchanged from the issue, cross-referenced)

1. Port `spec_ceiling_tokens_per_s` and `elementwise_time` into
   `budgets.py` (Section 3).
2. Add `sol_seconds(model, device, workload)`, built on
   `_projection_time` + corrected `_attention_time` (causal fix for
   filters, Section 4) + `elementwise_time`. Start with Option A
   (Section 9).
3. Add the `efficiency <= 1.0` assertion at report-build time
   (Section 5) — this is the actual ask behind "make sure we never
   report better than speed of light."
4. Extend `res.report` / the `quailb.py` row dict
   (`quailb.py:816-825`) with `tokens_per_s`, `docs_per_s`, `sol_s`,
   `efficiency`, `cost_dollars` (Sections 6-7).
5. Resolve the Modal billing rate key name with one live `rates()`
   call, wire in the fallback constant (Section 7).
6. Update the report table / `make_plots.py`-style generator with
   the new columns and a measured-vs-SOL chart per query.

Nothing above has been implemented yet — this document is the
research and design pass the issue asked for before touching code.

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

## 7. $/query — GPU, CPU, and memory, each with their own rate

```
cost_dollars = gpu_count * (wall_s + boot_s) * rate_gpu_per_s
             + cpu_memory_gb * (wall_s + boot_s) * rate_mem_per_gib_s
```

CPU core-seconds left out for now: the worker's `@app.function`
calls (`quail/quail/runtime/worker.py`) set `gpu=` and `memory=` but
never `cpu=`, so there's no explicit core count to multiply the CPU
rate against — only Modal's unrequested default. Worth resolving
before this ships, not assumed away.

- `gpu_count` = `EngineConfig.gpus` (`quail/quail/planner/plan.py:103`).
  Scope note: one model copy per GPU, no tensor-parallel sharing
  across cards (see project scope), so this is a flat multiply, not
  a TP-aware split.
- `cpu_memory_gb` = `EngineConfig.cpu_memory_gb` (`plan.py:104`,
  default 64, the benchmark driver runs at 80). Worker containers
  actually request 96 GiB (`memory=98304`) for one GPU — close to
  the ~100 GiB point past which the issue itself said memory cost
  stops being negligible, so it's included rather than skipped.
- Rates: hardcoded, checked in at
  `quail/quail/calibration/modal_rates.json` — given directly by the
  user from the Modal pricing page for this workspace, not pulled
  from the live API. In scope: `gpu_per_second["h100-sxm"]` =
  0.001097 (matches the issue's own fallback estimate of ~$3.95/hr
  almost exactly), `memory_per_gib_per_second` = 0.00000222.
  - Modal's SDK also exposes `modal.Workspace.from_context()
    .billing.rates()` (confirmed callable, v1.5.4, `quail/.venv`) —
    an async call returning current per-resource rates for the
    actual workspace, accounting for region and preemptibility. Not
    used as the primary source here since the user already supplied
    the numbers directly; worth a one-time live call later just to
    cross-check the checked-in file hasn't drifted.
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

## 11. Update: implemented and validated on real hardware (2026-08-23)

Items 1-4 above are built (`budgets.py`'s `sol_seconds` and its
supporting roofline functions; `session.py`'s per-query workload
walk; `quailb.py`'s new report columns and the hard efficiency
assertion). Item 5 uses the rates the user supplied directly
(`quail/calibration/modal_rates.json`) rather than a live
`Workspace.billing.rates()` call — see that file's provenance note.
Item 6 (the report generator) is built too, as of 2026-08-25 — see
section 16.

Validated with two real runs on Modal (`tests/gpu/sol_check.py`):
IMDB-1, IMDB-2, IMDB-5, BIO-2, FEV-5, cold + warm, sf=0.1,
qwen3-4b-fp8, one H100. All 10 runs landed 53-68% efficiency — well
under the 100% physical limit, in the range the old exploration's
~35%-full-loop measurement would predict for individual operators
running closer to the roofline than a full-loop average.

The first run caught a real bug in the SOL model, not just noise:
IMDB-2 and IMDB-5's warm passes restored KV for 517 and 675
documents from a store an earlier query in the same run had already
written. The initial `sol_seconds()` had no restore accounting, so
it charged every anchor a full causal-build cost regardless -
`sol_s` didn't move between cold and warm, but real restores made
`wall_s` genuinely faster, so **efficiency rose instead of fell**
(IMDB-5: 64% cold -> a wrong 73% warm). Section 5's original
reasoning called ignoring restores "the safe direction" - that was
backwards: an unmoving, too-large `sol_s` over a shrinking `wall_s`
is exactly how this check could have crossed 100% on a healthy run,
just not with quite enough restored volume in this particular
experiment. Fixed by netting each alias's `restored_tokens` out of
its causal-build candidates (largest documents first, since the
store only takes documents at or above its minimum length) before
computing the floor - rerun confirmed IMDB-5 now correctly reads 57%
warm (down from cold, as restores should make it), not 73%.

Full suite output for both runs (the buggy one and the fixed rerun)
is `results/sol_check_sf0.1_4b_uncorrected.json` and
`results/sol_check_sf0.1_4b_corrected.json` - each query's full
report dict, including `sol_s` / `sol_efficiency` / `store`
(restored/stored token counts) / `cost_dollars`.

| Query | Shape | Pass | Wall (s) | SOL (s) | Efficiency | Restored tok | Cost ($) |
|---|---|---|---|---|---|---|---|
| IMDB-1 | filter only | cold | 15.06 | 10.195 | 68% | 0 | 0.0806 |
| IMDB-1 | filter only | warm | 16.51 | 10.195 | 62% | 0 | 0.0210 |
| IMDB-2 | join only | cold | 53.18 | 35.549 | 67% | 0 | 0.0678 |
| IMDB-2 | join only | warm | 52.42 | 33.058 | 63% | 433,576 | 0.0668 |
| IMDB-5 | 3 filters + join | cold | 23.94 | 15.028 | 63% | 0 | 0.0305 |
| IMDB-5 | 3 filters + join | warm | 20.69 | 11.742 | 57% | 571,791 | 0.0264 |
| BIO-2 | join only | cold | 140.31 | 87.717 | 63% | 0 | 0.1788 |
| BIO-2 | join only | warm | 142.87 | 87.717 | 61% | 0 | 0.1821 |
| FEV-5 | 2-sided filter + join | cold | 1.84 | 1.135 | 62% | 0 | 0.0023 |
| FEV-5 | 2-sided filter + join | warm | 2.01 | 1.056 | 53% | 13,631 | 0.0026 |

(Corrected-code numbers throughout; the uncorrected run's IMDB-2 and
IMDB-5 warm efficiency read 68% and 73% respectively before the fix.)

## 12. Adversarial review (2026-08-24): one high-severity bug, three minor

A GPU run proves the common case works; it doesn't prove the check
can't be fooled by a query shape the run didn't happen to exercise.
A fresh review agent - not primed with any of the reasoning in this
document - was asked to try to break the implementation rather than
read and confirm it. It also noted that none of the 156 existing
tests exercised `sol_s`, `sol_efficiency`, `docs_per_s`, or
`cost_dollars` at all, which is exactly how the first finding shipped
unnoticed.

**High**: the `efficiency <= 100%` check (Section 5's whole point)
never actually stopped anything. `raise AssertionError(...)` sat
inside the same `try/except` in `quailb.py`'s row-building loop that
turns any exception into an `error` string in the results row -
indistinguishable from a Refusal or a network hiccup. `run_suite`
returned normally, exit 0, as if nothing had happened. Fixed with a
dedicated `SolViolation(RuntimeError)`, re-raised explicitly before
the generic `except Exception` clause so it can't be caught by it.

**Medium**: a join anchor's per-stage "frame" (the naming line
written into its kept KV, e.g. "the document above is...") had its
length used correctly as part of what partner suffixes read
*against*, but the write itself was never charged as its own chunk
of work in `sol_seconds()`'s input. Safe direction (makes the floor
slightly too generous, not too tight), but a real missing kernel
call. Fixed: one more streaming entry per anchor per join stage.

**Low** (two): the restore-correction heuristic (Section 11) used
whatever was left with no signal if `restored_tokens` for an alias
ever exceeded what the walk had charged as build candidates - not
reachable in the current query catalog, degrades safely, now leaves
a report remark instead of vanishing silently. And `_docs_per_s`
summed *every* stage-0 filter's evaluated count, which double-counts
across two different tables for a two-sided query (FEV-5/6, LEP-7);
fixed to use only the join anchor's own count when a join is present.

Checked and confirmed fine, no change: the LEP-2 self-join (one
table, two aliases) doesn't cross-contaminate the `built`/
`causal_by_alias` tracking; a zero-survivor upstream filter doesn't
crash the floor computation; the multi-stage filter chain's
accumulated context has no off-by-one; and aggregating FLOPs/bytes
before `max()` (Section 4's design choice) is mathematically
guaranteed to produce a value at or below the true floor, so it can
only make the invariant check more conservative, never a source of
false alarms.

All four fixes shipped with regression tests
(`test_sol_violation_aborts_run_suite`,
`test_run_suite_reports_sol_and_cost_fields`,
`test_docs_per_s_two_sided_query_uses_anchor_only`) exercising
`run_suite` end to end through a new `_execute` test seam (mirroring
`Query.run()`'s existing one), closing the coverage gap that let the
high-severity bug ship. 159 tests pass.

## 16. The report generator, and a second adversarial review (2026-08-25)

Item 6, finally built: `quail/quail/bench/sol_report.py`. `build_rows(suite)`
takes a `run_suite()` suite dict and produces one row per query with the
exact column set the issue specifies — Docs/Tokens from the cold pass (no
restores, the true full corpus), SOL(s)/Efficiency/Docs-per-s/Tok-per-s from
the warm pass (matching the issue's own "(warm)" labels), and cost reported
for both passes separately, per the issue's explicit point that cold's boot
time is a real cost. `render_markdown_table(rows)` formats it;
`plot_sol_comparison(rows, out_path)` draws the measured-vs-floor bar chart,
matplotlib imported lazily so the module doesn't require it — matching
`reports/make_plots.py`'s existing on-demand-dependency convention. A
`_docs_count` helper was split out of `quailb._docs_per_s` (a pure refactor)
so the table's raw "Docs" column and the existing rate could share one
implementation instead of two.

Run against the real validated data from section 14/15's GPU runs
(`results/sol_check_sf0.1_4b_corrected.json`), the tool reproduces every
number in this document's own table exactly. The chart is committed at
`reports/plots/sol_report_sf0.1_4b.png`.

29 tests shipped with the first version. A second adversarial review —
same method as section 12: a fresh agent, not primed with any of this
reasoning, asked to construct real failing inputs rather than read and nod
— found two real bugs in `render_markdown_table`, both edge cases the
first test suite's fixtures never exercised because they always described
plausible-looking data:

- An error message containing `|` (a real possibility — `KeyError` reprs
  quote dicts, exception text can embed almost anything) added an
  unescaped column separator, misaligning every cell after it in that row.
  Fixed by escaping `|` in the error text before interpolating it.
- A success-path row where `docs` is `None` (`_docs_count`'s fallback when
  a query has no stage-0 filter and no join) printed the Python string
  `"None"` instead of the `—` every other empty field uses — the one
  column that skipped its formatter. Fixed with a new `_fmt_int` helper.

Both fixes verified against the real data: table and chart output are
byte-identical to before the fix, confirming they only change behavior for
the previously-buggy edge cases. Two other things the review flagged were
confirmed not live bugs and documented in code rather than fixed: a
duplicate query id within one pass silently keeps the last one (`build_rows`
uses a plain dict comprehension), unreachable because `run_suite()` iterates
each id once per pass; and the chart doesn't visually flag an impossible
`sol_s > warm_s` row, unreachable because `SolViolation` (section 12) aborts
the run before such a row could exist in a suite this repo actually
produces.

192 tests pass with matplotlib installed (161 from before this section,
plus 29 for `sol_report.py`, 2 more for the `_docs_count` refactor — the 4
matplotlib-dependent tests skip cleanly without it installed, and were
separately verified to pass with `uv run --with matplotlib`).

All six items from the issue's original checklist are now built and
validated. What's left is exactly what section 7's fallback-constant note
already flagged: resolving the CPU core-count term for `$/query` if the
worker ever requests one, and a live `Workspace.billing.rates()`
cross-check against the hardcoded rates in `modal_rates.json` — neither
blocking, both explicitly deferred, not forgotten.

## 17. "How do you verify SOL is accurate?" — a three-phase answer (2026-08-25)

The claim "sol_s is accurate" splits into levels, and each got a different
verification method:

**Formula math sanity** (`tests/test_budgets_sol_properties.py`, 22 tests):
does each formula behave the way its own roofline derivation says it
should — monotonic, exactly linear where there's no `max()`, quadratic in
the compute-bound regime and not below the crossover, subadditive the way
the "aggregate before `max()`" design note claims, `compute_knee`/
`attention_crossover` landing exactly where their two sides are equal.
Caught a real bug immediately: `sol_seconds()` of a genuinely empty
workload returned ~0.00108s, not 0.0 — `_projection_time(chunk=0)`'s
weight-read memory term is chunk-independent, so it charged reading every
weight matrix even for a kernel that never launches. Fixed with a
`chunk==0` short-circuit; verified byte-identical output on real
non-zero-chunk data. `sol_seconds_breakdown()` was also split out of
`sol_seconds()` here (same arithmetic, four terms kept separate), needed
by both these tests and phase 3 below.

**Formula behavior, visualized** (`reports/plot_sol_formula_diagnostics.py`,
removed section 22 below): the same properties as pictures — causal
attention flat then quadratic at the ~24,639-token crossover, streaming
attention flat then linear (never quadratic — the whole reason it's a
separate function), projection's compute knee (~416 tokens) against
elementwise's dead-straight no-knee line. These were synthetic sweeps of
the formula's own math at fixed model/hardware constants, not this PR's
measured query results — the property tests above are what actually
verify this behavior; the plots were a picture of what those tests check
numerically. (An earlier version of this section also plotted the
subadditivity gap — the overstatement from pricing every item separately
and summing instead of aggregating before one `max()` — peaking at ~50%
when a compute-bound and a memory-bound item are equal-sized; that plot
was trimmed from the report at request, section 21, since section 4's
aggregate-before-`max()` design note already covers the same point in
text.)

**Real kernels vs. the formula's structure** (`tests/gpu/
torch_profiler_compare.py`, `reports/plot_torch_profiler_comparison.py`):
the level the first two can't reach — does the formula's internal
*shape*, not just its aggregate bound, match what the GPU actually
does? `session.py` now records `report["sol_breakdown"]` (the four
terms separately) alongside `sol_s`. One `torch.profiler` call per
query (not two — `sol_breakdown` comes from token counts, not timing,
so it's safe to read off the same profiled call; only `wall_s` would be
unreliable under profiling overhead, and this comparison doesn't need
`wall_s`). `Query.run()`'s existing `_execute` seam wraps the real
worker call via `worker.execute.local(payload)` — in-process, no RPC,
no changes to `worker.py`. Kernel-class rules were built from a real
exploratory trace (`tests/gpu/torch_profiler_explore.py`), not ported
from the old exploration's vLLM-based rules, which don't apply to this
engine's DeepGEMM/Triton/FlashAttention-3 kernels.

Result, consistent across all 5 validated queries regardless of shape:

| | measured (renormalized, excl. ~5-16% unmodeled) | formula |
|---|---|---|
| projection | closely matches, everywhere | closely matches, everywhere |
| attention | consistently *higher* than predicted | consistently *lower* than measured |
| elementwise | consistently *lower* than measured | consistently *higher* than predicted |

Projection matching closely is expected — `_projection_time` and DeepGEMM
are both doing the same GEMM math with nothing subtle in between. The
attention/elementwise gap is the real finding: attention costs relatively
more, and elementwise relatively less, than their formulas predict, on
every query shape tried (filter-only, join-only, a 3-filter chain into a
join, a different dataset's join, a two-sided filter-then-join) — not one
workload's noise. Two kernels this project doesn't model in any SOL term
(`qk_norm_rope` — RoPE, and `kv_row_scatter` — KV-write bookkeeping) sit in
the excluded "unmodeled" bucket, not inside either measured attention or
measured elementwise, so they don't explain this gap; it's about how the
two *modeled* shapes compare to each other, not about what's missing
entirely.

This doesn't threaten the efficiency invariant — `sol_s` stayed well under
`wall_s` on every query, same as every prior run — it's evidence about
*tightness*, not correctness of the bound. Worth a closer look at whether
`elementwise_time`'s memory-bound assumption holds as tightly in practice
as projection's does; not blocking, and not treated as a bug the way the
two adversarial reviews' findings were.

216 tests pass (214 + 2 for the renormalization logic in the comparison
plot).

## 18. A store methodology gap in the profiler comparison (2026-08-25)

`Session.store_enabled` defaults to `True` (`runtime/session.py`).
`quailb.py`'s cold/warm passes turn it off/on explicitly; section 16's
`torch_profiler_compare.py` and `torch_profiler_explore.py` never
called `set_store()` at all, so they silently ran with the store on.
Both scripts also "warm the container" by running IMDB-1 once before
profiling it - with the store on, that meant IMDB-1's profiled run
could restore its own KV from the warmup call instead of building it
fresh, which is not the same workload `sol_breakdown` assumes when
nothing is restored.

**Prediction before rerunning with `sess.set_store(False)` added to
both scripts:** IMDB-1, BIO-2, and FEV-5 shouldn't change much - BIO-2
and FEV-5 use tables no other query touches, and IMDB-1 restored 0
tokens even in `quailb.py`'s own warm pass. IMDB-2 and IMDB-5 (which
do restore under warm conditions per section 15) might get slower once
restores are forced off. The attention-high/elementwise-low pattern
should hold regardless, since BIO-2 and FEV-5 already showed it with
no possible restore path.

**Result**, rerun on Modal (`results/torch_profiler_compare_corrected.json`,
prior run kept at `results/torch_profiler_compare_uncorrected.json`):

| Query | sol_s before | sol_s after | wall_s before | wall_s after |
|---|---|---|---|---|
| IMDB-1 | 7.702 | 10.195 | 12.52 | 14.46 |
| IMDB-2 | 33.381 | 35.872 | 57.82 | 52.97 |
| IMDB-5 | 11.785 | 15.071 | 22.04 | 24.18 |
| BIO-2 | 87.764 | 87.764 | 147.59 | 139.76 |
| FEV-5 | 1.058 | 1.138 | 2.78 | 1.83 |

BIO-2's `sol_s` is identical before and after, confirming it had no
restore path either way. IMDB-1's jumped from 7.70s to 10.20s - it was
restoring essentially its whole corpus from the redundant self-warmup
call, and 10.195s now matches `results/sol_check_sf0.1_4b_corrected.json`'s
validated cold-pass number for IMDB-1 exactly. IMDB-2 and IMDB-5 also
increased, as predicted. The "unmodeled" kernel share also dropped
across every query, including BIO-2 and FEV-5 (5.36%→4.63%,
16.39%→6.27%) - larger than restores alone explain, most likely
because writing newly-built KV *into* the store (not just restoring
from it) is itself a memcpy the profiler classifies as unmodeled, and
that write traffic disappears entirely once the store is off for the
whole run, not just the restore side of it.

The attention-high/elementwise-low pattern (section 17) holds, and is
if anything sharper with the confound removed - now that "unmodeled"
is a smaller, more consistent slice, the renormalized comparison is
less diluted:

| Query | attn: measured ÷ formula | elem: measured ÷ formula | proj: measured ÷ formula |
|---|---|---|---|
| IMDB-1 | 2.12x | 0.66x | 1.11x |
| IMDB-2 | 2.29x | 0.51x | 1.13x |
| IMDB-5 | 2.01x | 0.61x | 1.11x |
| BIO-2 | 1.48x | 0.45x | 0.99x |
| FEV-5 | 2.17x | 0.53x | 1.12x |

BIO-2 - the one query untouched by the store bug at any point - sits
at the *low* end of the attention ratio, not the high end. That rules
out the store confound as the explanation for section 17's finding:
real causal attention kernels are running at roughly a third to a half
of the peak-FLOP rate `_causal_prefill_attention_time` assumes, fairly
consistently across query shapes, while the dense-projection formula
tracks real DeepGEMM time closely (0.99-1.13x) throughout.

A new figure, `plots/torch_profiler_totals.png`
(`plot_torch_profiler_comparison.py`'s `fig_totals`), makes the same
point a different way: for every query, `sol_s <= total kernel time
(torch.profiler) <= wall_s`, in that order, with no exceptions. The
gap from `sol_s` up to kernel time is real kernels running below peak
(mostly attention, per the table above); the gap from kernel time up
to `wall_s` is time spent outside any GPU kernel at all (Python,
scheduling, launch gaps) - small on every query here, kernel time
tracks within a few percent of wall_s throughout.

No new tests: the fix is two lines (`sess.set_store(False)`) in
scripts that only run on Modal, with no local-testable surface of
their own. Same 216 tests pass as section 17.

## 19. Cost and throughput, plotted (2026-08-25)

The report table (section 6) already has cost, docs/s, and tokens/s
columns; nothing in this section computes anything new, it just
charts the columns `sol_report.build_rows()` already produces from
`results/sol_check_sf0.1_4b_corrected.json`
(`reports/make_sol_cost_throughput_plots.py`):

Figure: plots/sol_cost_by_query.png

Cold vs. warm dollar cost per query. IMDB-1's cold bar ($0.081) is
disproportionately large next to its warm bar ($0.021) - not because
IMDB-1 itself is expensive, but because it happened to run first in
this suite and so is the one query charged the one-time model boot
cost (`boot_s`, ~48s). Every other query's cold pass had `boot_s = 0`.
BIO-2, the largest query by fresh tokens, costs the most either way
(~$0.18) and its cold/warm costs are nearly identical, since almost
all of its cost is steady-state GPU time, not boot.

Figure: plots/sol_docs_per_s.png

Documents/second, warm pass, linear (78-1,145 docs/s). IMDB-2 leads at 1,145 docs/s: it's a
join, and a join's per-document cost is dominated by short streaming
reads against an already-built anchor, not a full causal prefill.
FEV-5 is lowest (78 docs/s) - it's also the smallest query by document
count (57), so per-document fixed overhead (boot-adjacent setup,
short-query effects) has more relative weight.

Figure: plots/sol_tokens_per_s.png

Tokens/second, warm pass, linear (79k-109k, under 1.4x - no log
needed). Much tighter than docs/s across queries, because tokens/s
is closer to a hardware property (roughly how fast the GPU forwards
tokens) while docs/s also depends on how many tokens each query's
documents happen to contain.

## 20. Docs/s, tokens/s, and cost against the SOL number, not just wall time (2026-08-25)

Section 6 reports `sol_s` and efficiency against wall time, but
docs/s, tokens/s, and cost were still only ever the measured (warm)
numbers - nothing in the report said what those three would be if the
same workload ran at peak hardware speed instead. `sol_report.
build_rows()` now adds three fields: `docs_per_s_sol`,
`tokens_per_s_sol`, `cost_sol`.

**Which direction each one moves depends on whether it scales with
time or against it.** `sol_s` is never more than the measured
`wall_s` - nothing runs faster than peak. Cost is `time x rate`, same
direction, so `cost_sol` is never more than the measured cost either
- less time at the same rate is less money. Throughput is `count /
time` - a rate, the reciprocal of time - so it runs the other way:
`docs_per_s_sol` and `tokens_per_s_sol` are never less than the
measured rate - the same work in less time is a higher rate. Both
facts come from the one thing that's actually true (`sol_s <= wall_s`
), just applied to two different kinds of quantity.

**What this is not**: a pre-execution predictor. `sol_s` is computed
from the query's own real evaluated/restored counts after it runs
(`runtime/session.py`'s SOL walk), not from the planner's selectivity
estimate before running - `provided_selectivity` is null for every
query in this benchmark's current corpus, so there is no selectivity
model to predict from yet. These three fields are a best-case
reference for a query of this shape and size that has already run,
not a live estimate for one that hasn't.

**A real bug this surfaced**: the first version of `docs_per_s_sol`
used `docs` (the cold-pass corpus size) as the numerator, while the
already-shipped `docs_per_s_warm` field was computed by an older
`run_suite()` from the warm pass's own `_docs_count()` call. For four
of five queries these agree. For FEV-5 - the one two-sided query (a
filter on both the join's anchor and a partner table) - they didn't:
`results/sol_check_sf0.1_4b_corrected.json` was committed (`b3f9606`)
before the `_docs_count` two-sided-summing bug fix (`3ec98d5`, an
adversarial review two commits later), so its stored `docs_per_s` for
FEV-5 was stale - 85.3/78.1 (cold/warm), computed from the old buggy
sum of both tables' stage-0 counts (157), not the current, correct
anchor-only count (57). The symptom was exactly what a stale value
should produce: the SOL-derived rate came out *below* the measured
rate (54 vs. a stale 78.1), which cannot happen once `sol_s <= wall_s`
holds (section 12's invariant). Fixed the stale field directly in the
committed JSON - same underlying raw data (`evaluated` counts,
`wall_s`), recomputed with the current, already-fixed `_docs_count()`
(31.0/28.4, cold/warm) - not a new measurement, a correction of a
derived field the earlier fix never propagated to. `quailb.py`'s
`suite = dict(...)` now also persists `cpu_memory_gb` (only `gpus`
was saved before), so `cost_sol` doesn't have to guess the container
size for future runs; `build_rows()` falls back to 80 (the default
every suite committed before this field existed actually ran with)
when it's absent.

| Query | Docs/s (warm) | Docs/s (SOL) | Tok/s (warm) | Tok/s (SOL) | Cost (warm) | Cost (SOL) |
|---|---|---|---|---|---|---|
| IMDB-1 | 303 | 490 | 107k | 174k | $0.0210 | $0.0130 |
| IMDB-2 | 1,145 | 1,815 | 109k | 172k | $0.0668 | $0.0421 |
| IMDB-5 | 242 | 426 | 106k | 187k | $0.0264 | $0.0150 |
| BIO-2 | 860 | 1,400 | 79k | 129k | $0.1821 | $0.1118 |
| FEV-5 | 28 | 54 | 96k | 183k | $0.0026 | $0.0013 |

Figure: plots/sol_docs_per_s.png, plots/sol_tokens_per_s.png,
plots/sol_cost_estimate.png

The gap between measured and SOL tracks efficiency (section 6)
exactly, since the SOL-to-measured ratio is just `1 / efficiency`
(same `docs`/`tokens` divided by `sol_s` vs. `warm_s`). FEV-5 has
both the lowest efficiency (53%) and the widest relative gap (28
measured vs. 54 SOL-estimated docs/s, a 1.90x spread); IMDB-2 has the
highest efficiency (63%) and the narrowest gap (1.59x). No query
breaks that ordering.

**A second real bug this surfaced (2026-08-26)**: `tokens_per_s_sol`
divided `tokens` (the cold-pass fresh-token count) by `sol_s` from
`rate_src` (the warm pass). For the two queries with real KV-store
restores that quarter (IMDB-2, IMDB-5) and FEV-5, warm's `sol_s` is
discounted for restored documents but `tokens` still counted the full,
restore-free cold-pass corpus - two different passes' numbers divided
against each other. The `1 / efficiency` invariant this section states
held exactly for `docs_per_s_sol` and `cost_sol` (document count and
`sol_s` both track a single pass consistently) but not for
`tokens_per_s_sol`, which came out up to 26% too high (IMDB-5: 236k
instead of 187k). Fixed by dividing `sol_s` against the same pass's
own `fresh_tokens` (`rate_src`, not `size_src`) - the table above and
`plots/sol_tokens_per_s.png` are the corrected numbers.

6 new tests (`docs_per_s_sol`/`tokens_per_s_sol`/`cost_sol` computed
correctly, the direction invariant on real validated data - SOL rate
>= measured, SOL cost <= measured - `cpu_memory_gb` read from the
suite with the documented fallback, error rows return `None`, the new
`render_sol_estimate_table()`). 241 tests pass without matplotlib
installed as of this section (6 more skip cleanly); 247 pass with it -
the jump from the 216/222 in section 19 is `main`'s multi-join test
suite, picked up by the merge in section 18/19, not new tests from
this section.

## 21. Trimming the report and simplifying the language (2026-08-25)

Two plots removed, at request: `sol_formula_subadditivity_gap.png`
("why sol_seconds aggregates before max(), not after" - section 12's
design note already covers this in text) and `torch_profiler_
unmodeled_share.png` ("what SOL's formula doesn't claim to price at
all" - section 17's discussion already covers this). Fewer figures,
same findings, still in the text.

"Ceiling" and "floor" are gone from every user-facing label in this
feature (columns, plot titles/legends, table headers) - section 20
above rewritten to say what actually happens (SOL rate >= measured,
SOL cost <= measured) instead of naming it with two different words
depending on direction. Code fields renamed to match:
`docs_per_s_sol_ceiling` -> `docs_per_s_sol`, `tokens_per_s_sol_
ceiling` -> `tokens_per_s_sol`, `cost_sol_floor` -> `cost_sol`,
`render_sol_ceiling_table()` -> `render_sol_estimate_table()`. The
established meaning of "floor" for `sol_s` itself (the time no query
can run faster than) is unchanged - that one has no direction
ambiguity to cause confusion, since time only moves one way.

The three formula-scaling plots that survive (`sol_formula_causal_
attention_scaling.png`, `sol_formula_streaming_attention_scaling.png`,
`sol_formula_projection_vs_elementwise.png`) switched from log-log
axes to linear, at request. Log-log was what made the crossover
points (the compute knee at ~416 tokens, the causal-attention
crossover at ~24,639 tokens) visible at all against the formulas'
full valid range, which spans several orders of magnitude - so each
plot's x-axis now sweeps only to about 6x its own crossover point
(causal attention to ~148k tokens, projection/elementwise to ~2,500)
instead of the full range, keeping the crossover legible on a linear
axis instead of compressed against the origin. (These three, plus
`sol_formula_component_breakdown.png`, were removed entirely in
section 22 below - "survive" here only describes this section's own
point in time.)

## 22. Removing the formula-only diagnostic plots (2026-08-26)

`reports/plot_sol_formula_diagnostics.py` and its four outputs
(`sol_formula_causal_attention_scaling.png`, `sol_formula_streaming_
attention_scaling.png`, `sol_formula_projection_vs_elementwise.png`,
`sol_formula_component_breakdown.png`) are gone. All four were
synthetic sweeps of the formula's own math at fixed Qwen3-4B/H100
constants - useful as a picture of what section 17's property tests
(`test_budgets_sol_properties.py`) already check numerically, but not
built from this PR's actual measured query results the way every
other plot in this report is. The findings they illustrated (the
compute knee, the causal-attention crossover, the two-regime scaling)
are still covered in section 17's text and enforced by the property
tests; only the standalone pictures are gone. No test depended on the
removed script. 247 tests pass, unchanged.

## 23. A third adversarial review: two real bugs, one of them the restore heuristic's last gap (2026-08-26)

A third fresh review agent - same method as sections 12 and 16, not
primed with this document's reasoning, asked to try to break the
implementation rather than confirm it - found two real bugs beyond
the `tokens_per_s_sol` cross-pass bug already fixed in section 20's
addendum.

**Bug 1 - a real zero read as "missing"**: `sol_report.py`'s
`docs_per_s_sol`/`tokens_per_s_sol` guarded with `if docs and sol_s`,
and `_sol_cost()` with `if not sol_s` - truthiness, not a `None`
check. `0` is a real answer for a numerator (a query touching 0
documents, or `sol_s`'s own designed `0.0` for an empty workload -
section 17's `chunk == 0` short-circuit, covered by a property test),
not "no data." Only `sol_s`, the divisor, needed the guard. Fixed;
verified both paths directly (`docs=0, sol_s=0.2` now reports `0.0`
rather than `None`; `_sol_cost(0.0, ...)` now returns `0.0`).

**Bug 2 - the restore-correction heuristic's remaining gap**: section
12 already flagged one edge case in the largest-first restore guess
(Section 11) as low-severity and "not reachable." This review found a
sharper, reachable one: the guess assumes restored documents are the
longest ones in the current query's own batch, but the store's length
threshold (`store_min_doc_tokens`) is recomputed per query
(`planner/decide.py`), not fixed - a short document stored under an
earlier query's lower threshold can be restored here while a long,
never-before-seen document in this same query still has to be built
fresh. The review reproduced it directly: 5 documents (one ~2,000
tokens, four ~50 tokens), simulating that the four small ones were
restored (`restored_tokens=210`). The old code removed the one large
document instead (because it's largest), and `sol_s` collapsed 85% -
far more than a 9.5%-of-tokens restore should ever discount, with no
remark, since the old code only checked "did we remove enough total
tokens," never "did we remove the right ones." Because the error only
ever makes `sol_s` too *small*, it doesn't risk a false
`SolViolation` (that only fires when `sol_s` is too large) - it just
silently understates `sol_efficiency` and misprices the SOL-derived
cost/throughput columns.

The real fix, not another guess: `executor/loop.py`'s `run_filter`/
`run_join` already compute the exact restored set (`restored = {a for
a in range(n) if skey(a) in store}`) to produce the aggregate
`restored_tokens` count - the identity was there, just thrown away.
Both functions now also report `restored_ids` (the exact global
document ids restored, mapped through `store_ids` the same way
`skey()` already does). `runtime/session.py`'s causal-build buckets
now carry `(doc_id, length)` pairs instead of bare lengths, and the
restore-correction step removes exactly the ids the store reports,
falling back to the old length-guess (now explicitly flagged as a
guess in `report["remarks"]`) only when a caller reports
`restored_tokens` with no `restored_ids` at all. A mismatched id
(reported restored but not found among the walk's own candidates)
still degrades safely and still surfaces as a remark - the same
principle section 12's low-severity fix already established, now
applied to the common case, not just the overflow case.

Two merge sites in `runtime/worker.py` and one in `runtime/
coordinator.py` aggregate per-shard/per-stage store stats with a
plain `agg[key] = agg.get(key, 0) + v` - fine for counts, but `0 +
list` raises for the new `restored_ids` field, and a plain `+` would
duplicate an id two stages both restored. Pulled into one shared
`coordinator.merge_store_stats()` (the coordinator is already "pure
dict-and-list logic, CPU-tested," per its own module docstring) that
unions `restored_ids` and adds everything else, used by all three
call sites - `runtime/worker.py`'s single-GPU join stats, its
multi-GPU child, and its multi-GPU round merge.

5 new tests: three end-to-end through `runtime/session.py` (exact-id
removal distinguishes which document was actually discounted rather
than always stripping the largest; the no-`restored_ids` fallback
lands on the same document the exact path would have picked, cross-
checked against it directly; an unmatched id surfaces its remark and
discounts nothing), plus two direct tests of `coordinator.
merge_store_stats`/`merge_filter_round` unioning `restored_ids`
instead of raising. 252 tests pass (247 + 5).

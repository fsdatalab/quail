# The operator-grid flight

Two flights on the CUDA 13 image. Flight A measures the filter
operators under mixed selectivities and rides the step-budget A/B on
the side. Flight B measures the open-ended map operators at three
decode lengths. Both record the step trace (DOCENGINE_STEPTRACE), so
every cell yields the utilization figure of
plots/make_plot_steps.py: round fill against the step budget,
documents and filters in flight, KV occupancy against the pool, and
step gaps against packing CPU.

Nothing below launches itself. One `modal run` per flight.

## Flight A: filters

    modal run experiments/modal_opgrid.py

Setup: 10,000 documents (~340 tokens each, exact pass rates from the
[FLAGS] line), 5 filters, admission budget from the plan. Three
selectivity profiles (permissive 0.9/0.9/0.9/0.8/0.8,
selective_early 0.2/0.5/0.7/0.9/0.9, cliff 0.9/0.9/0.1/0.9/0.9),
four executors each (pipelined_filter, hybrid_filter, requests,
ask_everything), 3 repetitions, interleaved in one container. A
second engine boot in the same container re-runs the permissive
profile on the old flight recipe (2,564 for both caps) against the
plan-derived boot (step budget 25,305, sequence cap bounded at
4,096: the plan's raw cap OOM'd the first boot - per-sequence
engine overheads live outside the pool accounting - and that
failure is itself a banked finding of this flight).

Banks: results/engine/opgrid_filters.json,
opgrid_trace_{new,old}.jsonl.gz.

Predictions (stated in the instrument's docstring): pipelined near
the read floor; hybrid wins where the switch rule fires mid-chain
(selective_early, cliff) and ties on permissive; requests pays the
per-request tax; ask_everything pays the skipped tails; the boot A/B
lands within 2 percent; every filter trace shows zero decode
sequences and zero heuristic evictions.

Settles: which filter operator wins per selectivity regime; whether
the old 2,048-class step budget was binding (re-anchors ROUND_TOKENS
if so); the first measured decomposition of wall minus read floor at
corpus scale (ramp, drain, c0) from the trace.

## Flight B: open-ended maps (after A lands)

Setup: 2,000 documents, 5 prompts, output caps 16, 64, 256; both map
backends (run_map per-pair, run_map_forked one-request-per-document)
at every cap, 5 repetitions, one container. Uses api.map's existing
machinery; no code changes. Reports wall against read floor plus
decode floor, the planner's gen_tokens prediction per cell, pairwise
text agreement between backends (exact-match rate and first
divergence, since greedy decode under different batching is not
guaranteed identical), and the strict eviction counter (validates
decode-token admission accounting). The stress cell is the forked
backend at cap 256: five live sequences per document with growing
decode KV.

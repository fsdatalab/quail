# Operator grid, second pass: predictions before the runs

Written 2026-08-08, before any GPU time. The goal this pass: our
engine beats stock vLLM in every completed cell (both models; filter,
classifier map, open-ended map), and no run kills the engine. Code
changes this pass: two-part map admission with per-prompt release
(engine_client.py), the decode price gained a measured per-step cost
(cost.py: 17 ms fixed + 8 us per live sequence per step), the
sequence-cap arithmetic counts every live prompt for maps
(planner.py), and a finished request no longer strips cache entries
for blocks other live requests still hold (scheduler.py).

## What is already measured and needs no GPU time

| cell | ours | stock | margin |
|---|---|---|---|
| 4B filters, permissive | 39.4 s | 78.7 s | 2.0x |
| 4B filters, selective_early | 38.4 s | 43.0 s | 1.12x (thin: gating kills most docs at stage 1 for both sides) |
| 4B filters, cliff | 39.4 s | stock crashed at boot | re-fly stock below |
| 32B filters, permissive | 272.7 s | 966-982 s | 3.5x |
| 32B filters, selective_early | 246.1 s | 325-337 s | 1.3x |
| 32B filters, cliff | 264.4 s | 691.1 s | 2.6x |
| 4B classifier (ours = ask_everything) | 49.8 s | not measured | stock cell below |
| 32B classifier (ours = ask_everything) | 297.0 s | not measured | stock cell below |
| 4B maps stock, cap 16/64/256 | - | 106.3 / 128.0 / 479.3 s | ours re-flies below |

## Predictions for the cells that fly this pass

Corrected decode model: rate = width / (compute(width) + 17 ms +
8 us x width). Decode width from the admission arithmetic, bounded
by the 4,096 sequence cap.

| cell | prediction | comparison | call |
|---|---|---|---|
| 4B map cap 16, ours re-fly | 61.9 s | stock 106.3 | win ~1.7x |
| 4B map cap 64, ours re-fly | 115.8 s | stock 128.0 | THIN win ~1.1x: decode is ~55% of the wall and decode physics is the same for both sides |
| 4B map cap 256, ours re-fly | 390.6 s | stock 479.3 | win ~1.2x IF the EngineDeadError is gone; the cell banks a traceback this time |
| 4B stock classifier (permissive) | ~87 s (47.7 prefill + 0.79 ms/request toll x 50,000) | ours 49.8 | ours wins ~1.7x |
| 4B stock filters, cliff re-fly | ~135 s (3.2x reads, pool < corpus) | ours 39.4 | ours wins ~3.4x |
| 32B stock classifier (permissive) | 1,100-1,400 s (re-read thrash: pool holds 9% of the corpus; measured 4.5x reads on gated stock) | ours 297.0 | ours wins ~4x |
| 32B map cap 16, ours | 441 s | stock est. 350-450 s | NEAR TIE possible: prefill dominates and stock's i-major submission gets prefix hits until the pool churns |
| 32B map cap 64, ours | 706 s | stock est. 650-1,500 s (depends on re-read thrash) | win if stock thrashes; tie if not. Honest uncertainty - this cell decides it |
| 32B restore (persist32) | restore ~15 s at the 4B flight's 2.7 GB/s vs recompute ~26 s | - | restore wins ~1.7x; loses if CPU-tier bandwidth < 1.6 GB/s |

Not flown this pass: 32B map cap 256 (predicted ~1,924 s of H100 for
our side alone; deferred until the 4B cap-256 crash cause is named).
Stated so the gap is visible, not silent.

## Addendum, first re-fly round (same day)

Three crash causes are now named with numbers:

- The stock 4B crashes (cliff, classifier) are the attention
  kernel's workspace: flash_attn_varlen_func tried to allocate 4.22
  GiB (cliff: 3.82 GiB) with 2.79 GiB free outside the KV pool at
  0.92 utilization. Fixes applied: expandable allocator segments
  (the scale phase's own fix for this class) and stock boots drop to
  0.88 utilization. The deviation is itself a finding: stock cannot
  run these request shapes at 0.92 - our admission is what keeps our
  boots safe there.
- The persist32 store boot died in DeepGEMM warmup with a CUDA
  "OS call failed" - the second engine boot in one container forks
  its core from a CUDA-initialized parent (the in-process flag no
  longer takes effect on this vLLM). Fix: baseline and store stages
  run in separate containers; outcome identity is checked across
  them.
- A measured negative result: the reserve-plus-top-up admission
  (quarter-cap reserve, top-up at prompt start) made cap 16 THREE
  TIMES WORSE (206.2 s against 69.3, reads 2.347 against 1.438).
  Cause: a sibling delayed behind a top-up arrives after its
  document's first prompt already finished, so the document's KV is
  gone and the sibling re-reads all of it. The shipped design keeps
  launch order untouched and instead releases each finished prompt's
  charge immediately (release-at-prompt-end, not
  release-at-document-end).

## Addendum, second re-fly round (same day)

- Halving the stock step budget to make workspace room is
  self-defeating: the boot profiler hands the saved activation room
  straight to the KV pool, and free VRAM fell from 2.79 GiB to 138
  MiB (the cliff v4 OOM). The reliable stock lever is 0.88
  utilization.
- The expandable-segments corruption theory was wrong (the fork's
  review caught it): re-runs without the flag reproduce the same
  wrong counts. The current hypothesis is parser fragility, not
  numerics: stock scores its one answer token as text, so a "Y"
  token scores as NO; our cells judge by token ids and are immune.
  Fix flying now: every executor on both models samples under
  allowed_token_ids (the 32B protocol extended to 4B), which makes
  the text parse and the id judgment identical by construction. If
  wrong counts stay high even constrained, it is real numerics and
  gets its own investigation.
- The cap-64 "regression" (159.2 s against 132.2) was host CPU
  variance, not code: the step traces are byte-identical in shape
  (same 1,676 steps, same widths, same KV peak) and only the CPU
  times inflated, uniformly, by ~1.6x. Consequence: ours-against-
  stock is only valid same-container; the pair cells re-fly that
  way.
- Cap 256 no longer kills the engine: 637.7 s at 0.90 utilization,
  KV peak 764,000 of 946,800, reads 1.36, banked trace. The
  survival goal for this cell is met; the fair wall comparison is
  the same-container pair now flying.

## Close-out: the matrix as measured (2026-08-08, end of day)

Same-container pairs where marked; every cell ran without an
engine death.

| cell | ours | stock | verdict |
|---|---|---|---|
| 4B filter, permissive | 39.4 s | 78.7 s | WIN 2.0x |
| 4B filter, selective_early | 38.4 s | 43.0 s | WIN 1.12x |
| 4B filter, cliff | 39.4 s | 48.8-50.9 s (0.88 boot; its wrong answers HALVE its gated work, biasing the wall in stock's favor) | WIN >=1.24x, conservatively |
| 4B classifier | 49.8 s | 167.8-169.8 s vs sequential client (reads 4.94); 47.5-50.3 s vs COMPETENT client (reads 1.40) | TIE against a competent client; the 3.4x was the sequential client's re-reads |
| 4B map cap 16 | 70.9 s | 106.3 s (different container; margin 1.5x is beyond host variance) | WIN 1.5x |
| 4B map cap 64 (paired; final: traceless + parse cache) | 133.4 s | 130.8 s | LOSS 2%. The first paired run (139.1) carried our step trace - measurement tax stock never paid - and 1.85M per-step id re-parses (the map-900-steps profile). With both fixed, the residual ~2 ms/step is the strict guard, priority policy, and connector presence: real cost of capabilities maps do not use |
| 4B map cap 256 (paired) | 500.5 s | 480.0 s | LOSS 4% |
| 32B filter, permissive | 272.7 s | 966-982 s | WIN 3.5x |
| 32B filter, selective_early | 246.1 s | 325-337 s | WIN 1.3x |
| 32B filter, cliff | 264.4 s | 691.1 s | WIN 2.6x |
| 32B classifier | 297.0 s | 1,253.2 s vs sequential client (reads 5.72); 308.9 s vs COMPETENT client (reads 1.40, wrong 138 - clean, so the 0.88 numerics shift is 4B-only) | PARITY against a competent client (4 percent, within host variance). Predicted 340-380: missed low, same direction as the 4B miss - the stock model was systematically pessimistic about concurrent one-token workloads |
| 32B map cap 16 (paired; cap 64 dropped by decision for iteration speed) | 356.3 s | 356.1 s | TIE. Both sides read the corpus once (reads 1.36 both) and sit at the 32B compute floor. Stock's cache coped here because the naive client submits a document's five prompts together, so each document's KV lives just long enough; the thrash that loses it the filter cells (reads 4.5x) and classifier cells (5.7x) never starts. Our engine's edge is structure the access pattern defeats, not maps whose pattern is already cache-friendly |
| 32B restore vs recompute | restore 4.04-4.16 s | recompute 27.7 s | WIN 6.8x; write-through at ingest FREE (27.83 against 27.67 cold); outcomes identical across containers and restores |

The restore number took a spec correction, not new code: the
tiering spec's file-backed /dev/shm region cannot be pinned in
this sandbox and its unpinned fallback died natively (v8); the
PLAIN CPU spec allocates its pool with pin_memory=True and runs
the channel at 10.23 GB/s end to end. At that rate the tier also
clears the 4B threshold (7.2 GB/s KV generation), so the banked
"4B loses as predicted" was a property of the broken channel, not
the tier. CONFIRMED same evening (persist4b_pin_1k_cpu.json):
restore 2.28-2.39 s against recompute 4.54 - the 4B tier wins
about 2x, at the same 10.22 GB/s channel constant as the 32B run.
The banked persist2000.json conclusion is reversed on
measurement; the ledger entry must say the old conclusion was
channel-bound, not wrong about the tier.

The two losses are the same physical fact: at 4B with real decode,
both engines pay the same per-step software floor (17 ms fixed
plus 8 us per live sequence) and the same decode width, so our
~2-second prefill saving cannot cover a 6 percent gap. The honest
paths to flipping them are named, not vague: (1) the decode width
binder - measured width sat at exactly 1,898 sequences under two
different admission schemes, so admission is not what bounds it
and the binder is unidentified; (2) the parked async-scheduler
work. Model accuracy: with measured width substituted, the
corrected decode law predicts both paired walls within 3-5
percent; predicting width is the remaining modeling gap.

Accuracy caveat carried by the two 0.88 stock cells (4B cliff,
4B classifier): the fp8 checkpoint's answers shift ~24 percent at
the 0.88 memory fraction (clean at 0.92; parser and profile
exonerated by the constrained-sampler re-runs). Walls quoted
anyway; at cliff the shift REDUCES stock's work, so the win
stands conservatively. The fraction-numerics phenomenon is its
own follow-up flight.

## The decode-width binder, resolved (2026-08-08, late)

The instrumented cap-64 cell (queue depths in the step trace)
names the whole chain:

- The engine's running set pins at max_num_seqs = 4,096 every
  step, with ~1,000 more requests waiting. Not client pacing.
- Each decode step schedules only 2,047 of the 4,096: overlapped
  scheduling packs step N+1 while N executes, so a sequence
  scheduled in N cannot join N+1, and the running set splits into
  two alternating cohorts. Effective decode width is HALF the
  sequence cap. (This also explains the alternating 1,085/1,175
  step pattern in the original cap-256 death trace.)
- The lever test: booting with max_num_seqs 8,192 at 0.90
  utilization OOMs at engine init (2.32 GiB needed, 1.01 free) -
  the per-sequence boot overheads that motivated the 4,096 clamp.
  And even where a bigger cap boots, admission bounded by the KV
  pool holds in-flight requests near 5,100, so the reachable width
  (~2,550) prices out to a TIE with stock, not a win.

Consequence: at 4B cap 64 the remaining honest path to a win is
cutting per-step software cost, not widening. Two named options:
the parked async-scheduler work (the 17 ms fixed floor), and a
map-mode fast path in DocEngineScheduler - our packing CPU runs
20.6 ms per decode step against stock's ~8-10 (25.7 s of the
139 s wall) doing pin and chain bookkeeping that maps never use;
trimming it to stock's level prices to ~124-127 s against stock's
130.8. That trim is the identified next work item, not flown.

## The client-competence audit (2026-08-08, night)

Review point (Shreya): a document-first prompt and co-submitted
questions are obvious client-side optimizations, so every claimed
win must survive the question "could a competent stock client get
this for free?" Applying it:

- Maps: the tie stands; the fusion of five independent maps into
  one document-major pass is replicable by hand. The proposed
  prompt-major sequential-maps cell was a strawman; withdrawn.
- Classifier: the 3.4x/4.2x wins measure a SEQUENTIAL client
  (questions one at a time per document; the pool churns between
  them). A competent client co-submits the five questions - the
  maps access pattern, measured reads 1.36. The stock baseline
  re-flies with that client; predictions below.
- Filters: the wins survive - gating forces sequencing, and no
  client ordering keeps a document's KV resident across stages
  from outside the engine. The best client-side strategy is
  stage-major waves, and the banked four-arm 32B run already
  measured that matchup: chain 29.3 s against waves 42.1 and
  naive streaming 41.4 - 1.4x against the strongest client.
- Restore and the tier: untouched; no client fixes cross-query
  eviction or engine restarts.

The defensible headline, final after both competent-client cells
measured: about 1.4x on gated work against the best possible
client (2-3.5x against typical ones), PARITY on classifiers and
single-pass maps against a competent client (4B classifier
47.5-50.3 against 49.8; 32B classifier 308.9 against 297.0), and
2x to 6.8x on cross-query reuse - plus the robustness claim
(stock at 0.92 crashes outright on classifier and cliff shapes;
the baselines only run because this flight diagnosed their
workspace OOM for them). Every single-query multiplier beyond
1.4x in earlier drafts measured client quality, not the engine.
The durable differentiated value is the plan's gating, the
cross-query tier, and boot correctness - not raw single-query
throughput against a well-written client.

## Re-baseline batch: predictions before any of it flies

1. Competent-client classifier, 4B - MEASURED, and the prediction
   missed by 2x: 47.5-50.3 s against ours 49.8 (predicted 80-95).
   The toll term was wrong - concurrent one-token requests overlap
   their per-request cost with prefill completely, so the
   competent classifier is a maps-cap-1 workload at the read
   floor. Verdict: the 4B classifier is a TIE against a competent
   client (cross-container, so within host variance); the 3.4x
   was entirely the sequential client's re-reads. Bonus isolation:
   wrong stayed 12,016 while reads fell 4.94 to 1.40 - the
   0.88-fraction numerics shift is independent of access pattern
   too.
2. Competent-client classifier, 32B: predicted 340-380 s against
   ours 297.0 (win ~1.2x, the reads gap 1.14 against 1.36 at the
   32B prefill rate - worth ~57 s). The 4B miss forces this
   flight per the discipline, even though this prediction rests
   on the measured maps-16 shape, not the failed toll model.
   FLYING.
3. Warm map, 4B - at CAP 16, not 64, and the arithmetic says why:
   decode is untouched by the tier, so at cap 64 the warm win
   prices to ~4 seconds against warm stock - noise. At cap 16 the
   map is prefill-dominated and the cell discriminates: ours warm
   predicted ~50 s (restore 219 GB at 10.2 GB/s replaces the 42 s
   re-read) against ours cold 70.9 and stock warm ~90-95 (its GPU
   cache retains about a third of the corpus between queries).
   Expected win ~1.8x. Constraints stated: 10,000 documents are
   required (at 4,000 the corpus fits stock's GPU cache and the
   tier is pointless), which needs a 256 GiB container for the
   219 GB CPU pool; the 32B version is capacity-blocked at 10k
   documents (389 GB) and is not flown.
4. Composed maps (map B consumes map A's output) - the design
   paragraph. Semantics: per document, stage A generates from the
   document, stage B generates from document + A's output, and so
   on; the dependency is per document, so the corpus still
   pipelines. Stock's execution: resend document + A-output +
   B-instruction as a fresh prompt per stage; the document prefix
   hits cache only if nothing evicted between stages, and the
   A-output tokens always re-prefill (cheap - outputs are short).
   Our execution: the chain machinery minus the rewind - one
   living request per document whose KV stays resident; after
   stage A finishes, the scheduler appends B's instruction at the
   CURRENT boundary instead of erasing back to the document (a
   rewind-target variant in chainlogic, small, plus a parity test
   against the resend execution before any flight). Where the win
   lives: stock re-reads the document per stage when the corpus
   exceeds the pool (2 stages at 32B: reads ~2x, predicted ~1.7x
   win; at 4B the pool covers a third and the win is ~1.2-1.4x);
   our reads stay ~1x by construction. The warm tier composes:
   with the corpus KV in the tier, even stage A's document read
   becomes a restore. Mechanism first, parity test second,
   prediction table third, one cell fourth - in that order.
5. Plan-aware graph policy, settled without a cell: extending
   capture sizes to decode width is REJECTED by the map-900-steps
   profile (the fixed step floor is Python glue, nothing
   launch-shaped - graphs cannot help). Skipping capture for
   zero-decode plans survives at honest scope: ~10 s boot and 652
   MiB headroom, no wall change, and it would NOT have rescued
   stock at 0.92 (3.44 GiB free against the 4.22 the workspace
   wanted). Its value is that engine configuration becomes a plan
   output; verify inside any batch cell via boot time and a
   memory stat, no dedicated run.

## The Nsight Systems attempt, parked with a precise verdict
(2026-08-09)

Five attempts to trace the stock filter baselines. What was
established: nsys installs and runs in the container; CUDA API
interception works (the banked nsys_4b_filter_stats.txt holds a
60-second steady-state API summary, 1,347 kernel launches); but
CUPTI's GPU ACTIVITY records never arrive - "does not contain
CUDA kernel data" - reproduced down to a trivial 50-matmul trace
that itself ran fine under the wrapper. The named mechanism (the
2026-08-09 diagnostic): driver 580.95.05 pairs correctly with
nsys 2025.3.2, but `nsys status -e` reports "Timestamp counter
supported: No" under the gVisor kernel (4.19-gvisor, paranoid 3).
RESOLVED by Modal (2026-08-10): nsys collects GPU activity
through a driver path with permission requirements their sandbox
blocks - a different path than the torch profiler, which works.
The TSC flag was a co-symptom, not the mechanism. Their guidance:
Nsight Compute (ncu) should work for per-kernel depth with
--clock-control none (undocumented perms territory; report
issues). Division of instruments going forward: torch profiler
for system-level timelines (verified working end to end), ncu
only if a per-kernel question ever needs it - the one candidate
is decomposing PHI (why achieved prefill is 97,000 of the
275,000 spec: ncu on the fp8 GEMMs would name the bound). The claim nsys was meant to
confirm is now MEASURED via the torch profiler (vLLM's in-core
hooks, profiler_config argument): the stock 4B filter baseline
runs 48,321 kernels in a 15.4-second steady-state window at 99.5
percent GPU busy (torchprof_4b_filter on the volume; slimmed
trace in results/engine). The walls-vs-floor bound said gaps
under 7 percent; the timeline says 0.5. Prediction (at or above
85 percent) confirmed. The 32B twin measured the same day: 34,275
kernels over a 16.1-second window at 99.7 percent GPU busy
(torchprof_32b_filter). Both stock filter baselines are
GPU-saturated at the kernel-timeline level; the instrument
question is closed - torch profiler for timelines, ncu (with
--clock-control none, per Modal) reserved for per-kernel depth,
nsys retired on this platform with its cause named.

## Failure hypotheses being tested

- 4B cap-256 death: CUDA memory outside the KV pool (attention
  workspace at deep decode) is favored - the pool had 21% headroom at
  death. The strict guard raising on a preemption path is second.
  The re-fly banks tracebacks either way.
- Stock boot EngineDeadError is flaky and container-independent
  (4b_stock died alone in a fresh container; the same arguments ran
  fine in the permissive container). Re-fly with captured output.
- The cap-16 reads anomaly (1.438 vs the 1.374 block floor) was the
  shared-prefix cache stripping; with the scheduler fix, reads at cap
  16 should land near 1.374 (the finish-before-sibling-prefill window
  remains and is the pin-based follow-up).

## PHI decomposed (2026-08-10)

Two instruments, one budget. ncu (microbench at the prefill
shapes, --clock-control none per Modal, bf16 nvjet kernels -
caveat: not the engine's exact fp8 path): the GEMMs run at 91-93
percent of the compute ceiling with DRAM at 28-30 - the multiplies
are near-perfect. The torch trace of the real engine (48,321
kernels, 15.4 s window) splits the wall: GEMMs 58.3 percent, fp8
quantize/scale 17.9, normalization 12.1, elementwise 7.0,
attention 4.0. So PHI = 0.35 is the step MIX, not kernel
inefficiency: the ceiling only counts multiply time, and 42
percent of the wall is connective tissue - the largest single item
being the fp8 format's own conversion kernels. Predicts PHI's
movement: attention share grows with document length; the 18
percent quantize tax follows the weight format. Open one-cell
question the table makes precise: bf16 weights at 4B trade the
quantize tax for doubled weight traffic. The live-engine ncu
attempt timed out (ncu intercepts every boot kernel; the
microbench form is the usable recipe on this platform).

Busy-fraction matrix complete (2026-08-11): stock/ours at 4B
filter 99.5/99.5, 32B filter 99.7/99.4, 4B map cap 64 41.6/39.5.
Utilization never differs between engines; only total work does -
the timeline-level form of the work-elimination thesis.

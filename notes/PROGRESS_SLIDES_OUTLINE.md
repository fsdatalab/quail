# Progress update slides, outline only

Audience is the GPU and inference experts from meeting 1. Eleven
slides in the main deck, backup slides after. The structure is:
questions from the last meeting, what we measured and how, what
the measurements show, what we built because of them, results,
and the vision they motivate.

Every number below comes from notes/OPGRID2_PREDICTIONS.md,
notes/RESULTS.md, notes/TIERED_KV_DESIGN.md, notes/PINNED_POOL_PLAN.md,
or the figures in results/plots/.

## Slide 1. Questions from the last meeting

Figure: none.

Repeat the four questions that were asked:

- What do AI SQL workloads look like?
- Where does vLLM spend time on them?
- How far are we from the hardware speed of light?
- What would an engine designed around these workloads do
  differently?

## Slide 2. The workloads we measured, and the workload study still missing

Figure: none. Two small tables on the slide.

Setup table:

- 10,000 documents, roughly 300 tokens each
- One H100 SXM 80 GB per run, on Modal
- Qwen 4B and Qwen 32B, both fp8 checkpoints

Operator matrix:

| operator | answer length | stages/prompts per doc |
|----------|--------------|----------------------|
| AI.IF (filter) | 1 token (yes/no) | 2-4 stages in order |
| AI.CLASSIFY | 1 token per question | 5 questions per doc |
| AI.MAP | 16, 64, or 256 tokens | 1 prompt per doc |

Then explicitly say:

> This is an operator stress matrix, not yet a characterization
> of production AI SQL workloads. The real-workload distribution
> is the largest unfinished item from the previous meeting.

## Slide 3. How we profiled the engine

Figure: none. A compact instrument table.

| tool | what works here | what it gave us |
|------|----------------|-----------------|
| torch profiler | runs inside vLLM engine core via profiler_config | every GPU busy fraction in this deck, decode gap attribution, kernel-class split |
| ncu | works with --clock-control none, shape-matched microbenchmarks only (live engine times out) | GEMM speed-of-light: 91-93% of peak |
| step recorder | our instrumentation, one JSON record per scheduler step | queue depths, token counts, KV occupancy, CPU scheduling time per step |
| NVML / nvidia-smi | always available | coarse GPU utilization fallback |
| nsys | unavailable: sandbox blocks its driver-path GPU activity collection (platform-confirmed) | retired here; harness kept for other hosts |

## Slide 4. Profiling reveals two different regimes

Figure: results/plots/timeline_strips.png (primary). Supporting
figure: results/plots/gpu_busy_fractions.png.

### AI.IF and classification (prefill-only)

- Answers are obtained directly from the prefill pass. No decode
  loop.
- GPU is 99.5% busy (4B) and 99.7% busy (32B). The timeline
  strip is solid color.
- Opportunity: avoid repeated document prefill. There is no GPU
  idle time to win back by scheduling.

### Open-ended AI.MAP (decode-heavy)

- Repeated decode steps.
- GPU is only about 40% busy in both stock vLLM and DocEngine.
- The gaps come from host-side step preparation and small decode
  kernels.
- DocEngine has not improved this regime yet.

## Slide 5. Speed of light versus measured throughput

Figure: results/plots/roofline_4b.png (hollow markers = hardware
spec prediction, filled markers = measured). Backup:
results/plots/phi_budget.png (full kernel-class split).

How the spec ceiling is computed:
- H100 dense fp8 peak: 1,979 TFLOP/s
- Each token costs 2P FLOPs (P = 3.6 billion parameters at 4B)
- Ceiling = 1,979e12 / (2 x 3.6e9) = 275,000 tokens/s
- No measurement is in the ceiling. It is a hardware limit.

Measured:
- 97,000 prefill tokens/s (from end-to-end wall time on
  prefill-only filter runs, not a single-kernel measurement)
- PHI = 97,000 / 275,000 = 0.35
- GEMMs alone: 91-93% of peak (ncu microbenchmark at prefill
  shapes)
- Whole-step loss is the step mix: fp8 quantize/scale 17.9%,
  normalization 12.1%, elementwise 7.0%, attention 4.0%, other
  0.7%

For decode (backup detail):
- 36,500 tokens/s measured, 15% of peak
- Split: about 60% idle (host step preparation) and the busy
  40% at about 37% of peak (inferred from 15% / 0.40, not
  counter-measured)

> Optimizing only the GEMM will not close the end-to-end gap.

## Slide 6. KV regret: retain, reload, or recompute?

Figures: results/plots/restore_vs_recompute.png and
results/plots/channel_ladder.png.

- Measured host-to-GPU restore bandwidth: 10.2 GB/s end to end
  through vLLM's pinned pool path.
- At 32B: restore takes 4.1s, recompute takes 27.7s. 6.8x
  faster.
- At 4B: restore takes 2.3s, recompute takes 4.5s. 2.0x faster.
- The threshold law: a KV tier wins only when its measured
  bandwidth beats the GPU's KV regeneration rate (7.2 GB/s at
  4B, 1.7 GB/s at 32B).
- The channel ladder shows 23x variation between spec (64 GB/s)
  and the broken file-backed path (2.7 GB/s). Cost models must
  use measured constants, re-measured per deployment.

## Slide 7. What we built in response

Figure: results/plots/rewind_schematic.png as an inset. Backup:
results/plots/admission_sawtooth.png,
results/plots/workspace_vs_free.png.

Map each measured problem to the mechanism:

| observation | mechanism |
|-------------|-----------|
| filter stages repeatedly read document tokens | KV rewind: keep one live request per doc, erase to doc boundary after each stage, append the next question onto resident KV |
| AI.IF needs only a constrained label | zero-decode execution: answer token sampled from prefill with constrained output |
| request counts do not predict memory pressure | token-based admission: plan computes a token budget, requests enter only when their tokens fit |
| safe engine settings depend on the query | plan-derived boot: step budget, sequence cap, memory fraction, graph capture chosen from the query plan |
| restore can beat recompute | cost-based KV placement: write-through to host RAM at ingest, restore over PCIe for later queries |

Why no client program can replicate these:
- Cannot erase KV inside a living request or pin its residency
- Cannot see per-step pool occupancy (only request counts)
- Boot flags are fixed before the first request
- KV evicted from the GPU pool is unrecoverable through any API

## Slide 8. End-to-end results

Figure: results/plots/results_matrix.png.

### Gated filters (the win)

Methods compared (32B, 1,000 documents, four filters at 0.8
selectivity, four methods in one container):

| method | execution | wall time |
|--------|-----------|-----------|
| KV rewind (DocEngine) | one live request per doc, erase and append | 29.3s |
| document streaming (stock vLLM) | per-doc sequential requests | 41.4s |
| stage major waves (stock vLLM) | all docs through F1, survivors through F2, etc. | 42.1s |

The result:

> KV rewind is 1.4x faster than the best stock vLLM submission
> strategy. Gating forces sequencing that no submission order
> fixes. Only keeping KV inside a living request eliminates
> the rereads.

At 10,000 documents, the opgrid matrix ran KV rewind against
stock vLLM (the document-streaming strategy was stock's best at
that scale). The same 1.4x ratio held at 32B.

### Independent classifications (parity)

| method | execution | wall time (4B) |
|--------|-----------|---------------|
| document major co-submission (stock vLLM) | submit all 5 questions per doc together | 47.5-50.3s |
| DocEngine classification operator | plan-owned scheduler, constrained 1-token answer | 49.8s |
| sequential questions (stock vLLM) | one at a time, wait between | 167.8-169.8s |

The sequential strategy is a weak baseline. Against document
major co-submission, DocEngine is at parity. Show the sequential
result only as a diagnostic of how much submission order matters.

### Open-ended maps (parity)

| method | execution | wall time (4B, cap 64) |
|--------|-----------|----------------------|
| stock vLLM, document major | submit prompts together with bounded concurrency | 130.8s |
| DocEngine map | token-based admission, plan-derived boot | 133.4s |

Essentially tied. DocEngine's current overhead is about 2 ms per
step from the strict memory guard and connector bookkeeping.

### Cross-query KV restore (the other win)

- 4B: 2.0x faster (2.3s restore vs 4.5s recompute)
- 32B: 6.8x faster (4.1s restore vs 27.7s recompute)
- Write-through at ingest is free (27.83s vs 27.67s cold)

### Summary

- About 1.4x on gated filters against the best stock strategy
- Parity on classifiers and single-pass maps
- 2-6.8x on cross-query KV restore
- Admission lets a 256-token map finish where the engine
  previously crashed

## Slide 9. The measurements point to declarative physical operators

Figure: results/plots/system_block_diagram.png.

The vision, derived from the measurements:

> The user declares the query and deployment. The system selects
> a physical operator that specifies how inference and KV state
> should be executed.

Inputs:
- Query and data
- Model family and size
- GPU type and count
- Available host memory and storage tiers

Physical plan outputs:
- Prefill label, bounded decode, or variable decode
- KV disposition: rewind, fork, retain, restore, spill, discard,
  or recompute
- Admission and scheduling policy
- Data parallelism, tensor parallelism, and eventually
  prefill/decode placement

## Slide 10. What remains before we can claim the full vision

Figure: none.

Incomplete against the previous meeting's todos:
- Characterize real AI SQL workloads and operator fractions (the
  largest gap)
- Run Qwen 27B and Liquid 1B
- Measure H200, L40S, and RTX PRO 6000
- Produce a clean "actual versus infinite-cache/perfect-schedule"
  regret metric for every workload

Engineering next steps:
- Build bounded or structured decode for AI.EXTRACT
- Evaluate composed queries with multiple operators (code is
  built, parity test still gated)
- Reconcile tiered KV restore with the plan-owned scheduler
  (MultiConnector integration)

Longer-term parking lot (not in scope now):
- Speculative decoding for long decode
- Shared-prefix attention kernels (Hydragen-class)
- Model cascades
- Mamba-class models (fixed-size state breaks the KV threshold
  law)

## Slide 11. Asks

Figure: none.

- Check our roofline accounting (is splitting PHI by
  kernel-class share the right decomposition)
- Experience getting nsys GPU activity or pinned file-backed
  mappings inside a gVisor sandbox
- Whether H200 or L40S constants are worth measuring first
- The benchmark build with Arnav

## Backup slides

### B1. Admission detail

Figure: results/plots/admission_sawtooth.png.

- Token budget: 750,000 of 946,800 pool.
- KV cycles under the budget, no eviction, no OOM.
- Negative result: reserve + top-up made cap-16 maps 3x worse
  (206.2s vs 69.3s) because delayed siblings lose their
  document's KV.

### B2. Boot-memory crash mechanism

Figure: results/plots/workspace_vs_free.png.

- Flash-attention workspace demands 4.22 GiB outside the KV pool.
- At gpu_memory_utilization 0.92: only 2.79 GiB free. Crashes.
- At 0.88: 5.99 GiB free. Survives.
- 4B fp8 accuracy shifts about 24% with the boot fraction.

### B3. KV rewind block alignment detail

Figure: results/plots/rewind_schematic.png (larger version).

- vLLM's 16-token block is the allocation unit, not the validity
  unit. Within a living request, KV validity is tracked by token
  position.
- Rewind keeps every whole block below the document boundary and
  the straddling block. Destroys the straddling block's
  prefix-cache hash entry so stale content never serves a cache
  hit.
- Block misalignment costs only across requests (the measured
  1.374x read floor in maps), never within the living request.

### B4. PHI kernel-class decomposition

Figure: results/plots/phi_budget.png.

- GEMMs 58.3%, fp8 quantize/scale 17.9%, normalization 12.1%,
  elementwise/activation 7.0%, attention 4.0%, other 0.7%.
- Measured on stock vLLM, 4B filter, 48,321 kernels in a
  15.4-second window. Our engine's window shows the same mix.

### B5. Decode step cost model

- rate = width / (compute(width) + 17ms + 8us x width)
- compute(width) = 2P x width / (PHI x R_D)
- Constants fitted from a single step trace: 4B map cap-64,
  1,899 live sequences, 52.1 ms per step = 19.6 ms compute +
  15.2 ms packing CPU + 17.3 ms fixed host overhead
- Predicts the paired cap-64 walls within 3-5% when measured
  width is supplied. Predicting width from the query plan is
  the remaining modeling gap: the width turned out to be set by
  vLLM's overlapped scheduling (half of max_num_seqs = 2,047),
  not by our admission policy.

### B6. Step trace: the survival run

Figure: results/plots/steps_4b_map256_survived.png.

- 4B open-ended map, cap 256, 10,000 documents, 637.7s, no
  engine death.

### B7. The overlapped-scheduling cohort split

Figure: results/plots/steps_4b_map64_widthbinder.png.

- 4,096 sequences in the engine, but each step schedules only
  about 2,047
- While GPU step N runs, the CPU prepares step N+1. A sequence
  in N cannot join N+1, so the running set splits into two
  alternating groups.
- This is a diagnosis of the open-ended map workload, not an
  AI.IF finding (AI.IF has no decode loop).
- Raising max_num_seqs to 8,192 OOMs at boot (2.32 GiB needed,
  1.01 GiB free).

### B8. The copy.deepcopy receipt

- 22% of baseline CPU time was vLLM deepcopying SamplingParams
  per request.
- Bypassed with skip_clone=True on every request we issue.

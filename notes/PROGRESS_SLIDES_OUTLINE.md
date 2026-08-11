# Progress update slides, outline only

Audience is the GPU and inference experts from meeting 1. Fourteen
slides. Every number below comes from notes/OPGRID2_PREDICTIONS.md,
notes/RESULTS.md, notes/TIERED_KV_DESIGN.md, notes/PINNED_POOL_PLAN.md,
or the figures in results/plots/. Roofline framing follows the
"Inference Engineering" book (arithmetic intensity of the workload
against the hardware's ops-per-byte ratio).

Naming used throughout this outline. A driver is the client program
that submits requests to the serving engine. The two baselines run
the identical stock vLLM engine and differ only in submission
strategy: "vLLM with a basic driver" asks a document's questions one
at a time, and "vLLM with an optimized driver" puts the document
first in every prompt, submits a document's independent questions
together, and bounds open requests. Our four methods are "in-engine
gated chain with KV rewind", "plan-derived admission", "plan-derived
boot", and "tiered KV restore". Slide 2 carries the table mapping
repo terms to these names. The two baselines get their own slides
(6 and 7) before the four method slides (8 to 11).

## Slide 1. Progress update on LLM data operators over vLLM

Figure: none.

- One-line reminder of the project. We run SQL-style queries (filter,
  classify, map) over document collections by scheduling the work
  inside vLLM instead of through its client API.
- What this deck covers. You told us to stop trusting theoretical
  defaults and to profile, so the middle of the deck shows exactly
  how we profiled and what each measurement returned, with the math.
- The results summary up front. Single-query throughput is near
  parity against vLLM with an optimized driver. The wins that
  survive are cross-query KV reuse (up to 6.8x) and not crashing
  where stock vLLM crashes.
- The deck ends with the two baselines and the four methods behind
  those wins, the proposal they feed, and the asks.

## Slide 2. The setup, the memory sizes, and the names used in this deck

Figure: none. Two small tables on the slide, the setup numbers and
the naming table below.

- Corpus and hardware. 10,000 documents of about 300 tokens each, on
  one H100 SXM 80 GB per run, on Modal. Models are Qwen 4B and Qwen
  32B, both fp8 checkpoints.
- The three operator types. Filters are 2 to 4 yes/no stages applied
  in order with one-token answers, in three selectivity regimes
  ("permissive" passes most documents, "selective_early" removes
  most at stage 1, "cliff" concentrates the removal late).
  Classifiers ask 5 one-token questions per document, 50,000
  requests per query. Open-ended maps generate free text capped at
  16, 64, or 256 tokens per document.
- Engine budgets when they matter. The KV pool holds 946,800 tokens
  at 4B in the map runs, our admission budget is 750,000 of that,
  and the sequence cap is 4,096. KV means the KV cache, the model's
  stored per-token state, 73,728 bytes per token at 4B and 131,072
  at 32B.
- Memory terms used throughout. HBM is the GPU's on-package memory,
  80 GB here. "DRAM" in our tables means host CPU memory. For scale,
  at 10,000 documents the 4B corpus KV is 219 GB, about 3 times the
  GPU pool, and fits host memory. The 32B corpus KV is 389 GB or
  more and fits neither on one machine.
- One term to define before the naming table. A driver is the client
  program that submits requests to the serving engine. The engine is
  identical in both baselines; only the submission strategy differs.
- The naming table, since our repo's short names mean nothing
  outside it:

| repo term | deck term | one-line meaning |
|---|---|---|
| sequential client | vLLM with a basic driver | asks a document's questions one at a time, waiting for each answer; slide 6 |
| competent client | vLLM with an optimized driver | document first in every prompt, a document's questions submitted together, bounded open requests; slide 7 |
| pipelined policy (code name "chain") | in-engine gated chain with KV rewind | the filter executor; slide 8 |
| admission budget | plan-derived admission | the token-budget gate on what enters the engine; slide 9 |
| boot config from the plan | plan-derived boot | step budget, sequence cap, memory fraction, and graph capture chosen from the query plan; slide 10 |
| persist/restore | tiered KV restore | the CPU-memory KV cache written at ingest and restored by later queries; slide 11 |

## Slide 3. What you told us to do, and what we did

Figure: none. Two-column table, todo from meeting 1 on the left,
what we did on the right.

- Representative token counts and per-request metrics with rollups.
  Done. Documents are about 300 tokens, filter and classifier answers
  are one token, map answers are 16 to 256 tokens. A step recorder
  saves per-step token counts, queue depths, and KV in use, and the
  rollups are wall time, corpus read multiplier (1.40 for the
  optimized driver against 4.94 to 5.72 for the basic driver on
  classifiers), and per-step CPU cost.
- Kernel-level profiling of vLLM runs. Done, with one platform
  detour worth a single bullet. The torch profiler runs inside the
  vLLM engine core through the profiler_config argument, over
  15-second steady-state windows, with kernels binned by class. nsys
  intercepts the CUDA API on Modal but its GPU-activity records never
  arrive (Modal confirmed their sandbox blocks that collection path).
  ncu works only as a microbenchmark at the prefill shapes with
  --clock-control none, because on a live engine it intercepts every
  boot kernel and times out. nvidia-smi/NVML is the coarse fallback.
- Scheduling regret and gaps in GPU utilization. Done, with a
  mechanism. A decode step pays a measured 17 ms fixed cost plus 8
  microseconds per live sequence, and overlapped scheduling halves
  the effective decode width (width is the number of sequences
  generating together in one engine step). Details on slide 5.
- Speed-of-light and roofline latency estimates. Done. The achieved
  fraction of peak for 4B prefill is 97,000 of 275,000 tokens per
  second, and slide 5 decomposes the gap with ncu plus the torch
  trace.
- Overhead multiplier from a linear model over empirical runs. Done
  as the decode law rate = width / (compute(width) + 17 ms + 8 us x
  width). With measured width substituted it predicts the paired
  walls within 3 to 5 percent. Predicting the width itself is the
  remaining modeling gap.
- Models and GPUs (Qwen 4B and 27B-class fp8, H100/H200/L40S).
  Partially. 4B and 32B fp8 are measured on H100. L40S is in our
  device config but unmeasured, and H200 is deferred.

## Slide 4. Filters saturate the GPU and open-ended decode does not

Figure: results/plots/timeline_strips.png as the primary visual (one
second of GPU timeline; the filter row is solid, the map row is
broken by white strips, and the white strips are the host step
floor). Supporting figure: results/plots/gpu_busy_fractions.png (the
four busy fractions as bars).

- Method. Torch profiler inside the engine core, roughly 15-second
  steady-state windows, GPU busy fraction is the share of the window
  covered by kernels.
- The stock 4B filter baseline runs 48,321 kernels in a 15.4-second
  window at 99.5 percent GPU busy. The 32B twin runs 34,275 kernels
  in 16.1 seconds at 99.7 percent.
- The open-ended map (4B, 64-token cap) sits at 41.6 percent GPU
  busy for the stock engine with the optimized driver and 39.5
  percent for our engine. Both leave the GPU idle almost
  identically, so the idle time is the workload's physics, not an
  engine difference. The idle share is host software, the step
  floor that slide 5 measures and splits.
- Consequence for everything after this slide. A filter baseline at
  99.5 percent busy has no idle time to win back by scheduling. The
  only way to beat it is to compute fewer tokens.

## Slide 5. The roofline, spec against measured, and the host step floor

Figure: results/plots/roofline_4b.png (hollow markers are where the
hardware spec puts each phase, filled markers are what we measured).
Supporting figures: results/plots/steps_4b_map64_widthbinder.png
(queue depths and the packing-CPU panel) and the top two panels of
results/plots/board_map64_ours.png (kernel timeline plus the host
CUDA-API panel). Backup slide: results/plots/phi_budget.png with the
full kernel-class split.

- What the roofline lines are. The two lines are theory set by
  exactly two spec-sheet numbers: the peak math rate (the flat roof,
  1,979 TFLOP/s dense fp8) and the memory bandwidth (the slanted
  part, 3.35 TB/s). No measurement is in the lines. They are
  ceilings, not predictions, which is the whole argument for
  measuring. They meet at the ridge, 591 FLOP per byte. Hollow
  markers show where the spec puts each phase, filled markers show
  what we measured, and the spec puts both prefill and decode at
  width 2,047 on the roof.
- Width, defined once. Width is the number of sequences generating
  together in one engine step. Decode makes exactly one token per
  sequence per step, so width 2,047 means 2,047 documents advanced
  one token that step. Width sets decode's operations per byte,
  because the same loaded weights serve every sequence in the step,
  and it divides the per-step host cost.
- Prefill, spec against measured. The GEMMs alone run at 92 percent
  of peak (ncu microbenchmark at the prefill shapes, --clock-control
  none), but the whole prefill step achieves 97,000 tokens per
  second, 39 percent of peak. The achieved fraction of peak
  throughput (written PHI in our code) is 97,000 of 275,000 tokens
  per second. The gap is the step mix, led by fp8 quantize and scale
  at 18 percent of the wall and normalization at 12 (full
  kernel-class split, with GEMMs at 58.3 percent, on the backup
  slide).
- Decode, spec against measured, in two measured parts, and the
  idle part dominates. Achieved decode is 36,500 tokens per second,
  15 percent of peak. During decode the GPU executes kernels only
  about 40 percent of the wall (busy fractions 39.5 percent for
  ours, 41.6 for the optimized-driver baseline), and that idle 60
  percent is the host step floor, 17 ms fixed plus 8 microseconds
  per live sequence. The busy 40 percent itself averages only about
  37 percent of peak (15 percent overall divided by 0.40 busy),
  because decode-shaped work (small matrices, tiny kernels,
  attention reading KV) is intrinsically less efficient than
  prefill's large multiplies.
- Caveat on that split. The ncu microbenchmark measured
  prefill-shaped multiplies only, so the 37 percent during-busy
  figure is inferred from the busy-fraction arithmetic, not
  counter-measured. A decode-shaped ncu pass would pin it.
- The floor's mechanism, which is your "scheduling regret" ask.
  Overlapped scheduling packs step N+1 while step N executes, so the
  running set splits into two alternating groups and each step
  schedules 2,047 of the 4,096 running sequences. Effective width is
  half of max_num_seqs, and raising the cap to 8,192 fails boot with
  an out-of-memory error (2.32 GiB needed, 1.01 free). The host
  CUDA-API panel sits near zero during the idle gaps, so the gaps
  are scheduler Python, not launch traffic, and our packing code
  costs 20.6 ms of CPU per decode step against stock's 8 to 10. The
  decode law rate = width / (compute(width) + 17 ms + 8 us x width)
  predicts the paired walls within 3 to 5 percent once measured
  width is substituted.

## Slide 6. Baseline 1. vLLM with a basic driver

Figure: none.

- Definition, repeated from slide 2 because the section starts here.
  A driver is the client program that submits requests to the
  serving engine. Both baselines run the identical stock vLLM
  engine; only the submission strategy differs.
- What the basic driver does. For each document it asks the
  questions one at a time, waiting for each answer before sending
  the next. Between a document's questions the engine serves other
  documents, so the prefix cache evicts the document's KV.
- Measured. It re-reads the corpus 3.2 to 5.7 times, because each
  returning question re-prefills its document. On the 32B classifier
  that costs 1,253.2 seconds, compared with 308.9 seconds for the
  optimized driver on the same engine.

## Slide 7. Baseline 2. vLLM with an optimized driver

Figure: none.

- What the optimized driver does. It puts the document first in
  every prompt so shared prefixes can hit the prefix cache, submits
  a document's independent questions together so they arrive while
  the document's KV is still resident, and bounds the number of open
  requests. It is everything a careful engineer writes without
  touching the engine.
- Measured. It reads the corpus 1.36 to 1.40 times, and it ties us
  on classifiers (47.5 to 50.3 seconds against our 49.8 at 4B, 308.9
  against our 297.0 at 32B) and on single-pass maps.
- Where it stops. For gated filters its best strategy is stage-major
  waves (all documents through stage 1, then the survivors through
  stage 2), measured at 41.4 to 42.1 seconds at 32B, compared with
  our 29.3. No submission order keeps a document's KV resident
  across stages, and no driver recovers KV after eviction or across
  engine restarts. The four method slides live in exactly those
  gaps.

## Slide 8. Method. In-engine gated chain with KV rewind

Figure: none. A small drawn diagram fits here (document KV kept
resident, question-and-answer KV erased back to the boundary, the
straddling block half-kept).

- What it does. The filter executor keeps one living request per
  document. Each stage's question is appended onto the document's
  resident KV, and the one-token answer is judged at prefill by
  token id, so decode never happens.
- After a stage is judged, the question-and-answer KV is erased back
  to the document boundary, and for a surviving document the next
  question appends onto the same resident document KV. A document
  that fails a stage is dropped without further work.
- Block alignment, for the systems people in the room. vLLM's
  16-token block is the allocation unit, not the validity unit;
  within a living request, KV validity is tracked by token position.
  The rewind keeps every whole block below the document boundary and
  also keeps the block the boundary lands in, because its slots
  below the boundary are still valid. It destroys that straddling
  block's prefix-cache hash entry so stale content never serves a
  cache hit, and the next question writes into the straddling
  block's remaining slots.
- Block misalignment therefore costs only across requests, where
  prefix matching is whole-block (the measured 1.374x read floor in
  maps), and never within the living request.
- Why no driver can replicate it. A driver submits whole prompts. It
  cannot hold a document's KV resident across stages and cannot
  erase part of a request's KV. Its best strategy is the stage-major
  waves from slide 7, which depend on the prefix cache still holding
  each document and fail once the corpus exceeds the pool.
- Measured effect. Corpus reads are 1.14x, compared with 3.2 to 4.6x
  for the driver strategies. At 32B the wall is 29.3 seconds,
  compared with 41.4 seconds for the best driver strategy, about
  1.4x.

## Slide 9. Method. Plan-derived admission

Figure: the KV-tokens panel of results/plots/board_map64_ours.png
(KV in use stays under the 750,000 admission line while the pool
holds 946,800).

- What it does. Before any request is submitted, the plan computes
  how many KV tokens the query may hold in the engine at once
  (750,000 of the 946,800-token pool in the map runs). A request
  enters only when its tokens fit the budget, and a finished prompt
  releases its share immediately.
- Why no driver can replicate it. Bounded concurrency counts
  requests, not tokens, and a driver cannot see the pool's per-step
  occupancy. It either underfills the GPU or overfills it, and
  overfilling evicts KV that live requests still need or crashes the
  boot.
- Measured effect. The 256-token map that previously killed the
  engine finished at 637.7 seconds with a KV peak of 764,000 of
  946,800, and the final matrix ran with no engine deaths.
- A negative result we kept. Reserving budget and topping up per
  prompt made the 16-token map three times worse (206.2 seconds
  against 69.3), because a delayed sibling arrives after its
  document's KV is already gone. The shipped design leaves launch
  order alone and only releases early.

## Slide 10. Method. Plan-derived boot

Figure: none.

- What it does. The plan sets the engine's boot parameters from the
  query before launch: the step token budget (25,305 in the map
  runs), the sequence cap (4,096), the GPU memory fraction, and
  whether to capture CUDA graphs at all. Zero-decode plans skip
  capture, saving about 10 seconds of boot and 652 MiB of headroom
  with no wall change.
- Why no driver can replicate it. Boot flags are fixed before the
  first request exists, and the safe values depend on document
  counts, token counts, and answer lengths that only the query plan
  knows in advance. Hand-tuning finds them by crashing.
- Measured effect. Stock vLLM at 0.92 GPU memory utilization crashes
  on the classifier and cliff shapes, because the flash-attention
  workspace wants 4.22 GiB while only 2.79 GiB is free outside the
  KV pool. Our boots run at 0.92 on every shape in the matrix, and
  the stock baselines in this deck only run because we diagnosed
  that crash and dropped them to 0.88.
- A trap recorded for accuracy work. The 4B fp8 checkpoint's answers
  shift about 24 percent with the boot memory fraction (clean at
  0.92, shifted at 0.88), so accuracy comparisons must pin the
  fraction.

## Slide 11. Method. Tiered KV restore, and the measured channel behind it

Figure: the channel table from notes/TIERED_KV_DESIGN.md (tier, spec
bandwidth, measured bandwidth) on the slide. A two-bar comparison
(restore against recompute, both models) would be drawn new.

- What it does. At ingest the engine writes each document's KV
  through to a pinned host-memory pool while the first query
  computes, which costs 0.16 seconds, effectively free. A later
  query restores that KV over PCIe at 10.2 GB/s instead of
  re-reading the documents.
- Why no driver can replicate it. KV evicted from the GPU pool is
  unrecoverable through any client API, and nothing a driver does
  carries KV across engine restarts.
- Measured effect. At 32B, restoring 41.3 GB of KV (315,354 tokens)
  takes 4.1 seconds, compared with 27.7 seconds to recompute it, a
  6.8x win with bit-identical outcomes across containers. At 4B the
  restore is 2.3 seconds against 4.5, a 2x win, at the same 10.2
  GB/s channel constant.
- The channel, spec against measured. PCIe Gen5 promises 64 GB/s.
  Measured on the same containers: 55.4 GB/s for alloc-pinned
  memory, 10.2 GB/s end to end through vLLM's plain CPU offload path
  with a pinned pool, and 2.7 GB/s through vLLM's tiering path,
  which backs its pool with a /dev/shm file this sandbox cannot pin
  (cudaHostRegister returns error 304 for file-backed mappings).
  Pinning is worth 5.7x of the roughly 20x gap, and lazy page faults
  plus per-block transfer fragmentation cost the remaining 3.6x.
- The threshold law, and a reversed conclusion. A KV tier wins only
  when its measured bandwidth beats the rate at which the GPU
  regenerates KV by re-reading documents, 7.2 GB/s at 4B and 1.7 at
  32B. Through the broken 2.7 GB/s path the 4B tier lost and we
  recorded that as a conclusion; at 10.2 GB/s it reversed. Cost
  models must use measured constants, re-measured per deployment
  image, never spec numbers.

## Slide 12. Results against both baselines, and what the wins measure

Figure: the result matrix as a table on the slide. Optional
mechanism figure: panel b of results/plots/docengine_story.png (the
corpus read multiplier).

- Protocol. Paired cells run in one container, because host variance
  moved a wall 20 percent on byte-identical work, and they run
  untraced, because our step trace costs measurement time the
  baseline never pays.
- Against vLLM with a basic driver. Filters win 2.0x, 1.12x, and
  1.24x at 4B (permissive, selective_early, cliff) and 3.5x, 1.3x,
  and 2.6x at 32B. Classifiers appear to win 3.4x and 4.2x.
- Against vLLM with an optimized driver, the honest bar. The
  classifier wins vanish, because the optimized driver reads the
  corpus 1.40 times instead of 4.94 to 5.72 and lands at parity
  (47.5 to 50.3 seconds against our 49.8 at 4B, 308.9 against our
  297.0 at 32B). Maps are also parity (133.4 against 130.8 seconds
  at the 64-token cap, and a 356.3-against-356.1 tie at 32B with
  identical reads of 1.36).
- Filters survive the optimized driver at about 1.4x (29.3 seconds
  against 41.4), because gating forces sequencing that no
  submission order fixes. Only the in-engine gated chain with KV
  rewind keeps the document KV resident across stages.
- The pattern. Our speedups track the corpus read multiplier almost
  exactly, so all of the speedup comes from computing fewer tokens,
  and none of it comes from making any single token faster. Against
  a baseline that is 99.5 percent GPU busy, work elimination is the
  only possible win, and the query plan is the one component that
  knows before execution which prefill work would repeat.
- The defensible summary. About 1.4x on gated filter work against
  the optimized driver (2 to 3.5x against the basic one), parity on
  classifiers and single-pass maps, 2x to 6.8x from tiered KV
  restore, and boots that survive request shapes that crash stock at
  0.92.

## Slide 13. The proposal. A declarative execution engine for LLM data operators

Figure: none. A block diagram would be drawn new; no existing figure
fits.

- The system. An open-source declarative query execution engine for
  LLM data operators, implemented as a custom scheduler plus memory
  manager over vLLM. The user declares GPU type and count, model
  (qwen, deepseek, and llama3 families), CPU DRAM size, and the
  query with its data.
- The operator language. Filter, join, extract, summarize, and
  sort/rank. Each operator compiles to a decode strategy
  (zero-decode for filter, classifier, and join, because the answer
  token is known at prefill; fixed-length for extract; variable
  otherwise), a KV disposition (keep, spill, or discard), and a
  prefill/decode placement.
- The memory and scheduling layer. A tiered KV manager across GPU
  HBM, host DRAM, RDMA, and NVMe, run with buffer-pool discipline
  (fixed budgets, explicit admission, explicit eviction). Cost
  models choose the physical variant and keep-versus-recompute per
  document from measured channel constants, not spec sheets, and a
  plan-aware scheduler reorders work across operators to get reuse
  in before eviction.
- Benchmarks. Against Kalypso, naive vLLM, LOTUS and Palimpzest, and
  hosted LLM APIs, at 10,000 to 1,000,000 documents.
- Scope, justified by slide 4. The engine is optimized for
  prefill-heavy operators. Long open-ended maps are deliberately
  deferred, because both engines leave the GPU about 60 percent
  idle there and fixing that is decode-side work, not planning
  work.

## Slide 14. Asks and next steps

Figure: none.

- Next runs, in order. The warm-map cell at 4B with the 16-token cap
  (predicted about 50 seconds, compared with 70.9 cold and 90 to 95
  for warm stock; needs a 256 GiB container for the 219 GB CPU
  pool). Composed operators, where map B consumes map A's output,
  with a parity test before any timing run. Reconciling the tiered
  KV restore connector with our scheduler's plan-owned memory via
  vLLM's MultiConnector.
- The bf16-weights cell at 4B, with corrected framing. Weight
  capacity was never the issue, because the 4B weights are 8 GB in
  bf16 and always fit HBM. The tradeoff is per-step weight traffic
  against the 18 percent quantize tax, and at prefill's arithmetic
  intensity the traffic is amortized over about 25,000 tokens per
  step, so bf16 weights likely win for prefill-heavy 4B work.
- The benchmark build with Arnav.
- Asks for this group. Check our roofline accounting (is splitting
  the achieved fraction of peak by kernel-class share the
  decomposition you would use). Any experience getting nsys GPU
  activity or pinned file-backed mappings inside a gVisor sandbox.
  Whether H200 or L40S constants are worth measuring first.
- Follow-up paper parking lot, deliberately not in scope now:
  speculative decoding for long decode, Hydragen-class shared-prefix
  attention kernels, model cascades, and mamba-class models (their
  fixed-size state breaks the KV threshold law).

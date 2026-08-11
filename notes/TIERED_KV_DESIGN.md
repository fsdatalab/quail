# Tiered KV store: design and cost model

Written 2026-08-08. KV means the KV cache: the model's stored
per-token reading state, kappa bytes per token (73,728 at 4B,
131,072 at 32B, from configs.py). This document says where KV can
live, what moving it costs in our deployment, and how the planner
decides placement from the query plan.

## Tiers and channels

Four places KV can live, one channel to each. Spec numbers are
ceilings; the "measured here" column is what our deployment
actually moves, and it is the only column the cost model may use.

| tier | capacity | spec bandwidth | measured here | source |
|---|---|---|---|---|
| GPU HBM pool | 969,872 tokens at 4B, 318,320 at 32B (0.92 of 80 GB H100) | 3.35 TB/s | not the binder: reads are compute-bound at 97,889 tokens/s (4B), 12,611-13,123 (32B) | speed_limit.json, xengine.json, model32_1k.json |
| CPU DRAM tier | container RAM, requestable to ~330 GiB | 64 GB/s (PCIe Gen5 x16) | 55.4 GB/s alloc-pinned ceiling; 10.23 GB/s end to end through vLLM's PLAIN CPU spec (pinned pool - the shipped fast path once the right spec is chosen); 9.7 GB/s raw unpinned; 2.5-2.9 GB/s through the tiering spec's file-backed region (unpinnable here) | pinprobe.json, persist32_1k_cpu.json, persist2000.json |
| container-local NVMe | ~100 GB class | ~7 GB/s per Gen4 drive | 5.19 GB/s raw write; restore-through-disk not separately measured | persist2000.json disk probe |
| Modal volume / object storage | unbounded | 1-3 GB/s per client claimed | UNMEASURED - needs a calibration cell | - |

Channels not in the table: RDMA (network cards writing directly
into another machine's memory) does not exist on our single-node
Modal deployment; it enters only with prefill/decode
disaggregation, which is deferred below.

## The threshold law

A tier is worth reading from when its measured read bandwidth
exceeds the rate at which the GPU can regenerate KV by re-reading
the documents. That regeneration rate is kappa times the calibrated
prefill rate, which is kappa * PHI * R_D / (2 * P_active):

- 4B: 73,728 bytes/token * ~97,000 tokens/s = about 7.2 GB/s.
  Through the broken (tiering-spec) channel at 2.5-2.9 GB/s the
  tier lost 2 to 1, as the ratio predicted (persist2000.json:
  restore 18.4-20.3 seconds against recompute 10.5). Through the
  pinned channel the SAME tier wins: restore 2.28-2.39 seconds
  against recompute 4.54 (persist4b_pin_1k_cpu.json, 10.22 GB/s -
  the channel constant matches the 32B run's 10.23, model-
  independent as the model requires). One caveat only at 4B: the
  store engine's repeated queries disagreed on a few survivors
  (outcomes_identical_within_store false), the known borderline-
  call noise of this checkpoint; the 32B run was fully identical.
- 32B: 131,072 bytes/token * ~12,900 tokens/s = about 1.7 GB/s.
  MEASURED (persist32_1k_cpu.json, plain CPU spec with the pinned
  pool): restore 4.04-4.16 seconds against recompute 27.7 for
  315,354 tokens (41.3 GB of KV) - the tier wins 6.8x, the
  channel runs 10.23 GB/s end to end, the write-through at ingest
  costs 0.16 seconds (noise), and outcomes are identical across
  containers and restores. At 10.23 GB/s the tier also clears the
  4B threshold; the 4B confirmation cell (persist4b_pin) reverses
  or confirms the banked "4B loses" - which was measured through
  the broken channel.

The threshold moves with the model, not with the corpus size:
recompute cost and transfer cost both scale linearly in tokens, so
their ratio is a per-model constant until attention's quadratic
term matters (far beyond our 300-token documents).

## Why spec numbers are unusable: the 20x gap

PCIe Gen5 is 64 GB/s on paper and our measured CPU-tier restore is
2.5-2.9 GB/s, about 20x under spec. Three known contributors, in
this deployment:

- No pinned pages on this path. The platform itself is fine: the
  probe (pinprobe.json) allocates pinned memory (cudaHostAlloc)
  and moves 55.4 GB/s host to GPU, and a 48 GB pinned allocation
  succeeds. What fails is narrower: cudaHostRegister returns 304
  only for FILE-BACKED mmaps - including files on /dev/shm - while
  anonymous mmaps register fine. vLLM's connector mmaps a /dev/shm
  file (shared_offload_region.py) for cross-process sharing, so it
  hits exactly the unsupported case, falls back to unpinned DMA,
  and on top of that leaves the failed call's CUDA error uncleared
  so the next kernel launch dies; we patch the installed file
  in-container to clear it (modal_scale.py, persist32_run).
- Lazy page faults. The store's region pre-fault call
  (MADV_POPULATE_WRITE) is unsupported here, so pages fault in one
  at a time on first touch, serializing the early transfers.
- Fragmented transfers. KV moves per block (16 tokens, ~1-2 MB at
  32B but ~64 KB per layer slice at 4B), and per-operation costs
  do not shrink with the operation.

The probe decomposes the 20x: pinning is worth 5.7x (55.4 over
9.7), and page faults plus fragmentation cost the remaining ~3.6x
(9.7 over 2.7). The consequence is large: at 55.4 GB/s the CPU
tier clears even the 4B threshold (7.2 GB/s) with room to spare,
so an alloc-pinned host pool (the HiCache pattern, or an upstream
vLLM change to an anonymous shared mapping for single-node) makes
the tier viable for every model we serve, not only the 32B. Until
one of those exists, the cost model prices the channel at the
as-shipped 2.5-2.9 GB/s. The rule stands: every channel constant
is measured in deployment with one cheap calibration cell, the
same discipline as PHI for compute (97,000 over 275,000 on this
image; a different image or host gets a different constant and
must be re-measured).

## Cost-model parameters

- Device: type and count. DeviceConfig already carries memory
  size, memory bandwidth, and compute rate (H100-SXM-80GB and
  L40S-48GB exist). Multi-GPU sharding divides corpus and floors
  as today (multigpu2/4/8 results).
- Model: kappa by family formula, and active against total
  parameters.
  - Grouped-query attention (qwen, llama3): kappa = 2 * L * n_kv
    * d_h * bytes-per-value. Already in configs.py; llama3 is a
    config entry, no new derivation.
  - Latent attention (deepseek): a different formula, about 70 KB
    per token; must be added, and it sits at the favorable end of
    the threshold law.
  - Mixture of experts: FLOPs use P_active, KV bytes do not
    shrink, so recompute gets cheaper while transfer cost stays -
    the cost model must price compute with P_active and weight
    traffic with W_run (configs.py already separates W_run from
    P).
- Store: DRAM bytes and disk bytes (capacity terms - what fits
  decides the hit rate), plus one measured bandwidth per channel.
- Workload: the plan. Number of documents, tokens per document,
  prompts, per-stage selectivities, generation budget. Hit rate
  and the prefetch schedule are DERIVED from the plan, because a
  corpus query knows its future accesses. This is the difference
  from reactive systems (SGLang HiCache class): they observe
  reuse and tune write and prefetch policies by hand; the planner
  computes them.

## Decode-side constraint

A decode step pays a measured software floor: 17 ms fixed plus 8
microseconds per live sequence per step (the cap-64 step trace,
2026-08-08, now priced in cost.py decode_step_seconds). Any plan
that fetches KV during decode competes with that floor, not with
the spec bandwidth: at 2,000 live sequences a decode step already
takes ~52 ms, so a tier read must hide inside steps of that
length or it adds wall time directly.

## Write policy

Persist pays once at ingest: write-through while the first query
computes. Priced as kappa * tokens over the measured write
bandwidth. Measured at 4B: the writing query took 29.5 seconds
against an 11.0-second baseline, 18.5 seconds of write for ~53 GB,
about 2.9 GB/s - the write channel is the same 20x-under-spec
composite as the read channel. Write-back under pressure and
selective write are policies the planner can price later; neither
is needed while ingest-time write-through covers the corpus case.

## Scope and deferrals

- Supported in the model: qwen (measured), llama3 (config entry,
  same formulas), deepseek (needs the latent-attention kappa and
  the P_active split; worth it - it is the regime where slow
  tiers win).
- Deferred: mamba-class models. Their state is fixed-size per
  sequence and does not grow with tokens, so the KV-volume-
  against-recompute derivation does not apply at all; the hybrid
  allocator work is already parked in NEXT.md with a trigger.
- Deferred: prefill/decode disaggregation. The formula is the same
  threshold law with the network channel's measured bandwidth in
  place of the tier read rate; unpriced until that channel exists
  and is measured (no RDMA on single-node Modal).
- Precision bound: host variance measured up to 45 percent wall
  spread across containers (the ledger's re-baseline notes). No
  cost model on this substrate predicts tighter than that across
  containers; within one container the spread is small.

## Open questions

- Equality under restore: the 4B persist run's survivor lists
  were not identical across queries (persist2000.json,
  outcomes_identical false); consistent with borderline-call
  noise, but per-call diffs were never instrumented. Must be
  settled before restore is a correctness-neutral verb.
- The store runs on the stock scheduler; strict plan-owned memory
  and the tiering connector are not yet reconciled.
- ANSWERED same day (pinprobe.json): pinning is 5.7x of the 20x,
  page faults and fragmentation the rest, and the platform allows
  alloc-pinned pools at full speed - the 4B tier flips to "wins"
  as soon as the connector stops registering a file-backed mmap.
  Remaining piece: prototype the anonymous-shared-mapping change
  upstream, or an alloc-pinned staging pool in the connector.
- The Modal volume / object-storage calibration cell.
- The latent-attention kappa formula, confirmed against a real
  deepseek checkpoint.
- Hit-rate term for plans that repeat corpora across queries (the
  capacity term starts mattering only there).

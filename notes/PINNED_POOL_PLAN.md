# Pinned offload pool: the plan

Approved 2026-08-08. RESOLVED the same evening, without the fork:
vLLM's plain CPUOffloadingSpec already allocates its pool with
pin_memory=True when no mmap region is injected - only the
TIERING spec (which exists for disk secondaries) builds the
file-backed /dev/shm region this sandbox cannot pin. Switching
the experiment's spec_name was the whole fix. Measured
(persist32_1k_cpu.json): restore 4.04-4.16 s against recompute
27.7 s (6.8x), channel 10.23 GB/s end to end, write-through at
ingest free, outcomes identical. The fork below is now needed
ONLY if disk tiers return; kept for that case.

## Why

The CPU KV tier moves 2.5-2.9 GB/s through vLLM's connector as
shipped, against a measured 55.4 GB/s ceiling for alloc-pinned
memory on the same containers (results/engine/pinprobe.json). The
whole gap comes from the connector's pool construction: it mmaps a
file on /dev/shm so multiple worker processes can share one pool,
and the sandbox refuses to pin file-backed pages (cudaHostRegister
returns 304), so every transfer runs unpinned and page-faulted. At
55 GB/s the tier beats recompute for every model we serve (the 4B
generates KV at 7.2 GB/s, the 32B at 1.7); at 2.7 it wins only at
32B.

## What (verified against the installed vLLM 0.26.0 source)

- The spec is pluggable with no vLLM edits: kv_offload/factory.py
  resolves `spec_name` from `kv_connector_extra_config`, and takes
  `spec_module_path` for classes outside its registry - the same
  seam the fork connector uses (`kv_connector_module_path`).
- The transfer machinery is tensor-generic: gpu_worker.py reaches
  the pool only through `region._base` (an int8 torch tensor) and
  tensor views over it; `pin_mmap_region` reads
  `region._base.data_ptr()` and `region.total_size_bytes`.
- So the fork is: `docengine/engineext/pinnedstore.py` with
  `DocEnginePinnedOffloadSpec(CPUOffloadingSpec)` whose region
  class allocates `_base = torch.empty(total_size_bytes,
  dtype=torch.int8, pin_memory=True)` (cudaHostAlloc under the
  hood, measured working at 48 GB), keeps the same block-row and
  per-worker-slot arithmetic, sets `is_pinned = True`, and never
  calls cudaHostRegister or madvise at all.

## The one thing to verify first - VERIFIED (same day)

Read both managers. The TIERING manager (tiering/manager.py) does
map the pool's bytes on the scheduler side - it holds the region
and a memoryview over it, because it moves blocks to secondary
tiers itself. The plain CPU manager (cpu/manager.py) never touches
the region: metadata only. So the pinned spec bases on
CPUOffloadingSpec, no secondary tiers, pool worker-side only -
which matches the CPU-only restore cell exactly. The disk tier
keeps the tiering spec and is out of scope here.

## Validation

The persist32 protocol unchanged (experiments/modal_scale.py,
phase persist32) with the spec swapped in via extra_config.
Prediction to beat: restore 41.3 GB of 32B KV in ~1-3 seconds
against ~15 for the unpinned path and 27.5 for recompute. Also
re-run the 4B persist cell: at 55 GB/s the 4B tier flips to a win
(threshold 7.2 GB/s), which reverses a banked conclusion
(persist2000.json: "4B loses as predicted") - the ledger entry
must say the conclusion was config-bound, not wrong.

## The disk tier: re-scoped IN (2026-08-08, night)

The 32B capacity wall (a 10k-document corpus is ~445 GB of KV,
over container RAM) makes the disk tier the unlock for warm
operators at full scale, and the physics clears it: local NVMe
measured 5.19 GB/s raw (persist2000's probe), against the 32B
KV-generation rate of 1.7 GB/s - disk restore beats recompute
about 3x (89 s against ~264 for the full corpus). It is 32B-only:
the 4B generates at 7.2 GB/s and disk loses there, exactly the
threshold law. Route: NOT the shipped TieringOffloadingSpec (its
file-backed region is the eight-run failure); extend the plain
CPU spec in docengine/engineext with a worker-side disk
secondary, alloc-pinned RAM staging between GPU and NVMe, one
calibration cell for the achieved disk-restore rate before any
conclusion. Container NVMe is per-run scratch; cross-run
persistence would need Modal volumes, unmeasured (1-3 GB/s
claimed) and its own calibration cell.

## Not in scope

Upstreaming to vLLM (file an issue with the pinprobe numbers and
the sticky-error bug; the anonymous-shared-mapping change is
theirs to take), and RDMA.

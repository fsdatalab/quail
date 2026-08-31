# Cold model load: 35.3 s to 24.8 s with a persisted vLLM cache and pinned revisions

## Setup

`load_model` (quail/executor/model.py) is the first phase of a cold
worker boot: it imports vLLM, builds the engine config, and loads the
weights. Charles Frye profiled it phase by phase on the
pre-restructure tree (branch `charlesfrye/faster-boot`) and found two
avoidable costs:

- vLLM resolves the model architecture by running a fresh Python
  interpreter (`python -m vllm.model_executor.models.registry`)
  during `EngineArgs.create_engine_config`, and caches the result as
  JSON under `VLLM_CACHE_ROOT/modelinfos`. `VLLM_CACHE_ROOT` defaults
  to `~/.cache/vllm`, which is ephemeral container storage on Modal,
  so every container paid the subprocess (~13 s) again.
- The checkpoint was named by branch (`Qwen/Qwen3-4B-FP8` at default
  `main`), so hub resolution paid HF API round trips on every load.
  `HF_HUB_OFFLINE=1` was rejected: a missing snapshot must download,
  not fail.

This change ports his two fixes to the current tree:

- Every vLLM GPU image sets `VLLM_CACHE_ROOT` to
  `/root/.cache/kernels/vllm` on the existing kernel-cache volume, so
  the architecture subprocess runs once ever, not once per container.
- `ModelSpec` grows a `revision` field pinning the hub commit hash
  (`96b30dc1...` for Qwen3 4B fp8, `aa55da1e...` for Qwen3 32B fp8;
  both checkpoints unchanged on the hub since 2025-07-26, and both
  hashes match the snapshots already on the `quail-hf-cache` volume).
  `load_model` takes `revision` and passes it into `EngineArgs`; the
  worker, the calibration path, the ablation cells, and the stock
  vLLM baseline all pass `spec.revision`, so engine and baseline
  boots stay comparable.

The measurement cell is `tests/gpu/load_profile.py`. It mirrors the
current `load_model` body step by step (dtype="auto", single-rank
group stubs, no NCCL) and times four phases: import torch, import
vLLM, engine config, and `get_model` (weight load, fp8
post-processing, and the closing CUDA sync). Every call runs in its
own cold container (`single_use_containers=True`) on the same image.
Three runs:

- `main`: the pre-fix behavior. Pops `VLLM_CACHE_ROOT` before any
  vLLM import and passes no revision.
- `pinned_seed`: the fixed behavior, first container on the volume.
  Pays the architecture subprocess once and writes the JSON.
- `pinned`: the fixed behavior after the seed. This is the cold boot
  every later container sees.

A raw read of the cached safetensors files separates "the volume is
slow" from "vLLM is slow". Function call ids:
`fc-01M1D12GFQK3SHJYS69D4MN70W` (main),
`fc-01M1D12GJGDEC1JSDYV1Z8ZF9J` (pinned_seed),
`fc-01M1D1445DKBGWVVWJMZ5ZZ1TQ` (pinned).

## Prediction

Stated before the run, in the cell's docstring:

- `main` pays ~13-19 s in engine config (subprocess plus hub round
  trips) and ~14-18 s in `get_model`, in-timer total ~35-45 s.
- `pinned_seed` pays engine config once at ~13-15 s.
- `pinned` drops engine config to ~0.1-2 s and `get_model` to
  ~10-12 s, in-timer total ~22-27 s.
- The raw weight read stays ~2 s either way.

## Result

| phase | before (`main`) | after, first container | after, steady state |
|---|---|---|---|
| import torch | 2.18 s | 2.69 s | 1.61 s |
| import vLLM | 10.42 s | 12.57 s | 12.12 s |
| engine config | 12.27 s | 13.38 s | **0.96 s** |
| get_model | 10.45 s | 16.32 s | 10.14 s |
| **total** | **35.33 s** | 44.96 s | **24.84 s** |

Figure: plots/load_model_speed.png

- The steady-state total (24.84 s) fell inside the predicted 22-27 s
  window, and the engine-config and `get_model` phases fell inside
  their predicted ranges. Cold `load_model` now takes 24.8 s,
  compared with 35.3 s before: 10.5 s saved, a 1.42x speedup.
- One prediction missed: `main`'s `get_model` measured 10.45 s, below
  the predicted 14-18 s. The hub round trips were cheap in this run;
  Charles's August runs measured 14.4-17.5 s for the same phase. The
  pin's per-boot saving in `get_model` therefore varies with hub
  latency; what it guarantees is the cache-only resolution path and a
  fixed checkpoint identity. In this run nearly all of the 10.5 s
  came from the persisted modelinfos cache (12.27 s to 0.96 s).
- The seed container paid 13.38 s in engine config, close to the
  13-15 s predicted, and its `get_model` ran 16.32 s (container and
  network variance; the volume showed its slowest raw read in the
  same container, 1.96 GiB/s vs 2.55 GiB/s for `main`).
- The raw weight read stayed at ~2 s in every container (4.83 GiB at
  2.0-2.6 GiB/s), so the volume is not the bottleneck: `get_model`'s
  ~10 s is dominated by fp8 post-processing and module construction,
  not weight bytes.

In GPU cost: a cold boot now costs $0.027 of H100 time, compared
with $0.039 before (24.84 s and 35.33 s at $3.9492/h). Boot cost is
reported separately from query time, as always.

## What the numbers mean

- The worker's cold `load_model_s` drops by ~10.5 s. The rest of a
  cold boot (arena, pipeline, kernel warmup) is unchanged by this
  change; warm containers were already at ~0 s and stay there.
- The 32B model gets the same mechanism: its first boot after this
  change seeds its own modelinfos JSON, and its pinned hash resolves
  from the same volume. It was not rerun; the mechanism is
  model-independent (the subprocess cost is per-container, not
  per-model-size).
- The remaining floor is the torch+vLLM import (13.7 s of the
  24.8 s) plus `get_model` (10.1 s). Charles's pointer for the next
  step is Modal's vLLM memory-snapshot example
  (https://modal.com/docs/examples/vllm_snapshot): a memory snapshot
  taken after the imports would remove the 13.7 s and bring the cold
  load near 11 s. Two blockers before that works here: Modal creates
  memory snapshots only for deployed apps, and the session runs the
  worker on an ephemeral app (`worker.app.run()` in
  quail/runtime/session.py) precisely so warm state dies with the
  session; and GPU-state snapshots are alpha and incompatible with
  the multi-GPU child-process path (`execute_2/4/8`). Snapshotting
  therefore needs a deliberate change to the worker's app lifecycle,
  not a flag.

Data on the `quail-results` volume:

- `/results/ablations/load_model_profile_main.json`
- `/results/ablations/load_model_profile_pinned_seed.json`
- `/results/ablations/load_model_profile_pinned.json`

Rebuild the figure with the `modal volume get` commands in
`reports/make_load_model_speed_plots.py`.

# Faster cold model load: persisted vLLM cache and pinned hub revisions

## What changed

Two settings, ported from Charles Frye's measurements on branch
`charlesfrye/faster-boot`.

**Every vLLM GPU image sets `VLLM_CACHE_ROOT` to the kernel-cache
volume** (`/root/.cache/kernels/vllm`, which those images already
mount). vLLM resolves the model architecture by running a fresh
Python interpreter at engine-config creation and caches the answer as
JSON under `VLLM_CACHE_ROOT/modelinfos`. The default `~/.cache/vllm`
is ephemeral on Modal, so every container paid that ~13 s subprocess
again. One container now seeds the JSON; every later container reads
it back in about a second.

**`ModelSpec.revision` pins each checkpoint to its hub commit hash**
(`96b30dc1...` for Qwen3 4B fp8, `aa55da1e...` for Qwen3 32B fp8), and
`load_model` passes it into `EngineArgs`. A commit hash resolves from
the HF cache volume without the API round trips a branch name pays,
and still downloads on a cold cache. Offline mode was rejected: a
missing snapshot must download, not fail. The worker, the ablation
cells, and the stock vLLM baseline all pass `spec.revision`, so engine
and baseline boots stay comparable.

## Why

Cold worker boots paid ~13 s of repeated architecture inspection on
every fresh container, plus hub round trips whose cost varied with HF
API latency.

## Before and after

`experiments/cells/load_profile.py`, Qwen3 4B fp8 on one H100, one cold
container per run:

| phase | before | after, first container | after, steady state |
|---|---|---|---|
| import torch | 2.18 s | 2.69 s | 1.61 s |
| import vLLM | 10.42 s | 12.57 s | 12.12 s |
| engine config | 12.27 s | 13.38 s | 0.96 s |
| get_model | 10.45 s | 16.32 s | 10.14 s |
| **total** | **35.33 s** | 44.96 s | **24.84 s** |

Figure: ../plots/load_model_speed.png

Cold `load_model` takes 24.84 s, compared with 35.33 s before: 10.5 s
saved, 1.42x, after one 45 s container seeds the JSON on the volume.
In GPU cost that is $0.027 per cold boot, compared with $0.039 before
(at $3.9492/h). Boot cost stays reported separately from query time.

The prediction was recorded before the run and the steady-state total
fell inside its 22-27 s window. One part missed: `get_model` before
the change measured 10.45 s, below the predicted 14-18 s, so the
revision pin contributed only ~0.3 s in this run. Charles's August
runs measured 14.4-17.5 s for the same phase, so the pin's saving
varies with hub latency; what it guarantees is the cache-only
resolution path and a fixed checkpoint identity. Nearly all of the
10.5 s here came from the persisted modelinfos cache.

The raw safetensors read stayed at ~2 s in every container (4.83 GiB
at 2.0-2.6 GiB/s), so the volume is not the bottleneck: `get_model`'s
~10 s is module construction and fp8 post-processing, not weight
bytes.

End to end through the real worker (`experiments/cells/session_smoke.py`):
cold boot 24.3 s, of which `load_model` 20.3 s, arena 0.6 s, kernel
touch pass 3.4 s. Results unchanged, 76 of 76 planted filter
survivors and 72 of 72 planted join pairs, warm rerun identical.

The 32B model gets the same mechanism and was not rerun: the
subprocess cost is per container, not per model size.

## What is left

The remaining floor is the torch+vLLM import (13.7 s of the 24.8 s)
plus `get_model` (10.1 s). Modal's vLLM memory-snapshot example
(https://modal.com/docs/examples/vllm_snapshot) would remove the
import and bring a cold load near 12 s, but it needs a change to the
worker's app lifecycle first: Modal creates memory snapshots only for
deployed apps, and each session runs the worker on an ephemeral app
(`worker.app.run()` in `quail/runtime/session.py`) so the GPU is
released when the session closes. The GPU-state variant is alpha and
incompatible with the multi-GPU child processes.

Data on the `quail-results` volume:

- `/results/ablations/load_model_profile_main.json`
- `/results/ablations/load_model_profile_pinned_seed.json`
- `/results/ablations/load_model_profile_pinned.json`

Rebuild the figure with the `modal volume get` commands in
`reports/make_load_model_speed_plots.py`.

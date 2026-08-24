# Quail

A query engine for AI_FILTER and AI_JOIN over document collections.
Filter queries and joins only. The models are Qwen3 4B fp8 and Qwen3
32B fp8, one H100 per model copy, and workers run through Modal.

## Layout

- `quail/specs/` has the model and device structs. Every planner
  input comes from them.
- `quail/planner/` has the budget arithmetic, the physical plan,
  and the planner decisions (stage order, anchor choice, sharding,
  access method). It loads `a` and `a2` from the JSON files below.
  KV is always bf16.
- `quail/calibration/` has the measured constants per (model, device)
  pair: `a`, `a2`, and the host channel bandwidth table.
- `quail/catalog.py`, `quail/logical.py`, `quail/sqlfront/`, and
  `quail/builder.py` are the providers, logical operators, and the
  two query entry points (AI SQL and the builder API).
- `quail/executor/` is the packed executor: chunk packing, admission,
  the paged KV arena, attention kernels, the overlapped loop, and
  weight loading. The GPU parts run only inside the Modal image.
- `quail/runtime/` is the run side: the session (plan, payload,
  recombination), the multi-GPU coordinator, and the Modal worker.
- `quail/bench/` has the QUAIL-B benchmark queries.
- `tests/` has CPU tests. `tests/gpu/` has the Modal GPU cells -
  milestone gates, smokes, and benchmarks - which cost GPU time and
  run only when invoked explicitly.

## Setup and tests

```
cd quail
uv sync
uv run pytest
```

GPU cells (Modal, H100). Tee output to a file per house rule:

```
uv run modal run tests/gpu/milestone1.py::run_probe 2>&1 | tee results/m1_probe.log
```

Committed result summaries (the JSON files reports cite) are in
`results/`; raw per-item run records live on the `quail-results`
Modal volume. Teed logs stay local and are not committed.

## Calibration

The planner's restore break-even uses two constants per (model,
device) pair, stored in `quail/calibration/{model}_{device}.json`:

- `a_s_per_token`: seconds per fresh token in the packed loop.
- `a2_s_per_token2`: the quadratic attention coefficient. It bends
  the restore break-even for long documents.

A pair without a file is spec-ratio-scaled from the 4B/H100
anchor. The plan's `calibration_source` field says where the values
came from. There is no measure entry on this branch; the length
sweep (`calibrate.py`) was removed so it can be rewritten.

## Adding a new model

Add a spec struct in `quail/specs/` (about 15 lines). Every field
except `params` and `w_mem_bytes` comes from the model's HF
`config.json`. Register the spec in `quail/specs/__init__.py`.

The executor's fused kernels assume the Qwen3 architecture:
QK-norm before rope, gated SiLU MLP, and fp8 block-quantized
weights. Another Qwen3-family fp8 checkpoint works without
changes. A different model family needs kernel-path changes first.

Until a new measure path exists, a new pair uses the spec-scaled
defaults from the 4B/H100 JSON.

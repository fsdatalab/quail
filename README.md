# Quail

A query engine for AI_FILTER and AI_JOIN over document collections.
Filter queries and joins only. The models are Qwen3 4B fp8 and Qwen3
32B fp8, one H100 per model copy, and workers run through Modal.

## Layout

- `quail/specs/` has the model and device structs. Every planner
  input comes from them.
- `quail/planner/` has the budget arithmetic, calibration loading
  and measurement, the physical plan, and the planner decisions
  (stage order, anchor choice, sharding). KV is always bf16.
- `quail/calibration/` has the measured constants per (model, device)
  pair: `a`, `a2`, and the host channel bandwidth table.
- `quail/catalog.py`, `quail/logical.py`, `quail/sqlfront/`, and
  `quail/builder.py` are the providers, logical operators, and the
  two query entry points (AI SQL and the builder API).
- `quail/executor/` is the packed executor: chunk packing, admission,
  the paged KV arena, attention kernels, the overlapped loop, and
  weight loading. The GPU parts run only inside the Modal image.
- `quail/runtime/` is the run side: the session (plan, payload,
  recombination), the multi-GPU coordinator, the Modal worker, and
  the calibrate entry.
- `quail/bench/` has the QUAIL-B benchmark queries. Its
  [README](quail/bench/README.md) explains how to run the benchmark and
  label a new predicate.
- `ablations/` and `baselines/` have experiment entry points and the
  stock vLLM comparison code.
- `reports/` has experiment reports and plots. `results/` has the
  committed summaries that those reports read.
- `plans/` has the original engine design.
- `tests/` has CPU tests. `tests/gpu/` has the Modal GPU cells -
  milestone gates, smokes, and benchmarks - which cost GPU time and
  run only when invoked explicitly.

## Setup and tests

```
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

Two measured constants per (model, device) pair are stored in
`quail/calibration/{model}_{device}.json`:

- `a_s_per_token`: wall seconds per fresh token in the packed loop
  (the serving rate, with all overhead included).
- `a2_s_per_token2`: the quadratic attention coefficient for long
  documents.

A pair without a calibration file gets defaults scaled from the anchor
measurement (Qwen3 4B on H100) using spec ratios. The plan's
`calibration_source` field says where the values came from.

To measure a pair, run the calibrate entry on the engine app:

```
uv run modal run quail/runtime/calibrate.py --model qwen3-4b-fp8 --device h100-sxm 2>&1 | tee results/calibrate.log
```

The measure step (`quail.planner.calibrate.measure`) does the following:

- It sweeps document length (256, 1,024, 4,096, and 8,192 tokens,
  about 1.5M fresh tokens per point) through the packed filter and
  fits `t(h) = a + a2*h` by least squares.
- It re-probes the host copy channels (pinned and unpinned, both
  directions, 2 GiB timed copies) for comparison against
  `quail/calibration/channels.json`.

The result goes to `results/calibrate.json` only. To write it where
the planner reads it, pass `--commit`:

```
uv run modal run quail/runtime/calibrate.py --model qwen3-4b-fp8 --device h100-sxm --commit 2>&1 | tee results/calibrate.log
```

## Adding a new model

There are three steps, and one constraint to know about.

First, add a spec struct in `quail/specs/` (about 15 lines). Every
field except `params` and `w_mem_bytes` comes from the model's HF
`config.json`. Register the spec in `quail/specs/__init__.py`.

Second, run the calibrate entry with `--model` set to the new spec
name and `--commit`. One run measures `a` and `a2` and writes the
calibration file. The Modal function is wired to H100 today; a new
device needs a `gpu=` mapping in `quail/runtime/calibrate.py` as
well as a device spec.

Third, know that the executor's fused kernels assume the Qwen3
architecture: QK-norm before rope, gated SiLU MLP, and fp8
block-quantized weights. Another Qwen3-family fp8 checkpoint works
without changes. A different model family needs kernel-path changes
first.

State the prediction before the run and compare after, per the house
rule. The calibrate entry prints the previously loaded constants
(`loaded_before`) next to the fresh fit for exactly that comparison.

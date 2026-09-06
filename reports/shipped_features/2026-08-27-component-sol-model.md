# SoL uses named model components

## What changed

The planner and SoL report now price three Qwen3 components:

- `attn_proj` counts Q, K, V, and output projection FLOPs and weight bytes.
- `mlp` counts gate, up, and down projection FLOPs and weight bytes.
- `attention` counts attention pair FLOPs and KV bytes.

Each component supplies its FLOPs, bytes, and precision. The generic
calculation takes the larger of compute time and memory time for each
component, then adds the component times.

The code is split by responsibility. `planner/work.py` counts query work.
`planner/qwen3_cost.py` defines the Qwen3 components.
`planner/roofline.py` contains the generic component calculation.
`planner/sol.py` applies the existing query-wide ideal packing rule and adds
the component times.

The resident model footprint remains in `ModelSpec.W_mem` for GPU capacity
and KV admission calculations. The SoL calculation no longer treats every
resident model byte as weight traffic. It counts only the weights belonging
to the modeled attention projection and MLP components.

KV accounting is unchanged. Quail counts both KV written and KV read because
KV rewind depends on both operations.

## Why

The previous implementation combined all dense work in one record and used
the measured resident model footprint for dense memory traffic. The component
model names the source of every modeled FLOP and byte. It also keeps model
architecture code separate from the generic hardware calculation.

The model follows the component structure used by the Modal speculative
decoding calculator. Query operators do not create separate component limits
or extra weight reads. Aggregate query work still uses ideal query-wide
packing.

## SoL values

Every filter order, join order, and anchor stayed the same for the 35 QUAIL-B
queries at scale factor 0.1. The total SoL time changed as follows:

| Model | Before | After | Difference |
|---|---:|---:|---:|
| Qwen3 4B fp8 | 324.6505 s | 324.6364 s | 0.0141 s lower, 0.0043% |
| Qwen3 32B fp8 | 2,470.1103 s | 2,470.0616 s | 0.0487 s lower, 0.0020% |

The modeled weight bytes are below the previous resident footprint, but the
dense components are compute bound for every query. The time decrease comes
from omitting norms from dense FLOPs, which matches the component model's
scope.

The values are analytical. No GPU run was needed. The result data is at
`/results/sol/sol_quailb_sf0.1.json` on the `quail-results` volume. The full
table and plots are in the deleted report `2026-08-26-sol-quailb.md`
(in git history). These values describe earlier query definitions.

## Verification

`uv run pytest -q` passes 174 tests. Tests check the component names, the sum
of component times, and the use of modeled component weight bytes instead of
the resident model footprint.

# Device leftover: vLLM utilization and DiffusionGemma chunk

Follow-up to #181. The ticket still wants stock/pipelined vLLM and
DiffusionGemma on L40S. Two constants, set for H100 leftover, block
that. This is not a public user API. The values should come from the
device (and model), not a single literal.

Recorded behavior: `docs/content/docs/architecture/device-memory.mdx`.

## What is hard-coded


| Constant                        | Where                                | Today                                            |
| ------------------------------- | ------------------------------------ | ------------------------------------------------ |
| `GPU_MEMORY_UTILIZATION = 0.91` | `quail/backends/vllm.py`             | Every device. Set in #79. vLLM's default is 0.9. |
| `chunk_cap_tokens = 65_536`     | `quail/specs/diffusion_gemma_26b.py` | Every device. `chunk_budget` never exceeds it.   |


The 0.91 pool is weights + peak activations + KV. CUDA graphs and
EngineCore warmup (`logits.clone`, 1.16 GiB) allocate after it.

On L40S (44.42 GiB visible) 0.91 left ~4 GiB. Qwen3 4B vLLM died with
550 MiB free. On H100, 9% is ~7 GB and boots.

DiffusionGemma weights fit L40S (27.68 GB, one card). One 65,536-token
chunk's sliding KV needs 13.63 GB; 5.82 GB remains. Planner refuses
`sliding_kv_needs_more_memory`. H100 leftover 36.51 GB; B200 142.43 GB.

## Plan

Keep H100 and B200 at today's numbers. Derive L40S (and the next 48 GB
card) from leftover bytes.

### 1. vLLM utilization

Add a leftover reserve, not a new magic fraction.

- Name the extras: CUDA-graph capture + EngineCore warmup. Measure
~6 GiB needed on L40S (covers the 1.16 GiB clone plus graphs).
- `gpu_memory_utilization(device) = 1 - reserve / mem_bytes`, floored
so the pool still holds weights. L40S → about 0.865. H100/B200 keep
0.91 until they miss.
- Put `reserve` on `DeviceSpec` (bytes, 0 means “use the 0.91 default”).
`VLLMEngine.llm_kwargs` reads that. No `EngineConfig` field unless a
cell must override for one run.
- Do not change the shared 0.91 default for H100.



### 2. DiffusionGemma chunk

The cap is a model field but the fit is `(model, device)`.

- `chunk_budget` already takes both. After `min(..., chunk_cap_tokens)`,
clamp so `sliding_kv_fits` is true, or refuse if even the compute
knee does not fit.
- Largest L40S chunk that fits today is ~45,397 tokens. Use 32,768
(needs 6.82 GB, leftover 11.72 GB). H100/B200 stay at 65,536.
- Write the chosen chunk on the physical plan / suite metadata.
- `moe_configs.py` only has H100 tiles at 32,768 and 65,536. If L40S
uses 32,768, measure Ada tiles or fall back to vLLM defaults and say
so.



### 3. Checks

- CPU: L40S DiffusionGemma plans IMDB-1 / FEV-1 (not a Refusal).
`chunk_budget` is 32,768 on L40S and 65,536 on H100/B200.
- CPU: vLLM `llm_kwargs` utilization is ~0.865 on L40S and 0.91 on
H100.
- GPU: Qwen3 4B stock + pipelined vLLM boots on L40S. One
DiffusionGemma filter on L40S Quail. Cite volume paths.



## Out of scope

- Changing 0.91 or 65,536 on H100/B200.
- NVFP4.
- Tensor-parallel weights.
- A session/SQL knob for utilization or chunk. Bench cells can pass a
spec override if needed.



## Acceptance

L40S runs the #181 configs the ticket named: Quail + both vLLM on Qwen3
4B, and Quail on DiffusionGemma, without shrinking H100/B200 packs or
pools. Report L40S utilization and chunk next to the numbers.
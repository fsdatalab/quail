# Device leftover: vLLM utilization

Follow-up to #181. vLLM stays at 0.91 unless a model spec asks for
a reserve. DiffusionGemma on L40S is out of scope. This is not a
public user API.

## DiffusionGemma on L40S

Do not run it, and do not shrink its chunk to make the planner accept
it.

The 65,536-token cap does not fit L40S leftover. One chunk's sliding
KV needs 13.63 GB, and 5.82 GB remains. The planner refuses with
`sliding_kv_needs_more_memory`. A 32,768-token pack would fit that
KV. That does not make the model run.

Five of the thirty layers keep every token with 512-wide heads.
FlashAttention 2 and 3 stop at 256. Quail sends those layers through
FlashAttention 4 over paged KV. FlashAttention 4 does not run on an
L40S. `flash_attention_version` selects FlashAttention 2 for compute
capability 8.9. A run that planned at 32,768 still died in kernel
warmup, after the weights had loaded:

```
AssertionError: paged KV not supported on SM 8.0
```

That run is `/results/benchmarks/quailb/20261015T120134Z-1aa1a30d`.
The failure is the kernel, not the chunk. H100 and B200 stay at
65,536. The SF 0.1 suite skips DiffusionGemma on L40S.

## vLLM utilization

vLLM's pool is weights, peak activations, and KV. CUDA graphs and
EngineCore warmup (`logits.clone`, 1.16 GiB) allocate after it.
`gpu_memory_utilization` in `quail/baselines/vllm.py` is 0.91 unless
the model spec sets `vllm_reserve_bytes`. A reserve lowers the
fraction to `min(0.91, 1 - reserve / device memory)`. It never raises
the 0.91 pool. No `EngineConfig` field.

Qwen3 4B sets a 6 GiB reserve. On an L40S that is 0.865. On H100
(80 GB) and B200, 9% is already more than 6 GiB, so those stay at
0.91. Other models leave the field at 0 and stay at 0.91 on every
device. Qwen3 4B stock vLLM on L40S completed IMDB-1 at 0.865:
`/results/benchmarks/quailb/20261015T131311Z-636400dd`. A fixed 0.91
on that card left about 4 GiB, and the same model died with 550 MiB
free.

## Checks

- CPU: a DiffusionGemma plan on L40S is a Refusal.
- CPU: Qwen3 4B utilization is about 0.865 on L40S and 0.91 on H100
  and B200. A model with no reserve is 0.91 on every device.
- GPU: Qwen3 4B stock vLLM on L40S completed IMDB-1. Pipelined vLLM
  on L40S is not run yet.

## Out of scope

- DiffusionGemma on L40S, including a smaller chunk.
- Changing the 65,536-token chunk cap on H100 or B200.
- NVFP4.
- Tensor-parallel weights.
- A session or SQL knob for utilization.

## Acceptance

L40S runs Quail and both vLLM backends on Qwen3 4B. DiffusionGemma
stays on H100 and B200. Report the L40S utilization next to the
numbers.

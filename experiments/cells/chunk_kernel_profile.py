"""Time one filter chunk's forward pass and list its CUDA kernels.

Loads a model into Quail's executor, packs one unified chunk of
synthetic documents (each a document prefix saved at its end plus an
eight-token suffix, as a filter packs them), runs it three times to
warm the kernels, then profiles three more runs with torch.profiler
and prints the kernels by CUDA time, the GPU time per run, and the
wall time per run.

Run from the repository root and tee every line:

    uv run modal run --detach experiments/cells/chunk_kernel_profile.py \
      --models qwen3.8-27b-fp8,qwen3-32b-fp8 --documents 100 --tokens 300 \
      2>&1 | tee results/chunk-kernel-profile.log
"""

import json
import time

import modal

from quail.bench.images import gpu_image

app = modal.App("quail-milestone1")
image = gpu_image()
volumes = {
    "/root/.cache/huggingface": modal.Volume.from_name(
        "quail-hf-cache", create_if_missing=True),
    "/root/.cache/kernels": modal.Volume.from_name(
        "quail-kernel-cache", create_if_missing=True),
}
SUFFIX_TOKENS = 8
WARMUP_RUNS = 3
PROFILED_RUNS = 3


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def profile_chunk(model_name: str, documents: int, tokens: int,
                  top: int) -> dict:
    """Profile one chunk of `documents` documents of `tokens` tokens each."""
    import torch

    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.chunk import pack_chunk
    from quail.backends.quail.executor.model import load_model
    from quail.backends.quail.executor.models import build_pipeline
    from quail.cost.budgets import PAGE_TOKENS
    from quail.specs import MODELS

    spec = MODELS[model_name]
    model = load_model(spec.hf_name, revision=spec.revision,
                       max_batched_tokens=documents * (tokens + SUFFIX_TOKENS),
                       language_model_only=spec.language_model_only)
    stateful = bool(spec.state_bytes)
    arena = KVArena(
        n_layers=spec.layers, n_pages=4096, page_tokens=PAGE_TOKENS,
        n_kv=spec.n_kv, d_head=spec.d_head, dtype=torch.bfloat16,
        layer_kv=spec.kv_shapes,
        state_layers=spec.linear_layer_set if stateful else (),
        n_state_slots=2 * documents + 1 if stateful else 0,
        state_shape=spec.state_shape if stateful else None,
        conv_shape=spec.conv_shape if stateful else None)
    pipeline = build_pipeline(spec, model, arena)
    generator = torch.Generator().manual_seed(20260818)
    groups = []
    for index in range(documents):
        ids = torch.randint(1000, 100_000, (tokens + SUFFIX_TOKENS,),
                            generator=generator).tolist()
        key = ("doc", index)
        arena.activate(key, tokens, capacity_tokens=tokens + 64,
                       base_tokens=tokens, slots=2 if stateful else 0)
        group = dict(key=key, prefix=ids[:tokens], f=tokens,
                     suffixes=[ids[tokens:]], write_suffix_tokens=4,
                     single=True)
        if stateful:
            group["save_at"] = (tokens, tokens + 4)
        groups.append(group)
    chunk = pack_chunk(torch, arena, groups, attention_mode="unified")

    def run():
        with torch.inference_mode():
            out = pipeline.forward_chunk(chunk)
        torch.cuda.synchronize()
        return out

    for _ in range(WARMUP_RUNS):
        run()
    walls = []
    with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(PROFILED_RUNS):
            t0 = time.perf_counter()
            run()
            walls.append(time.perf_counter() - t0)
    averages = prof.key_averages()
    cuda_total_us = sum(e.self_device_time_total for e in averages)
    rows = sorted(averages, key=lambda e: e.self_device_time_total,
                  reverse=True)[:top]
    kernels = [dict(name=e.key[:90], calls=e.count,
                    cuda_ms_per_run=round(e.self_device_time_total / 1000
                                          / PROFILED_RUNS, 2),
                    share=round(e.self_device_time_total / max(1, cuda_total_us), 3))
               for e in rows if e.self_device_time_total > 0]
    return dict(model=model_name, documents=documents, tokens=tokens,
                chunk_tokens=chunk.tokens, waves=(len(chunk.meta["state"]["waves"])
                                                  if stateful else None),
                wall_s_per_run=round(sum(walls) / len(walls), 3),
                gpu_s_per_run=round(cuda_total_us / 1e6 / PROFILED_RUNS, 3),
                kernels=kernels)


@app.local_entrypoint()
def main(models: str = "qwen3.8-27b-fp8,qwen3-32b-fp8", documents: int = 100,
         tokens: int = 300, top: int = 25):
    calls = []
    for name in models.split(","):
        call = profile_chunk.spawn(name.strip(), documents, tokens, top)
        print(f"function call id: {call.object_id} ({name.strip()})", flush=True)
        calls.append(call)
    for call in calls:
        print(json.dumps(call.get(), indent=1), flush=True)

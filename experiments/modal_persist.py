"""Persisted KV: is restoring a corpus cheaper than recomputing it?

The second query over the same documents can either re-read them
through prefill or load their saved KV back from host memory. Which
one wins is a bandwidth question with a measurable threshold: restore
beats recompute when

    channel bandwidth  >  kappa * prefill rate

where kappa is the KV bytes per token and the prefill rate is how
fast the GPU regenerates that KV from text. At 4B fp8 that threshold
is 73,728 bytes/token * 97,000 tokens/s = 7.2 GB/s. A bigger model
lowers the threshold (it prefills slower), which is why restore wins
more easily as models grow.

Two containers, because a second engine boot in one container dies in
DeepGEMM warmup (the core forks from a CUDA-initialized parent):

  stage "baseline"  cold query, then reset the prefix cache and run
                    the same query again - the recompute cost
  stage "store"     cold query with the CPU offload connector
                    writing KV through, then two restores

The store stage uses vLLM's OffloadingConnector with its plain CPU
spec, which allocates its pool with pin_memory=True (cudaHostAlloc).
The probe (modal_pinprobe.py) measures that memory at 55.4 GB/s for
a raw copy, against a 64 GB/s PCIe Gen5 spec; the end-to-end restore
through the connector measures about 10.2 GB/s, so roughly 5x is
lost inside the connector's per-block transfer loop. That gap is
open - it is the "might be super handicapped" TODO.

Not the tiering spec: it backs its pool with a file-backed /dev/shm
region this sandbox cannot pin (cudaHostRegister returns 304 on
file-backed mappings) and its unpinned fallback died natively
mid-query. Plain CPU spec needs no workarounds.

Run both stages, then compare:
    modal run experiments/modal_persist.py --stage baseline
    modal run experiments/modal_persist.py --stage store
"""

import json
import os

import modal

from workload import MODEL, N_FILTERS, hf_cache, image, results_vol

app = modal.App("quail-persist")
persist_image = image.add_local_python_source("workload")


@app.function(image=persist_image, gpu="H100!", timeout=3600,
              memory=131072,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
async def persist_run(n_docs: int = 1000, stage: str = "baseline",
                      cpu_gb: int = 96) -> dict:
    import inspect
    import time as _time

    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM as Engine

    from quail.configs import MODELS
    from quail.runtime.engine_client import run_filter_chain
    from workload import CFG_NAME, build_corpus, yes_no_ids

    tok = AutoTokenizer.from_pretrained(MODEL)
    yes_ids, no_ids = yes_no_ids(tok)
    body_ids, q_ids, _flags = build_corpus(tok, n_docs)
    corpus = sum(len(b) for b in body_ids)
    kappa = MODELS[CFG_NAME].kappa
    kv_bytes = corpus * kappa
    sp = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1,
                        allowed_token_ids=sorted(yes_ids | no_ids))
    # This cell runs on the STOCK scheduler: the offload connector
    # manages the KV lifecycle, and reconciling it with plan-owned
    # memory is open work. The point here is the channel, not the
    # executor.
    pool_budget = 700_000

    report = dict(n_docs=n_docs, stage=stage, model=MODEL,
                  corpus_tokens=corpus, kappa=kappa,
                  kv_bytes_estimate=kv_bytes)
    print(f"[persist] stage {stage}: corpus {corpus:,} tokens, KV "
          f"about {kv_bytes / 1e9:.1f} GB", flush=True)

    def engine_args(store):
        kw = dict(model=MODEL, kv_cache_dtype="fp8", max_model_len=4608,
                  gpu_memory_utilization=0.92,
                  enable_prefix_caching=True, disable_log_stats=True)
        if store:
            kw["kv_transfer_config"] = KVTransferConfig(
                kv_connector="OffloadingConnector", kv_role="kv_both",
                kv_connector_extra_config=dict(
                    cpu_bytes_to_use=cpu_gb * (1 << 30)))
        return AsyncEngineArgs(**kw)

    async def one_query(engine, tag):
        return await run_filter_chain(engine, sp, body_ids, q_ids,
                                      pool_budget, tag=tag)

    async def reset(engine):
        res = engine.reset_prefix_cache()
        if inspect.isawaitable(res):
            await res

    if stage == "baseline":
        engine = Engine.from_engine_args(engine_args(store=False))
        base = await one_query(engine, "pb")
        await reset(engine)
        base2 = await one_query(engine, "pb2")
        report["baseline_cold_s"] = round(base["wall"], 2)
        report["baseline_recompute_s"] = round(base2["wall"], 2)
        report["survivors"] = base["survivors"]
        print(f"[persist] baseline cold {base['wall']:.2f}s, "
              f"recompute after reset {base2['wall']:.2f}s", flush=True)
    else:
        engine = Engine.from_engine_args(engine_args(store=True))
        q1 = await one_query(engine, "ps1")
        _time.sleep(8)                 # let offload writes drain
        await reset(engine)
        q2 = await one_query(engine, "ps2")
        await reset(engine)
        q3 = await one_query(engine, "ps3")
        report["store_cold_offload_s"] = round(q1["wall"], 2)
        report["store_restore_s"] = round(q2["wall"], 2)
        report["store_restore2_s"] = round(q3["wall"], 2)
        best = min(q2["wall"], q3["wall"])
        report["restore_effective_GBps"] = round(kv_bytes / best / 1e9, 2)
        report["outcomes_identical_within_store"] = (
            q1["survivors"] == q2["survivors"] == q3["survivors"])
        report["survivors"] = q1["survivors"]
        print(f"[persist] with store: cold+offload {q1['wall']:.2f}s, "
              f"restore {q2['wall']:.2f}s then {q3['wall']:.2f}s "
              f"({report['restore_effective_GBps']} GB/s effective)",
              flush=True)
    try:
        engine.shutdown()
    except Exception:
        pass
    with open(f"/results/persist_{stage}.json", "w") as f:
        json.dump(report, f, indent=2)
    results_vol.commit()
    return report


@app.local_entrypoint()
def main(n_docs: int = 1000, stage: str = "baseline", cpu_gb: int = 96,
         out: str = ""):
    data = persist_run.remote(n_docs, stage, cpu_gb)
    path = out or f"results/engine/persist_{stage}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"saved {path}")

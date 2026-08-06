"""GPU memory snapshots for the experiment engine (Modal alpha).

Engine boot (weights, DeepGEMM warmup, FlashInfer autotune) costs 4
to 6 minutes per run and dwarfs the measurements. A GPU memory
snapshot captures the booted engine once; later runs restore it in
seconds. Alpha feature: single GPU only, deployed apps only,
incompatible with the multi-GPU phases. The classic per-run boot in
modal_scale.py stays as the fallback.

Usage:
    modal deploy experiments/modal_snap.py
    then call SnapEngine.specsmoke via modal.Cls.from_name
    (first call boots and captures; later cold starts restore)
"""
import os
import sys

import modal

# deployed from the repo root (so the docengine package resolves for
# mounting); modal_scale lives next to this file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modal_scale import (FLAG_SEED, IMAGE_STAMP, MODEL,  # noqa: E402
                         _build_pool, _flags_line, _question, hf_cache,
                         image)

app = modal.App("docengine-snap")
snap_image = image.add_local_python_source("modal_scale")


@app.cls(image=snap_image, gpu="H100!", timeout=1800,
         enable_memory_snapshot=True,
         experimental_options={"enable_gpu_snapshot": True},
         volumes={"/root/.cache/huggingface": hf_cache})
class SnapEngine:
    @modal.enter(snap=True)
    def boot(self):
        import os
        import time as _time

        os.environ["DOCENGINE_SINGLE_TENANT"] = "1"
        os.environ["DOCENGINE_STEPSTATS"] = "1"
        t0 = _time.time()
        try:
            from vllm.v1.engine.async_llm import AsyncLLM as Engine
        except ImportError:
            from vllm import AsyncLLMEngine as Engine
        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
        except ImportError:
            from vllm import AsyncEngineArgs
        self.engine = Engine.from_engine_args(AsyncEngineArgs(
            model=MODEL, kv_cache_dtype="fp8", max_model_len=4608,
            gpu_memory_utilization=0.92, enable_prefix_caching=True,
            disable_log_stats=True, scheduling_policy="priority",
            scheduler_cls="docengine.engineext.scheduler."
                          "DocEngineScheduler"))
        self.boot_seconds = round(_time.time() - t0, 1)
        print(f"[snap] engine boot {self.boot_seconds}s (captured in "
              f"the snapshot)", flush=True)

    @modal.method()
    async def specsmoke(self, n_docs: int = 500) -> dict:
        """The fork validation body, on the snapshotted engine: 4
        filters, selectivity 1, one-token answers; pipelined, forked
        speculation, and sequential speculation on one corpus."""
        import inspect
        import time as _time

        import numpy as np
        from transformers import AutoTokenizer
        from vllm import SamplingParams

        from docengine.runtime.engine_client import (
            run_filter_chain_engine)

        t_ready = _time.time()
        engine = self.engine
        docs = _build_pool(n_docs)
        tok = AutoTokenizer.from_pretrained(MODEL)
        n, s = 4, 1.0
        rng = np.random.default_rng(FLAG_SEED + 11)
        flags = (rng.random((len(docs), n)) < s).astype(int)
        bodies = [d + _flags_line(f) for d, f in zip(docs, flags)]
        body_ids = tok(bodies, add_special_tokens=False)["input_ids"]
        q_ids = [tok(_question(j + 1),
                     add_special_tokens=False)["input_ids"]
                 for j in range(n)]
        yes_ids = set()
        for w in ("YES", " YES", "Yes", " Yes", "Y", " Y"):
            ids = tok(w, add_special_tokens=False)["input_ids"]
            if ids:
                yes_ids.add(ids[0])
        corpus = sum(len(b) for b in body_ids)

        async def reset():
            res = engine.reset_prefix_cache()
            if inspect.isawaitable(res):
                await res

        sp = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1,
                            skip_clone=True)
        report = dict(n_docs=n_docs, n_filters=n, s=s, model=MODEL,
                      corpus_tokens=int(corpus), image=IMAGE_STAMP,
                      snapshot=True, boot_seconds=self.boot_seconds)
        runs = {}
        await reset()
        runs["pipelined"] = await run_filter_chain_engine(
            engine, sp, body_ids, q_ids, 700_000, yes_ids, tag="ps")
        await reset()
        runs["spec_forked"] = await run_filter_chain_engine(
            engine, sp, body_ids, q_ids, 700_000, yes_ids, tag="sf",
            spec=True)
        await reset()
        runs["spec_sequential"] = await run_filter_chain_engine(
            engine, sp, body_ids, q_ids, 700_000, yes_ids, tag="sq",
            spec=True, forked=False)
        for name, r in runs.items():
            wrong = sum(1 for k, v in r["answers"].items()
                        if v != flags[k[0]][k[1] - 1])
            reads = round((r.get("prompt_tokens", 0)
                           - r.get("cached_tokens", 0)) / corpus, 3)
            report[name] = dict(wall=round(r["wall"], 2),
                                requests=r["requests"],
                                survivors=len(r["survivors"]),
                                wrong=wrong, reads=reads)
            print(f"[snap-smoke] {name}: {r['wall']:.2f}s, reads "
                  f"{reads}, wrong {wrong}", flush=True)
        sf = runs["spec_forked"]["answers"]
        sq = runs["spec_sequential"]["answers"]
        report["fork_vs_sequential_mismatch"] = sum(
            1 for k in sq if sf.get(k) != sq[k])
        report["measure_seconds"] = round(_time.time() - t_ready, 1)
        print(f"[snap-smoke] fork vs sequential mismatches "
              f"{report['fork_vs_sequential_mismatch']}/{len(sq)}, "
              f"measurement span {report['measure_seconds']}s",
              flush=True)
        return report

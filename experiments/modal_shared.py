"""Shared corpus scans across concurrent queries, on an H100 through
Modal.

Phase "shared": 2,000 documents, each carrying 32 planted flags, and
q in {1, 2, 4, 8} synthetic query sets (four filters each at 0.8
selectivity, every query reading its own four flag columns, so no two
queries share a question). Each q runs twice on the same engine with a
prefix-cache reset between arms:

  shared    one corpus pass through run_shared_scan: every document is
            admitted once, its q chain requests launch together and
            share the document KV through the prefix cache, and its
            blocks stay pinned until the last query releases them.
  separate  the sum of q single-query chain-mode runs, cache reset
            before each, the way q independent queries would run today.

The printed table compares makespans, read multipliers ((prompt minus
cached) tokens over corpus tokens: 1.0 means the corpus was prefilled
exactly once), and truth-scored wrong answer counts.

Run with:
  modal run experiments/modal_shared.py --phase shared
"""

import gzip
import json

import modal

app = modal.App("docengine-shared")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # Pinned: the scheduler subclass reaches into a non-public engine
    # interface, so a silent version jump on image rebuild could break
    # it mid-study. Every recorded result is stamped with this version.
    .pip_install("vllm==0.26.0", "huggingface_hub", "pandas", "pyarrow",
                 "numpy", "yappi")
    .env({"VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("docengine")
)
hf_cache = modal.Volume.from_name("docengine-hf-cache", create_if_missing=True)

MODEL = "Qwen/Qwen3-4B-FP8"
WORKLOAD_SEED = 20260731
FLAG_SEED = 424242
N_FLAGS = 32              # 8 queries x 4 filters, all columns distinct


def _build_pool(n_docs):
    """Reproduce the repo's seeded 10k sample, then take its first n_docs."""
    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download

    frames = []
    for split in ("train", "test"):
        path = hf_hub_download(
            "stanfordnlp/imdb",
            f"plain_text/{split}-00000-of-00001.parquet",
            repo_type="dataset")
        frames.append(pd.read_parquet(path)["text"])
    pool = list(frames[0]) + list(frames[1])
    rng = np.random.default_rng(WORKLOAD_SEED)
    idx = sorted(rng.choice(len(pool), size=10_000, replace=False))
    return [pool[i] for i in idx[:n_docs]]


def _flags_line(flags):
    return "\n\n[FLAGS] " + " ".join(
        f"FLAG_{j+1}={'YES' if f else 'NO'}" for j, f in enumerate(flags))


def _question(j):
    return (f"\n\nExample: if the line said [FLAGS] FLAG_9=NO, then FLAG_9 "
            f"has value NO.\nInstruction: output only the value of FLAG_{j} "
            f"from the [FLAGS] line above.\nFLAG_{j}=")


@app.function(image=image, gpu="H100!", timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache})
async def shared_run(n_docs: int = 2000) -> dict:
    import inspect
    import os

    import numpy as np
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    from vllm.v1.engine.async_llm import AsyncLLM as Engine
    from vllm.engine.arg_utils import AsyncEngineArgs

    from docengine.runtime.engine_client import (run_filter_chain_engine,
                                                 run_shared_scan)

    os.environ["DOCENGINE_SINGLE_TENANT"] = "1"
    n, s = 4, 0.8
    docs = _build_pool(n_docs)
    rng = np.random.default_rng(FLAG_SEED + 77)
    flags = (rng.random((len(docs), N_FLAGS)) < s).astype(int)
    bodies = [d + _flags_line(f) for d, f in zip(docs, flags)]
    tok = AutoTokenizer.from_pretrained(MODEL)
    body_ids = tok(bodies, add_special_tokens=False)["input_ids"]
    corpus = sum(len(b) for b in body_ids)
    # query k reads flag columns 4k+1 .. 4k+4; no overlap between queries
    q_ids_of = [[tok(_question(4 * k + j + 1),
                     add_special_tokens=False)["input_ids"]
                 for j in range(n)] for k in range(8)]
    yes_ids = set()
    for w in ("YES", " YES", "Yes", " Yes", "Y", " Y"):
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if ids:
            yes_ids.add(ids[0])

    sp = SamplingParams(temperature=0.0, max_tokens=1, skip_clone=True)
    engine = Engine.from_engine_args(AsyncEngineArgs(
        model=MODEL, kv_cache_dtype="fp8", max_model_len=4608,
        gpu_memory_utilization=0.92, enable_prefix_caching=True,
        disable_log_stats=True, scheduling_policy="priority",
        scheduler_cls="docengine.engineext.scheduler.DocEngineScheduler"))
    pool = 981_728
    try:
        cc = engine.vllm_config.cache_config
        if cc.num_gpu_blocks:
            pool = int(cc.num_gpu_blocks) * int(cc.block_size)
    except Exception:
        pass
    budget = int(0.85 * pool)
    print(f"[shared] corpus {corpus} tokens, pool {pool}, budget {budget}",
          flush=True)

    async def reset():
        res = engine.reset_prefix_cache()
        if inspect.isawaitable(res):
            await res

    def wrong(answers, k):
        return sum(1 for (i, j), a in answers.items()
                   if a != flags[i][4 * k + j - 1])

    rows = []
    for q in (1, 2, 4, 8):
        queries = [dict(q_ids=q_ids_of[k], yes_ids=yes_ids)
                   for k in range(q)]

        await reset()
        sh = await run_shared_scan(engine, sp, body_ids, queries,
                                   budget, tag=f"sh{q}")
        sh_wrong = sum(wrong(sh["queries"][k]["answers"], k)
                       for k in range(q))
        sh_calls = sum(len(sh["queries"][k]["answers"]) for k in range(q))

        # the baseline: the same q queries as today's single-query
        # chain runs, one after another, cache reset between
        sep_wall = 0.0
        sep_prompt = sep_cached = sep_wrong = sep_calls = 0
        sep_survivors = []
        for k in range(q):
            await reset()
            r = await run_filter_chain_engine(
                engine, sp, body_ids, queries[k]["q_ids"], budget,
                yes_ids, tag=f"sp{q}x{k}")
            sep_wall += r["wall"]
            sep_prompt += r["prompt_tokens"]
            sep_cached += r["cached_tokens"]
            sep_wrong += wrong(r["answers"], k)
            sep_calls += len(r["answers"])
            sep_survivors.append(r["survivors"])
            same = r["survivors"] == sh["queries"][k]["survivors"]
            if not same:
                print(f"[shared] q={q} query {k}: shared and separate "
                      f"survivors differ "
                      f"({len(sh['queries'][k]['survivors'])} vs "
                      f"{len(r['survivors'])})", flush=True)

        sep_mult = (sep_prompt - sep_cached) / corpus
        rows.append(dict(
            q=q, shared_s=round(sh["wall"], 2),
            separate_s=round(sep_wall, 2),
            speedup=round(sep_wall / max(sh["wall"], 1e-9), 2),
            shared_read_mult=round(sh["read_multiplier"], 3),
            separate_read_mult=round(sep_mult, 3),
            shared_wrong=sh_wrong, shared_calls=sh_calls,
            separate_wrong=sep_wrong, separate_calls=sep_calls,
            shared_requests=sh["requests"],
            query_walls=[round(x["wall"], 2) for x in sh["queries"]],
            shared_survivors=[len(x["survivors"]) for x in sh["queries"]],
            separate_survivors=[len(x) for x in sep_survivors],
            survivors_match=[sh["queries"][k]["survivors"]
                             == sep_survivors[k] for k in range(q)]))
        r = rows[-1]
        print(f"[shared] q={q}: shared {r['shared_s']}s vs separate "
              f"{r['separate_s']}s ({r['speedup']}x), reads "
              f"{r['shared_read_mult']}x vs {r['separate_read_mult']}x "
              f"corpus, wrong {r['shared_wrong']}/{r['shared_calls']} vs "
              f"{r['separate_wrong']}/{r['separate_calls']}", flush=True)

    print(f"[shared] {'q':>2} {'shared':>8} {'separate':>9} {'x':>5} "
          f"{'rd_sh':>6} {'rd_sep':>7} {'wrong_sh':>9} {'wrong_sep':>10}",
          flush=True)
    for r in rows:
        print(f"[shared] {r['q']:>2} {r['shared_s']:>8} "
              f"{r['separate_s']:>9} {r['speedup']:>5} "
              f"{r['shared_read_mult']:>6} {r['separate_read_mult']:>7} "
              f"{r['shared_wrong']:>9} {r['separate_wrong']:>10}",
              flush=True)

    try:
        engine.shutdown()
    except Exception:
        pass
    import vllm
    return dict(model=MODEL, n_docs=n_docs, n_filters=n, s=s,
                n_flags=N_FLAGS, corpus_tokens=int(corpus),
                budget=budget, kv_tokens=pool,
                vllm_version=vllm.__version__, rows=rows)


@app.local_entrypoint()
def main(phase: str = "shared", n_docs: int = 0, out: str = ""):
    import os
    if phase == "shared":
        data = shared_run.remote(n_docs or 2000)
        path = out or "results/engine/shared2000.json.gz"
    else:
        raise SystemExit(f"unknown phase {phase}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".gz"):
        with gzip.open(path, "wt") as f:
            json.dump(data, f)
    else:
        with open(path, "w") as f:
            json.dump(data, f)
    print(f"saved {path}")

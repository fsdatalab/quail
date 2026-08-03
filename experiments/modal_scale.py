"""Steps 1 and 2 of the scheduler plan, on a real H100 through Modal.

Three independent experiments, selected by --phase:

  speed     Step 1a. The engine's reading speed limit. Sequential engine
            configurations (tokens packed per internal step, concurrent
            request limit, prefix caching on or off), each measured on
            reading-only jobs: short documents pre-converted to token
            numbers, the same documents as raw text (isolating repeated
            tokenization), and four-document concatenations (isolating
            short-sequence effects).

  overhead  Step 1b. The per request overhead split. One async engine,
            a four filter query over 2,000 documents, four arms in a two
            by two design: prompts as raw text versus pre-converted token
            numbers, and staged execution (send a stage, wait for every
            answer) versus streaming (each document advances the moment
            its own answer arrives). Each arm runs cold and then again
            warm, and the warm pass divided by its request count is that
            arm's per request floor.

  scale     Step 2. The 10,000 document cold experiment. The corpus
            needs about four times the card's note capacity, so execution
            order decides how much is silently re-read. Each
            configuration runs as naive stage-order waves and as the
            analytical builder's capacity-blocked schedule driven batch
            by batch (manifest mode), with the prefix cache reset before
            every run. The engine's cached-token counters report the
            re-read mass directly.

Run with:
  modal run experiments/modal_scale.py --phase speed
  modal run experiments/modal_scale.py --phase overhead --n-docs 400
  modal run experiments/modal_scale.py --phase overhead
  modal run experiments/modal_scale.py --phase scale
"""

import gzip
import json
import time

import modal

app = modal.App("docengine-scale")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "huggingface_hub", "pandas", "pyarrow", "numpy",
                 "yappi")
    .env({"VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("docengine")
)
hf_cache = modal.Volume.from_name("docengine-hf-cache", create_if_missing=True)

MODEL = "Qwen/Qwen3-4B-FP8"
WORKLOAD_SEED = 20260731
FLAG_SEED = 424242
CEIL = 275_000            # dense FP8 prefill ceiling, tokens per second


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


def _task_prefix(j):
    return (f"You will see a document ending in a [FLAGS] metadata line. "
            f"Your task: report whether FLAG_{j} is set to YES.\n\n"
            f"Document:\n")


def _task_suffix(j):
    return f"\n\nFrom the [FLAGS] line, FLAG_{j}="


def _answer_of(out):
    txt = out.outputs[0].text.strip().upper()
    return 1 if txt.startswith("Y") else 0


def _kv_tokens(llm):
    """The engine's actual KV pool size in tokens, if reachable."""
    for path in (("llm_engine", "cache_config"),
                 ("llm_engine", "vllm_config", "cache_config")):
        obj = llm
        try:
            for a in path:
                obj = getattr(obj, a)
            if obj.num_gpu_blocks:
                return int(obj.num_gpu_blocks) * int(obj.block_size)
        except Exception:
            continue
    return None


def _sched_cfg(llm):
    try:
        sc = llm.llm_engine.vllm_config.scheduler_config
        return dict(max_num_batched_tokens=int(sc.max_num_batched_tokens),
                    max_num_seqs=int(sc.max_num_seqs))
    except Exception:
        return {}


# ---------------------------------------------------------------- step 1a

@app.function(image=image, gpu="H100!", timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache})
def speed_limit(n_docs: int = 4000) -> dict:
    import gc

    import torch
    from vllm import LLM, SamplingParams

    docs = _build_pool(n_docs)
    sp = SamplingParams(temperature=0.0, max_tokens=1)
    configs = [
        dict(name="defaults_apc", apc=True, mnbt=None, seqs=None, util=0.92),
        dict(name="t8k_s256", apc=False, mnbt=8192, seqs=256, util=0.90),
        dict(name="t16k_s512", apc=False, mnbt=16384, seqs=512, util=0.90),
        dict(name="t32k_s1024", apc=False, mnbt=32768, seqs=1024, util=0.90),
        dict(name="t32k_s1024_apc", apc=True, mnbt=32768, seqs=1024,
             util=0.90),
    ]
    short_ids = None
    results = []
    for cfg in configs:
        kwargs = dict(model=MODEL, kv_cache_dtype="fp8", max_model_len=8192,
                      gpu_memory_utilization=cfg["util"],
                      enable_prefix_caching=cfg["apc"])
        if cfg["mnbt"]:
            kwargs["max_num_batched_tokens"] = cfg["mnbt"]
        if cfg["seqs"]:
            kwargs["max_num_seqs"] = cfg["seqs"]
        llm = LLM(**kwargs)
        if short_ids is None:
            tok = llm.get_tokenizer()
            short_ids = tok(docs, add_special_tokens=False)["input_ids"]
            long_ids = [sum(short_ids[i:i + 4], [])[:8000]
                        for i in range(0, len(short_ids), 4)]
        arms = [
            ("short_ids", [{"prompt_token_ids": x} for x in short_ids]),
            ("long_ids", [{"prompt_token_ids": x} for x in long_ids]),
            ("short_text", docs),
        ]
        runs = []
        for name, prompts in arms:
            t0 = time.time()
            outs = llm.generate(prompts, sp, use_tqdm=False)
            dt = time.time() - t0
            toks = sum(len(o.prompt_token_ids) for o in outs)
            cached = sum(getattr(o, "num_cached_tokens", 0) or 0
                         for o in outs)
            rate = (toks - cached) / dt
            runs.append(dict(arm=name, s=dt, prompt_tokens=toks,
                             cached_tokens=cached, rate=rate,
                             pct_of_ceiling=rate / CEIL))
            print(f"[speed] {cfg['name']} {name}: {toks} tokens in "
                  f"{dt:.2f}s = {rate:,.0f} tok/s "
                  f"({100 * rate / CEIL:.1f}% of ceiling)", flush=True)
            if cfg["apc"]:
                try:
                    llm.reset_prefix_cache()
                except Exception:
                    pass
        results.append(dict(cfg=cfg, sched=_sched_cfg(llm),
                            kv_tokens=_kv_tokens(llm), runs=runs))
        try:
            llm.shutdown()
        except Exception:
            pass
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(8)
    import vllm
    return dict(model=MODEL, n_docs=n_docs, vllm_version=vllm.__version__,
                ceiling=CEIL, results=results)


# ---------------------------------------------------------------- step 1b

@app.function(image=image, gpu="H100!", timeout=2400,
              volumes={"/root/.cache/huggingface": hf_cache})
async def overhead_split(n_docs: int = 2000) -> dict:
    import asyncio
    import inspect

    import numpy as np
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    try:
        from vllm.v1.engine.async_llm import AsyncLLM as Engine
    except ImportError:
        from vllm import AsyncLLMEngine as Engine
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
    except ImportError:
        from vllm import AsyncEngineArgs

    n, s_vec = 4, (0.8, 0.8, 0.8, 0.8)
    docs = _build_pool(n_docs)
    rng = np.random.default_rng(FLAG_SEED + 1000 * n + int(100 * s_vec[0]))
    flags = (rng.random((len(docs), n)) < np.asarray(s_vec)).astype(int)
    bodies = [d + _flags_line(f) for d, f in zip(docs, flags)]

    tok = AutoTokenizer.from_pretrained(MODEL)
    body_ids = tok(bodies, add_special_tokens=False)["input_ids"]
    q_ids = [tok(_question(j + 1), add_special_tokens=False)["input_ids"]
             for j in range(n)]

    engine = Engine.from_engine_args(AsyncEngineArgs(
        model=MODEL, kv_cache_dtype="fp8", max_model_len=4608,
        gpu_memory_utilization=0.92, enable_prefix_caching=True,
        disable_log_stats=True))
    sp = SamplingParams(temperature=0.0, max_tokens=1)

    async def reset_cache():
        res = engine.reset_prefix_cache()
        if inspect.isawaitable(res):
            res = await res
        return res

    async def ask(prompt, rid):
        final = None
        async for out in engine.generate(prompt, sp, rid):
            final = out
        return final

    def prompt_for(i, j, form):
        if form == "ids":
            return {"prompt_token_ids": body_ids[i] + q_ids[j - 1]}
        return bodies[i] + _question(j)

    counters = dict(requests=0, prompt_tokens=0, cached_tokens=0)

    def count(out):
        counters["requests"] += 1
        counters["prompt_tokens"] += len(out.prompt_token_ids)
        counters["cached_tokens"] += getattr(out, "num_cached_tokens", 0) or 0

    async def one_pass(mode, form, tag):
        for key in counters:
            counters[key] = 0
        answers = {}
        t0 = time.time()
        if mode == "staged":
            alive = list(range(len(docs)))
            for j in range(1, n + 1):
                outs = await asyncio.gather(
                    *[ask(prompt_for(i, j, form), f"{tag}-{i}-{j}")
                      for i in alive])
                nxt = []
                for i, out in zip(alive, outs):
                    count(out)
                    a = _answer_of(out)
                    answers[(i, j)] = a
                    if a:
                        nxt.append(i)
                alive = nxt
        else:
            async def chain(i):
                for j in range(1, n + 1):
                    out = await ask(prompt_for(i, j, form), f"{tag}-{i}-{j}")
                    count(out)
                    a = _answer_of(out)
                    answers[(i, j)] = a
                    if not a:
                        break
            await asyncio.gather(*[chain(i) for i in range(len(docs))])
        wall = time.time() - t0
        agree = sum(1 for (i, j), a in answers.items()
                    if a == flags[i][j - 1]) / max(1, len(answers))
        return dict(wall=wall, agreement=agree, **counters)

    arms = []
    for mode in ("staged", "stream"):
        for form in ("text", "ids"):
            await reset_cache()
            cold = await one_pass(mode, form, f"{mode}-{form}-c")
            warm = await one_pass(mode, form, f"{mode}-{form}-w")
            arms.append(dict(mode=mode, form=form, cold=cold, warm=warm))
            print(f"[overhead] {mode}/{form}: cold {cold['wall']:.2f}s "
                  f"({cold['requests']} reqs), warm {warm['wall']:.2f}s = "
                  f"{1000 * warm['wall'] / warm['requests']:.2f} ms/req, "
                  f"agree {cold['agreement']:.3f}", flush=True)

    try:
        engine.shutdown()
    except Exception:
        pass
    import vllm
    return dict(model=MODEL, n_docs=n_docs, n=n, s=list(s_vec),
                vllm_version=vllm.__version__, arms=arms)


# ------------------------------------------------------------- phase C

@app.function(image=image, gpu="H100!", timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache})
async def pinned_run(variant: str = "pinned", n_docs: int = 10000,
                     junk_rate: float = 25.0, junk_len: int = 800,
                     junk_only: bool = False) -> dict:
    """Phase C acceptance: the in-engine scheduler (pinning plus
    priorities) against the stock engine, each with and without an
    adversarial co-tenant stream that hammers the cache."""
    import asyncio
    import inspect

    import numpy as np
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    try:
        from vllm.v1.engine.async_llm import AsyncLLM as Engine
    except ImportError:
        from vllm import AsyncLLMEngine as Engine
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
    except ImportError:
        from vllm import AsyncEngineArgs

    from docengine.runtime.engine_client import EngineTags, run_filter_chain

    import os
    os.environ["DOCENGINE_SINGLE_TENANT"] = "0"   # co-tenant is legitimate
    ext = variant == "pinned"
    kwargs = dict(model=MODEL, kv_cache_dtype="fp8", max_model_len=4608,
                  gpu_memory_utilization=0.92, enable_prefix_caching=True,
                  disable_log_stats=True)
    if ext:
        kwargs["scheduling_policy"] = "priority"
        kwargs["scheduler_cls"] = \
            "docengine.engineext.scheduler.DocEngineScheduler"
    docs = _build_pool(n_docs)
    engine = Engine.from_engine_args(AsyncEngineArgs(**kwargs))
    tok = AutoTokenizer.from_pretrained(MODEL)
    sp = SamplingParams(temperature=0.0, max_tokens=1, skip_clone=True)
    pool = 981_728
    try:
        cc = engine.vllm_config.cache_config
        if cc.num_gpu_blocks:
            pool = int(cc.num_gpu_blocks) * int(cc.block_size)
    except Exception:
        pass
    budget = int((0.85 if ext else 0.9) * pool)
    print(f"[pinned] variant={variant} pool={pool} budget={budget}",
          flush=True)

    async def reset_cache():
        for _ in range(15):
            res = engine.reset_prefix_cache()
            if inspect.isawaitable(res):
                res = await res
            if res is not False:
                return
            await asyncio.sleep(1.0)
        raise RuntimeError("prefix cache reset kept failing")

    async def cotenant(stop, stats, rate=junk_rate, length=junk_len):
        rng = np.random.default_rng(4321)
        jobs = []

        async def one(uid, ids):
            t0 = time.time()
            kw = {"priority": 2} if ext else {}
            async for _ in engine.generate({"prompt_token_ids": ids}, sp,
                                           f"junk-{uid}", **kw):
                pass
            stats["done"] += 1
            stats["lat_s"] += time.time() - t0

        uid = 0
        while not stop.is_set():
            uid += 1
            ids = rng.integers(1000, 100_000, size=length).tolist()
            jobs.append(asyncio.create_task(one(uid, ids)))
            await asyncio.sleep(1.0 / rate)
        await asyncio.gather(*jobs, return_exceptions=True)

    if junk_only:
        grid = [(4, (0.8,) * 4, False), (4, (0.8,) * 4, True)]
    else:
        grid = [(2, (0.5, 0.5), False), (4, (0.8,) * 4, False),
                (4, (0.95,) * 4, False), (4, (0.8,) * 4, True)] if ext \
            else [(4, (0.8,) * 4, False), (4, (0.8,) * 4, True)]

    results = []
    for n, s_vec, junk in grid:
        rng = np.random.default_rng(FLAG_SEED + 1000 * n + int(100 * s_vec[0]))
        flags = (rng.random((len(docs), n)) < np.asarray(s_vec)).astype(int)
        bodies = [d + _flags_line(f) for d, f in zip(docs, flags)]
        body_ids = tok(bodies, add_special_tokens=False)["input_ids"]
        q_ids = [tok(_question(j + 1), add_special_tokens=False)["input_ids"]
                 for j in range(n)]
        await reset_cache()
        stop = asyncio.Event()
        stats = dict(done=0, lat_s=0.0)
        bg = asyncio.create_task(cotenant(stop, stats)) if junk else None
        res = await run_filter_chain(
            engine, sp, body_ids, q_ids, budget, lookahead=1,
            tag=f"n{n}s{int(100 * s_vec[0])}",
            tags=EngineTags() if ext else None, use_priority=ext)
        if bg is not None:
            stop.set()
            await bg
        answers = {f"{i},{j}": a for (i, j), a in res["answers"].items()}
        agree = sum(1 for (i, j), a in res["answers"].items()
                    if a == flags[i][j - 1]) / max(1, len(answers))
        results.append(dict(
            n=n, s=list(s_vec), policy="block", k=1, mode=variant,
            junk=junk, junk_rate=junk_rate, junk_len=junk_len,
            junk_stats=stats, makespan=res["wall"],
            waves=[dict(stage=1, requests=res["requests"], s=res["wall"],
                        prompt_tokens=res["prompt_tokens"],
                        cached_tokens=res["cached_tokens"])],
            d_tok=[len(x) for x in body_ids],
            p_tok=[len(q) for q in q_ids],
            p_task=[0] * n, answers=answers, flags=flags.tolist(),
            answer_agreement=agree, budget=budget, kv_tokens=pool))
        hit = res["cached_tokens"] / max(1, res["prompt_tokens"])
        print(f"[pinned] {variant} n={n} s={s_vec[0]} junk={junk}: "
              f"{res['wall']:.1f}s, hit {100 * hit:.1f}%, "
              f"junk done {stats['done']}, agree {agree:.3f}", flush=True)

    try:
        engine.shutdown()
    except Exception:
        pass
    import vllm
    return dict(model=MODEL, n_docs=n_docs, variant=variant,
                vllm_version=vllm.__version__, results=results)


# ---------------------------------------------------------- profiling

@app.function(image=image, gpu="H100!", timeout=3000,
              volumes={"/root/.cache/huggingface": hf_cache})
async def profile_run(n_docs: int = 4000) -> dict:
    """Split the per-request software bundle without attach profiling
    (the sandbox forbids it). Three runs of the same workload:

      A  engine core in its own process, unprofiled: the shipped
         baseline makespan.
      B  engine core in-process, unprofiled: the makespan delta A - B
         is the direct price of cross-process serialization.
      C  engine core in-process under yappi (all threads, CPU clock):
         a ranked table of where the remaining software time goes,
         with waits excluded by the CPU clock."""
    import gc
    import inspect
    import os

    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    try:
        from vllm.v1.engine.async_llm import AsyncLLM as Engine
    except ImportError:
        from vllm import AsyncLLMEngine as Engine
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
    except ImportError:
        from vllm import AsyncEngineArgs

    from docengine.runtime.engine_client import EngineTags, run_filter_chain

    os.environ["DOCENGINE_SINGLE_TENANT"] = "1"
    n, s_vec = 4, (0.8,) * 4
    docs = _build_pool(n_docs)
    tok = AutoTokenizer.from_pretrained(MODEL)
    sp = SamplingParams(temperature=0.0, max_tokens=1, skip_clone=True)
    rng = np.random.default_rng(FLAG_SEED + 1000 * n + int(100 * s_vec[0]))
    flags = (rng.random((len(docs), n)) < np.asarray(s_vec)).astype(int)
    bodies = [d + _flags_line(f) for d, f in zip(docs, flags)]
    body_ids = tok(bodies, add_special_tokens=False)["input_ids"]
    q_ids = [tok(_question(j + 1), add_special_tokens=False)["input_ids"]
             for j in range(n)]

    def make_engine():
        return Engine.from_engine_args(AsyncEngineArgs(
            model=MODEL, kv_cache_dtype="fp8", max_model_len=4608,
            gpu_memory_utilization=0.92, enable_prefix_caching=True,
        disable_log_stats=True,
            scheduling_policy="priority",
            scheduler_cls="docengine.engineext.scheduler."
                          "DocEngineScheduler"))

    async def one_query(engine, tag):
        res = engine.reset_prefix_cache()
        if inspect.isawaitable(res):
            await res
        out = await run_filter_chain(engine, sp, body_ids, q_ids,
                                     budget_tokens=830_000, lookahead=1,
                                     tag=tag, tags=EngineTags(),
                                     use_priority=True)
        return out

    report = dict(n_docs=n_docs)

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    engine = make_engine()
    out = await one_query(engine, "profa")
    report["A_multiproc_s"] = out["wall"]
    report["requests"] = out["requests"]
    print(f"[profile] A multiproc: {out['wall']:.2f}s "
          f"({out['requests']} requests)", flush=True)
    try:
        engine.shutdown()
    except Exception:
        pass
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    import time as _t
    _t.sleep(8)

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    engine = make_engine()
    out = await one_query(engine, "profb")
    report["B_inproc_s"] = out["wall"]
    print(f"[profile] B in-process: {out['wall']:.2f}s "
          f"(IPC price ~ {report['A_multiproc_s'] - out['wall']:+.2f}s)",
          flush=True)

    import yappi
    yappi.set_clock_type("cpu")
    yappi.start()
    out = await one_query(engine, "profc")
    yappi.stop()
    report["C_profiled_s"] = out["wall"]
    stats = yappi.get_func_stats()
    rows = []
    for st in stats:
        rows.append(dict(fn=f"{st.module.split('/')[-1]}:{st.name}",
                         self_s=st.tsub, total_s=st.ttot,
                         calls=st.ncall))
    rows.sort(key=lambda r: -r["self_s"])
    total_cpu = sum(r["self_s"] for r in rows) or 1.0
    report["top"] = rows[:40]
    report["total_cpu_s"] = total_cpu
    print(f"[profile] C profiled run: {out['wall']:.2f}s, total CPU "
          f"{total_cpu:.1f}s", flush=True)
    for r in rows[:30]:
        print(f"[profile] {100 * r['self_s'] / total_cpu:5.1f}% "
              f"{r['self_s']:7.2f}s self {r['calls']:>9} calls  "
              f"{r['fn'][:80]}", flush=True)
    try:
        engine.shutdown()
    except Exception:
        pass
    return report


@app.function(image=image, gpu="H100!", timeout=2400,
              volumes={"/root/.cache/huggingface": hf_cache})
async def chainprof_run(n_docs: int = 4000) -> dict:
    """Name where chain mode's extra seconds go. The 10k flight showed
    chain mode at 56.8 seconds against request mode's 52.4: each
    continuation (stop, rewind, park, full worker re-sync) costs more
    than the per-request toll it replaces. This runs both modes on one
    in-process engine, each under the CPU-clock profiler, and returns
    ranked tables so the difference has function names."""
    import gc
    import inspect
    import os

    import numpy as np
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    try:
        from vllm.v1.engine.async_llm import AsyncLLM as Engine
    except ImportError:
        from vllm import AsyncLLMEngine as Engine
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
    except ImportError:
        from vllm import AsyncEngineArgs

    from docengine.runtime.engine_client import (EngineTags,
                                                 run_filter_chain,
                                                 run_filter_chain_engine)

    os.environ["DOCENGINE_SINGLE_TENANT"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    n, s = 4, 0.8
    docs = _build_pool(n_docs)
    tok = AutoTokenizer.from_pretrained(MODEL)
    sp = SamplingParams(temperature=0.0, max_tokens=1, skip_clone=True)
    rng = np.random.default_rng(FLAG_SEED + 7)
    flags = (rng.random((len(docs), n)) < s).astype(int)
    bodies = [d + _flags_line(f) for d, f in zip(docs, flags)]
    body_ids = tok(bodies, add_special_tokens=False)["input_ids"]
    q_ids = [tok(_question(j + 1), add_special_tokens=False)["input_ids"]
             for j in range(n)]
    yes_ids = set()
    for w in ("YES", " YES", "Yes", " Yes", "Y", " Y"):
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if ids:
            yes_ids.add(ids[0])

    engine = Engine.from_engine_args(AsyncEngineArgs(
        model=MODEL, kv_cache_dtype="fp8", max_model_len=4608,
        gpu_memory_utilization=0.92, enable_prefix_caching=True,
        disable_log_stats=True, scheduling_policy="priority",
        scheduler_cls="docengine.engineext.scheduler.DocEngineScheduler"))

    async def reset_cache():
        res = engine.reset_prefix_cache()
        if inspect.isawaitable(res):
            await res

    import yappi
    yappi.set_clock_type("cpu")

    def table():
        rows = []
        for st in yappi.get_func_stats():
            rows.append(dict(fn=f"{st.module.split('/')[-1]}:{st.name}",
                             self_s=st.tsub, total_s=st.ttot,
                             calls=st.ncall))
        rows.sort(key=lambda r: -r["self_s"])
        return rows

    report = dict(n_docs=n_docs)
    for mode in ("request", "chain"):
        await reset_cache()
        gc.collect()
        yappi.clear_stats()
        yappi.start()
        if mode == "request":
            out = await run_filter_chain(
                engine, sp, body_ids, q_ids, budget_tokens=830_000,
                lookahead=1, tag="cpa", tags=EngineTags(),
                use_priority=True)
        else:
            out = await run_filter_chain_engine(
                engine, sp, body_ids, q_ids, 830_000, yes_ids, tag="cpb")
        yappi.stop()
        rows = table()
        total_cpu = sum(r["self_s"] for r in rows) or 1.0
        report[mode] = dict(wall=out["wall"], requests=out["requests"],
                            total_cpu_s=total_cpu, top=rows[:60])
        print(f"[chainprof] {mode} mode: {out['wall']:.2f}s wall, "
              f"{out['requests']} requests, total CPU {total_cpu:.1f}s",
              flush=True)
        for r in rows[:30]:
            print(f"[chainprof] {mode} {100 * r['self_s'] / total_cpu:5.1f}% "
                  f"{r['self_s']:7.2f}s self {r['calls']:>9} calls  "
                  f"{r['fn'][:80]}", flush=True)
    try:
        engine.shutdown()
    except Exception:
        pass
    return report


# ------------------------------------------------------ long documents

@app.function(image=image, gpu="H100!", timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache})
async def longdoc_run() -> dict:
    """Long-document validation on the shipped configuration: the model
    predicts the read floor with policy differences compressed. Uses a
    YaRN rope-scaling override to reach 100k contexts (native limit is
    32k); noted in the results."""
    import inspect
    import os

    import numpy as np
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    try:
        from vllm.v1.engine.async_llm import AsyncLLM as Engine
    except ImportError:
        from vllm import AsyncLLMEngine as Engine
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
    except ImportError:
        from vllm import AsyncEngineArgs

    from docengine.runtime.engine_client import EngineTags, run_filter_chain

    os.environ["DOCENGINE_SINGLE_TENANT"] = "1"
    tok = AutoTokenizer.from_pretrained(MODEL)
    pool = _build_pool(10000)
    lens = [len(x) for x in
            tok(pool, add_special_tokens=False)["input_ids"]]

    def build_long(count, target):
        docs, i = [], 0
        for _ in range(count):
            parts, total = [], 0
            while total < target - 400:
                parts.append(pool[i % len(pool)])
                total += lens[i % len(pool)] + 2
                i += 1
            docs.append("\n\n".join(parts))
        return docs

    engine = Engine.from_engine_args(AsyncEngineArgs(
        model=MODEL, kv_cache_dtype="fp8", max_model_len=102_400,
        gpu_memory_utilization=0.92, enable_prefix_caching=True,
        disable_log_stats=True,
        scheduling_policy="priority",
        scheduler_cls="docengine.engineext.scheduler.DocEngineScheduler",
        hf_overrides={"rope_scaling": {
            "rope_type": "yarn", "factor": 4.0,
            "original_max_position_embeddings": 32768}}))
    sp = SamplingParams(temperature=0.0, max_tokens=1, skip_clone=True)
    n, s = 2, 0.7

    async def reset_cache():
        res = engine.reset_prefix_cache()
        if inspect.isawaitable(res):
            await res

    results = []
    for count, target, k in ((100, 30_000, 1), (100, 30_000, 2),
                             (30, 100_000, 1)):
        docs = build_long(count, target)
        rng = np.random.default_rng(FLAG_SEED + count)
        flags = (rng.random((count, n)) < s).astype(int)
        bodies = [d + _flags_line(f) for d, f in zip(docs, flags)]
        body_ids = tok(bodies, add_special_tokens=False)["input_ids"]
        q_ids = [tok(_question(j + 1), add_special_tokens=False)
                 ["input_ids"] for j in range(n)]
        corpus = sum(len(x) for x in body_ids)
        await reset_cache()
        res = await run_filter_chain(engine, sp, body_ids, q_ids,
                                     budget_tokens=830_000, lookahead=k,
                                     tag=f"ld{target}k{k}",
                                     tags=EngineTags(), use_priority=True)
        agree = sum(1 for (i, j), a in res["answers"].items()
                    if a == flags[i][j - 1]) / max(1, len(res["answers"]))
        floor = corpus / 80_000.0
        hit = res["cached_tokens"] / max(1, res["prompt_tokens"])
        results.append(dict(count=count, target=target, k=k,
                            corpus_tokens=corpus, makespan=res["wall"],
                            floor_s=floor, ratio=res["wall"] / floor,
                            cache_hit=hit, agreement=agree,
                            requests=res["requests"]))
        print(f"[longdoc] {count}x{target} k={k}: {res['wall']:.1f}s, "
              f"floor {floor:.1f}s, ratio {res['wall'] / floor:.2f}, "
              f"hit {100 * hit:.1f}%, agree {agree:.3f}", flush=True)
    try:
        engine.shutdown()
    except Exception:
        pass
    import vllm
    return dict(model=MODEL, vllm_version=vllm.__version__,
                rope_scaling="yarn x4", results=results)




# ------------------------------------------------------------ chain proof

@app.function(image=image, gpu="H100!", timeout=2400,
              volumes={"/root/.cache/huggingface": hf_cache})
async def chain_run(n_docs: int = 50, n_filters: int = 2,
                    s: float = 0.7, profile_core: int = 0) -> dict:
    """Sequence truncation proof at any scale: the same documents run
    the old way (one request per filter) and the new way (one request
    per document, the engine rewinding between filters). Pass means the
    surviving documents match exactly and the rewind count equals the
    number of chain continuations."""
    import inspect
    import os

    import numpy as np
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    try:
        from vllm.v1.engine.async_llm import AsyncLLM as Engine
    except ImportError:
        from vllm import AsyncLLMEngine as Engine
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
    except ImportError:
        from vllm import AsyncEngineArgs

    from docengine.runtime.engine_client import (EngineTags,
                                                 run_filter_chain,
                                                 run_filter_chain_engine)

    os.environ["DOCENGINE_SINGLE_TENANT"] = "1"
    if profile_core:
        os.environ["DOCENGINE_PROFILE"] = "1"
    docs = _build_pool(n_docs)
    engine = Engine.from_engine_args(AsyncEngineArgs(
        model=MODEL, kv_cache_dtype="fp8", max_model_len=4608,
        gpu_memory_utilization=0.92, enable_prefix_caching=True,
        disable_log_stats=True, scheduling_policy="priority",
        scheduler_cls="docengine.engineext.scheduler.DocEngineScheduler"))
    tok = AutoTokenizer.from_pretrained(MODEL)
    sp = SamplingParams(temperature=0.0, max_tokens=1, skip_clone=True)
    n = n_filters
    rng = np.random.default_rng(FLAG_SEED + 7)
    flags = (rng.random((len(docs), n)) < s).astype(int)
    bodies = [d + _flags_line(f) for d, f in zip(docs, flags)]
    body_ids = tok(bodies, add_special_tokens=False)["input_ids"]
    q_ids = [tok(_question(j + 1), add_special_tokens=False)["input_ids"]
             for j in range(n)]
    yes_ids = set()
    for w in ("YES", " YES", "Yes", " Yes", "Y", " Y"):
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if ids:
            yes_ids.add(ids[0])

    async def reset_cache():
        res = engine.reset_prefix_cache()
        if inspect.isawaitable(res):
            await res

    a = await run_filter_chain(engine, sp, body_ids, q_ids, 830_000,
                               lookahead=1, tag="rm", tags=EngineTags(),
                               use_priority=True)
    await reset_cache()
    b = await run_filter_chain_engine(engine, sp, body_ids, q_ids,
                                      830_000, yes_ids, tag="cm")
    raw = b["answers"].pop(("raw", 0), None)
    same_surv = a["survivors"] == b["survivors"]
    shared = [k for k in a["answers"] if k in b["answers"]]
    agree = sum(1 for k in shared if a["answers"][k] == b["answers"][k])
    flips = sorted(k for k in shared if a["answers"][k] != b["answers"][k])
    if flips:
        print(f"[chain] flipped answers (doc, stage): {flips[:20]}",
              flush=True)
    print(f"[chain] request mode: {a['requests']} requests, "
          f"{a['wall']:.2f}s; chain mode: {b['requests']} requests, "
          f"{b['wall']:.2f}s", flush=True)
    print(f"[chain] survivors match: {same_surv} "
          f"({len(a['survivors'])} vs {len(b['survivors'])}); shared "
          f"answers agree {agree}/{len(shared)}", flush=True)
    print(f"[chain] doc0 raw snapshots: {raw}", flush=True)
    try:
        engine.shutdown()
    except Exception:
        pass
    return dict(n_docs=n_docs, n_filters=n_filters, s=s,
                request_mode=dict(requests=a["requests"], wall=a["wall"],
                                  survivors=a["survivors"]),
                chain_mode=dict(requests=b["requests"], wall=b["wall"],
                                survivors=b["survivors"]),
                survivors_match=same_surv,
                answers_agree=[agree, len(shared)],
                flipped=[list(k) for k in flips[:50]],
                doc0_raw=[list(x) for x in (raw or ())])

# ---------------------------------------------------- single tenant mode

@app.function(image=image, gpu="H100!", timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache})
async def strict_run(n_docs: int = 10000) -> dict:
    """Shipping-mode validation: single tenant strict, where the
    recency rule is unreachable and any firing raises. Runs the
    10k grid and one untagged canary request that must be refused."""
    import asyncio  # noqa: F401
    import inspect
    import os

    import numpy as np
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    try:
        from vllm.v1.engine.async_llm import AsyncLLM as Engine
    except ImportError:
        from vllm import AsyncLLMEngine as Engine
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
    except ImportError:
        from vllm import AsyncEngineArgs

    from docengine.runtime.engine_client import EngineTags, run_filter_chain

    os.environ["DOCENGINE_SINGLE_TENANT"] = "1"
    docs = _build_pool(n_docs)
    engine = Engine.from_engine_args(AsyncEngineArgs(
        model=MODEL, kv_cache_dtype="fp8", max_model_len=4608,
        gpu_memory_utilization=0.92, enable_prefix_caching=True,
        disable_log_stats=True,
        scheduling_policy="priority",
        scheduler_cls="docengine.engineext.scheduler.DocEngineScheduler"))
    tok = AutoTokenizer.from_pretrained(MODEL)
    sp = SamplingParams(temperature=0.0, max_tokens=1, skip_clone=True)
    pool = 981_728
    try:
        cc = engine.vllm_config.cache_config
        if cc.num_gpu_blocks:
            pool = int(cc.num_gpu_blocks) * int(cc.block_size)
    except Exception:
        pass
    budget = int(0.85 * pool)

    async def reset_cache():
        res = engine.reset_prefix_cache()
        if inspect.isawaitable(res):
            await res

    # canary: an untagged request must be refused, not served. The
    # scheduler aborts it before its first step, so no final output
    # ever reaches this generator; a bounded wait treats never-served
    # as refused and cleans up the client-side request state.
    canary = None
    rejected = False

    async def _canary():
        nonlocal canary
        async for out in engine.generate({"prompt_token_ids": [100] * 8},
                                         sp, "canary-1"):
            canary = out

    try:
        await asyncio.wait_for(_canary(), timeout=60)
        rejected = bool(canary and canary.outputs
                        and canary.outputs[0].finish_reason == "abort")
    except (asyncio.TimeoutError, Exception):
        rejected = canary is None
        try:
            res = engine.abort("canary-1")
            if inspect.isawaitable(res):
                await res
        except Exception:
            pass
    print(f"[strict] untagged canary refused: {rejected}", flush=True)

    results = []
    for n, s_vec in ((2, (0.5, 0.5)), (4, (0.8,) * 4), (4, (0.95,) * 4)):
        rng = np.random.default_rng(FLAG_SEED + 1000 * n + int(100 * s_vec[0]))
        flags = (rng.random((len(docs), n)) < np.asarray(s_vec)).astype(int)
        bodies = [d + _flags_line(f) for d, f in zip(docs, flags)]
        body_ids = tok(bodies, add_special_tokens=False)["input_ids"]
        q_ids = [tok(_question(j + 1), add_special_tokens=False)["input_ids"]
                 for j in range(n)]
        await reset_cache()
        res = await run_filter_chain(
            engine, sp, body_ids, q_ids, budget, lookahead=1,
            tag=f"n{n}s{int(100 * s_vec[0])}", tags=EngineTags(),
            use_priority=True)
        answers = {f"{i},{j}": a for (i, j), a in res["answers"].items()}
        agree = sum(1 for (i, j), a in res["answers"].items()
                    if a == flags[i][j - 1]) / max(1, len(answers))
        results.append(dict(
            n=n, s=list(s_vec), policy="block", k=1, mode="strict",
            junk=False, makespan=res["wall"],
            waves=[dict(stage=1, requests=res["requests"], s=res["wall"],
                        prompt_tokens=res["prompt_tokens"],
                        cached_tokens=res["cached_tokens"])],
            d_tok=[len(x) for x in body_ids],
            p_tok=[len(q) for q in q_ids], p_task=[0] * n,
            answers=answers, flags=flags.tolist(),
            answer_agreement=agree, budget=budget, kv_tokens=pool,
            canary_rejected=rejected))
        hit = res["cached_tokens"] / max(1, res["prompt_tokens"])
        print(f"[strict] n={n} s={s_vec[0]}: {res['wall']:.1f}s, "
              f"hit {100 * hit:.1f}%, agree {agree:.3f}", flush=True)

    try:
        engine.shutdown()
    except Exception:
        pass
    import vllm
    return dict(model=MODEL, n_docs=n_docs, vllm_version=vllm.__version__,
                canary_rejected=rejected, results=results)


# ------------------------------------------------------- beneath the stack

@app.function(image=image, gpu="H100!", timeout=1800,
              volumes={"/root/.cache/huggingface": hf_cache})
def model_floor() -> dict:
    """Measure reading speed with no serving stack at all, to split
    vLLM's 80k tokens per second into silicon, model, and serving tax.

    Part A: the four per-layer matrix multiplies of Qwen3-4B at their
    exact shapes, in FP8 and bf16. Their per-token cost is 7.27e9
    operations, which matches the 2P pricing convention, so measured
    arithmetic rate divides directly into a tokens-per-second
    equivalent: the kernel-only ceiling.

    Part B: a bare forward pass of the bf16 model through the plain
    transformers library on packed 352-token rows of the real corpus,
    no engine, no scheduler, no per-request objects. The ratio of this
    to the bf16 kernel-only ceiling is the model tax (attention,
    normalization, memory-bound ops, launches). Applying the same tax
    to the FP8 kernel ceiling projects the FP8 model floor, and vLLM's
    measured 80k against that floor is the serving tax."""
    import torch

    dev = "cuda"
    SHAPES = [(2560, 6144), (4096, 2560), (2560, 19456), (9728, 2560)]
    LAYERS = 36
    FLOP_TOK = 2 * LAYERS * sum(k * n for k, n in SHAPES)   # ~7.27e9

    def bench(fn, iters=50, warmup=10):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
        t0.record()
        for _ in range(iters):
            fn()
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) / 1000 / iters

    def gemm_tok_rate(dtype, M):
        t = 0.0
        for k, n in SHAPES:
            if dtype == "fp8":
                a = torch.randn(M, k, device=dev).to(torch.float8_e4m3fn)
                b = torch.randn(n, k, device=dev).to(torch.float8_e4m3fn).t()
                s = torch.ones((), device=dev)

                def fn(a=a, b=b, s=s):
                    r = torch._scaled_mm(a, b, scale_a=s, scale_b=s,
                                         out_dtype=torch.bfloat16)
                    return r[0] if isinstance(r, tuple) else r
            else:
                a = torch.randn(M, k, device=dev, dtype=torch.bfloat16)
                b = torch.randn(k, n, device=dev, dtype=torch.bfloat16)

                def fn(a=a, b=b):
                    return a @ b
            t += bench(fn)
        toks = M / (t * LAYERS)          # M tokens need LAYERS x these GEMMs
        flops = FLOP_TOK * toks
        return toks, flops / 1e12

    out = dict(flop_per_token=FLOP_TOK, gemm={})
    for dtype in ("fp8", "bf16"):
        for M in (2048, 8192, 16384):
            toks, tf = gemm_tok_rate(dtype, M)
            out["gemm"][f"{dtype}_M{M}"] = dict(tok_s=toks, tflops=tf)
            print(f"[floor] GEMM {dtype} M={M}: {toks:,.0f} tok/s equiv "
                  f"({tf:.0f} TFLOP/s)", flush=True)

    # Part B: bare bf16 model forward on packed real-corpus rows
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    docs = _build_pool(2000)
    ids = tok("\n\n".join(docs), add_special_tokens=False)["input_ids"]
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to(dev).eval()
    out["bare_model"] = {}
    with torch.inference_mode():
        for B in (16, 64):
            rows = B * 352
            x = torch.tensor(ids[:rows], device=dev).view(B, 352)
            sec = bench(lambda: model(x, use_cache=False), iters=20,
                        warmup=5)
            rate = rows / sec
            out["bare_model"][f"bf16_B{B}x352"] = rate
            print(f"[floor] bare bf16 model B={B}x352: {rate:,.0f} tok/s",
                  flush=True)
    return out


# ---------------------------------------------------------------- step 3

@app.function(image=image, gpu="H100!", timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache})
async def client_run(n_docs: int = 10000) -> dict:
    import inspect

    import numpy as np
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    try:
        from vllm.v1.engine.async_llm import AsyncLLM as Engine
    except ImportError:
        from vllm import AsyncLLMEngine as Engine
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
    except ImportError:
        from vllm import AsyncEngineArgs

    from docengine.runtime.engine_client import run_filter_chain

    docs = _build_pool(n_docs)
    engine = Engine.from_engine_args(AsyncEngineArgs(
        model=MODEL, kv_cache_dtype="fp8", max_model_len=4608,
        gpu_memory_utilization=0.92, enable_prefix_caching=True))
    tok = AutoTokenizer.from_pretrained(MODEL)
    sp = SamplingParams(temperature=0.0, max_tokens=1, skip_clone=True)
    pool = 981_728            # measured KV pool for this config (scale runs)
    try:
        cc = engine.vllm_config.cache_config
        if cc.num_gpu_blocks:
            pool = int(cc.num_gpu_blocks) * int(cc.block_size)
    except Exception:
        pass
    budget = int(0.9 * pool)
    print(f"[client] KV pool {pool} tokens, admission budget {budget}",
          flush=True)

    async def reset_cache():
        res = engine.reset_prefix_cache()
        if inspect.isawaitable(res):
            await res

    results = []
    grid = ((2, (0.5, 0.5), 1), (2, (0.5, 0.5), 2),
            (4, (0.8,) * 4, 1), (4, (0.95,) * 4, 1), (4, (0.95,) * 4, 2))
    for n, s_vec, k in grid:
        rng = np.random.default_rng(FLAG_SEED + 1000 * n + int(100 * s_vec[0]))
        flags = (rng.random((len(docs), n)) < np.asarray(s_vec)).astype(int)
        bodies = [d + _flags_line(f) for d, f in zip(docs, flags)]
        body_ids = tok(bodies, add_special_tokens=False)["input_ids"]
        q_ids = [tok(_question(j + 1), add_special_tokens=False)["input_ids"]
                 for j in range(n)]
        p_task = [len(tok(_task_prefix(j + 1) + _task_suffix(j + 1),
                          add_special_tokens=False)["input_ids"])
                  for j in range(n)]
        await reset_cache()
        res = await run_filter_chain(
            engine, sp, body_ids, q_ids, budget, lookahead=k,
            tag=f"n{n}s{int(100 * s_vec[0])}k{k}")
        answers = {f"{i},{j}": a for (i, j), a in res["answers"].items()}
        agree = sum(1 for (i, j), a in res["answers"].items()
                    if a == flags[i][j - 1]) / max(1, len(answers))
        results.append(dict(
            n=n, s=list(s_vec), policy="block", k=k, mode="client",
            makespan=res["wall"],
            waves=[dict(stage=1, requests=res["requests"], s=res["wall"],
                        prompt_tokens=res["prompt_tokens"],
                        cached_tokens=res["cached_tokens"])],
            d_tok=[len(x) for x in body_ids],
            p_tok=[len(q) for q in q_ids], p_task=p_task,
            answers=answers, flags=flags.tolist(),
            answer_agreement=agree, budget=budget, kv_tokens=pool))
        print(f"[client] n={n} s={s_vec} k={k}: {res['wall']:.1f}s, "
              f"{res['requests']} requests, "
              f"hit {100 * res['cached_tokens'] / max(1, res['prompt_tokens']):.1f}%, "
              f"agreement {agree:.3f}", flush=True)

    try:
        engine.shutdown()
    except Exception:
        pass
    import vllm
    return dict(model=MODEL, n_docs=n_docs, vllm_version=vllm.__version__,
                results=results)


# ---------------------------------------------------------------- step 2

@app.function(image=image, gpu="H100!", timeout=5400,
              volumes={"/root/.cache/huggingface": hf_cache})
def scale10k(n_docs: int = 10000) -> dict:
    import numpy as np
    from vllm import LLM, SamplingParams

    from docengine.configs import DEVICES, MODELS
    from docengine.instance import Instance
    from docengine.sched.blockwise import schedule_blockwise

    t0 = time.time()
    docs = _build_pool(n_docs)
    llm = LLM(model=MODEL, kv_cache_dtype="fp8", max_model_len=4608,
              gpu_memory_utilization=0.92, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=1)
    load_s = time.time() - t0
    kv = _kv_tokens(llm)
    cap = int(0.9 * kv) if kv else 800_000
    print(f"[scale] engine KV pool: {kv} tokens, builder cap {cap}",
          flush=True)

    def wave(prompts):
        t = time.time()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.time() - t
        cached = sum(getattr(o, "num_cached_tokens", 0) or 0 for o in outs)
        toks = sum(len(o.prompt_token_ids) for o in outs)
        return outs, dt, toks, cached

    cfgs = []
    for n, s_vec in ((2, (0.5, 0.5)), (4, (0.8,) * 4), (4, (0.95,) * 4)):
        cfgs.append(dict(n=n, s=s_vec, policy="task", k=0, mode="waves"))
        cfgs.append(dict(n=n, s=s_vec, policy="block", k=1, mode="waves"))
        cfgs.append(dict(n=n, s=s_vec, policy="block", k=1, mode="manifest"))
        if n == 2:
            cfgs.append(dict(n=n, s=s_vec, policy="block", k=2,
                             mode="waves"))
            cfgs.append(dict(n=n, s=s_vec, policy="block", k=2,
                             mode="manifest"))

    results = []
    for cfg in cfgs:
        n, s_vec, policy, k = cfg["n"], cfg["s"], cfg["policy"], cfg["k"]
        rng = np.random.default_rng(FLAG_SEED + 1000 * n + int(100 * s_vec[0]))
        flags = (rng.random((len(docs), n)) < np.asarray(s_vec)).astype(int)
        bodies = [d + _flags_line(f) for d, f in zip(docs, flags)]
        d_tok = [len(x) for x in
                 tok(bodies, add_special_tokens=False)["input_ids"]]
        p_tok = [len(tok(_question(j + 1),
                         add_special_tokens=False)["input_ids"])
                 for j in range(n)]
        p_task = [len(tok(_task_prefix(j + 1) + _task_suffix(j + 1),
                          add_special_tokens=False)["input_ids"])
                  for j in range(n)]

        assert llm.reset_prefix_cache(), "prefix cache reset failed"

        waves = []
        answers = {}
        if cfg["mode"] == "manifest":
            inst = Instance(model=MODELS["Qwen3-4B-FP8"],
                            device=DEVICES["H100-SXM-80GB"],
                            d=tuple(d_tok),
                            p=tuple(x + 1 for x in p_tok),
                            s=tuple(s_vec), delta=max(d_tok) + 1)
            X = np.array(flags, dtype=np.int8)
            recs = schedule_blockwise(inst, X, k, cap=cap)
            for rec in recs:
                prompts, branch_of = [], []
                branched = {op["doc"] for op in rec["ops"]
                            if op["kind"] == "branch"}
                for op in rec["ops"]:
                    i, jj = op["doc"], op["stage"]
                    if op["kind"] == "branch":
                        prompts.append(bodies[i] + _question(jj))
                        branch_of.append((i, jj))
                    elif op["kind"] == "doc_chunk" and i not in branched:
                        prompts.append(bodies[i])
                        branch_of.append((i, 0))
                if not prompts:
                    continue
                # Submission order matters under LRU eviction: requests
                # whose prefixes are already resident (branches owed to
                # docs read in earlier batches) must run BEFORE new
                # document prefills, or the new writes evict exactly the
                # bodies the branches are about to reuse. Within the new
                # docs, branches go stage-major so a doc's later branch
                # reuses its earlier branch's prefix in flight. Bare
                # prefills for future batches go last, leaving them the
                # most recently touched.
                chunked = {op["doc"] for op in rec["ops"]
                           if op["kind"] == "doc_chunk"}

                def order_key(t):
                    i, jj = branch_of[t]
                    if jj == 0:
                        return (2, 0, t)
                    if i not in chunked:
                        return (0, jj, t)
                    return (1, jj, t)

                order = sorted(range(len(prompts)), key=order_key)
                prompts = [prompts[t] for t in order]
                branch_of = [branch_of[t] for t in order]
                outs, dt, toks, cached = wave(prompts)
                waves.append(dict(stage=rec["t"], requests=len(prompts),
                                  s=dt, prompt_tokens=toks,
                                  cached_tokens=cached))
                for (i, jj), o in zip(branch_of, outs):
                    if jj > 0:
                        answers[f"{i},{jj}"] = _answer_of(o)
        elif policy == "task":
            alive = list(range(len(docs)))
            for j in range(1, n + 1):
                if not alive:
                    break
                prompts = [_task_prefix(j) + bodies[i] + _task_suffix(j)
                           for i in alive]
                outs, dt, toks, cached = wave(prompts)
                waves.append(dict(stage=j, requests=len(prompts), s=dt,
                                  prompt_tokens=toks, cached_tokens=cached))
                nxt = []
                for i, o in zip(alive, outs):
                    a = _answer_of(o)
                    answers[f"{i},{j}"] = a
                    if a:
                        nxt.append(i)
                alive = nxt
        else:
            frontier = {i: 1 for i in range(len(docs))}
            wave_no = 0
            while frontier:
                wave_no += 1
                reqs = []
                for off in range(k):
                    for i, j0 in frontier.items():
                        kk = min(k, n - j0 + 1)
                        if off < kk:
                            reqs.append((i, j0, kk, j0 + off))
                prompts = [bodies[i] + _question(jj)
                           for (i, _j0, _kk, jj) in reqs]
                outs, dt, toks, cached = wave(prompts)
                waves.append(dict(stage=wave_no, requests=len(prompts),
                                  s=dt, prompt_tokens=toks,
                                  cached_tokens=cached))
                nxt = {}
                by_doc = {}
                for (i, j0, kk, jj), o in zip(reqs, outs):
                    a = _answer_of(o)
                    answers[f"{i},{jj}"] = a
                    by_doc.setdefault(i, []).append((jj, a))
                for i, arr in by_doc.items():
                    arr.sort()
                    j0, kk = frontier[i], len(arr)
                    passes = 0
                    for _jj, a in arr:
                        if a:
                            passes += 1
                        else:
                            break
                    if passes == kk and j0 + kk <= n:
                        nxt[i] = j0 + kk
                frontier = nxt

        agree = sum(1 for key, a in answers.items()
                    for i, j in [map(int, key.split(","))]
                    if a == flags[i][j - 1])
        results.append(dict(
            n=n, s=list(s_vec), policy=policy, k=k, mode=cfg["mode"],
            makespan=sum(w["s"] for w in waves), waves=waves,
            d_tok=d_tok, p_tok=p_tok, p_task=p_task,
            answers=answers, flags=flags.tolist(),
            answer_agreement=agree / max(1, len(answers)),
            kv_tokens=kv, cap=cap))
        print(f"[scale] n={n} s={s_vec} {policy} k={k} {cfg['mode']}: "
              f"{results[-1]['makespan']:.1f}s over {len(waves)} waves, "
              f"agreement {results[-1]['answer_agreement']:.3f}", flush=True)

    import vllm
    return dict(model=MODEL, n_docs=n_docs, load_s=load_s,
                vllm_version=vllm.__version__, results=results)


@app.local_entrypoint()
def main(phase: str = "speed", n_docs: int = 0, out: str = ""):
    import os
    if phase == "speed":
        data = speed_limit.remote(n_docs or 4000)
        path = out or "results/engine/speed_limit.json"
    elif phase == "overhead":
        nd = n_docs or 2000
        data = overhead_split.remote(nd)
        path = out or (f"results/engine/overhead_{nd}.json"
                       if nd != 2000 else "results/engine/overhead.json")
    elif phase == "scale":
        data = scale10k.remote(n_docs or 10000)
        path = out or "results/engine/scale10k.json.gz"
    elif phase == "client":
        data = client_run.remote(n_docs or 10000)
        path = out or "results/engine/client10k.json.gz"
    elif phase == "floor":
        data = model_floor.remote()
        path = out or "results/engine/model_floor.json"
    elif phase == "pinned":
        nd = n_docs or 10000
        hp = pinned_run.spawn("pinned", nd)
        hs = pinned_run.spawn("stock", nd)
        dp, ds = hp.get(), hs.get()
        data = dict(model=dp["model"], n_docs=nd,
                    vllm_version=dp["vllm_version"],
                    results=dp["results"] + ds["results"])
        path = out or ("results/engine/pinned10k.json.gz" if nd == 10000
                       else f"results/engine/pinned_{nd}.json.gz")
    elif phase == "pinned2":
        nd = n_docs or 10000
        hp = pinned_run.spawn("pinned", nd, 60.0, 1500, True)
        hs = pinned_run.spawn("stock", nd, 60.0, 1500, True)
        dp, ds = hp.get(), hs.get()
        data = dict(model=dp["model"], n_docs=nd,
                    vllm_version=dp["vllm_version"],
                    results=dp["results"] + ds["results"])
        path = out or "results/engine/pinned10k_hard.json.gz"
    elif phase == "pinned3":
        nd = n_docs or 10000
        data = pinned_run.remote("pinned", nd, 60.0, 1500, True)
        path = out or "results/engine/pinned10k_v3.json.gz"
    elif phase == "strict":
        nd = n_docs or 10000
        data = strict_run.remote(nd)
        path = out or ("results/engine/strict10k.json.gz" if nd == 10000
                       else f"results/engine/strict_{nd}.json.gz")
    elif phase == "chain":
        data = chain_run.remote(n_docs or 50)
        path = out or "results/engine/chain_smoke.json"
    elif phase == "chain4":
        data = chain_run.remote(n_docs or 50, 4, 0.8)
        path = out or "results/engine/chain4_smoke.json"
    elif phase == "chain10k":
        data = chain_run.remote(n_docs or 10000, 4, 0.8)
        path = out or "results/engine/chain10k.json"
    elif phase == "chainprof":
        data = chainprof_run.remote(n_docs or 4000)
        path = out or "results/engine/chainprof.json"
    elif phase == "chaincore":
        data = chain_run.remote(n_docs or 10000, 4, 0.8, 1)
        path = out or "results/engine/chaincore10k.json"
    elif phase == "profile":
        data = profile_run.remote(n_docs or 4000)
        path = out or "results/engine/profile.json"
    elif phase == "longdoc":
        data = longdoc_run.remote()
        path = out or "results/engine/longdoc.json"
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

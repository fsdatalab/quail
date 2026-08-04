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

Fidelity discriminator. The first flight showed survivor sets
diverging in 6 of 8 queries at q=8; this run decides whether that is
borderline noise or a routing bug. Every call samples with logprobs=2
in BOTH arms, so each sampled answer token carries its top-2 logprob
gap. Per-call records (query, doc, stage, token id, gap in nats) are
banked for both arms, and a call-by-call classification is printed
and banked per q: agreements, then disagreements split into confident
flips (both arms' gaps above GAP_NATS with different tokens - a
routing bug to find before the multi-query speedup ships) versus
near-ties (either gap at or under GAP_NATS - the claim ships with a
measured tolerance). The protocol is otherwise unchanged: cache reset
between arms, refcounted pins, same budgets.

Run with:
  modal run experiments/modal_shared.py --phase shared
"""

import gzip
import json

import modal

app = modal.App("docengine-shared")

# CUDA devel base, same recipe as the xengine vllm_new arm: nvcc is
# present, so FlashInfer can JIT its kernels (the old slim image
# could not, and read 80,556 tok/s where this base reads 97,220).
# Same vllm pin, same env, same code - the toolchain is the only
# variable versus the pre-rebase banked run. Results carry
# IMAGE_STAMP so post-rebase JSONs are recognizable.
IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"
IMAGE_STAMP = dict(base=IMAGE_BASE, toolchain="cuda13-devel")

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
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
# A flip past this top-2 logprob gap is not float noise (same
# threshold as the fusion gate in modal_fused.py).
GAP_NATS = 0.2


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


class _FinalRecorder:
    """Wraps the engine so the client library's calls pass through
    unchanged while the final RequestOutput of every request is kept,
    keyed by the request id's last |-part (the client's own suffix).
    The client library stays untouched; this file reads the sampled
    tokens and their top-2 logprob gaps out of the kept outputs."""

    def __init__(self, engine):
        self._engine = engine
        self.finals = {}

    def generate(self, prompt, sampling_params, request_id, **kw):
        agen = self._engine.generate(prompt, sampling_params,
                                     request_id, **kw)
        key = request_id.split("|")[-1]
        finals = self.finals

        async def _wrap():
            final = None
            async for out in agen:
                final = out
                yield out
            finals[key] = final

        return _wrap()

    def __getattr__(self, name):
        return getattr(self._engine, name)


def _call_records(final, n_stages):
    """Per-stage (sampled token id, top-2 logprob gap in nats) from a
    chain request's cumulative output. The engine's rewind only
    appends to the record, so position j holds stage j+1's answer
    token, and logprobs[j] its top-2 table. A missing table gives
    gap None, counted separately by the classifier - never silently
    folded into a near-tie."""
    if final is None or not final.outputs:
        return []
    out = final.outputs[0]
    toks = list(out.token_ids or ())[:n_stages]
    lps = out.logprobs or []
    recs = []
    for j, t in enumerate(toks):
        gap = None
        if j < len(lps) and lps[j]:
            vals = sorted((e.logprob for e in lps[j].values()),
                          reverse=True)
            if len(vals) >= 2:
                gap = round(float(vals[0] - vals[1]), 4)
        recs.append((int(t), gap))
    return recs


def _collect_calls(finals, parse, n_stages):
    """{(query, doc, stage): (token, gap)} from recorded finals.
    parse maps a request-id suffix to (query, doc) or None."""
    calls = {}
    for key, final in finals.items():
        ki = parse(key)
        if ki is None:
            continue
        k, i = ki
        for j, rec in enumerate(_call_records(final, n_stages)):
            calls[(k, i, j + 1)] = rec
    return calls


def _classify(sh_calls, sp_calls, q):
    """Call-by-call comparison of the two arms, one row per query.
    agree: same sampled token. confident flip: tokens differ and BOTH
    arms preferred theirs by more than GAP_NATS. near_tie: tokens
    differ and either gap is at or under GAP_NATS. gap_missing:
    tokens differ but a gap could not be read. only_shared and
    only_separate count calls the other arm never made (survivor
    divergence upstream cuts a chain short)."""
    rows = []
    for k in range(q):
        keys_sh = {c for c in sh_calls if c[0] == k}
        keys_sp = {c for c in sp_calls if c[0] == k}
        both = keys_sh & keys_sp
        agree = conf = near = missing = 0
        flips = []
        for c in sorted(both):
            t1, g1 = sh_calls[c]
            t2, g2 = sp_calls[c]
            if t1 == t2:
                agree += 1
                continue
            detail = dict(doc=c[1], stage=c[2], shared_token=t1,
                          separate_token=t2, shared_gap=g1,
                          separate_gap=g2)
            if g1 is None or g2 is None:
                missing += 1
                detail["kind"] = "gap_missing"
            elif g1 > GAP_NATS and g2 > GAP_NATS:
                conf += 1
                detail["kind"] = "confident"
            else:
                near += 1
                detail["kind"] = "near_tie"
            flips.append(detail)
        rows.append(dict(query=k, calls_both=len(both), agree=agree,
                         confident_flips=conf, near_ties=near,
                         gap_missing=missing,
                         only_shared=len(keys_sh - keys_sp),
                         only_separate=len(keys_sp - keys_sh),
                         flips=flips))
    return rows


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

    # logprobs=2 is the confident-flip discriminator: every sampled
    # answer token comes back with its top-2 logprob gap, in both
    # arms, so each disagreement can be classified as a confident
    # flip (both arms sure, past GAP_NATS) or a near-tie. Identical
    # in both arms, so the comparison stays fair.
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=2,
                        skip_clone=True)
    engine = Engine.from_engine_args(AsyncEngineArgs(
        model=MODEL, kv_cache_dtype="fp8", max_model_len=4608,
        gpu_memory_utilization=0.92, enable_prefix_caching=True,
        disable_log_stats=True, scheduling_policy="priority",
        scheduler_cls="docengine.engineext.scheduler.DocEngineScheduler"))
    rec = _FinalRecorder(engine)
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

        def parse_sh(key, want=f"sh{q}"):
            parts = key.split("-")
            if len(parts) != 3 or parts[0] != want:
                return None
            try:
                return int(parts[2]), int(parts[1])   # (query, doc)
            except ValueError:
                return None

        await reset()
        rec.finals = {}
        sh = await run_shared_scan(rec, sp, body_ids, queries,
                                   budget, tag=f"sh{q}")
        sh_lp = _collect_calls(rec.finals, parse_sh, n)
        sh_wrong = sum(wrong(sh["queries"][k]["answers"], k)
                       for k in range(q))
        sh_calls = sum(len(sh["queries"][k]["answers"]) for k in range(q))

        # the baseline: the same q queries as today's single-query
        # chain runs, one after another, cache reset between
        sep_wall = 0.0
        sep_prompt = sep_cached = sep_wrong = sep_calls = 0
        sep_survivors = []
        sp_lp = {}
        for k in range(q):

            def parse_sp(key, want=f"sp{q}x{k}", kk=k):
                parts = key.split("-")
                if (len(parts) != 3 or parts[0] != want
                        or parts[2] != "0"):
                    return None
                try:
                    return kk, int(parts[1])          # (query, doc)
                except ValueError:
                    return None

            await reset()
            rec.finals = {}
            r = await run_filter_chain_engine(
                rec, sp, body_ids, queries[k]["q_ids"], budget,
                yes_ids, tag=f"sp{q}x{k}")
            sp_lp.update(_collect_calls(rec.finals, parse_sp, n))
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
        fid = _classify(sh_lp, sp_lp, q)
        tot = {f: sum(fr[f] for fr in fid)
               for f in ("calls_both", "agree", "confident_flips",
                         "near_ties", "gap_missing", "only_shared",
                         "only_separate")}
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
                             == sep_survivors[k] for k in range(q)],
            fidelity=fid, fidelity_totals=tot,
            calls=dict(columns=["query", "doc", "stage", "token_id",
                                "gap_nats"],
                       shared=[[c[0], c[1], c[2], v[0], v[1]]
                               for c, v in sorted(sh_lp.items())],
                       separate=[[c[0], c[1], c[2], v[0], v[1]]
                                 for c, v in sorted(sp_lp.items())])))
        r = rows[-1]
        print(f"[shared] q={q}: shared {r['shared_s']}s vs separate "
              f"{r['separate_s']}s ({r['speedup']}x), reads "
              f"{r['shared_read_mult']}x vs {r['separate_read_mult']}x "
              f"corpus, wrong {r['shared_wrong']}/{r['shared_calls']} vs "
              f"{r['separate_wrong']}/{r['separate_calls']}", flush=True)
        print(f"[shared] q={q} fidelity {'k':>2} {'both':>6} "
              f"{'agree':>6} {'conf':>5} {'near':>5} {'nogap':>6} "
              f"{'sh_only':>8} {'sp_only':>8}", flush=True)
        for fr in fid:
            print(f"[shared] q={q} fidelity {fr['query']:>2} "
                  f"{fr['calls_both']:>6} {fr['agree']:>6} "
                  f"{fr['confident_flips']:>5} {fr['near_ties']:>5} "
                  f"{fr['gap_missing']:>6} {fr['only_shared']:>8} "
                  f"{fr['only_separate']:>8}", flush=True)
        print(f"[shared] q={q} verdict: {tot['confident_flips']} "
              f"confident flips, {tot['near_ties']} near-ties, "
              f"{tot['gap_missing']} with no readable gap, "
              f"{tot['agree']} agreements over {tot['calls_both']} "
              f"paired calls", flush=True)

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
                budget=budget, kv_tokens=pool, gap_nats=GAP_NATS,
                vllm_version=vllm.__version__, image=IMAGE_STAMP,
                rows=rows)


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

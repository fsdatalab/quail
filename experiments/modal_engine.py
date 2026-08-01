"""Measured layer: run the scheduling policies on a real H100 with vLLM and
Qwen3-4B-FP8, under controlled selectivity, and return everything the
analytical comparison needs.

Selectivity is induced: each document gets a trailing metadata line
[FLAGS] FLAG_1=YES FLAG_2=NO ... with flag j set YES with probability s_j
from a recorded seed. Filter j asks the model to read flag j and complete
"The answer is" with YES or NO at temperature 0, so outcomes are planted
and verifiable.

Policies are wave structures over one shared vLLM engine:
  task-first    [task prompt][document][cue]; every stage re-prefills docs
  pipeline      [document][question]; stage j+1 gated on stage j outcomes;
                document KV reuse via vLLM automatic prefix caching
  lookahead k   same template, all k branch questions issued ungated

Run with:
  modal run experiments/modal_engine.py --smoke true
  modal run experiments/modal_engine.py
"""

import json
import time

import modal

app = modal.App("docengine-engine")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "huggingface_hub", "pandas", "pyarrow", "numpy")
    .env({"VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("docengine")
)
hf_cache = modal.Volume.from_name("docengine-hf-cache", create_if_missing=True)

MODEL = "Qwen/Qwen3-4B-FP8"
WORKLOAD_SEED = 20260731
FLAG_SEED = 424242


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
    return (f"\n\nInstruction: output only the value of FLAG_{j} from the "
            f"[FLAGS] line above.\nFLAG_{j}=")


def _task_prefix(j):
    return (f"You will see a document ending in a [FLAGS] metadata line. "
            f"Your task: report whether FLAG_{j} is set to YES.\n\n"
            f"Document:\n")


def _task_suffix(j):
    return f"\n\nFrom the [FLAGS] line, FLAG_{j}="


@app.function(image=image, gpu="H100!", timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache})
def run_grid(configs: list, n_docs: int = 2000) -> dict:
    import numpy as np
    from vllm import LLM, SamplingParams

    t0 = time.time()
    docs = _build_pool(n_docs)
    llm = LLM(model=MODEL, kv_cache_dtype="fp8",
              max_model_len=4352, gpu_memory_utilization=0.92,
              enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=1)
    load_s = time.time() - t0

    def n_tokens(text):
        return len(tok.encode(text, add_special_tokens=False))

    def answer_of(out):
        txt = out.outputs[0].text.strip().upper()
        return 1 if txt.startswith("Y") else 0

    def wave(prompts):
        t = time.time()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.time() - t
        cached = sum(getattr(o, "num_cached_tokens", 0) or 0 for o in outs)
        toks = sum(len(o.prompt_token_ids) for o in outs)
        return outs, dt, toks, cached

    results = []
    for cfg in configs:
        n, s_vec, policy, k = cfg["n"], cfg["s"], cfg["policy"], cfg["k"]
        rng = np.random.default_rng(FLAG_SEED + 1000 * n + int(100 * s_vec[0]))
        flags = (rng.random((len(docs), n)) < np.asarray(s_vec)).astype(int)
        bodies = [d + _flags_line(f) for d, f in zip(docs, flags)]
        d_tok = [n_tokens(b) for b in bodies]
        p_tok = [n_tokens(_question(j + 1)) for j in range(n)]
        p_task = [n_tokens(_task_prefix(j + 1)) + n_tokens(_task_suffix(j + 1))
                  for j in range(n)]

        waves = []
        answers = {}          # (doc, stage) -> model's 0/1
        if cfg.get("mode") == "manifest":
            # drive the engine batch-for-batch from the analytical builder's
            # schedule, with outcomes planted (the offline coupled scenario)
            from docengine.configs import DEVICES, MODELS
            from docengine.instance import Instance
            from docengine.sched.blockwise import (schedule_blockwise,
                                                   schedule_taskfirst)
            inst = Instance(model=MODELS["Qwen3-4B-FP8"],
                            device=DEVICES["H100-SXM-80GB"],
                            d=tuple(d_tok),
                            p=tuple(x + 1 for x in
                                    (p_task if policy == "task" else p_tok)),
                            s=tuple(s_vec), delta=max(d_tok) + 1)
            X = np.array(flags, dtype=np.int8)
            recs = (schedule_taskfirst(inst, X) if policy == "task"
                    else schedule_blockwise(inst, X, k))
            for rec in recs:
                prompts = []
                branch_of = []
                branched = {op["doc"] for op in rec["ops"]
                            if op["kind"] == "branch"}
                for op in rec["ops"]:
                    i, jj = op["doc"], op["stage"]
                    if op["kind"] == "branch":
                        prompts.append(bodies[i] + _question(jj))
                        branch_of.append((i, jj))
                    elif op["kind"] == "doc_chunk" and policy == "task":
                        prompts.append(_task_prefix(jj) + bodies[i]
                                       + _task_suffix(jj))
                        branch_of.append((i, jj))
                    elif (op["kind"] == "doc_chunk" and policy != "task"
                          and i not in branched):
                        prompts.append(bodies[i])   # prefill-only, cache doc
                        branch_of.append((i, 0))
                if not prompts:
                    continue
                outs, dt, toks, cached = wave(prompts)
                waves.append(dict(stage=rec["t"], requests=len(prompts), s=dt,
                                  prompt_tokens=toks, cached_tokens=cached))
                for (i, jj), o in zip(branch_of, outs):
                    if jj > 0:
                        answers[f"{i},{jj}"] = answer_of(o)
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
                    a = answer_of(o)
                    answers[f"{i},{j}"] = a
                    if a:
                        nxt.append(i)
                alive = nxt
        else:
            # blockwise lookahead k over the doc-first template
            frontier = {i: 1 for i in range(len(docs))}
            wave_no = 0
            while frontier:
                wave_no += 1
                reqs = []
                for i, j0 in frontier.items():
                    kk = min(k, n - j0 + 1)
                    for jj in range(j0, j0 + kk):
                        reqs.append((i, j0, kk, jj))
                prompts = [bodies[i] + _question(jj)
                           for (i, _j0, _kk, jj) in reqs]
                outs, dt, toks, cached = wave(prompts)
                waves.append(dict(stage=wave_no, requests=len(prompts), s=dt,
                                  prompt_tokens=toks, cached_tokens=cached))
                nxt = {}
                by_doc = {}
                for (i, j0, kk, jj), o in zip(reqs, outs):
                    a = answer_of(o)
                    answers[f"{i},{jj}"] = a
                    by_doc.setdefault(i, []).append((jj, a))
                for i, arr in by_doc.items():
                    arr.sort()
                    j0 = frontier[i]
                    kk = len(arr)
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
            n=n, s=list(s_vec), policy=policy, k=k,
            mode=cfg.get("mode", "waves"),
            makespan=sum(w["s"] for w in waves), waves=waves,
            d_tok=d_tok, p_tok=p_tok, p_task=p_task,
            answers=answers, flags=flags.tolist(),
            answer_agreement=agree / max(1, len(answers)),
        ))
        print(f"[grid] n={n} s={s_vec} {policy}: "
              f"{results[-1]['makespan']:.1f}s over {len(waves)} waves, "
              f"agreement {results[-1]['answer_agreement']:.3f}", flush=True)

    return dict(model=MODEL, n_docs=n_docs, load_s=load_s, results=results)


def _grid(smoke: bool):
    if smoke:
        return [dict(n=2, s=(0.5, 0.5), policy=p, k=k)
                for p, k in (("task", 0), ("block", 1), ("block", 2))], 50
    cfgs = []
    for s1 in (0.25, 0.5, 0.8):
        for p, k in (("task", 0), ("block", 1), ("block", 2)):
            cfgs.append(dict(n=2, s=(s1, 0.5), policy=p, k=k))
    for s in (0.7, 0.9):
        for p, k in (("task", 0), ("block", 1), ("block", 3)):
            cfgs.append(dict(n=3, s=(s,) * 3, policy=p, k=k))
    for s in (0.8, 0.95):
        for p, k in (("task", 0), ("block", 1), ("block", 2), ("block", 4)):
            cfgs.append(dict(n=4, s=(s,) * 4, policy=p, k=k))
    for p, k in (("task", 0), ("block", 1), ("block", 2)):
        cfgs.append(dict(n=2, s=(0.5, 0.5), policy=p, k=k, mode="manifest"))
    for p, k in (("task", 0), ("block", 1), ("block", 4)):
        cfgs.append(dict(n=4, s=(0.8,) * 4, policy=p, k=k, mode="manifest"))
    return cfgs, 2000


@app.local_entrypoint()
def main(smoke: bool = False, out: str = ""):
    cfgs, n_docs = _grid(smoke)
    data = run_grid.remote(cfgs, n_docs)
    path = out or ("results/engine/smoke.json" if smoke
                   else "results/engine/grid.json")
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"saved {path}")
    for r in data["results"]:
        print(f"n={r['n']} s={r['s']} {r['policy']} k={r['k']}: "
              f"{r['makespan']:.2f}s, agreement {r['answer_agreement']:.3f}")


@app.function(image=image, gpu="H100!", timeout=1200,
              volumes={"/root/.cache/huggingface": hf_cache})
def probe() -> list:
    """Print raw completions for candidate answer-extraction patterns."""
    from vllm import LLM, SamplingParams
    docs = _build_pool(3)
    llm = LLM(model=MODEL, kv_cache_dtype="fp8", max_model_len=4352,
              gpu_memory_utilization=0.92, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=8)
    b_yes = docs[0] + _flags_line([1, 0])
    b_no = docs[1] + _flags_line([0, 1])
    A = "\n\n[ANSWER] FLAG_1="
    B = ("\n\nInstruction: output only the value of FLAG_1 from the "
         "[FLAGS] line above.\nFLAG_1=")
    C = "\n\nQ: FLAG_1?\nA: FLAG_1="
    variants = {
        "A_yes": b_yes + A, "A_no": b_no + A,
        "B_yes": b_yes + B, "B_no": b_no + B,
        "C_yes": b_yes + C, "C_no": b_no + C,
    }
    outs = llm.generate(list(variants.values()), sp, use_tqdm=False)
    report = []
    for (name, _p), o in zip(variants.items(), outs):
        toks = [llm.get_tokenizer().decode([t]) for t in o.outputs[0].token_ids]
        report.append((name, repr(o.outputs[0].text), [repr(t) for t in toks]))
        print(name, "->", repr(o.outputs[0].text), "tokens:", toks, flush=True)
    return report

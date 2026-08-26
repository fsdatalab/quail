"""Boot profiling: compile pass, touch pass, py-spy recording, and query timing on GPU."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

from baselines.boot_stats import aggregate as _aggregate
from corpus import MODEL

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "pandas", "pyarrow",
                 "numpy", "datasets", "py-spy")
    .env({"VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "TRITON_CACHE_DIR": "/root/.cache/kernels/triton"})
    .add_local_python_source("quail", "corpus", "baselines")
    .add_local_dir("quail/calibration",
                   remote_path="/root/quail/calibration")
)

# House rule: attach to the existing milestone1 app; never invent a
# new Modal app name.
app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)

GPU_KW = dict(image=image, gpu="H100!", memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})

PREDICTION = (
    "compile boot: generator sweep + unseen-config compiles, minutes "
    "once ever; touch boot warm_kernels_s 2-4 s (vs 3.7-4.9 s swept); "
    "cold boot_s load_model-dominated (28-38 s); query best fast wall "
    "28.3 s +-3%, 0 wrong, under stock's 33.4 s")

# The committed numbers this run is checked against.
REFERENCE = dict(
    quail_query=dict(no_arena_wall_s=28.30, arena_wall_s=28.65,
                     wrong=0, source="results/m1_filter1.json"),
    stock_query=dict(
        wall_s=33.41,
        submission="separate requests per document, "
                   "document-cap admission",
        source="results/baseline_filter1.json"),
    old_boot=dict(warm_kernels_s=(3.68, 4.9),
                  boot_s=(33.61, 45.3),
                  source="results/boot_profile.json"))


# ------------------------------------------------------------- py-spy

def _pyspy_start(out_path: str):
    """Attach py-spy to this process. Returns a stop() callable that yields a status dict."""
    import signal
    import subprocess

    try:
        # same-uid attach needs ptrace scope 0 on hardened kernels
        with open("/proc/sys/kernel/yama/ptrace_scope", "w") as f:
            f.write("0")
    except OSError:
        pass
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    proc = subprocess.Popen(
        ["py-spy", "record", "--pid", str(os.getpid()),
         "--format", "speedscope", "--output", out_path,
         "--rate", "100", "--idle"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    time.sleep(1.0)          # let the sampler attach before the work

    def stop() -> dict:
        if proc.poll() is not None:
            err = (proc.stderr.read() or b"").decode()[-400:]
            return dict(ok=False, path=None, error=err.strip())
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            return dict(ok=False, path=None, error="py-spy hung")
        err = (proc.stderr.read() or b"").decode()[-400:]
        ok = proc.returncode == 0 and os.path.exists(out_path)
        return dict(ok=ok, path=out_path if ok else None,
                    error=None if ok else err.strip())

    return stop


def _speedscope_top(path: str, n: int = 15,
                    thread: str = "MainThread") -> dict:
    """Extract top-n functions by self time from a py-spy speedscope file."""
    with open(path) as f:
        data = json.load(f)
    frames = data["shared"]["frames"]
    profiles = [p for p in data["profiles"]
                if p.get("type") == "sampled"]
    named = [p for p in profiles if thread in p.get("name", "")]
    self_s: dict[int, float] = {}
    cum_s: dict[int, float] = {}
    total = 0.0
    for prof in (named or profiles):
        for stack, w in zip(prof["samples"], prof["weights"]):
            if not stack:
                continue
            total += w
            self_s[stack[-1]] = self_s.get(stack[-1], 0.0) + w
            for fi in set(stack):
                cum_s[fi] = cum_s.get(fi, 0.0) + w
    top = sorted(self_s.items(), key=lambda kv: -kv[1])[:n]
    out = []
    for fi, s in top:
        fr = frames[fi]
        where = f"{fr.get('file', '?')}:{fr.get('line', '?')}"
        out.append(dict(func=fr.get("name", "?"), at=where,
                        self_s=round(s, 2),
                        cum_s=round(cum_s.get(fi, 0.0), 2)))
    return dict(total_sampled_s=round(total, 2), top=out)


# --------------------------------------------------------- quail boot

def _round_boot(boot: dict) -> dict:
    out = dict(boot)
    for k, v in list(out.items()):
        if isinstance(v, float):
            out[k] = round(v, 2)
    return out


def _quail_boot_once(*, reuse: dict | None,
                     force_compile: bool = False,
                     model_key: str = "qwen3-4b-fp8") -> tuple[dict, dict]:
    """Run one Quail boot. reuse=None is cold; reuse=state is warm."""
    import torch
    import torch.nn.functional as F

    from quail.executor.arena import KVArena
    from quail.executor.attention import (FILTER_ATTENTION,
                                          Pipeline)
    from quail.executor.loop import Answerer, AsyncAnswers, warm_kernels
    from quail.executor.model import load_model
    from quail.planner import budgets
    from quail.specs import DEVICES, MODELS
    from transformers import AutoTokenizer

    boot = dict(kind="warm", load_model_s=0.0, arena_s=0.0,
                pipeline_s=0.0, warm_kernels_s=0.0)
    t_boot = time.perf_counter()
    spec = MODELS[model_key]
    device = DEVICES["h100-sxm"]

    if reuse is None:
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
        t0 = time.perf_counter()
        model = load_model(spec.hf_name)
        boot["load_model_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        chunk_tokens = budgets.chunk_budget(spec, device)
        arena_tok = budgets.arena_tokens(spec, device, chunk_tokens)
        arena = KVArena(n_layers=spec.layers,
                        n_pages=arena_tok // budgets.PAGE_TOKENS,
                        page_tokens=budgets.PAGE_TOKENS,
                        n_kv=spec.n_kv, d_head=spec.d_head,
                        dtype=torch.bfloat16)
        boot["arena_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        pipeline = Pipeline(
            model, arena, attention_mode=FILTER_ATTENTION)
        boot["pipeline_s"] = time.perf_counter() - t0
        answerer = Answerer(torch, F, model, tokenizer)
        async_ans = AsyncAnswers(torch, answerer)
        chunk_tokens = min(chunk_tokens, pipeline.max_chunk_tokens)
        state = dict(torch=torch, tokenizer=tokenizer, model=model,
                     arena=arena, pipeline=pipeline,
                     async_ans=async_ans,
                     chunk_tokens=chunk_tokens, warmed=False)
        boot["kind"] = "cold"
    else:
        state = reuse
        torch = state["torch"]
        arena = state["arena"]
        pipeline = state["pipeline"]
        async_ans = state["async_ans"]
        chunk_tokens = state["chunk_tokens"]

    if not state["warmed"]:
        t0 = time.perf_counter()
        with torch.inference_mode():
            warm = warm_kernels(torch, arena, pipeline, async_ans,
                                chunk_tokens,
                                model_name=spec.hf_name,
                                force_compile=force_compile)
        torch.cuda.synchronize()
        kernel_cache.commit()
        boot["warm_kernels_s"] = time.perf_counter() - t0
        boot["warm_tier"] = warm["tier"]
        state["warmed"] = True
        boot["kind"] = "cold"

    boot["boot_s"] = time.perf_counter() - t_boot
    return state, _round_boot(boot)


def _profiled_boot(tag: str, *, force_compile: bool,
                   spy: bool = True,
                   model_key: str = "qwen3-4b-fp8") -> tuple[dict, dict]:
    """Cold boot with optional py-spy profiling."""
    if not spy:
        state, cold = _quail_boot_once(reuse=None,
                                       force_compile=force_compile,
                                       model_key=model_key)
        return state, dict(cold=cold, pyspy=dict(ok=False,
                                                 path=None,
                                                 error="disabled"))
    spy_path = f"/results/boot/pyspy_{tag}.speedscope.json"
    stop = _pyspy_start(spy_path)
    state, cold = _quail_boot_once(reuse=None,
                                   force_compile=force_compile,
                                   model_key=model_key)
    spy_row = stop()
    if spy_row["ok"]:
        try:
            spy_row["summary"] = _speedscope_top(spy_path)
        except (OSError, ValueError, KeyError) as e:
            spy_row["summary"] = dict(error=repr(e))
        results_vol.commit()
    row = dict(cold=cold, pyspy=spy_row)
    return state, row


# -------------------------------------------------------------- cells

@app.function(timeout=7200, **GPU_KW)
def compile_trial(model_key: str = "qwen3-4b-fp8",
                  spy: bool = True) -> dict:
    """Run the one-time compile pass with timing and py-spy recording."""
    _, row = _profiled_boot(f"{model_key}-compile",
                            force_compile=True, spy=spy,
                            model_key=model_key)
    row.update(side="quail_compile", trial=0)
    print(f"[boot_tiered] compile: {row['cold']}", flush=True)
    return row


@app.function(timeout=7200, max_containers=8, **GPU_KW)
def touch_trial(trial: int = 0, reps: int = 2,
                spy: bool = True,
                model_key: str = "qwen3-4b-fp8") -> dict:
    """Run one touch-pass boot trial with warm reuse and a filter query on both paths."""
    from corpus import build_corpus
    from quail.executor.attention import FILTER_ATTENTION
    from quail.executor.loop import run_filter

    state, row = _profiled_boot(f"{model_key}-touch{trial}",
                                force_compile=False, spy=spy,
                                model_key=model_key)
    _, warm = _quail_boot_once(reuse=state, model_key=model_key)
    row.update(side="quail", trial=trial, warm=warm)
    print(f"[boot_tiered] touch trial {trial}: cold={row['cold']} "
          f"warm={warm}", flush=True)

    torch = state["torch"]
    pipeline = state["pipeline"]
    body_ids, q_ids, flags = build_corpus(state["tokenizer"], 10000,
                                          n_filters=1)
    pipeline.attention_mode = FILTER_ATTENTION
    runs = []
    for rep in range(reps):
        for mode, aw in (("arena", True), ("no_arena", False)):
            t0 = time.perf_counter()
            with torch.inference_mode():
                answers, spans, tokens = run_filter(
                    torch, state["arena"], pipeline,
                    state["async_ans"], body_ids, q_ids,
                    state["chunk_tokens"], arena_writes=aw)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            runs.append(dict(
                rep=rep, mode=mode, wall=round(wall, 2),
                fresh_tokens=tokens,
                tok_s=round(tokens / wall, 1),
                wrong=sum(bit != int(flags[d][0])
                          for d, r in answers.items() for bit in r),
                chunks=len(spans),
                gpu_s=round(sum(e0.elapsed_time(e1)
                                for _, e0, e1 in spans) / 1e3, 2)))
            print(f"[boot_tiered] trial {trial} query {runs[-1]}",
                  flush=True)
    row["query"] = runs
    return row


@app.function(timeout=3600, max_containers=8, **GPU_KW)
def stock_boot_trial(trial: int = 0,
                     model_key: str = "qwen3-4b-fp8") -> dict:
    """Run one stock vLLM boot trial: cold LLM init, then warm reuse."""
    from baselines.stock_boot import time_llm_boot, warm_boot_dict
    from quail.specs import MODELS

    # Same knobs as the committed stock filter baseline.
    llm, cold = time_llm_boot(
        model=MODELS[model_key].hf_name,
        max_num_batched_tokens=25_305,
        max_num_seqs=2648, gpu_memory_utilization=0.92,
        enable_prefix_caching=True, disable_log_stats=True)
    _ = llm.llm_engine
    warm = warm_boot_dict()
    row = dict(side="stock", trial=trial, cold=cold, warm=warm)
    print(f"[boot_tiered] stock trial {trial}: cold={cold}",
          flush=True)
    return row


@app.function(timeout=600, image=image, memory=4096,
              volumes={"/results": results_vol})
def write_report(report: dict, name: str = "boot_tiered") -> str:
    os.makedirs("/results/boot", exist_ok=True)
    with open(f"/results/boot/{name}.json", "w") as f:
        json.dump(report, f, indent=2)
    results_vol.commit()
    return json.dumps(report, indent=2)


@app.local_entrypoint()
def main(touch_trials: int = 3, reps: int = 2,
         stock_trials: int = 0, spy: bool = True,
         skip_compile: bool = False,
         name: str = "boot_tiered",
         model: str = "qwen3-4b-fp8"):
    """Run the compile pass, then touch trials in parallel, with optional stock boot trials."""
    print(f"[boot_tiered] prediction: {PREDICTION}", flush=True)

    if skip_compile:
        compile_row = dict(skipped=True)
    else:
        h = compile_trial.spawn(model, spy)
        print(f"[boot_tiered] compile fc={h.object_id}", flush=True)
        compile_row = h.get()

    t_handles = [touch_trial.spawn(i, reps, spy, model)
                 for i in range(touch_trials)]
    s_handles = [stock_boot_trial.spawn(i, model)
                 for i in range(stock_trials)]
    for hh in t_handles + s_handles:
        print(f"[boot_tiered] fc={hh.object_id}", flush=True)
    touch_rows = [hh.get() for hh in t_handles]
    stock_rows = [hh.get() for hh in s_handles]

    query_runs = [r for row in touch_rows for r in row["query"]]
    best = {m: min((r["wall"] for r in query_runs
                    if r["mode"] == m), default=None)
            for m in ("arena", "no_arena")}
    report = dict(
        cell="boot_tiered", model=model, gpu="H100!", vllm="0.26.0",
        prediction=PREDICTION, reference=REFERENCE,
        compile=compile_row,
        touch=_aggregate("quail", touch_rows),
        stock=(_aggregate("stock", stock_rows) if stock_rows
               else dict(reused="results/boot_profile.json")),
        query=dict(runs=query_runs, best_wall=best,
                   wrong=sum(r["wrong"] for r in query_runs)))
    payload = write_report.remote(report, name)
    out = f"results/{name}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(payload)
    print(f"[boot_tiered] saved {out}", flush=True)

    if not skip_compile:
        ct = compile_row["cold"]
        print(f"[boot_tiered] compile boot: warm_kernels_s="
              f"{ct['warm_kernels_s']} boot_s={ct['boot_s']}",
              flush=True)
    agg = report["touch"]["cold"]
    print(f"[boot_tiered] touch boots: warm_kernels_s "
          f"mean={agg['warm_kernels_s']['mean']} boot_s "
          f"mean={agg['boot_s']['mean']}", flush=True)
    print(f"[boot_tiered] query best {best} wrong="
          f"{report['query']['wrong']} (ref: no_arena 28.30, "
          f"stock 33.41)", flush=True)

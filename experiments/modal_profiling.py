"""The four instruments behind the speed-of-light and profiling slides.

    batchsweep         UNPROFILED walls at several max_num_batched_tokens
                       values, through the synchronous API. This is the
                       data the step model T(B) = a*B + b is fitted to.
                       Unprofiled on purpose: a profiler changes the
                       timing it measures, so the regression must not
                       run under one.

    torchprof          vLLM's in-core torch profiler over a 15-second
                       steady-state window. Gives the GPU busy fraction
                       and the per-kernel-class split that says what a
                       is made of. Runs inside the engine core, the one
                       process a client-side profiler cannot see.

    component_profile  the same profiler at one batch size, with the
                       trace parsed into kernel classes on the way out.
                       Launch several in parallel (one per B) to see
                       how the class mix moves with batch size.

    ncubench           Nsight Compute speed-of-light on standalone
                       GEMMs at the 4B prefill shapes. Says the GEMM
                       kernels themselves run at 91-93 percent of the
                       compute ceiling, so the gap to peak is the step
                       MIX, not slow kernels.
    ncureport          render a banked ncu report's details page.

Platform note: nsys does not work here. Its GPU-activity collection
uses a driver path the sandbox blocks, so it returns CUDA API rows and
no kernels. torch.profiler's CUPTI path works, and ncu works with
--clock-control none but only as a microbenchmark - against a live
engine it intercepts every boot kernel and times out.

Run:
    modal run experiments/modal_profiling.py::batchsweep
    modal run experiments/modal_profiling.py::torchprof
    modal run experiments/modal_profiling.py::run_component_profiles
    modal run experiments/modal_profiling.py::ncubench
"""

import os

import modal

from workload import MODEL, hf_cache, image, results_vol

app = modal.App("quail-profiling")
prof_image = image.add_local_python_source("workload")


# -----------------------------------------------------------------
# 1. The batch-size sweep: unprofiled walls for the step model.
# -----------------------------------------------------------------

@app.function(image=prof_image, gpu="H100!", timeout=7200, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def batchsweep(n_docs: int = 10000,
               batch_sizes: str = "512,1024,2048,4096,8192,16384,25305"
               ) -> str:
    """Measure each token budget with one CUDA graph size.

    All points run in one container. The LLM is cleared between points,
    so every point uses the same H100 with a fresh model instance. Graph
    capture matches the token budget through 8192 and is capped there.
    """
    import gc
    import json
    import time

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from workload import MODEL, build_corpus

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    body_ids, question_ids, _flags = build_corpus(tokenizer, n_docs)
    prompts = [
        {"prompt_token_ids": body_ids[i] + question_ids[0]}
        for i in range(n_docs)
    ]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)
    gpu_memory_utilization = 0.88
    outpath = "/results/batchsweep_exact_cudagraph.json"

    results = []
    for batch_size in (int(value) for value in batch_sizes.split(",")):
        max_num_seqs = min(4096, batch_size)
        capture_size = min(batch_size, 8192)
        print(
            f"\n[bsweep] B={batch_size}, max_num_seqs={max_num_seqs}, "
            f"capture_sizes=[{capture_size}]",
            flush=True,
        )
        llm = LLM(
            model=MODEL,
            kv_cache_dtype="fp8",
            max_model_len=4608,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=batch_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=False,
            disable_log_stats=True,
            compilation_config={
                "max_cudagraph_capture_size": capture_size,
                "cudagraph_capture_sizes": [capture_size],
            },
        )
        started = time.time()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        wall = time.time() - started
        prompt_tokens = sum(len(output.prompt_token_ids) for output in outputs)
        cached_tokens = sum(
            getattr(output, "num_cached_tokens", 0) or 0
            for output in outputs
        )
        uncached_tokens = prompt_tokens - cached_tokens
        row = {
            "B": batch_size,
            "max_num_seqs": max_num_seqs,
            "cudagraph_capture_sizes": [capture_size],
            "gpu_memory_utilization": gpu_memory_utilization,
            "n_docs": n_docs,
            "wall": round(wall, 3),
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "uncached_tokens": uncached_tokens,
            "rate": round(uncached_tokens / wall, 1),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        with open(outpath, "w") as output:
            json.dump(results, output, indent=2)
        results_vol.commit()

        del llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)

    with open(outpath) as output:
        data = output.read()
    print(data, flush=True)
    return data


# -----------------------------------------------------------------
# 1a. Does the dynamic FP8 kernel choice cause the B=1024 gap?
# -----------------------------------------------------------------

@app.function(image=prof_image, gpu="H100!", timeout=7200, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def deepgemm_choice_probe() -> str:
    """Compare normal B=1024 with the FP8 size check bypassed.

    The three runs share one container. A normal repeat after the
    forced run checks for drift on the same H100.
    """
    import gc
    import json
    import pathlib
    import site
    import time

    import torch
    from transformers import AutoTokenizer
    from workload import MODEL, build_corpus

    matches = []
    for root in site.getsitepackages():
        candidate = pathlib.Path(root) / (
            "vllm/model_executor/kernels/linear/scaled_mm/flashinfer.py"
        )
        if candidate.exists():
            matches.append(candidate)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one vLLM flashinfer.py, found {matches}"
        )

    source_path = matches[0]
    source = source_path.read_text()
    source = source.replace(
        "from typing import ClassVar\n\nimport torch",
        "import os\nfrom typing import ClassVar\n\nimport torch",
        1,
    )
    old = "if envs.VLLM_BATCH_INVARIANT:\n        return run_deepgemm"
    new = (
        "if (envs.VLLM_BATCH_INVARIANT or "
        "os.getenv('QUAIL_FORCE_DEEPGEMM') == '1'):\n"
        "        return run_deepgemm"
    )
    if source.count(old) != 1:
        raise RuntimeError(
            "vLLM dynamic FP8 branch did not match expected source"
        )
    source_path.write_text(source.replace(old, new, 1))

    from vllm import LLM, SamplingParams

    n_docs = 10000
    batch_size = 1024
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    body_ids, question_ids, _flags = build_corpus(tokenizer, n_docs)
    prompts = [
        {"prompt_token_ids": body_ids[i] + question_ids[0]}
        for i in range(n_docs)
    ]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)

    results = []
    for name, force in (
        ("normal_1", "0"),
        ("direct_deepgemm", "1"),
        ("normal_2", "0"),
    ):
        os.environ["QUAIL_FORCE_DEEPGEMM"] = force
        print(f"\n[deepgemm choice] {name}, force={force}", flush=True)
        llm = LLM(
            model=MODEL,
            kv_cache_dtype="fp8",
            max_model_len=4608,
            max_num_seqs=1024,
            max_num_batched_tokens=batch_size,
            gpu_memory_utilization=0.88,
            enable_prefix_caching=False,
            disable_log_stats=True,
        )
        started = time.time()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        wall = time.time() - started
        tokens = sum(len(output.prompt_token_ids) for output in outputs)
        row = {
            "name": name,
            "force_direct_deepgemm": force == "1",
            "B": batch_size,
            "wall": round(wall, 3),
            "prompt_tokens": tokens,
            "rate": round(tokens / wall, 1),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)

    outpath = "/results/deepgemm_choice_B1024.json"
    with open(outpath, "w") as output:
        json.dump(results, output, indent=2)
    results_vol.commit()
    with open(outpath) as output:
        data = output.read()
    print(data, flush=True)
    return data


@app.function(image=prof_image, gpu="H100!", timeout=7200, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def cudagraph_cap_probe() -> str:
    """Test whether vLLM's default 512-token graph cap causes the dip."""
    import gc
    import json
    import time

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from workload import MODEL, build_corpus

    n_docs = 10000
    batch_size = 1024
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    body_ids, question_ids, _flags = build_corpus(tokenizer, n_docs)
    prompts = [
        {"prompt_token_ids": body_ids[i] + question_ids[0]}
        for i in range(n_docs)
    ]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)

    results = []
    for name, capture_limit in (
        ("default_1", None),
        ("capture_1024", 1024),
        ("default_2", None),
    ):
        print(
            f"\n[cudagraph cap] {name}, limit={capture_limit}",
            flush=True,
        )
        extra = {}
        if capture_limit is not None:
            extra["compilation_config"] = {
                "max_cudagraph_capture_size": capture_limit,
            }
        llm = LLM(
            model=MODEL,
            kv_cache_dtype="fp8",
            max_model_len=4608,
            max_num_seqs=1024,
            max_num_batched_tokens=batch_size,
            gpu_memory_utilization=0.88,
            enable_prefix_caching=False,
            disable_log_stats=True,
            **extra,
        )
        started = time.time()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        wall = time.time() - started
        tokens = sum(len(output.prompt_token_ids) for output in outputs)
        row = {
            "name": name,
            "max_cudagraph_capture_size": capture_limit,
            "B": batch_size,
            "wall": round(wall, 3),
            "prompt_tokens": tokens,
            "rate": round(tokens / wall, 1),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)

    outpath = "/results/cudagraph_cap_B1024.json"
    with open(outpath, "w") as output:
        json.dump(results, output, indent=2)
    results_vol.commit()
    with open(outpath) as output:
        data = output.read()
    print(data, flush=True)
    return data


# -----------------------------------------------------------------
# 1b. Rule out max_num_seqs as the cause of the B=1024 anomaly.
#
# batchsweep sets max_num_seqs = min(4096, B), so below B=4096 it
# moves TWO knobs at once. Its B=1024 point read 39,287 tok/s where
# the fitted step model predicts 72,600, while B=512 (60,448) and
# B=2048 (76,595) sit on the line - but that point also ran with a
# sequence cap of 1024, so it never measured the token budget alone.
#
# This re-runs the same B values with the sequence cap pinned at 512,
# so B is the only thing that moves. The dip remained in this control.
# The CUDA graph capture limit test above identified the actual cause.
# -----------------------------------------------------------------

KNOBGRID_RUNNER = r'''
import gc, json, sys, time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
sys.path.insert(0, "/root")
from workload import MODEL, build_corpus

n_docs = int(sys.argv[1])
outpath = sys.argv[2]
grid = [tuple(int(x) for x in pair.split(":"))
        for pair in sys.argv[3].split(",")]

tok = AutoTokenizer.from_pretrained(MODEL)
body_ids, q_ids, _flags = build_corpus(tok, n_docs)
sp = SamplingParams(temperature=0.0, max_tokens=1)
prompts = [{"prompt_token_ids": body_ids[i] + q_ids[0]}
           for i in range(n_docs)]
total = sum(len(p["prompt_token_ids"]) for p in prompts)

results = []
for B, seqs in grid:
    print(f"\n[knobgrid] === B={B}, max_num_seqs={seqs} ===", flush=True)
    try:
        llm = LLM(model=MODEL, kv_cache_dtype="fp8", max_model_len=4608,
                  max_num_seqs=seqs, max_num_batched_tokens=B,
                  gpu_memory_utilization=0.88,
                  enable_prefix_caching=False, disable_log_stats=True)
        t0 = time.time()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        wall = time.time() - t0
        toks = sum(len(o.prompt_token_ids) for o in outs)
        cached = sum(getattr(o, "num_cached_tokens", 0) or 0 for o in outs)
        row = dict(B=B, max_num_seqs=seqs, wall=round(wall, 3),
                   prompt_tokens=toks, cached_tokens=cached,
                   rate=round((toks - cached) / wall, 1))
        print(f"[knobgrid] B={B} seqs={seqs}: {wall:.2f}s, "
              f"{row['rate']:,.0f} tok/s", flush=True)
        del llm
    except Exception as e:
        row = dict(B=B, max_num_seqs=seqs, error=f"{type(e).__name__}: {e}")
        print(f"[knobgrid] B={B} seqs={seqs} FAILED: {e}", flush=True)
    results.append(row)
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
    time.sleep(5)

with open(outpath, "w") as f:
    json.dump(dict(n_docs=n_docs, corpus_tokens=total, cells=results),
              f, indent=2)
print(f"\n[knobgrid] done", flush=True)
'''


@app.function(image=prof_image, gpu="H100!", timeout=7200, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def knobgrid(n_docs: int = 10000, grid: str = "") -> str:
    """The batch sweep's low-B points with max_num_seqs pinned, so
    max_num_batched_tokens is the only variable. One container, one
    corpus."""
    import subprocess

    if not grid:
        # The cap must be pinned at or below every B: vLLM rejects a
        # config with max_num_batched_tokens < max_num_seqs, which is
        # exactly why batchsweep tied them together. Pin it at 512,
        # the smallest B tested, so the B=512 cell is byte-identical
        # to batchsweep's B=512 cell and anchors the comparison; after
        # that only B moves. The cap cannot bind at any of these
        # sizes - a 512 to 2048 token step holds 1.4 to 5.6 requests
        # of ~366 tokens, far under 512 running. B=1024 runs twice to
        # show the within-container spread.
        grid = ",".join(f"{b}:512" for b in (512, 1024, 2048, 1024))

    outpath = "/results/knobgrid.json"
    with open("/tmp/knobgrid_runner.py", "w") as f:
        f.write(KNOBGRID_RUNNER)
    r = subprocess.run(["python", "/tmp/knobgrid_runner.py",
                        str(n_docs), outpath, grid])
    results_vol.commit()
    with open(outpath) as f:
        data = f.read()
    print(f"[knobgrid] exited {r.returncode}", flush=True)
    print(data, flush=True)
    return data


# -----------------------------------------------------------------
# 2/3. The torch profiler, whole-window and per-component forms.
# -----------------------------------------------------------------

TORCHPROF_RUNNER = r'''
import asyncio, glob, gzip, inspect, json, os, sys, time
os.environ["QUAIL_SINGLE_TENANT"] = "1"
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM as Engine
sys.path.insert(0, "/root")
from workload import MODEL, N_FILTERS, build_corpus, yes_no_ids

n_docs, outdir, side = int(sys.argv[1]), sys.argv[2], sys.argv[3]
B = int(sys.argv[4])
n_filters = int(sys.argv[5])

tok = AutoTokenizer.from_pretrained(MODEL)
yes_ids, no_ids = yes_no_ids(tok)
body_ids, q_ids, _flags = build_corpus(tok, n_docs)
sp = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1,
                    allowed_token_ids=sorted(yes_ids | no_ids))

# this vLLM takes profiling as an engine argument, not the old
# VLLM_TORCH_PROFILER_DIR env var
try:
    from vllm.config import ProfilerConfig
    _pc = ProfilerConfig(profiler="torch", torch_profiler_dir=outdir)
except Exception:
    _pc = dict(profiler="torch", torch_profiler_dir=outdir)

extra = ({"scheduler_cls": "quail.engineext.scheduler.QuailScheduler"}
         if side == "rewind" else {})
engine = Engine.from_engine_args(AsyncEngineArgs(
    model=MODEL, kv_cache_dtype="fp8", max_model_len=4608,
    max_num_seqs=min(4096, B), max_num_batched_tokens=B,
    gpu_memory_utilization=0.92 if side == "rewind" else 0.88,
    enable_prefix_caching=True, disable_log_stats=True,
    profiler_config=_pc, **extra))

async def maybe(x):
    if inspect.isawaitable(x):
        return await x
    return x

async def main():
    if side == "rewind":
        from quail.runtime.engine_client import run_filter_chain_engine
        work = run_filter_chain_engine(engine, sp, body_ids,
                                       q_ids[:n_filters], 750_000,
                                       yes_ids, tag="tp")
    else:
        sem = asyncio.Semaphore(2048)
        async def ask(ids, rid):
            async for _out in engine.generate({"prompt_token_ids": ids},
                                              sp, rid):
                pass
        async def one_doc(i):
            async with sem:
                for j in range(n_filters):
                    await ask(body_ids[i] + q_ids[j], f"tp-{i}-{j}")
        work = asyncio.gather(*(one_doc(i) for i in range(n_docs)))

    async def window():
        # a 15-second window well past the ramp. Failures print
        # loudly instead of banking an empty directory.
        try:
            await asyncio.sleep(15)
            await maybe(engine.start_profile())
            await asyncio.sleep(15)
            await maybe(engine.stop_profile())
            print("[torchprof] window ok", flush=True)
        except Exception as e:
            print(f"[torchprof] WINDOW FAILED: {type(e).__name__}: {e}",
                  flush=True)

    t0 = time.time()
    await asyncio.gather(window(), work)
    print(f"[torchprof] wall {time.time() - t0:.2f}s", flush=True)

try:
    asyncio.run(main())
finally:
    engine.shutdown()

# summarize the trace into kernel classes on the way out
traces = glob.glob(outdir + "/*rank0*")
if traces:
    opener = gzip.open if traces[0].endswith(".gz") else open
    with opener(traces[0], "rt") as f:
        events = json.load(f)["traceEvents"]
    gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
    kernels = [e for e in events
               if e.get("cat") in gpu_cats and e.get("dur", 0) > 0]
    total = sum(e["dur"] for e in kernels)
    classes = dict(gemm=0, quantize=0, norm=0, elementwise=0,
                   attention=0, other=0)
    for e in kernels:
        k = e.get("name", "").lower()
        if "gemm" in k or "cutlass" in k or "nvjet" in k:
            classes["gemm"] += e["dur"]
        elif "quant" in k or "scale" in k or "cast" in k:
            classes["quantize"] += e["dur"]
        elif "norm" in k or "rms" in k:
            classes["norm"] += e["dur"]
        elif ("attn" in k or "attention" in k or "flash" in k
              or "fmha" in k):
            classes["attention"] += e["dur"]
        elif ("silu" in k or "gelu" in k or "add" in k or "mul" in k
              or "residual" in k):
            classes["elementwise"] += e["dur"]
        else:
            classes["other"] += e["dur"]
    # GPU busy fraction: the union of kernel intervals over the window
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels)
    merged = []
    for a, b in iv:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    span = merged[-1][1] - merged[0][0]
    busy = sum(b - a for a, b in merged)
    out = dict(B=B, side=side, n_docs=n_docs, n_filters=n_filters,
               n_kernels=len(kernels), total_kernel_us=total,
               window_s=round(span / 1e6, 2),
               gpu_busy_frac=round(busy / span, 4))
    for cls, us in classes.items():
        out[f"{cls}_us"] = us
        out[f"{cls}_frac"] = round(us / total, 4) if total else 0
    print(json.dumps(out, indent=2), flush=True)
    with open(f"/results/torchprof_{side}_B{B}.json", "w") as f:
        json.dump(out, f, indent=2)
'''


def _run_torchprof(n_docs, side, B, n_filters, tag):
    import subprocess
    outdir = f"/results/torchprof_{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open("/tmp/tp_runner.py", "w") as f:
        f.write(TORCHPROF_RUNNER)
    r = subprocess.run(["python", "/tmp/tp_runner.py", str(n_docs),
                        outdir, side, str(B), str(n_filters)])
    results_vol.commit()
    try:
        with open(f"/results/torchprof_{side}_B{B}.json") as f:
            return f.read()
    except FileNotFoundError:
        return f"FAILED exit {r.returncode}"


@app.function(image=prof_image, gpu="H100!", timeout=5400, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def torchprof(n_docs: int = 10000, side: str = "stock",
              B: int = 25305, n_filters: int = 5) -> str:
    """The GPU timeline instrument: vLLM's torch profiler running
    inside the engine core over a 15-second steady-state window.
    side "stock" runs per-document sequential requests, "rewind" runs
    the chain executor on the Quail scheduler."""
    return _run_torchprof(n_docs, side, B, n_filters, f"{side}_B{B}")


@app.function(image=prof_image, gpu="H100!", timeout=3600, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def component_profile(b: int = 25305, n_docs: int = 10000) -> str:
    """One profiled run at batch size B, one filter, trace parsed
    into kernel classes. Launch several in parallel (see
    run_component_profiles) to watch the class mix move with B."""
    return _run_torchprof(n_docs, "stock", b, 1, f"comp_B{b}")


@app.function(image=prof_image, gpu="H100!", timeout=7200, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def component_profile_512_1024(n_docs: int = 10000) -> str:
    """Profile B=512 then B=1024 in one container on one H100."""
    import json

    results = []
    for b in (512, 1024):
        result = _run_torchprof(
            n_docs, "stock", b, 1, f"matched_512_1024_B{b}"
        )
        results.append(json.loads(result))
    return json.dumps(results, indent=2)


@app.local_entrypoint()
def run_component_profiles(batch_sizes: str = "512,4096,25305",
                           n_docs: int = 10000):
    """Profiled runs at several batch sizes, in parallel containers so
    the wall clock is one run instead of several."""
    handles = [component_profile.spawn(b=int(b), n_docs=n_docs)
               for b in batch_sizes.split(",")]
    for h in handles:
        print(h.get())


# -----------------------------------------------------------------
# 4. Nsight Compute: are the GEMM kernels themselves any good?
# -----------------------------------------------------------------

def _find_ncu():
    import glob
    import subprocess
    subprocess.run(
        "apt-get update -qq && apt-get install -y -qq "
        "cuda-nsight-compute-13-0 >/dev/null 2>&1 || "
        "apt-get install -y -qq nsight-compute >/dev/null 2>&1",
        shell=True)
    for c in (glob.glob("/opt/nvidia/nsight-compute/*/ncu")
              + ["/usr/local/cuda/bin/ncu", "/usr/local/bin/ncu"]):
        if os.path.exists(c):
            return c
    return None


@app.function(image=prof_image, gpu="H100!", timeout=1800,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def ncubench() -> str:
    """Speed-of-light on standalone GEMMs at the 4B prefill shapes.
    Microbenchmark form: against a live engine ncu intercepts every
    boot kernel and times out, so the usable recipe is isolated
    matmuls at the shapes a 25,305-token step actually runs."""
    import subprocess

    ncu = _find_ncu()
    if ncu is None:
        return "ncu not found"
    with open("/tmp/gemm.py", "w") as f:
        f.write(
            "import torch\n"
            "M, K = 25305, 2560   # step tokens x 4B hidden\n"
            "for N in (4096, 2560, 13824):  # qkv-ish, o, mlp-ish\n"
            "    a = torch.randn(M, K, device='cuda',"
            " dtype=torch.bfloat16)\n"
            "    b = torch.randn(K, N, device='cuda',"
            " dtype=torch.bfloat16)\n"
            "    for _ in range(5): c = a @ b\n"
            "try:\n"
            "    af = a[:, :2560].to(torch.float8_e4m3fn)\n"
            "    bf = b[:2560, :].t().contiguous().t()"
            ".to(torch.float8_e4m3fn)\n"
            "    s = torch.tensor(1.0, device='cuda')\n"
            "    for _ in range(5):\n"
            "        torch._scaled_mm(af, bf, scale_a=s, scale_b=s,"
            " out_dtype=torch.bfloat16)\n"
            "except Exception as e:\n"
            "    print('fp8 path skipped:', e)\n"
            "torch.cuda.synchronize(); print('bench done')\n")
    r = subprocess.run(
        [ncu, "--clock-control", "none", "--launch-skip", "3",
         "--launch-count", "12",
         "-k", "regex:gemm|Gemm|nvjet|cutlass|scaled",
         "--section", "SpeedOfLight",
         "--export", "/results/ncu_gemm_bench", "--force-overwrite",
         "python", "/tmp/gemm.py"],
        capture_output=True, text=True, timeout=1500)
    with open("/results/ncu_gemm_bench_summary.txt", "w") as f:
        f.write(r.stdout)
    results_vol.commit()
    tail = r.stdout[-7000:] + "\n--- stderr ---\n" + r.stderr[-1200:]
    print(tail, flush=True)
    return tail[:6000]


@app.function(image=prof_image, gpu="H100!", timeout=900,
              volumes={"/results": results_vol})
def ncureport() -> str:
    """Render the banked ncu report's details page: the per-kernel
    Compute (SM) and DRAM throughput rows. DRAM at 28-30 percent
    against Compute at 91-93 is the expected shape for a
    compute-bound GEMM - most of its memory traffic is served by L2,
    and the math unit, not the memory system, is the limit."""
    import glob
    import subprocess

    subprocess.run(
        "apt-get update -qq && apt-get install -y -qq "
        "cuda-nsight-compute-13-0 >/dev/null 2>&1", shell=True)
    ncu = glob.glob("/opt/nvidia/nsight-compute/*/ncu")[0]
    r = subprocess.run(
        [ncu, "--import", "/results/ncu_gemm_bench.ncu-rep",
         "--page", "details"], capture_output=True, text=True)
    keep = [ln for ln in r.stdout.splitlines()
            if any(k in ln for k in ("nvjet", "Compute (SM)",
                                     "Memory Throughput",
                                     "DRAM Throughput", "Duration",
                                     "Elapsed Cycles"))]
    out = "\n".join(keep[:60]) or r.stdout[-2000:] + r.stderr[-500:]
    print(out, flush=True)
    return out[:6000]

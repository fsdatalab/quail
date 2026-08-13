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

BSWEEP_RUNNER = r'''
import gc, json, sys, time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
sys.path.insert(0, "/root")
from workload import MODEL, build_corpus, yes_no_ids

n_docs = int(sys.argv[1])
batch_sizes = [int(x) for x in sys.argv[2].split(",")]
outpath = sys.argv[3]

tok = AutoTokenizer.from_pretrained(MODEL)
body_ids, q_ids, _flags = build_corpus(tok, n_docs)
sp = SamplingParams(temperature=0.0, max_tokens=1)

results = []
for B in batch_sizes:
    print(f"\n[bsweep] === B={B}, sync LLM, {n_docs} docs ===", flush=True)
    llm = LLM(model=MODEL, kv_cache_dtype="fp8", max_model_len=4608,
              max_num_seqs=min(4096, B), max_num_batched_tokens=B,
              gpu_memory_utilization=0.88,
              enable_prefix_caching=False, disable_log_stats=True)
    # one filter only: every request is one document plus one
    # question, so nothing shares a prefix and the measurement is
    # pure prefill throughput with no cache effects
    prompts = [{"prompt_token_ids": body_ids[i] + q_ids[0]}
               for i in range(n_docs)]
    t0 = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    wall = time.time() - t0
    toks = sum(len(o.prompt_token_ids) for o in outs)
    cached = sum(getattr(o, "num_cached_tokens", 0) or 0 for o in outs)
    uncached = toks - cached
    row = dict(B=B, n_docs=n_docs, wall=round(wall, 3),
               prompt_tokens=toks, cached_tokens=cached,
               uncached_tokens=uncached, rate=round(uncached / wall, 1))
    results.append(row)
    print(f"[bsweep B={B}] wall {wall:.2f}s, {toks:,} prompt toks, "
          f"rate {uncached / wall:,.0f} tok/s", flush=True)
    del llm
    gc.collect()
    import torch; torch.cuda.empty_cache()
    time.sleep(5)

with open(outpath, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[bsweep] done: {json.dumps(results, indent=2)}", flush=True)
'''


@app.function(image=prof_image, gpu="H100!", timeout=7200, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def batchsweep(n_docs: int = 10000,
               batch_sizes: str = "512,1024,2048,4096,8192,16384,25305"
               ) -> str:
    """Walls at several max_num_batched_tokens values WITHOUT a
    profiler, all in one container so host variance cannot move the
    comparison. Fit 1/throughput = a + b/B to the result."""
    import subprocess

    outpath = "/results/batchsweep.json"
    with open("/tmp/bsweep_runner.py", "w") as f:
        f.write(BSWEEP_RUNNER)
    r = subprocess.run(["python", "/tmp/bsweep_runner.py",
                        str(n_docs), batch_sizes, outpath])
    results_vol.commit()
    with open(outpath) as f:
        data = f.read()
    print(f"[batchsweep] exited {r.returncode}", flush=True)
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

asyncio.run(main())

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
def component_profile(B: int = 25305, n_docs: int = 10000) -> str:
    """One profiled run at batch size B, one filter, trace parsed
    into kernel classes. Launch several in parallel (see
    run_component_profiles) to watch the class mix move with B."""
    return _run_torchprof(n_docs, "stock", B, 1, f"comp_B{B}")


@app.local_entrypoint()
def run_component_profiles(batch_sizes: str = "512,4096,25305",
                           n_docs: int = 10000):
    """Profiled runs at several batch sizes, in parallel containers so
    the wall clock is one run instead of several."""
    handles = [component_profile.spawn(B=int(b), n_docs=n_docs)
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

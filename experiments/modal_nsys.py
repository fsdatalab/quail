"""Nsight Systems traces of the STOCK filter baselines: where the
baseline's wall goes at the CUDA level (kernel busy fraction, gaps,
memcpy share). Complements the yappi CPU profile of our scheduler.

Predictions, stated before the run: steady-state GPU busy at or
above ~85 percent on both tiers; fp8 GEMM prefill kernels dominate;
memcpy negligible; inter-step gaps a few milliseconds absolute on
both models (host-side per-step costs), so a smaller relative share
at 32B. A gappy 4B timeline would instead mean the stock baseline
is host-limited and its banked walls carry recoverable headroom.

CPU sampling and context-switch tracing stay off: the sandbox does
not expose perf counters. The .nsys-rep files bank to the volume
for the GUI; text summaries bank alongside for analysis without it.

Run:
    modal run experiments/modal_nsys.py --model 4b
    modal run experiments/modal_nsys.py --model 32b
"""
import os
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modal_scale import (FLAG_SEED, MODEL, MODEL32,  # noqa: E402
                         _build_pool, _flags_line, _question, hf_cache,
                         image)

app = modal.App("docengine-nsys")
nsys_image = image.add_local_python_source("modal_scale")
results_vol = modal.Volume.from_name("docengine-results",
                                     create_if_missing=True)

RUNNER = r'''
import asyncio, os, sys
import numpy as np
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM as Engine
sys.path.insert(0, "/root")
from modal_scale import FLAG_SEED, MODEL, MODEL32, _build_pool, \
    _flags_line, _question

model_key, n_docs = sys.argv[1], int(sys.argv[2])
model = MODEL if model_key == "4b" else MODEL32
tok = AutoTokenizer.from_pretrained(model)
yes_ids, no_ids = set(), set()
for w in ("YES", " YES", "Yes", " Yes", "Y", " Y"):
    ids = tok(w, add_special_tokens=False)["input_ids"]
    if ids: yes_ids.add(ids[0])
for w in ("NO", " NO", "No", " No", "N", " N"):
    ids = tok(w, add_special_tokens=False)["input_ids"]
    if ids: no_ids.add(ids[0])
sels = (0.9, 0.9, 0.9, 0.8, 0.8)          # the permissive profile
rng = np.random.default_rng(FLAG_SEED + 100)
flags = (rng.random((n_docs, 5)) < np.array(sels)[None, :]).astype(int)
docs = _build_pool(n_docs)
bodies = [d + _flags_line(f) for d, f in zip(docs, flags)]
body_ids = tok(bodies, add_special_tokens=False)["input_ids"]
q_ids = [tok(_question(j + 1), add_special_tokens=False)["input_ids"]
         for j in range(5)]
sp = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1,
                    skip_clone=True,
                    allowed_token_ids=sorted(yes_ids | no_ids))
# the shipped stock baseline configuration: 0.88, full step budget
engine = Engine.from_engine_args(AsyncEngineArgs(
    model=model, kv_cache_dtype="fp8", max_model_len=4608,
    max_num_seqs=4096,
    max_num_batched_tokens=25305 if model_key == "4b" else 9521,
    gpu_memory_utilization=0.88, enable_prefix_caching=True,
    disable_log_stats=True))

async def main():
    sem = asyncio.Semaphore(4096)
    async def ask(ids, rid):
        final = None
        async for out in engine.generate(
                {"prompt_token_ids": ids}, sp, rid):
            final = out
        t = final.outputs[0].text.upper()
        iy, ino = t.find("YES"), t.find("NO")
        return 1 if iy >= 0 and (ino < 0 or iy < ino) else 0
    async def chain(i):
        async with sem:
            for j in range(5):
                if not await ask(body_ids[i] + q_ids[j], f"n-{i}-{j}"):
                    return
    import time
    t0 = time.time()
    await asyncio.gather(*(chain(i) for i in range(n_docs)))
    print(f"[nsys-runner] wall {time.time() - t0:.2f}s", flush=True)

asyncio.run(main())
'''


@app.function(image=nsys_image, gpu="H100!", timeout=5400,
              memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def profile(model_key: str = "4b", n_docs: int = 10000) -> str:
    import shutil
    import subprocess

    nsys = shutil.which("nsys")
    if nsys is None:
        for cand in ("/usr/local/cuda/bin/nsys",
                     "/opt/nvidia/nsight-systems/bin/nsys"):
            if os.path.exists(cand):
                nsys = cand
                break
    if nsys is None:
        subprocess.run(
            "apt-get update -qq && apt-get install -y -qq "
            "cuda-nsight-systems-13-0 || apt-get install -y -qq "
            "nsight-systems-cli", shell=True, check=True)
        nsys = shutil.which("nsys") or "/usr/local/cuda/bin/nsys"
    with open("/tmp/runner.py", "w") as f:
        f.write(RUNNER)
    rep = f"/tmp/nsys_{model_key}_filter"
    # stream output live: the first attempt died silently with the
    # console captured, so nothing reached the Modal log
    # bounded capture: a 60-second steady-state window starting
    # after boot, so trace finalization never runs under full load
    # (the unbounded captures died mid-run four ways)
    r = subprocess.run(
        [nsys, "profile", "--trace=cuda", "--sample=none",
         "--cpuctxsw=none", "--delay", "150", "--duration", "60",
         "-o", rep, "--force-overwrite=true",
         "python", "/tmp/runner.py", model_key, str(n_docs)])
    print(f"[nsys] profile exited {r.returncode}", flush=True)
    stats = subprocess.run(
        [nsys, "stats", "--report",
         "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum",
         rep + ".nsys-rep"], capture_output=True, text=True)
    with open(f"/results/nsys_{model_key}_filter_stats.txt", "w") as f:
        f.write(stats.stdout)
    shutil.copy(rep + ".nsys-rep",
                f"/results/nsys_{model_key}_filter.nsys-rep")
    results_vol.commit()
    print(f"[nsys] banked nsys_{model_key}_filter.nsys-rep and stats",
          flush=True)
    return stats.stdout[:6000]


@app.local_entrypoint()
def main(model: str = "4b", n_docs: int = 0):
    nd = n_docs or (10000 if model == "4b" else 2000)
    out = profile.remote(model, nd)
    print(out)


@app.function(image=nsys_image, gpu="H100!", timeout=900)
def diag() -> str:
    """Two-minute discriminator: nsys status -e (what the
    environment supports) plus a one-matmul trace (does CUPTI
    kernel data arrive at all). Modal confirms nsys is supported,
    so an empty kernel table here means our stack, most likely the
    nsys/CUPTI version against the host driver."""
    import shutil
    import subprocess

    subprocess.run(
        "apt-get update -qq && apt-get install -y -qq "
        "cuda-nsight-systems-13-0 || apt-get install -y -qq "
        "nsight-systems-cli", shell=True)
    nsys = shutil.which("nsys") or "/usr/local/bin/nsys"
    out = []
    v = subprocess.run(["nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader"],
                       capture_output=True, text=True)
    out.append("driver: " + v.stdout.strip())
    e = subprocess.run([nsys, "status", "-e"], capture_output=True,
                       text=True)
    out.append("===== nsys status -e =====\n" + e.stdout + e.stderr)
    with open("/tmp/k.py", "w") as f:
        f.write("import torch\n"
                "a = torch.randn(4096, 4096, device='cuda')\n"
                "for _ in range(50): a = a @ a * 0 + a\n"
                "torch.cuda.synchronize()\nprint('kernels done')\n")
    r = subprocess.run([nsys, "profile", "--trace=cuda",
                        "--sample=none", "-o", "/tmp/diag",
                        "--force-overwrite=true", "python",
                        "/tmp/k.py"], capture_output=True, text=True)
    out.append("===== profile tail =====\n" + r.stdout[-1200:]
               + r.stderr[-600:])
    s = subprocess.run([nsys, "stats", "--report", "cuda_gpu_kern_sum",
                        "/tmp/diag.nsys-rep"], capture_output=True,
                       text=True)
    out.append("===== kern_sum =====\n" + s.stdout[:2500]
               + s.stderr[:400])
    print("\n".join(out), flush=True)
    return "\n".join(out)


@app.function(image=nsys_image, gpu="H100!", timeout=900,
              volumes={"/results": results_vol})
def diag2() -> str:
    """Round two, after the timestamp-counter finding: (a) does nsys
    expose a clock-source escape hatch; (b) does the PyTorch
    profiler - same CUPTI activity interface, its own clock
    handling - capture kernel events where nsys drops them. If (b)
    works, the baseline timelines come from torch.profiler and nsys
    is unnecessary here."""
    import subprocess

    out = []
    subprocess.run(
        "apt-get update -qq && apt-get install -y -qq "
        "cuda-nsight-systems-13-0 >/dev/null 2>&1", shell=True)
    h = subprocess.run(["nsys", "profile", "--help"],
                       capture_output=True, text=True)
    clockish = [l for l in h.stdout.splitlines()
                if "clock" in l.lower() or "timestamp" in l.lower()
                or "tsc" in l.lower()]
    out.append("===== nsys clock-related options =====\n"
               + ("\n".join(clockish) or "(none found)"))

    import torch
    from torch.profiler import ProfilerActivity, profile
    a = torch.randn(4096, 4096, device="cuda")
    with profile(activities=[ProfilerActivity.CPU,
                             ProfilerActivity.CUDA]) as prof:
        for _ in range(50):
            a = a @ a * 0 + a
        torch.cuda.synchronize()
    events = prof.key_averages()
    cuda_rows = [e for e in events
                 if getattr(e, "device_time_total", 0) > 0
                 or "gemm" in e.key.lower() or "sgemm" in e.key.lower()]
    out.append(f"===== torch.profiler =====\n"
               f"total event kinds: {len(events)}; kinds with GPU "
               f"time: {len(cuda_rows)}")
    out.append(events.table(sort_by="cuda_time_total",
                            row_limit=8))
    prof.export_chrome_trace("/results/torchprof_smoke.json")
    out.append("chrome trace banked: torchprof_smoke.json")
    s = "\n".join(out)
    print(s, flush=True)
    return s


TORCHPROF_RUNNER = r'''
import asyncio, inspect, os, sys, time
import numpy as np
os.environ["VLLM_TORCH_PROFILER_DIR"] = sys.argv[3]
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM as Engine
sys.path.insert(0, "/root")
from modal_scale import FLAG_SEED, MODEL, MODEL32, _build_pool, \
    _flags_line, _question

model_key, n_docs = sys.argv[1], int(sys.argv[2])
model = MODEL if model_key == "4b" else MODEL32
tok = AutoTokenizer.from_pretrained(model)
yes_ids, no_ids = set(), set()
for w in ("YES", " YES", "Yes", " Yes", "Y", " Y"):
    ids = tok(w, add_special_tokens=False)["input_ids"]
    if ids: yes_ids.add(ids[0])
for w in ("NO", " NO", "No", " No", "N", " N"):
    ids = tok(w, add_special_tokens=False)["input_ids"]
    if ids: no_ids.add(ids[0])
rng = np.random.default_rng(FLAG_SEED + 100)
flags = (rng.random((n_docs, 5))
         < np.array((0.9, 0.9, 0.9, 0.8, 0.8))[None, :]).astype(int)
docs = _build_pool(n_docs)
bodies = [d + _flags_line(f) for d, f in zip(docs, flags)]
body_ids = tok(bodies, add_special_tokens=False)["input_ids"]
q_ids = [tok(_question(j + 1), add_special_tokens=False)["input_ids"]
         for j in range(5)]
sp = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1,
                    skip_clone=True,
                    allowed_token_ids=sorted(yes_ids | no_ids))
# this vLLM takes profiling as an engine argument, not the old
# VLLM_TORCH_PROFILER_DIR env var (the hardened run printed the
# exact requirement)
try:
    from vllm.config import ProfilerConfig
    _pc = ProfilerConfig(profiler="torch",
                         torch_profiler_dir=sys.argv[3])
except Exception:
    _pc = dict(profiler="torch", torch_profiler_dir=sys.argv[3])
engine = Engine.from_engine_args(AsyncEngineArgs(
    model=model, kv_cache_dtype="fp8", max_model_len=4608,
    max_num_seqs=4096,
    max_num_batched_tokens=25305 if model_key == "4b" else 9521,
    gpu_memory_utilization=0.88, enable_prefix_caching=True,
    disable_log_stats=True, profiler_config=_pc))

async def maybe(x):
    if inspect.isawaitable(x):
        return await x
    return x

async def main():
    sem = asyncio.Semaphore(4096)
    async def ask(ids, rid):
        final = None
        async for out in engine.generate(
                {"prompt_token_ids": ids}, sp, rid):
            final = out
        t = final.outputs[0].text.upper()
        iy, ino = t.find("YES"), t.find("NO")
        return 1 if iy >= 0 and (ino < 0 or iy < ino) else 0
    async def chain(i):
        async with sem:
            for j in range(5):
                if not await ask(body_ids[i] + q_ids[j], f"n-{i}-{j}"):
                    return
    async def window():
        # a 15-second steady-state window, well past the ramp.
        # Failures print loudly instead of killing the run silently
        # (the first flight banked an empty directory with no clue).
        try:
            await asyncio.sleep(20)
            print("[torchprof] has start_profile:",
                  hasattr(engine, "start_profile"), flush=True)
            print("[torchprof] start_profile", flush=True)
            await maybe(engine.start_profile())
            await asyncio.sleep(15)
            await maybe(engine.stop_profile())
            print("[torchprof] stop_profile ok", flush=True)
        except Exception as e:
            print(f"[torchprof] WINDOW FAILED: {type(e).__name__}: "
                  f"{e}", flush=True)
    t0 = time.time()
    await asyncio.gather(window(),
                         *(chain(i) for i in range(n_docs)))
    print(f"[torchprof-runner] wall {time.time() - t0:.2f}s",
          flush=True)
    import glob as _g
    print("[torchprof-runner] outdir:",
          _g.glob(os.environ["VLLM_TORCH_PROFILER_DIR"] + "/*"),
          flush=True)

asyncio.run(main())
'''


@app.function(image=nsys_image, gpu="H100!", timeout=5400,
              memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def torchprof(model_key: str = "4b", n_docs: int = 10000,
              side: str = "stock") -> str:
    """The working GPU-timeline instrument: vLLM's native torch
    profiler runs INSIDE the engine core (where the CUDA lives) and
    exports a chrome trace of a 15-second steady-state window.
    torch.profiler's CUPTI path keeps kernel records that nsys's
    clock correlation drops in this sandbox (diag2, 2026-08-09).
    side "ours" runs the chain executor on our scheduler - the
    rewind machinery's trace."""
    import glob
    import subprocess

    outdir = (f"/results/torchprof_{model_key}_filter_{side}"
              if side != "stock"
              else f"/results/torchprof_{model_key}_filter")
    os.makedirs(outdir, exist_ok=True)
    with open("/tmp/tp_runner.py", "w") as f:
        f.write(TORCHPROF_RUNNER)
    r = subprocess.run(["python", "/tmp/tp_runner.py", model_key,
                        str(n_docs), outdir, side])
    results_vol.commit()
    traces = glob.glob(outdir + "/*")
    print(f"[torchprof] exited {r.returncode}; banked: {traces}",
          flush=True)
    return str(traces)


MAPPROF_RUNNER = r'''
import asyncio, inspect, os, sys, time
os.environ["DOCENGINE_SINGLE_TENANT"] = "1"
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM as Engine
sys.path.insert(0, "/root")
from modal_scale import MODEL, _build_pool

side, n_docs, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cap = int(sys.argv[4])
tok = AutoTokenizer.from_pretrained(MODEL)
docs = _build_pool(n_docs)
body_ids = tok(docs, add_special_tokens=False)["input_ids"]
prompts = (
    "\n\nInstruction: give a one-sentence summary of the review "
    "above.\nSummary:",
    "\n\nInstruction: in one sentence, state the reviewer's overall "
    "sentiment and why.\nAnswer:",
    "\n\nInstruction: name the movie or show being reviewed, if "
    "stated; otherwise say unknown.\nAnswer:",
    "\n\nInstruction: guess the genre of the movie in at most five "
    "words.\nGenre:",
    "\n\nInstruction: quote the single phrase from the review that "
    "best captures its tone.\nQuote:",
)
p_ids = [tok(p, add_special_tokens=False)["input_ids"]
         for p in prompts]
try:
    from vllm.config import ProfilerConfig
    _pc = ProfilerConfig(profiler="torch", torch_profiler_dir=outdir)
except Exception:
    _pc = dict(profiler="torch", torch_profiler_dir=outdir)
extra = {}
if side == "ours":
    extra = dict(
        scheduling_policy="priority",
        kv_transfer_config=KVTransferConfig(
            kv_connector="DocEngineForkConnector",
            kv_connector_module_path="docengine.engineext.forkconnector",
            kv_role="kv_both"),
        scheduler_cls="docengine.engineext.scheduler.DocEngineScheduler")
engine = Engine.from_engine_args(AsyncEngineArgs(
    model=MODEL, kv_cache_dtype="fp8", max_model_len=4608,
    max_num_seqs=4096, max_num_batched_tokens=25305,
    gpu_memory_utilization=0.90 if side == "ours" else 0.92,
    enable_prefix_caching=True, disable_log_stats=True,
    profiler_config=_pc, **extra))
sp = SamplingParams(temperature=0.0, max_tokens=cap)

async def maybe(x):
    if inspect.isawaitable(x):
        return await x
    return x

async def window():
    try:
        # cap-64 runs ~130-160s; t+45 lands in the mixed
        # prefill-and-decode steady state on either side
        await asyncio.sleep(45)
        print("[mapprof] start_profile", flush=True)
        await maybe(engine.start_profile())
        await asyncio.sleep(15)
        await maybe(engine.stop_profile())
        print("[mapprof] stop_profile ok", flush=True)
    except Exception as e:
        print(f"[mapprof] WINDOW FAILED: {type(e).__name__}: {e}",
              flush=True)

async def main():
    t0 = time.time()
    if side == "ours":
        from docengine.runtime.engine_client import run_map
        work = run_map(engine, sp, body_ids, p_ids, 749_782, tag="mp")
        await asyncio.gather(window(), work)
    else:
        sem = asyncio.Semaphore(4096)
        async def gen(i, j):
            async with sem:
                async for _ in engine.generate(
                        {"prompt_token_ids": body_ids[i] + p_ids[j]},
                        sp, f"mp-{i}-{j}"):
                    pass
        await asyncio.gather(window(),
                             *(gen(i, j) for i in range(n_docs)
                               for j in range(5)))
    print(f"[mapprof-runner] wall {time.time() - t0:.2f}s", flush=True)
    import glob as _g
    print("[mapprof-runner] outdir:", _g.glob(outdir + "/*"),
          flush=True)

asyncio.run(main())
'''


@app.function(image=nsys_image, gpu="H100!", timeout=5400,
              memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def mapprof(side: str = "stock", n_docs: int = 10000,
            cap: int = 64) -> str:
    """Torch-profiler window on the cap-64 open-ended map, either
    side: the decode stretches are where the per-step software floor
    lives, so unlike the filter traces the stream row here should
    show real gaps - the anatomy of the 2 percent loss, drawn."""
    import glob
    import subprocess

    outdir = f"/results/torchprof_{side}_map{cap}"
    os.makedirs(outdir, exist_ok=True)
    with open("/tmp/mp_runner.py", "w") as f:
        f.write(MAPPROF_RUNNER)
    r = subprocess.run(["python", "/tmp/mp_runner.py", side,
                        str(n_docs), outdir, str(cap)])
    results_vol.commit()
    traces = glob.glob(outdir + "/*")
    print(f"[mapprof] exited {r.returncode}; banked: {traces}",
          flush=True)
    return str(traces)


@app.function(image=nsys_image, gpu="H100!", timeout=3600,
              memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def ncuprof(model_key: str = "4b", n_docs: int = 200) -> str:
    """Nsight Compute on the stock filter workload's GEMM kernels:
    the PHI decomposition. Per Modal: ncu should work with
    --clock-control none (undocumented territory; failures get
    reported upstream). Replay overhead is why the corpus is tiny
    and only ~15 launches are captured, past warmup."""
    import glob
    import subprocess

    subprocess.run(
        "apt-get update -qq && apt-get install -y -qq "
        "cuda-nsight-compute-13-0 >/dev/null 2>&1 || "
        "apt-get install -y -qq nsight-compute >/dev/null 2>&1",
        shell=True)
    ncu = None
    for c in (glob.glob("/opt/nvidia/nsight-compute/*/ncu")
              + ["/usr/local/cuda/bin/ncu", "/usr/local/bin/ncu"]):
        if os.path.exists(c):
            ncu = c
            break
    if ncu is None:
        return "ncu not found after install"
    with open("/tmp/runner.py", "w") as f:
        f.write(RUNNER)
    cmd = [ncu, "--clock-control", "none",
           "--target-processes", "all",
           "--launch-skip", "300", "--launch-count", "15",
           "-k", "regex:gemm|Gemm|nvjet|cutlass",
           "--section", "SpeedOfLight",
           "--section", "Occupancy",
           "--export", f"/results/ncu_{model_key}_filter",
           "--force-overwrite",
           "python", "/tmp/runner.py", model_key, str(n_docs)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = r.stdout[-8000:] + "\n--- stderr ---\n" + r.stderr[-2000:]
    with open(f"/results/ncu_{model_key}_filter_summary.txt",
              "w") as f:
        f.write(r.stdout)
    results_vol.commit()
    print(tail, flush=True)
    return tail[:6000]


@app.function(image=nsys_image, gpu="H100!", timeout=1800,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def ncubench() -> str:
    """The PHI decomposition, microbench form: ncu speed-of-light on
    standalone GEMMs at the 4B prefill shapes (the live-engine
    attempt timed out - ncu intercepts every boot kernel). Shapes:
    a 25,305-token step against the 4B's projections. bf16 matmul
    and fp8 scaled_mm both measured where available."""
    import glob
    import subprocess

    subprocess.run(
        "apt-get update -qq && apt-get install -y -qq "
        "cuda-nsight-compute-13-0 >/dev/null 2>&1 || "
        "apt-get install -y -qq nsight-compute >/dev/null 2>&1",
        shell=True)
    ncu = None
    for c in (glob.glob("/opt/nvidia/nsight-compute/*/ncu")
              + ["/usr/local/cuda/bin/ncu", "/usr/local/bin/ncu"]):
        if os.path.exists(c):
            ncu = c
            break
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
         "--launch-count", "12", "-k", "regex:gemm|Gemm|nvjet|cutlass|scaled",
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


@app.function(image=nsys_image, gpu="H100!", timeout=900,
              volumes={"/results": results_vol})
def ncureport() -> str:
    """Render the banked ncu report's details page (the summary txt
    only captured progress lines)."""
    import glob
    import subprocess

    subprocess.run(
        "apt-get update -qq && apt-get install -y -qq "
        "cuda-nsight-compute-13-0 >/dev/null 2>&1", shell=True)
    ncu = glob.glob("/opt/nvidia/nsight-compute/*/ncu")[0]
    r = subprocess.run(
        [ncu, "--import", "/results/ncu_gemm_bench.ncu-rep",
         "--page", "details"], capture_output=True, text=True)
    keep = [l for l in r.stdout.splitlines()
            if any(k in l for k in ("nvjet", "Compute (SM)",
                                    "Memory Throughput",
                                    "DRAM Throughput", "Duration",
                                    "Elapsed Cycles"))]
    out = "\n".join(keep[:60]) or r.stdout[-2000:] + r.stderr[-500:]
    print(out, flush=True)
    return out[:6000]

"""Make vLLM's fusion passes fire at the best single-filter config.

The fusion A/B/C (modal_fusion.py) found the passes on but never firing.
Reading the pinned vllm 0.26.0 source located two independent causes:

  1. Norm side: the model's RMSNorm emits the maybe_inplace overload of
     the vllm_ir fused_add_rms_norm op, but the fusion patterns trace
     the functional overload. Different overload, zero matches at all
     36 fused-add sites. The fix is a one-token source patch in
     rms_quant_fusion.py that makes the patterns trace the same
     overload the model emits, gated by QUAIL_PATTERN_MAYBE_INPLACE.

  2. Act side: with the default custom_ops base "none", SiluAndMul
     lowers to native ops that inductor compiles into its own kernel
     (the profile's triton_poi_fused_mul_silu kernel). The silu+quant
     matcher adapts to the native form, but produced zero matches in
     the A/B/C, so the custom-op form is the hypothesis fix: enable
     "+silu_and_mul" so the graph carries the _C.silu_and_mul anchor
     node. Config only.

Three cells in one container on one H100, at the committed control
setting (25,305 batched tokens, CUDA graph at [8192], prefix caching
off). Each cell boots in its own subprocess so the source patch and
env gates apply cleanly:

  control         stock behavior (patched file present, gate off)
  act_fused       control + custom_ops ["+silu_and_mul"]
  act_norm_fused  act_fused + QUAIL_PATTERN_MAYBE_INPLACE=1

Prediction, stated before the run. The per-layer memory saving from
fusing is about 1,504 MB per 25,305-token step: 984 MB at the SiLU
site and 260 MB at each of the two norm sites. Against yesterday's
in-container control of 96,212 tokens per second:
  - control repeats 96,212 within 1 percent.
  - act_fused: if silu+quant fusion fires, its profile shows
    silu_and_mul_per_block_quant kernels and throughput rises 3.5 to
    6.5 percent. If it does not fire, the cell stays within 1 percent
    and isolates the plain kernel swap.
  - act_norm_fused: the norm patterns fire, rms_norm_per_block_quant
    kernels appear, the two plain fused_add_rms_norm triton kernels
    vanish (the QK-norm+RoPE composite stays: not quant-adjacent),
    and throughput ends 6 to 10 percent over control, 102,000 to
    106,000 tokens per second.
  - wrong answers stay within a few tens of the control's 2,990 of
    10,000; small nonzero disagreement counts against control are
    expected because the fused kernels round differently.
  Contingency: if a cell's expected fused kernels do not appear, the
  next step is one boot with VLLM_LOGGING_LEVEL=DEBUG and a graph
  dump to read the actual nodes, not more guessing.

Result: neither prediction branch happened - nothing fired anywhere.
All three cells ran the same 13,839 kernels with zero fused
norm+quant or silu+quant time. The maybe_inplace patch applied and
its patterns registered, but matched nothing, so at least one more
mismatch hides behind the overload one. Means were control 95,599,
act_fused 94,720, act_norm_fused 94,583 tokens per second, but every
cell drifted downward across its own reps by about 2.4 percent
(96.9k on first reps to 94.5k on last), so the 1 percent differences
between cells are inside the container's drift and no speed effect
is resolvable either way. The one clearly real change: the
silu_and_mul custom op swapped the SiLU kernel implementation and
flipped 1,768 of 10,000 answers against control (wrong moved 2,990
to 2,954) - repeating the packed run's lesson that this workload's
YES/NO margins flip on single-kernel rounding differences. Per the
contingency, the graphdump function below reads the compiler's
actual post-grad graph instead of guessing further.

Run:
    modal run experiments/modal_fusion_fix.py::main
    modal run experiments/modal_fusion_fix.py::graphdump_main
"""

import os

import modal

from workload import IMAGE_BASE, MODEL, hf_cache, results_vol

FLASH_ATTN_4_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "fa4-v4.0.0.beta26/flash_attn_4-4.0.0b26-py3-none-any.whl"
)
BEST_BATCH_TOKENS = 25_305
VLLM_GRAPH_TOKENS = 8_192

fix_image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "pandas", "pyarrow",
                 "numpy", "yappi", "accelerate")
    .pip_install("kernels==0.16.0")
    .pip_install(FLASH_ATTN_4_WHEEL, extra_options="--no-deps")
    .env({"VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("workload")
)

app = modal.App("quail-fusion-fix")


CELL_RUNNER = r'''
import gc, glob, gzip, json, os, sys, time

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, "/root")
from workload import MODEL, build_corpus, yes_no_ids

cell = sys.argv[1]
n_docs = int(sys.argv[2])
reps = int(sys.argv[3])
batch_tokens = int(sys.argv[4])
profile_docs = int(sys.argv[5])
outpath = sys.argv[6]

# Kernel classes, first match wins; attention before gemm because the
# sm90 FlashAttention mainloop is a cutlass::device_kernel.
RULES = (
    ("attention", ("attn", "attention", "flash", "fmha")),
    ("gemm", ("gemm", "cutlass", "nvjet")),
    ("quantize", ("quant", "scale", "cast")),
    ("norm", ("norm", "rms")),
    ("elementwise", ("silu", "gelu", "add", "mul", "residual")),
)

def classify(name):
    k = name.lower()
    for cls, keys in RULES:
        if any(s in k for s in keys):
            return cls
    return "other"

def summarize(events, top_n=20):
    gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
    classes = dict(gemm=0, quantize=0, norm=0, elementwise=0,
                   attention=0, other=0)
    fused_norm = fused_act = 0
    fused_calls = {}
    by_name = {}
    total = n = 0
    for e in events:
        if e.get("cat") not in gpu_cats or e.get("dur", 0) <= 0:
            continue
        name = e.get("name", "")
        dur = e["dur"]
        k = name.lower()
        classes[classify(name)] += dur
        if "quant" in k and ("norm" in k or "rms" in k):
            fused_norm += dur
            fused_calls[name[:80]] = fused_calls.get(name[:80], 0) + 1
        if "quant" in k and "silu" in k:
            fused_act += dur
            fused_calls[name[:80]] = fused_calls.get(name[:80], 0) + 1
        by_name[name] = by_name.get(name, 0) + dur
        total += dur
        n += 1
    out = dict(n_kernels=n, total_kernel_us=total)
    for cls, us in classes.items():
        out[f"{cls}_us"] = us
        out[f"{cls}_frac"] = round(us / total, 4) if total else 0
    out["fused_norm_quant_us"] = fused_norm
    out["fused_norm_quant_frac"] = round(fused_norm / total, 4) if total else 0
    out["fused_act_quant_us"] = fused_act
    out["fused_act_quant_frac"] = round(fused_act / total, 4) if total else 0
    out["fused_kernel_calls"] = fused_calls
    out["top_kernels"] = [
        [classify(nm), us, nm[:140]]
        for nm, us in sorted(by_name.items(), key=lambda kv: -kv[1])[:top_n]
    ]
    return out

tokenizer = AutoTokenizer.from_pretrained(MODEL)
body_ids, question_ids, flags = build_corpus(tokenizer, n_docs)
prompts = [body_ids[i] + question_ids[0] for i in range(n_docs)]
expected = [int(flags[i][0]) for i in range(n_docs)]
total_tokens = sum(map(len, prompts))
profile_tokens = sum(map(len, prompts[:profile_docs]))
yes_ids, no_ids = yes_no_ids(tokenizer)
allowed = sorted(yes_ids | no_ids)
vllm_prompts = [{"prompt_token_ids": p} for p in prompts]
sampling = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1,
                          allowed_token_ids=allowed)
graph_tokens = min(batch_tokens, 8192)

compilation = {
    "max_cudagraph_capture_size": graph_tokens,
    "cudagraph_capture_sizes": [graph_tokens],
}
if os.getenv("QUAIL_CUSTOM_SILU") == "1":
    compilation["custom_ops"] = ["+silu_and_mul"]

outdir = f"/tmp/prof_{cell}"
os.makedirs(outdir, exist_ok=True)
try:
    from vllm.config import ProfilerConfig
    prof_cfg = ProfilerConfig(profiler="torch", torch_profiler_dir=outdir)
except Exception:
    prof_cfg = dict(profiler="torch", torch_profiler_dir=outdir)

llm = LLM(
    model=MODEL,
    kv_cache_dtype="fp8",
    max_model_len=4608,
    max_num_seqs=4096,
    max_num_batched_tokens=batch_tokens,
    gpu_memory_utilization=0.88,
    enable_prefix_caching=False,
    disable_log_stats=True,
    compilation_config=compilation,
    profiler_config=prof_cfg,
)

try:
    cfg = llm.llm_engine.vllm_config
    cc = cfg.compilation_config
    pc = cc.pass_config
    resolved = {
        "optimization_level": int(cfg.optimization_level),
        "custom_ops": list(cc.custom_ops),
        "fuse_norm_quant": bool(pc.fuse_norm_quant),
        "fuse_act_quant": bool(pc.fuse_act_quant),
        "cudagraph_capture_sizes": sorted(cc.cudagraph_capture_sizes or []),
    }
except Exception as e:
    resolved = {"error": f"{type(e).__name__}: {e}"}
resolved["QUAIL_CUSTOM_SILU"] = os.getenv("QUAIL_CUSTOM_SILU", "")
resolved["QUAIL_PATTERN_MAYBE_INPLACE"] = os.getenv(
    "QUAIL_PATTERN_MAYBE_INPLACE", "")
print(f"[fix:{cell}] resolved {json.dumps(resolved)}", flush=True)

llm.generate(vllm_prompts[:64], sampling, use_tqdm=False)
runs = []
predicted = []
for rep in range(reps):
    t0 = time.perf_counter()
    outs = llm.generate(vllm_prompts, sampling, use_tqdm=False)
    wall = time.perf_counter() - t0
    predicted = [1 if int(o.outputs[0].token_ids[0]) in yes_ids else 0
                 for o in outs]
    wrong = sum(a != b for a, b in zip(predicted, expected))
    row = dict(method=cell, rep=rep, wall=round(wall, 4),
               tokens_per_second=round(total_tokens / wall, 1), wrong=wrong)
    runs.append(row)
    print(f"[fix:{cell}] {json.dumps(row)}", flush=True)

profile = {}
try:
    llm.start_profile()
    llm.generate(vllm_prompts[:profile_docs], sampling, use_tqdm=False)
    llm.stop_profile()
    path, size = None, -1
    for _ in range(90):
        time.sleep(2)
        traces = (glob.glob(outdir + "/*rank0*")
                  or glob.glob(outdir + "/*"))
        if not traces:
            continue
        cand = max(traces, key=os.path.getmtime)
        now = os.path.getsize(cand)
        if cand == path and now == size and now > 0:
            break
        path, size = cand, now
    if path is None:
        profile = {"error": f"no trace appeared in {outdir}"}
    else:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            events = json.load(f)["traceEvents"]
        profile = summarize(events)
        profile["profiled_docs"] = profile_docs
        profile["profiled_tokens"] = profile_tokens
        profile["kernel_us_per_token"] = round(
            profile["total_kernel_us"] / profile_tokens, 3)
except Exception as e:
    profile = {"error": f"{type(e).__name__}: {e}"}
keep = {k: v for k, v in profile.items() if k != "top_kernels"}
print(f"[fix:{cell}] profile {json.dumps(keep)}", flush=True)

with open(outpath, "w") as f:
    json.dump(dict(cell=cell, resolved=resolved, runs=runs,
                   profile=profile, predictions=predicted), f)
'''


PATCH_CALL = "vllm.ir.ops.fused_add_rms_norm("
PATCH_HELPER = '''import vllm.ir.ops


def _quail_fused_add_rms_norm():
    # The model emits the maybe_inplace overload of this IR op; the
    # stock patterns trace the functional overload and never match.
    # QUAIL_PATTERN_MAYBE_INPLACE=1 makes the patterns trace the same
    # overload the model emits.
    import os
    if os.getenv("QUAIL_PATTERN_MAYBE_INPLACE") == "1":
        return vllm.ir.ops.fused_add_rms_norm.maybe_inplace
    return vllm.ir.ops.fused_add_rms_norm
'''


def _apply_pattern_patch():
    """Patch the installed rms_quant_fusion.py so its fused-add
    patterns can trace the overload the model emits, off by default."""
    import pathlib
    import site

    matches = []
    for root in site.getsitepackages():
        candidate = pathlib.Path(root) / (
            "vllm/compilation/passes/fusion/rms_quant_fusion.py"
        )
        if candidate.exists():
            matches.append(candidate)
    if len(matches) != 1:
        raise RuntimeError(f"expected one rms_quant_fusion.py, got {matches}")
    path = matches[0]
    source = path.read_text()
    if "_quail_fused_add_rms_norm" in source:
        return "already patched"
    count = source.count(PATCH_CALL)
    if count != 3:
        raise RuntimeError(
            f"expected 3 fused_add_rms_norm pattern call sites, found {count}"
        )
    source = source.replace(PATCH_CALL, "_quail_fused_add_rms_norm()(")
    if source.count("import vllm.ir.ops\n") != 1:
        raise RuntimeError("could not anchor the patch helper")
    source = source.replace("import vllm.ir.ops\n", PATCH_HELPER, 1)
    path.write_text(source)
    return f"patched {path}"


@app.function(image=fix_image, gpu="H100!", timeout=10800, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def fix_ab(n_docs: int = 10_000, reps: int = 3,
           batch_tokens: int = BEST_BATCH_TOKENS,
           profile_docs: int = 1024) -> str:
    import json
    import subprocess
    import time

    patch_note = _apply_pattern_patch()
    print(f"[fix] {patch_note}", flush=True)

    with open("/tmp/cell_runner.py", "w") as f:
        f.write(CELL_RUNNER)

    cells = (
        ("control", {}),
        ("act_fused", {"QUAIL_CUSTOM_SILU": "1"}),
        ("act_norm_fused", {"QUAIL_CUSTOM_SILU": "1",
                            "QUAIL_PATTERN_MAYBE_INPLACE": "1"}),
    )
    report = {
        "model": MODEL,
        "n_docs": n_docs,
        "batch_tokens": batch_tokens,
        "patch": patch_note,
        "cells": [name for name, _ in cells],
        "prediction": (
            "act_fused +3.5 to 6.5 percent if silu+quant fires (else "
            "within 1 percent); act_norm_fused +6 to 10 percent total "
            "with rms_norm_per_block_quant kernels present; wrong "
            "within a few tens of 2,990"
        ),
        "runs": [],
        "profiles": {},
        "resolved": {},
    }

    predictions = {}
    for name, env_extra in cells:
        outpath = f"/tmp/cell_{name}.json"
        env = dict(os.environ)
        env.update(env_extra)
        print(f"\n[fix] === {name}: {env_extra or 'stock env'} ===",
              flush=True)
        r = subprocess.run(
            ["python", "/tmp/cell_runner.py", name, str(n_docs), str(reps),
             str(batch_tokens), str(profile_docs), outpath],
            env=env,
        )
        if r.returncode != 0:
            raise RuntimeError(f"cell {name} exited {r.returncode}")
        with open(outpath) as f:
            cell = json.load(f)
        report["runs"].extend(cell["runs"])
        report["profiles"][name] = cell["profile"]
        report["resolved"][name] = cell["resolved"]
        predictions[name] = cell["predictions"]
        time.sleep(5)

    control_predictions = predictions["control"]
    for row in report["runs"]:
        if row["method"] != "control":
            row["disagrees_with_control"] = sum(
                a != b for a, b in
                zip(predictions[row["method"]], control_predictions)
            )

    rates = {}
    for name, _ in cells:
        values = [row["tokens_per_second"] for row in report["runs"]
                  if row["method"] == name]
        rates[name] = round(sum(values) / len(values), 1)
    report["mean_tokens_per_second"] = rates
    report["relative_to_control"] = {
        name: round(rates[name] / rates["control"], 4) for name, _ in cells
    }

    outpath = "/results/fusion_fix.json"
    with open(outpath, "w") as f:
        json.dump(report, f, indent=2)
    results_vol.commit()
    report_json = json.dumps(report, indent=2)
    print(report_json, flush=True)
    return report_json


GRAPHDUMP_RUNNER = r'''
import sys

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, "/root")
from workload import MODEL, build_corpus, yes_no_ids

tok = AutoTokenizer.from_pretrained(MODEL)
body_ids, q_ids, _flags = build_corpus(tok, 64)
yes, no = yes_no_ids(tok)
sp = SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1,
                    allowed_token_ids=sorted(yes | no))
prompts = [{"prompt_token_ids": body_ids[i] + q_ids[0]}
           for i in range(64)]
llm = LLM(
    model=MODEL,
    kv_cache_dtype="fp8",
    max_model_len=4608,
    max_num_seqs=4096,
    max_num_batched_tokens=25305,
    gpu_memory_utilization=0.88,
    enable_prefix_caching=False,
    disable_log_stats=True,
    # capture off: the dump needs the compile passes, not the graphs,
    # and capture plus DEBUG logging is what timed out the first try
    compilation_config={"cudagraph_mode": 0,
                        "custom_ops": ["+silu_and_mul"]},
)
llm.generate(prompts, sp, use_tqdm=False)
print("[graphdump] boot and generate done", flush=True)
'''


@app.function(image=fix_image, gpu="H100!", timeout=3600, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def graphdump() -> str:
    """One boot at the full-fix config with vLLM's compile debug dump
    on, so the post-grad graphs say which nodes are really there.

    Prediction: the dumped graph's norm sites show the auto-
    functionalized maybe_inplace overload (which the patched patterns
    now trace), and the surviving mismatch is on the quant side - the
    quant appears either as a wrapper op the patterns do not trace or
    as _C.per_token_group_fp8_quant with argument constants that
    differ from the pattern's. Falsifier: if both nodes look exactly
    as the patterns trace them, the miss is in pattern normalization
    itself, and the dumped pattern files against the dumped graph
    lines show the literal difference either way.

    Result: the quant side was the miss, in a stronger form than
    either named option - the quant is not a mismatched node, it is
    not a node at all. The pre-grad graph shows every decoder layer
    as fused_add_rms_norm.maybe_inplace feeding one opaque
    dynamic_flashinfer_deepgemm_blockscale_gemm op, with the SiLU
    consuming that op's output directly. That op is the registered
    custom linear for block-FP8 on this build (defined in vllm
    model_executor/kernels/linear/scaled_mm/flashinfer.py, which
    calls per_token_group_quant_fp8 inside its implementation), so
    the quant kernel the profiles show runs from inside the linear
    op at runtime and never exists in the compiled graph. The
    RMSNorm+quant and SiLU+quant patterns therefore have nothing to
    match against this backend: no flag or pattern patch can make
    them fire, and the fusion passes and the FlashInfer-DeepGEMM
    linear path are mutually exclusive for this checkpoint in vllm
    0.26.0. Reaching the shipped fused kernels now means either
    swapping the linear backend (giving up the 91-93 percent
    efficient DeepGEMM kernels that hold 51 percent of the runtime -
    a measurable but probably losing trade) or calling the fused
    kernels explicitly outside the pattern machinery. The runner hit
    its own 2,400 s timeout (dump writing is slow even without DEBUG
    logging), and the 193 partial dump files were sufficient."""
    import glob
    import json
    import re
    import subprocess

    patch_note = _apply_pattern_patch()
    print(f"[graphdump] {patch_note}", flush=True)
    with open("/tmp/graphdump_runner.py", "w") as f:
        f.write(GRAPHDUMP_RUNNER)

    # The dump is driven by VLLM_DEBUG_DUMP_PATH alone. DEBUG logging
    # is not needed for it and made the first try time out.
    env = dict(os.environ)
    env.update({
        "VLLM_DEBUG_DUMP_PATH": "/tmp/gdump",
        "QUAIL_CUSTOM_SILU": "1",
        "QUAIL_PATTERN_MAYBE_INPLACE": "1",
    })
    runner_exit = None
    with open("/tmp/graphdump.log", "w") as log:
        try:
            r = subprocess.run(["python", "/tmp/graphdump_runner.py"],
                               env=env, stdout=log,
                               stderr=subprocess.STDOUT, timeout=2400)
            runner_exit = r.returncode
        except subprocess.TimeoutExpired:
            runner_exit = "timeout, analyzing partial dumps"

    match_lines = []
    with open("/tmp/graphdump.log") as log:
        for line in log:
            low = line.lower()
            if ("replaced" in low or "match" in low) and "fusion" in low:
                match_lines.append(line.strip()[:240])
            elif "quail" in low or "graphdump" in low:
                match_lines.append(line.strip()[:240])
    match_lines = match_lines[:80]

    dump_files = sorted(
        p for p in glob.glob("/tmp/gdump/**", recursive=True)
        if os.path.isfile(p)
    )
    keys = ("fused_add_rms_norm", "silu", "per_token_group", "quant_fp8")
    samples = {k: [] for k in keys}
    op_tokens = set()
    op_regexes = (
        re.compile(r"vllm_ir\.[A-Za-z_0-9]+\.[A-Za-z_0-9]+"),
        re.compile(r"_C\.[A-Za-z_0-9]+"),
        re.compile(r"torch\.ops\.vllm\.[A-Za-z_0-9]+"),
    )
    for path in dump_files:
        base = os.path.basename(path)
        try:
            with open(path, errors="replace") as f:
                for line in f:
                    hit = [k for k in keys if k in line]
                    if not hit:
                        continue
                    for rx in op_regexes:
                        op_tokens.update(rx.findall(line))
                    for k in hit:
                        if len(samples[k]) < 14:
                            samples[k].append(f"{base}: "
                                              + line.strip()[:240])
        except OSError:
            continue

    result = {
        "patch": patch_note,
        "runner_exit": runner_exit,
        "log_match_lines": match_lines,
        "n_dump_files": len(dump_files),
        "dump_files": [os.path.basename(p) for p in dump_files][:60],
        "op_tokens": sorted(op_tokens)[:60],
        "samples": samples,
    }
    outpath = "/results/fusion_graphdump.json"
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    result_json = json.dumps(result, indent=2)
    print(result_json, flush=True)
    return result_json


@app.local_entrypoint()
def main(n_docs: int = 10_000, reps: int = 3,
         batch_tokens: int = BEST_BATCH_TOKENS,
         profile_docs: int = 1024,
         out: str = "results/engine/fusion_fix.json"):
    import json

    report = json.loads(fix_ab.remote(n_docs, reps, batch_tokens,
                                      profile_docs))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as output:
        json.dump(report, output, indent=2)
    print(f"saved {out}")


@app.local_entrypoint()
def graphdump_main(out: str = "results/engine/fusion_graphdump.json"):
    import json

    result = json.loads(graphdump.remote())
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as output:
        json.dump(result, output, indent=2)
    print(f"saved {out}")

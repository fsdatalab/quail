"""Measure what vLLM's fusion passes are worth at the best single-filter config.

Three boots of the same engine in one container on one H100, all at the
committed single-filter control setting (25,305 batched tokens, CUDA
graph at [8192], prefix caching off): the control with default pass
flags, the norm+quant and act+quant fusion passes forced off, and the
same passes forced on. Each boot runs three timed repetitions over the
full corpus plus one profiled repetition over the first 1,024 documents.
The report carries the resolved pass flags, the kernel-class shares, and
the top kernel names, so the run itself says which kernels executed.

Premise, from reading the pinned vllm 0.26.0 source: the control already
has these fusion passes ON. The default optimization level is O2; at O2
the fuse_norm_quant and fuse_act_quant defaults resolve true when the
quant_fp8 custom op is active; and this checkpoint's block-quantized FP8
weights ([128, 128]) make vLLM enable quant_fp8 on its own. Both passes
carry patterns for the checkpoint's dynamic 128-group scheme and CUDA
fused kernels (rms_norm_per_block_quant, silu_and_mul_per_block_quant).
If that is right, the committed kernel mix (quantize 18.3, norm 12.2
percent of kernel time at B=25,305) already counts the fused kernels
under "quantize", because the classifier files any name containing
"quant" there.

Prediction, stated before the run:
  - control repeats the committed 97,637 input tokens per second within
    1 percent, and its profile shows fused kernel names pairing
    norm/silu with quant.
  - fusion_on equals control within repetition noise, under 1 percent.
  - fusion_off runs 6 to 10 percent slower, about 89,000 to 92,000
    tokens per second. Computed: unfusing adds one bf16 write plus one
    read of the normed tensor at each of 2 norm sites per layer (260 MB
    per site per step) and of the SiLU output (984 MB per layer per
    step); about 54 GB per 25,305-token step over 36 layers, at least
    16 ms against the 263 ms step at the full 3.35 TB/s, more at the
    bandwidth these small kernels actually reach.
  - wrong answers stay near the control's 2,990 of 10,000. Fused and
    unfused group quantization round differently, so a small
    disagreement count against control is expected, not a failure.
  Falsifier: if the control profile shows no fused kernel names, the
  premise is wrong; then fusion_on should beat control by the 10 to 20
  percent the original proposal expected, and fusion_off should equal
  control.

Run:
    modal run experiments/modal_fusion.py::bankedtrace_main
    modal run experiments/modal_fusion.py::main
"""

import os

import modal

from workload import IMAGE_BASE, MODEL, hf_cache, results_vol

# Mirrors the packed-forward experiment's image, so the control cell
# reruns the committed 97,637 tok/s measurement on identical software.
FLASH_ATTN_4_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "fa4-v4.0.0.beta26/flash_attn_4-4.0.0b26-py3-none-any.whl"
)
BEST_BATCH_TOKENS = 25_305
VLLM_GRAPH_TOKENS = 8_192

fusion_image = (
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

banked_image = (modal.Image.debian_slim(python_version="3.12")
                .add_local_python_source("workload"))

app = modal.App("quail-fusion")

# Kernel classes, first match wins. Attention must match before the
# gemm patterns: the sm90 FlashAttention mainloop is a
# cutlass::device_kernel, and the gemm patterns would claim it.
KERNEL_CLASS_RULES = (
    ("attention", ("attn", "attention", "flash", "fmha")),
    ("gemm", ("gemm", "cutlass", "nvjet")),
    ("quantize", ("quant", "scale", "cast")),
    ("norm", ("norm", "rms")),
    ("elementwise", ("silu", "gelu", "add", "mul", "residual")),
)


def classify_kernel(name):
    k = name.lower()
    for cls, keys in KERNEL_CLASS_RULES:
        if any(s in k for s in keys):
            return cls
    return "other"


def is_fused_norm_quant(name):
    k = name.lower()
    return "quant" in k and ("norm" in k or "rms" in k)


def is_fused_act_quant(name):
    k = name.lower()
    return "quant" in k and "silu" in k


def summarize_kernel_events(events, top_n=20):
    """Class shares, fused-kernel shares, and top kernel names from a
    chrome trace's event list."""
    gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
    classes = dict(gemm=0, quantize=0, norm=0, elementwise=0,
                   attention=0, other=0)
    fused_norm_us = 0
    fused_act_us = 0
    by_name = {}
    total = 0
    n_kernels = 0
    for e in events:
        if e.get("cat") not in gpu_cats or e.get("dur", 0) <= 0:
            continue
        name = e.get("name", "")
        dur = e["dur"]
        classes[classify_kernel(name)] += dur
        if is_fused_norm_quant(name):
            fused_norm_us += dur
        if is_fused_act_quant(name):
            fused_act_us += dur
        by_name[name] = by_name.get(name, 0) + dur
        total += dur
        n_kernels += 1
    out = dict(n_kernels=n_kernels, total_kernel_us=total)
    for cls, us in classes.items():
        out[f"{cls}_us"] = us
        out[f"{cls}_frac"] = round(us / total, 4) if total else 0
    out["fused_norm_quant_us"] = fused_norm_us
    out["fused_norm_quant_frac"] = (
        round(fused_norm_us / total, 4) if total else 0
    )
    out["fused_act_quant_us"] = fused_act_us
    out["fused_act_quant_frac"] = (
        round(fused_act_us / total, 4) if total else 0
    )
    out["top_kernels"] = [
        [classify_kernel(n), us, n[:140]]
        for n, us in sorted(by_name.items(), key=lambda kv: -kv[1])[:top_n]
    ]
    return out


@app.function(image=banked_image, timeout=900, memory=32768,
              volumes={"/results": results_vol})
def bankedtrace(pattern: str = "torchprof_*stock*/*rank0*") -> str:
    """Read the banked production-boot trace and list its top kernels.

    No GPU: this parses the trace the five-filter torchprof run already
    saved to the results volume. The boot that produced it used the same
    default pass flags as the single-filter control, so fused kernel
    names here mean the control is fused too. Prediction: they are
    present."""
    import glob
    import gzip
    import json

    candidates = (pattern, "torchprof_*stock*/*rank0*",
                  "torchprof_*rank0*", "torchprof_*/*rank0*")
    matches = []
    for candidate in candidates:
        matches = sorted(glob.glob(f"/results/{candidate}"))
        if matches:
            break
    if not matches:
        found = sorted(glob.glob("/results/torchprof*")
                       + glob.glob("/results/torchprof*/*"))[:30]
        report = {"pattern": pattern, "matches": [],
                  "note": "no trace matched; volume has", "found": found}
        print(json.dumps(report, indent=2), flush=True)
        return json.dumps(report)

    path = matches[-1]
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        events = json.load(f)["traceEvents"]
    out = summarize_kernel_events(events, top_n=30)
    out["trace"] = path
    out["all_matches"] = matches
    print(json.dumps(out, indent=2), flush=True)
    return json.dumps(out)


@app.function(image=fusion_image, gpu="H100!", timeout=7200, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def compare(n_docs: int = 10_000, reps: int = 3,
            batch_tokens: int = BEST_BATCH_TOKENS,
            profile_docs: int = 1024) -> str:
    import gc
    import glob
    import gzip
    import json
    import time

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from workload import build_corpus, yes_no_ids

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    body_ids, question_ids, flags = build_corpus(tokenizer, n_docs)
    prompts = [body_ids[i] + question_ids[0] for i in range(n_docs)]
    expected = [int(flags[i][0]) for i in range(n_docs)]
    total_prompt_tokens = sum(map(len, prompts))
    profile_tokens = sum(map(len, prompts[:profile_docs]))
    yes_ids, no_ids = yes_no_ids(tokenizer)
    allowed_ids = sorted(yes_ids | no_ids)
    vllm_prompts = [{"prompt_token_ids": prompt} for prompt in prompts]
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        min_tokens=1,
        allowed_token_ids=allowed_ids,
    )
    graph_tokens = min(batch_tokens, VLLM_GRAPH_TOKENS)

    # this vLLM takes profiling as an engine argument, not the old
    # VLLM_TORCH_PROFILER_DIR env var
    def profiler_config(outdir):
        try:
            from vllm.config import ProfilerConfig
            return ProfilerConfig(profiler="torch",
                                  torch_profiler_dir=outdir)
        except Exception:
            return dict(profiler="torch", torch_profiler_dir=outdir)

    def resolved_compilation(llm):
        """The pass flags and custom ops the boot actually resolved to.
        The report must carry these: the control's flags are defaults,
        and the premise is about what the defaults resolve to."""
        try:
            cfg = llm.llm_engine.vllm_config
            cc = cfg.compilation_config
            pc = cc.pass_config
            return {
                "optimization_level": int(cfg.optimization_level),
                "compilation_mode": int(cc.mode),
                "custom_ops": list(cc.custom_ops),
                "cudagraph_capture_sizes": sorted(
                    cc.cudagraph_capture_sizes or []),
                "fuse_norm_quant": bool(pc.fuse_norm_quant),
                "fuse_act_quant": bool(pc.fuse_act_quant),
                "fuse_attn_quant": bool(pc.fuse_attn_quant),
                "eliminate_noops": bool(pc.eliminate_noops),
            }
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def profiled_rep(llm, outdir, cell):
        """One engine-side profiled pass over the first profile_docs
        documents, parsed to kernel classes and names, trace deleted."""
        try:
            llm.start_profile()
            llm.generate(vllm_prompts[:profile_docs], sampling,
                         use_tqdm=False)
            llm.stop_profile()
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        # stop_profile returns before the engine core finishes writing
        # the trace; wait until a file appears and stops growing
        path, size = None, -1
        for _ in range(90):
            time.sleep(2)
            traces = (glob.glob(outdir + "/*rank0*")
                      or glob.glob(outdir + "/*"))
            if not traces:
                continue
            candidate = max(traces, key=os.path.getmtime)
            now = os.path.getsize(candidate)
            if candidate == path and now == size and now > 0:
                break
            path, size = candidate, now
        if path is None:
            return {"error": f"no trace appeared in {outdir}"}
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            events = json.load(f)["traceEvents"]
        out = summarize_kernel_events(events)
        out["profiled_docs"] = profile_docs
        out["profiled_tokens"] = profile_tokens
        out["kernel_us_per_token"] = round(
            out["total_kernel_us"] / profile_tokens, 3)
        for stale in glob.glob(outdir + "/*"):
            os.unlink(stale)
        print(f"[fusion] {cell} profile: " + json.dumps(
            {k: out[k] for k in out if k != "top_kernels"}), flush=True)
        for row in out["top_kernels"][:10]:
            print(f"[fusion] {cell} top kernel: {row}", flush=True)
        return out

    cells = (
        ("control", None),
        ("fusion_off", {"fuse_norm_quant": False, "fuse_act_quant": False}),
        ("fusion_on", {"fuse_norm_quant": True, "fuse_act_quant": True}),
    )
    report = {
        "model": MODEL,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "n_docs": n_docs,
        "prompt_tokens": total_prompt_tokens,
        "batch_tokens": batch_tokens,
        "cells": [name for name, _ in cells],
        "prediction": (
            "control is already fused (vllm 0.26.0 resolves "
            "fuse_norm_quant and fuse_act_quant true at the default O2 "
            "for this block-FP8 checkpoint); fusion_on == control "
            "within 1 percent; fusion_off 6 to 10 percent slower; "
            "wrong stays near 2,990 of 10,000"
        ),
        "runs": [],
        "profiles": {},
        "resolved": {},
    }
    print(
        f"[fusion] {n_docs:,} prompts, {total_prompt_tokens:,} tokens, "
        f"cells {[name for name, _ in cells]}",
        flush=True,
    )

    control_predictions = None
    for name, pass_overrides in cells:
        compilation = {
            "max_cudagraph_capture_size": graph_tokens,
            "cudagraph_capture_sizes": [graph_tokens],
        }
        if pass_overrides is not None:
            compilation["pass_config"] = dict(pass_overrides)
        outdir = f"/tmp/prof_{name}"
        os.makedirs(outdir, exist_ok=True)
        print(f"\n[fusion] === {name}: pass_config="
              f"{pass_overrides or 'defaults'} ===", flush=True)
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
            profiler_config=profiler_config(outdir),
        )
        report["resolved"][name] = resolved_compilation(llm)
        print(f"[fusion] {name} resolved: "
              f"{json.dumps(report['resolved'][name])}", flush=True)

        llm.generate(vllm_prompts[:64], sampling, use_tqdm=False)
        for rep in range(reps):
            started = time.perf_counter()
            outputs = llm.generate(vllm_prompts, sampling, use_tqdm=False)
            wall = time.perf_counter() - started
            predicted = []
            for output in outputs:
                token_id = int(output.outputs[0].token_ids[0])
                predicted.append(1 if token_id in yes_ids else 0)
            wrong = sum(a != b for a, b in zip(predicted, expected))
            row = {
                "method": name,
                "rep": rep,
                "wall": round(wall, 4),
                "tokens_per_second": round(total_prompt_tokens / wall, 1),
                "wrong": wrong,
            }
            if control_predictions is not None:
                row["disagrees_with_control"] = sum(
                    a != b for a, b in zip(predicted, control_predictions)
                )
            report["runs"].append(row)
            print(f"[fusion] {row}", flush=True)
        if name == "control":
            control_predictions = list(predicted)

        report["profiles"][name] = profiled_rep(llm, outdir, name)

        del llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)

    rates = {}
    for name, _ in cells:
        values = [row["tokens_per_second"] for row in report["runs"]
                  if row["method"] == name]
        rates[name] = round(sum(values) / len(values), 1)
    report["mean_tokens_per_second"] = rates
    report["relative_to_control"] = {
        name: round(rates[name] / rates["control"], 4)
        for name, _ in cells
    }

    outpath = "/results/fusion_ab.json"
    with open(outpath, "w") as output:
        json.dump(report, output, indent=2)
    results_vol.commit()
    report_json = json.dumps(report, indent=2)
    print(report_json, flush=True)
    return report_json


@app.local_entrypoint()
def bankedtrace_main(pattern: str = "torchprof_*stock*/*rank0*",
                     out: str = "results/engine/fusion_bankedtrace.json"):
    import json

    result = json.loads(bankedtrace.remote(pattern))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as output:
        json.dump(result, output, indent=2)
    print(f"saved {out}")


@app.local_entrypoint()
def main(n_docs: int = 10_000, reps: int = 3,
         batch_tokens: int = BEST_BATCH_TOKENS,
         profile_docs: int = 1024,
         out: str = "results/engine/fusion_ab.json"):
    import json

    report = json.loads(compare.remote(n_docs, reps, batch_tokens,
                                       profile_docs))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as output:
        json.dump(report, output, indent=2)
    print(f"saved {out}")

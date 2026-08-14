"""A/B: vLLM 0.26's shipped fusion passes, on against off, one filter.

Both runs execute the first planted filter over the full 10,000-document
corpus in one Modal container on one H100, using the committed best
single-filter configuration (25,305 batch tokens, CUDA graph sizes
[8192], prefix caching off). The control passes exactly the committed
compilation config. The fusion run adds only two things: the pass flags
that turn on vLLM's RMSNorm+quant, SiluMul+quant, and QK-norm+RoPE
fusions, and the custom ops those patterns match on. vLLM 0.26 ships
fused kernels for this checkpoint's quantization scheme (fp8 with
128-wide activation groups): rms_norm_per_block_quant and
silu_and_mul_per_block_quant.

Why this experiment: the corrected production kernel mix at B=25,305
spends 18.3 percent of GPU time in separate quantize kernels, 12.2 in
normalization, and 6.7 in elementwise — 37 percent between the matrix
multiplies. The fusion passes are the cheapest possible attack on that
time: configuration only, no new kernels.

Prediction, stated before the run:
- Control reproduces the committed anchor, 97,637 input tokens per
  second, within the ±2 percent container spread.
- If the fusion patterns match the production graph, the fused kernel
  names appear in the profile and the rate rises 10 to 20 percent
  (107k–117k tokens per second): the norm and part of the quantize and
  silu time fuse away.
- If the patterns do not match (the block-quant path can dispatch
  through Triton kernels the matcher does not see), the rate moves
  less than 2 percent and the profile still shows the separate
  per_token_group_quant_8bit_kernel and vllm::rms_norm_kernel.
- Gate: a rate change is believed only with kernel-name evidence, and
  the wrong-answer count must stay within 0.5 points of the control's
  2,990 of 10,000 (fused kernels may shift borderline answers; the
  error rate must not).

Reading the profile: the fused kernels contain "quant" in their names,
so the quantize class absorbs the fused norm and silu time and its
fraction can RISE under fusion. Evidence of firing is the fused names
in top_kernels and a fall in norm_frac plus elementwise_frac, not the
quantize fraction.

Run:
    modal run experiments/modal_fusion_filter.py 2>&1 | tee /tmp/fusion_filter.log
"""

import modal

from workload import MODEL, hf_cache, image, results_vol

BEST_BATCH_TOKENS = 25_305
VLLM_GRAPH_TOKENS = 8_192

CONTROL_COMPILATION = {
    "max_cudagraph_capture_size": VLLM_GRAPH_TOKENS,
    "cudagraph_capture_sizes": [VLLM_GRAPH_TOKENS],
}

# The pattern matchers only see ops that run as vLLM custom ops, so the
# fusion arm must also enable them; on its own that swap is a second
# treatment, which is why the profile, not the rate alone, attributes
# any change.
FUSION_COMPILATION = {
    **CONTROL_COMPILATION,
    "custom_ops": ["+rms_norm", "+silu_and_mul", "+quant_fp8"],
    "pass_config": {
        "fuse_norm_quant": True,
        "fuse_act_quant": True,
        "enable_qk_norm_rope_fusion": True,
    },
}

# vllm 0.26 pass-flag names, checked against its source: fuse_norm_quant,
# fuse_act_quant, enable_qk_norm_rope_fusion; eliminate_noops defaults
# True and fusion requires it. fuse_attn_quant is left off: it demands
# use_inductor_graph_partition and attention outside splitting_ops,
# which would change the CUDA-graph configuration under test.

FUSED_KERNEL_MARKS = ("per_block_quant", "qk_norm_rope")

app = modal.App("quail-fusion-filter")


def _classify(name):
    """The torchprof kernel classes. Attention before gemm: the sm90
    FlashAttention mainloop is a cutlass::device_kernel, and the gemm
    patterns would claim it."""
    k = name.lower()
    if "attn" in k or "attention" in k or "flash" in k or "fmha" in k:
        return "attention"
    if "gemm" in k or "cutlass" in k or "nvjet" in k:
        return "gemm"
    if "quant" in k or "scale" in k or "cast" in k:
        return "quantize"
    if "norm" in k or "rms" in k:
        return "norm"
    if ("silu" in k or "gelu" in k or "add" in k or "mul" in k
            or "residual" in k):
        return "elementwise"
    return "other"


def _summarize_trace(outdir, top_n=10):
    """Kernel-class shares and top kernels from the newest trace in
    outdir, then delete the traces (they run to ~100 MB)."""
    import glob
    import gzip
    import json
    import os
    import shutil
    import time

    # stop_profile can return before the worker finishes exporting the
    # trace, so wait for a file and retry one mid-write parse failure.
    traces = []
    for _ in range(60):
        traces = sorted(
            (p for p in glob.glob(outdir + "/**/*", recursive=True)
             if os.path.isfile(p)),
            key=os.path.getmtime)
        if traces:
            break
        time.sleep(1)
    if not traces:
        return {"error": f"no trace written in {outdir}"}
    path = traces[-1]
    opener = gzip.open if path.endswith(".gz") else open
    events = None
    for attempt in range(2):
        time.sleep(3)
        try:
            with opener(path, "rt") as f:
                events = json.load(f)["traceEvents"]
            break
        except Exception as e:
            if attempt:
                shutil.rmtree(outdir, ignore_errors=True)
                return {"error": f"trace parse failed: "
                                 f"{type(e).__name__}: {e}"}
            time.sleep(15)
    gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
    kernels = [e for e in events
               if e.get("cat") in gpu_cats and e.get("dur", 0) > 0]
    total = sum(e["dur"] for e in kernels)
    classes = {}
    by_name = {}
    for e in kernels:
        cls = _classify(e.get("name", ""))
        classes[cls] = classes.get(cls, 0) + e["dur"]
        key = e.get("name", "")[:140]
        by_name[key] = by_name.get(key, 0) + e["dur"]
    top = sorted(by_name.items(), key=lambda kv: -kv[1])[:top_n]
    out = {
        "trace": os.path.basename(path),
        "n_kernels": len(kernels),
        "total_kernel_us": round(total, 1),
        "fused_kernels_present": sorted(
            {mark for mark in FUSED_KERNEL_MARKS
             for name in by_name if mark in name.lower()}),
    }
    for cls, us in sorted(classes.items()):
        out[f"{cls}_frac"] = round(us / total, 4) if total else 0
    out["top_kernels"] = [
        [_classify(name), round(us, 1), name] for name, us in top]
    shutil.rmtree(outdir, ignore_errors=True)
    return out


def _boot_snapshot(llm):
    """What the engine actually resolved the config to, best effort:
    with engine multiprocessing the client-side copy may not carry the
    worker's custom-op counters."""
    try:
        cc = llm.llm_engine.vllm_config.compilation_config
        pc = cc.pass_config
        return {
            "custom_ops": list(cc.custom_ops),
            "pass_config": {
                k: getattr(pc, k, None)
                for k in ("fuse_norm_quant", "fuse_act_quant",
                          "enable_qk_norm_rope_fusion", "eliminate_noops")},
        }
    except Exception as e:
        return {"unavailable": f"{type(e).__name__}: {e}"}


@app.function(image=image, gpu="H100!", timeout=7200, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def compare(n_docs: int = 10_000, reps: int = 3,
            batch_tokens: int = BEST_BATCH_TOKENS,
            prof_docs: int = 2_000) -> str:
    import gc
    import json
    import time

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    from workload import build_corpus, yes_no_ids

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    body_ids, question_ids, flags = build_corpus(tokenizer, n_docs)
    prompts = [body_ids[i] + question_ids[0] for i in range(n_docs)]
    expected = [int(flags[i][0]) for i in range(n_docs)]
    total_prompt_tokens = sum(map(len, prompts))
    prof_tokens = sum(map(len, prompts[:prof_docs]))
    yes_ids, no_ids = yes_no_ids(tokenizer)
    allowed_ids = sorted(yes_ids | no_ids)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        min_tokens=1,
        allowed_token_ids=allowed_ids,
    )
    vllm_prompts = [{"prompt_token_ids": prompt} for prompt in prompts]

    try:
        from vllm.config import ProfilerConfig
        def profiler_cfg(outdir):
            return ProfilerConfig(profiler="torch",
                                  torch_profiler_dir=outdir)
    except Exception:
        def profiler_cfg(outdir):
            return dict(profiler="torch", torch_profiler_dir=outdir)

    report = {
        "model": MODEL,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "n_docs": n_docs,
        "prompt_tokens": total_prompt_tokens,
        "batch_tokens": batch_tokens,
        "prof_docs": prof_docs,
        "prof_tokens": prof_tokens,
        "prediction": ("control reproduces 97,637 tok/s +-2 percent; "
                       "fusion is +10 to +20 percent if the fused "
                       "kernel names appear, unchanged if they do not"),
        "arms": {},
        "runs": [],
    }
    print(f"[fusion filter] {n_docs:,} prompts, "
          f"{total_prompt_tokens:,} tokens", flush=True)

    control_predictions = None
    for arm_name, compilation in (("control", CONTROL_COMPILATION),
                                  ("fusion", FUSION_COMPILATION)):
        prof_dir = f"/tmp/prof_{arm_name}"
        boot_started = time.perf_counter()
        llm = LLM(
            model=MODEL,
            kv_cache_dtype="fp8",
            max_model_len=4608,
            max_num_seqs=4096,
            max_num_batched_tokens=batch_tokens,
            gpu_memory_utilization=0.88,
            enable_prefix_caching=False,
            disable_log_stats=True,
            compilation_config=dict(compilation),
            profiler_config=profiler_cfg(prof_dir),
        )
        boot_s = time.perf_counter() - boot_started
        arm = {"compilation_config": compilation,
               "boot_s": round(boot_s, 1),
               "resolved": _boot_snapshot(llm)}
        print(f"[fusion filter] {arm_name} booted in {boot_s:.0f}s: "
              f"{arm['resolved']}", flush=True)

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
                "method": arm_name,
                "rep": rep,
                "wall": round(wall, 4),
                "tokens_per_second": round(total_prompt_tokens / wall, 1),
                "wrong": wrong,
            }
            if arm_name == "control":
                control_predictions = predicted
            else:
                row["disagrees_with_control"] = sum(
                    a != b for a, b in zip(predicted, control_predictions))
            report["runs"].append(row)
            print(f"[fusion filter] {row}", flush=True)

        # Profiled slice, outside the timed reps: the profiler adds
        # overhead, so its wall is reported separately and never enters
        # the rate comparison.
        llm.start_profile()
        started = time.perf_counter()
        llm.generate(vllm_prompts[:prof_docs], sampling, use_tqdm=False)
        prof_wall = time.perf_counter() - started
        llm.stop_profile()
        arm["profiled_wall_s"] = round(prof_wall, 2)
        arm["profile"] = _summarize_trace(prof_dir)
        print(f"[fusion filter] {arm_name} profile: "
              f"{json.dumps(arm['profile'], indent=1)}", flush=True)
        report["arms"][arm_name] = arm

        del llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)

    rates = {}
    for method in ("control", "fusion"):
        values = [row["tokens_per_second"] for row in report["runs"]
                  if row["method"] == method]
        rates[method] = round(sum(values) / len(values), 1)
    report["mean_tokens_per_second"] = rates
    report["fusion_speedup"] = round(rates["fusion"] / rates["control"], 4)

    outpath = "/results/fusion_filter.json"
    with open(outpath, "w") as output:
        json.dump(report, output, indent=2)
    results_vol.commit()
    report_json = json.dumps(report, indent=2)
    print(report_json, flush=True)
    return report_json


@app.local_entrypoint()
def main(n_docs: int = 10_000, reps: int = 3,
         batch_tokens: int = BEST_BATCH_TOKENS, prof_docs: int = 2_000,
         out: str = "results/engine/fusion_filter.json"):
    import json
    import os

    report = json.loads(compare.remote(n_docs, reps, batch_tokens,
                                       prof_docs))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as output:
        json.dump(report, output, indent=2)
    print(f"saved {out}")

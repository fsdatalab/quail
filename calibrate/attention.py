r"""Measure a GPU's attention and memory-bandwidth ceilings at a head layout.

Times upstream FlashAttention 3 and FlashInfer, bf16, one layer per
call, on one GPU, with calibrate.timing's sustained window. Three kinds
of point:

  packed  causal attention over 65,536 tokens of equal-length documents,
          sweeping the document length
  cached  query rows reading cached KV in 16-token pages, non-causal,
          sweeping query rows and context length; each call reads about
          2 GB of KV, with every sequence on its own pages ("private") or
          all on the same pages ("shared")
  copy    one 4 GB device-to-device tensor copy

FLOPs are 4 * query heads * head dimension per query-key pair; a causal
document of L tokens has L(L+1)/2 pairs. The smoke run times one point
of each kind for 0.5 s and checks both libraries against torch's
scaled_dot_product_attention.

    mkdir -p results/benchmark
    log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-calibrate-attention.log"
    uv run modal run --detach calibrate/attention.py --device h100-sxm \
        --model qwen3-4b-fp8 --smoke 2>&1 | tee "$log"

Drop --smoke for the full sweep. Results go under
/results/calibrate/<device>/attention/<stamp>/, and the full sweep's
summary to /results/calibrate/<device>/attention_<q>_<kv>_<d>.json.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import modal

from calibrate.timing import Telemetry, measure_window, settle_clock

PAGE_TOKENS = 16
PACKED_TOKENS = 65536
PACKED_LENGTHS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
CACHED_QUERY_ROWS = (16, 32, 64, 128, 256, 512, 1024)
CACHED_CONTEXTS = (1024, 4096, 8192, 32768)
CACHED_KV_BYTES = 2 * 2**30
CACHED_MIN_SEQUENCES = 8
COPY_BYTES = 4 * 2**30
SMOKE_WINDOW_S, SMOKE_WARM_S = 0.5, 0.2
LIBRARIES = ("fa3", "flashinfer")
RESULTS = Path("/results/calibrate")

app = modal.App("quail-milestone1")
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04",
                              add_python="3.12")
    .pip_install("torch==2.13.0",
                 index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("flash-attn-3==3.0.0", extra_options="--no-deps",
                 index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("flashinfer-python==0.7.1", "nvidia-ml-py", "numpy")
    .add_local_python_source("calibrate")
)


def summary_name(layout) -> str:
    """Return the summary file name for a (query heads, KV heads, head dim) layout."""
    n_q, n_kv, d_head = layout
    return f"attention_{n_q}_{n_kv}_{d_head}.json"


def points(smoke: bool) -> list[dict]:
    """Return the points to measure, copy first.

    Args:
        smoke: Return one point of each kind instead of the full sweep.

    Returns:
        One dict per point, naming its kind and shape.
    """
    if smoke:
        return [{"kind": "copy"}, {"kind": "packed", "length": 1024},
                {"kind": "cached", "query_rows": 64, "context": 8192,
                 "pages": "private"},
                {"kind": "cached", "query_rows": 64, "context": 8192,
                 "pages": "shared"}]
    out = [{"kind": "copy"}]
    out += [{"kind": "packed", "length": n} for n in PACKED_LENGTHS]
    out += [{"kind": "cached", "query_rows": q, "context": c, "pages": p}
            for c in CACHED_CONTEXTS for q in CACHED_QUERY_ROWS
            for p in ("private", "shared")]
    return out


def packed_case(torch, layout, length):
    """Return inputs, FLOPs, and bytes for causal attention over equal documents."""
    n_q, n_kv, d_head = layout
    docs = PACKED_TOKENS // length
    total = docs * length
    q = torch.randn(total, n_q, d_head, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(total, n_kv, d_head, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    cu = torch.arange(0, total + 1, length, dtype=torch.int32, device="cuda")
    pairs = docs * length * (length + 1) // 2
    nbytes = (2 * q.numel() + k.numel() + v.numel()) * 2
    return {"q": q, "k": k, "v": v, "cu": cu, "docs": docs, "pairs": pairs,
            "flops": 4 * n_q * d_head * pairs, "bytes": nbytes}


def cached_case(torch, layout, query_rows, context, pages):
    """Return inputs, FLOPs, and bytes for query rows reading paged cached KV."""
    n_q, n_kv, d_head = layout
    kv_bytes_per_token = 2 * n_kv * d_head * 2
    seqs = max(CACHED_MIN_SEQUENCES,
               CACHED_KV_BYTES // (context * kv_bytes_per_token))
    per_seq = context // PAGE_TOKENS
    n_pages = per_seq if pages == "shared" else seqs * per_seq
    shape = (n_pages, PAGE_TOKENS, n_kv, d_head)
    k_cache = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn_like(k_cache)
    if pages == "shared":
        table = torch.arange(per_seq, dtype=torch.int32, device="cuda").expand(
            seqs, per_seq).contiguous()
    else:
        table = torch.arange(seqs * per_seq, dtype=torch.int32,
                             device="cuda").view(seqs, per_seq)
    q = torch.randn(seqs, query_rows, n_q, d_head, dtype=torch.bfloat16,
                    device="cuda")
    pairs = seqs * query_rows * context
    kv_bytes = seqs * context * kv_bytes_per_token
    return {"q": q, "k_cache": k_cache, "v_cache": v_cache, "table": table,
            "seqs": seqs, "pairs": pairs, "flops": 4 * n_q * d_head * pairs,
            "kv_bytes": kv_bytes, "bytes": kv_bytes + 2 * q.numel() * 2,
            "unique_kv_bytes": n_pages * PAGE_TOKENS * kv_bytes_per_token}


def fa3_op(torch, layout, point, case):
    """Return the FlashAttention 3 call for one point."""
    from flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache

    if point["kind"] == "packed":
        length = point["length"]
        return lambda: flash_attn_varlen_func(
            case["q"], case["k"], case["v"], case["cu"], case["cu"], length,
            length, causal=True)
    lens = torch.full((case["seqs"],), point["context"], dtype=torch.int32,
                      device="cuda")
    return lambda: flash_attn_with_kvcache(
        case["q"], case["k_cache"], case["v_cache"], cache_seqlens=lens,
        page_table=case["table"], causal=False, num_splits=0)


def flashinfer_op(torch, layout, point, case):
    """Return the FlashInfer call for one point, planned once."""
    import flashinfer

    n_q, n_kv, d_head = layout
    workspace = torch.empty(512 * 2**20, dtype=torch.uint8, device="cuda")
    if point["kind"] == "packed":
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
        wrapper.plan(case["cu"], case["cu"], n_q, n_kv, d_head,
                     causal=True, q_data_type=torch.bfloat16)
        return lambda: wrapper.run(case["q"], case["k"], case["v"])
    seqs, rows = case["seqs"], point["query_rows"]
    per_seq = case["table"].shape[1]
    qo = torch.arange(0, (seqs + 1) * rows, rows, dtype=torch.int32, device="cuda")
    kv = torch.arange(0, (seqs + 1) * per_seq, per_seq, dtype=torch.int32,
                      device="cuda")
    last = torch.full((seqs,), PAGE_TOKENS, dtype=torch.int32, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(qo, kv, case["table"].flatten(), last, n_q, n_kv,
                 d_head, PAGE_TOKENS, causal=False, q_data_type=torch.bfloat16)
    q = case["q"].view(seqs * rows, n_q, d_head)
    kv_cache = (case["k_cache"], case["v_cache"])
    return lambda: wrapper.run(q, kv_cache)


def reference_error(torch, layout, point, case, out) -> float:
    """Return the largest absolute difference from torch SDPA on two sequences."""
    import torch.nn.functional as F

    n_q, _, d_head = layout
    out = (out[0] if isinstance(out, tuple) else out).float()
    worst = 0.0
    for i in range(2):
        if point["kind"] == "packed":
            span = slice(i * point["length"], (i + 1) * point["length"])
            q, k, v = case["q"][span], case["k"][span], case["v"][span]
            got = out[span]
            causal = True
        else:
            pages = case["table"][i]
            q = case["q"][i]
            k = case["k_cache"][pages].flatten(0, 1)
            v = case["v_cache"][pages].flatten(0, 1)
            got = out.view(case["seqs"], -1, n_q, d_head)[i]
            causal = False
        want = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
            is_causal=causal, enable_gqa=True).transpose(0, 1).float()
        worst = max(worst, (got - want).abs().max().item())
    return worst


@app.function(image=image, gpu="H100!", timeout=2 * 3600,
              volumes={"/results": results_vol})
def measure(stamp: str, device: str, model: str, layout: tuple, smoke: bool,
            prediction: str) -> dict:
    """Measure every point with every library on one GPU and save the summary.

    Args:
        stamp: The UTC timestamp naming this run's directory.
        device: The quail.specs device name.
        model: The quail.specs model name the head layout is from.
        layout: (query heads, KV heads, head dimension).
        smoke: Measure one point of each kind briefly and check correctness.
        prediction: The prediction stated before the run.

    Returns:
        The summary also written to the results volume.
    """
    import torch

    layout = tuple(layout)
    run_dir = RESULTS / device / "attention" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / ("smoke.json" if smoke else "summary.json")
    window = (SMOKE_WINDOW_S, SMOKE_WARM_S) if smoke else ()
    telem = Telemetry()
    versions = {"torch": torch.__version__, "cuda": torch.version.cuda}
    for name in ("flash_attn_3", "flashinfer-python"):
        try:
            from importlib.metadata import version
            versions[name] = version(name)
        except Exception as error:
            versions[name] = f"not installed: {error}"
    summary = {"stamp": stamp, "device": device, "model": model,
               "layout": list(layout), "smoke": smoke, "gpu": telem.info(),
               "versions": versions, "prediction": prediction, "points": []}
    print(json.dumps({"gpu": summary["gpu"], "versions": versions}), flush=True)
    summary["settled_mhz"] = settle_clock(torch, telem)

    for point in points(smoke):
        row = dict(point)
        if point["kind"] == "copy":
            src = torch.ones(COPY_BYTES, dtype=torch.uint8, device="cuda")
            dst = torch.empty_like(src)
            row["copy"] = measure_window(torch, lambda: dst.copy_(src), 0,
                                         2 * COPY_BYTES, False, telem, *window)
            del src, dst
        else:
            case = (packed_case(torch, layout, point["length"])
                    if point["kind"] == "packed"
                    else cached_case(torch, layout, point["query_rows"],
                                     point["context"], point["pages"]))
            row.update({key: case[key] for key in case
                        if isinstance(case[key], int)})
            # shared pages measure L2 reuse, so their KV must stay in L2
            flush = not (point["kind"] == "cached" and point["pages"] == "shared")
            for lib, make in (("fa3", fa3_op), ("flashinfer", flashinfer_op)):
                try:
                    started = time.time()
                    op = make(torch, layout, point, case)
                    out = op()
                    torch.cuda.synchronize()
                    row[f"{lib}_setup_s"] = time.time() - started
                    if smoke:
                        row[f"{lib}_max_abs_error"] = reference_error(
                            torch, layout, point, case, out)
                    del out
                    row[lib] = measure_window(torch, op, case["flops"],
                                              case["bytes"], flush, telem,
                                              *window)
                except Exception as error:
                    row[lib] = {"error": f"{type(error).__name__}: {error}"[:500]}
                    traceback.print_exc()
            del case
        torch.cuda.empty_cache()
        summary["points"].append(row)
        print(json.dumps(row, default=str), flush=True)
        out_path.write_text(json.dumps(summary, indent=2, default=str))
        results_vol.commit()
    if not smoke:
        (RESULTS / device / summary_name(layout)).write_text(
            json.dumps(summary, indent=2, default=str))
        results_vol.commit()
    return summary


@app.local_entrypoint()
def main(device: str = "h100-sxm", model: str = "qwen3-4b-fp8",
         smoke: bool = False, prediction: str = ""):
    """Start the measurement on the device's GPU and print one line per point."""
    from datetime import datetime, timezone

    from calibrate.targets import TARGETS, head_layout
    from quail.specs import MODELS

    target = TARGETS[device]
    layout = head_layout(MODELS[model])
    print(f"prediction: {prediction or '(none stated)'}", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    call = measure.with_options(
        gpu=target.modal_gpu,
        env={"FLASHINFER_CUDA_ARCH_LIST": target.flashinfer_arch},
    ).spawn(stamp, device, model, layout, smoke, prediction)
    print(f"function call id: {call.object_id}", flush=True)
    print(f"results: {RESULTS / device / 'attention' / stamp}", flush=True)
    summary = call.get()
    for row in summary["points"]:
        shape = {k: row[k] for k in ("kind", "length", "query_rows", "context",
                                     "pages") if k in row}
        cells = []
        for lib in ("copy", *LIBRARIES):
            r = row.get(lib)
            if r and "error" in r:
                cells.append(f"{lib}: {r['error'][:80]}")
            elif r:
                cells.append(f"{lib}: {r['tflops']:.0f} TFLOPS {r['gbps']:.0f} GB/s "
                             f"{r['mhz'] or 0:.0f} MHz steady={r['steady']}")
        print(json.dumps(shape), " | ".join(cells), flush=True)

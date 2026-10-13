r"""Measure a GPU's attention and memory-bandwidth ceilings at one head layout.

Times upstream FlashAttention 3 and FlashInfer, bf16, one layer per
call, on one GPU, with calibrate.timing's sustained window. Three kinds
of point, set by AttentionSweep:

  packed  causal attention over a fixed number of tokens of
          equal-length documents, sweeping the document length
  cached  query rows reading cached KV in pages, non-causal, sweeping
          query rows and context length; each call reads a fixed amount
          of KV, with every sequence on its own pages ("private") or all
          on the same pages ("shared")
  copy    one device-to-device tensor copy

FLOPs are 4 * query heads * head dimension per query-key pair; a causal
document of L tokens has L(L+1)/2 pairs. The smoke run times one point
of each kind briefly and checks both libraries against torch's
scaled_dot_product_attention.

    mkdir -p results/benchmark
    log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-calibrate-attention.log"
    uv run modal run --detach calibrate/attention.py --device h100-sxm \
        --heads 32,8,128 --smoke 2>&1 | tee "$log"

--heads is query heads, KV heads, and head dimension. Drop --smoke for
the full sweep. Results go under
/results/calibrate/<device>/attention/<stamp>/, and the full sweep's
summary to /results/calibrate/<device>/attention_<q>_<kv>_<d>.json.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import modal

from calibrate.timing import Telemetry, Window, measure_window, settle_clock

# the results volume is mounted at /results
RESULTS = Path("/results/calibrate")


@dataclass(frozen=True)
class AttentionSweep:
    """The points an attention run measures, and how long it times each.

    Attributes:
        packed_tokens: Tokens per packed call, split into equal documents.
        packed_lengths: Document lengths of the packed points.
        page_tokens: Tokens per KV page of the cached points.
        cached_query_rows: Query rows per sequence of the cached points.
        cached_contexts: Cached tokens per sequence of the cached points.
        cached_pages: "private", "shared", or both.
        cached_kv_bytes: KV one cached call reads; sets the sequence count.
        cached_min_sequences: Fewest sequences in a cached call.
        copy_bytes: Size of the copied tensor.
        window: The timing window of every point.
    """

    packed_tokens: int = 65536
    packed_lengths: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096, 8192,
                                       16384, 32768)
    page_tokens: int = 16
    cached_query_rows: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024)
    cached_contexts: tuple[int, ...] = (1024, 4096, 8192, 32768)
    cached_pages: tuple[str, ...] = ("private", "shared")
    cached_kv_bytes: int = 2 * 2**30
    cached_min_sequences: int = 8
    copy_bytes: int = 4 * 2**30
    window: Window = field(default_factory=Window)

    @classmethod
    def smoke(cls) -> AttentionSweep:
        """Return one point of each kind, timed for 0.5 s."""
        return cls(packed_lengths=(1024,), cached_query_rows=(64,),
                   cached_contexts=(8192,),
                   window=Window(seconds=0.5, warm_seconds=0.2))

    def points(self) -> list[dict]:
        """Return one dict per point, naming its kind and shape, copy first."""
        out = [{"kind": "copy"}]
        out += [{"kind": "packed", "length": n} for n in self.packed_lengths]
        out += [{"kind": "cached", "query_rows": q, "context": c, "pages": p}
                for c in self.cached_contexts for q in self.cached_query_rows
                for p in self.cached_pages]
        return out


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


def summary_name(heads) -> str:
    """Return the summary file name for (query heads, KV heads, head dim)."""
    n_q, n_kv, d_head = heads
    return f"attention_{n_q}_{n_kv}_{d_head}.json"


def packed_case(torch, heads, sweep, length):
    """Return inputs, FLOPs, and bytes for causal attention over equal documents."""
    n_q, n_kv, d_head = heads
    docs = sweep.packed_tokens // length
    total = docs * length
    q = torch.randn(total, n_q, d_head, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(total, n_kv, d_head, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    cu = torch.arange(0, total + 1, length, dtype=torch.int32, device="cuda")
    pairs = docs * length * (length + 1) // 2
    nbytes = (2 * q.numel() + k.numel() + v.numel()) * 2
    return {"q": q, "k": k, "v": v, "cu": cu, "docs": docs, "pairs": pairs,
            "flops": 4 * n_q * d_head * pairs, "bytes": nbytes}


def cached_case(torch, heads, sweep, query_rows, context, pages):
    """Return inputs, FLOPs, and bytes for query rows reading paged cached KV."""
    n_q, n_kv, d_head = heads
    page_tokens = sweep.page_tokens
    kv_bytes_per_token = 2 * n_kv * d_head * 2
    seqs = max(sweep.cached_min_sequences,
               sweep.cached_kv_bytes // (context * kv_bytes_per_token))
    per_seq = context // page_tokens
    n_pages = per_seq if pages == "shared" else seqs * per_seq
    shape = (n_pages, page_tokens, n_kv, d_head)
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
            "unique_kv_bytes": n_pages * page_tokens * kv_bytes_per_token}


def fa3_op(torch, heads, point, case):
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


def flashinfer_op(torch, heads, point, case):
    """Return the FlashInfer call for one point, planned once."""
    import flashinfer

    n_q, n_kv, d_head = heads
    workspace = torch.empty(512 * 2**20, dtype=torch.uint8, device="cuda")
    if point["kind"] == "packed":
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
        wrapper.plan(case["cu"], case["cu"], n_q, n_kv, d_head,
                     causal=True, q_data_type=torch.bfloat16)
        return lambda: wrapper.run(case["q"], case["k"], case["v"])
    seqs, rows = case["seqs"], point["query_rows"]
    per_seq = case["table"].shape[1]
    page_tokens = case["k_cache"].shape[1]
    qo = torch.arange(0, (seqs + 1) * rows, rows, dtype=torch.int32, device="cuda")
    kv = torch.arange(0, (seqs + 1) * per_seq, per_seq, dtype=torch.int32,
                      device="cuda")
    last = torch.full((seqs,), page_tokens, dtype=torch.int32, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(qo, kv, case["table"].flatten(), last, n_q, n_kv,
                 d_head, page_tokens, causal=False, q_data_type=torch.bfloat16)
    q = case["q"].view(seqs * rows, n_q, d_head)
    kv_cache = (case["k_cache"], case["v_cache"])
    return lambda: wrapper.run(q, kv_cache)


def reference_error(torch, heads, point, case, out) -> float:
    """Return the largest absolute difference from torch SDPA on two sequences."""
    import torch.nn.functional as F

    n_q, _, d_head = heads
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
def measure(stamp: str, device: str, heads: tuple, smoke: bool,
            prediction: str) -> dict:
    """Measure every point with every library on one GPU and save the summary.

    Args:
        stamp: The UTC timestamp naming this run's directory.
        device: The quail.specs device name.
        heads: (query heads, KV heads, head dimension).
        smoke: Measure AttentionSweep.smoke() and check each library
            against torch, keeping the summary in the run's directory;
            otherwise measure AttentionSweep().
        prediction: The prediction stated before the run.

    Returns:
        The summary also written to the results volume.
    """
    import torch

    heads = tuple(heads)
    sweep = AttentionSweep.smoke() if smoke else AttentionSweep()
    run_dir = RESULTS / device / "attention" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / ("smoke.json" if smoke else "summary.json")
    telem = Telemetry()
    versions = {"torch": torch.__version__, "cuda": torch.version.cuda}
    for name in ("flash_attn_3", "flashinfer-python"):
        try:
            from importlib.metadata import version
            versions[name] = version(name)
        except Exception as error:
            versions[name] = f"not installed: {error}"
    summary = {"stamp": stamp, "device": device, "heads": list(heads),
               "sweep": asdict(sweep), "smoke": smoke, "gpu": telem.info(),
               "versions": versions, "prediction": prediction, "points": []}
    print(json.dumps({"gpu": summary["gpu"], "versions": versions}), flush=True)
    summary["settled_mhz"] = settle_clock(torch, telem)

    for point in sweep.points():
        row = dict(point)
        if point["kind"] == "copy":
            src = torch.ones(sweep.copy_bytes, dtype=torch.uint8, device="cuda")
            dst = torch.empty_like(src)
            row["copy"] = measure_window(torch, lambda: dst.copy_(src), 0,
                                         2 * sweep.copy_bytes, False, telem,
                                         sweep.window)
            del src, dst
        else:
            case = (packed_case(torch, heads, sweep, point["length"])
                    if point["kind"] == "packed"
                    else cached_case(torch, heads, sweep, point["query_rows"],
                                     point["context"], point["pages"]))
            row.update({key: case[key] for key in case
                        if isinstance(case[key], int)})
            # shared pages measure L2 reuse, so their KV must stay in L2
            flush = not (point["kind"] == "cached" and point["pages"] == "shared")
            for lib, make in (("fa3", fa3_op), ("flashinfer", flashinfer_op)):
                try:
                    started = time.time()
                    op = make(torch, heads, point, case)
                    out = op()
                    torch.cuda.synchronize()
                    row[f"{lib}_setup_s"] = time.time() - started
                    if smoke:
                        row[f"{lib}_max_abs_error"] = reference_error(
                            torch, heads, point, case, out)
                    del out
                    row[lib] = measure_window(torch, op, case["flops"],
                                              case["bytes"], flush, telem,
                                              sweep.window)
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
        (RESULTS / device / summary_name(heads)).write_text(
            json.dumps(summary, indent=2, default=str))
        results_vol.commit()
    return summary


@app.local_entrypoint()
def main(heads: str, device: str = "h100-sxm", smoke: bool = False,
         prediction: str = ""):
    """Start the measurement on the device's GPU and print one line per point.

    Args:
        heads: Query heads, KV heads, and head dimension, such as "32,8,128".
        device: The calibrate target, by quail.specs device name.
        smoke: Measure one point of each kind briefly and check correctness.
        prediction: The prediction stated before the run.
    """
    from datetime import datetime, timezone

    from calibrate.targets import TARGETS

    target = TARGETS[device]
    n_q, n_kv, d_head = (int(part) for part in heads.split(","))
    print(f"prediction: {prediction or '(none stated)'}", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    call = measure.with_options(
        gpu=target.modal_gpu,
        env={"FLASHINFER_CUDA_ARCH_LIST": target.flashinfer_arch},
    ).spawn(stamp, device, (n_q, n_kv, d_head), smoke, prediction)
    print(f"function call id: {call.object_id}", flush=True)
    print(f"results: {RESULTS / device / 'attention' / stamp}", flush=True)
    summary = call.get()
    for row in summary["points"]:
        shape = {k: row[k] for k in ("kind", "length", "query_rows", "context",
                                     "pages") if k in row}
        cells = []
        for lib in ("copy", "fa3", "flashinfer"):
            r = row.get(lib)
            if r and "error" in r:
                cells.append(f"{lib}: {r['error'][:80]}")
            elif r:
                cells.append(f"{lib}: {r['tflops']:.0f} TFLOPS {r['gbps']:.0f} GB/s "
                             f"{r['mhz'] or 0:.0f} MHz steady={r['steady']}")
        print(json.dumps(shape), " | ".join(cells), flush=True)

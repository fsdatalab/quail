"""Measure H100 attention and memory-bandwidth ceilings, independent of Quail.

Times upstream FlashAttention 3 and FlashInfer at Qwen3 4B's head layout
(32 query heads, 8 KV heads, head dimension 128, bf16), one layer per
call, on one H100. Nothing here imports quail. Three kinds of point:

  packed  causal attention over 65,536 tokens of equal-length documents,
          sweeping the document length
  cached  query rows reading cached KV in 16-token pages, non-causal,
          sweeping query rows and context length; each call reads about
          2 GB of KV, with every sequence on its own pages ("private") or
          all on the same pages ("shared")
  copy    one 4 GB device-to-device tensor copy

Each point is timed the way mamf-finder times MSMF: a warm-up, then a
2 s window of per-call events in 8 sub-windows, steady when the second
half is at most 1% slower than the first. FLOPs are 16,384 per query-key
pair; a causal document of L tokens has L(L+1)/2 pairs. The smoke run
times one point of each kind for 0.5 s and checks both libraries against
torch's scaled_dot_product_attention.

    mkdir -p results/benchmark
    log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-attention-ceilings.log"
    uv run modal run --detach experiments/attention_ceilings.py --smoke \
        2>&1 | tee "$log"

Drop --smoke for the full sweep. Results go under
/results/ablations/attention_ceilings/<stamp>/, and the full sweep's
summary to /results/ablations/attention_ceilings_h100.json.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path

import modal

NHEADS_Q, NHEADS_KV, HEAD_DIM = 32, 8, 128
FLOPS_PER_PAIR = 4 * NHEADS_Q * HEAD_DIM
KV_BYTES_PER_TOKEN = 2 * NHEADS_KV * HEAD_DIM * 2
PAGE_TOKENS = 16

PACKED_TOKENS = 65536
PACKED_LENGTHS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
CACHED_QUERY_ROWS = (16, 32, 64, 128, 256, 512, 1024)
CACHED_CONTEXTS = (1024, 4096, 8192, 32768)
CACHED_KV_BYTES = 2 * 2**30
CACHED_MIN_SEQUENCES = 8
COPY_BYTES = 4 * 2**30

WINDOW_S, WARM_S, SUB_WINDOWS = 2.0, 1.0, 8
SMOKE_WINDOW_S, SMOKE_WARM_S = 0.5, 0.2
MAX_DROP_PCT, MAX_SPREAD_PCT = 1.0, 10.0
SETTLE_S = 20.0
LIBRARIES = ("fa3", "flashinfer")

RUNS_DIR = Path("/results/ablations/attention_ceilings")
SUMMARY_PATH = Path("/results/ablations/attention_ceilings_h100.json")

PREDICTION_TEXT = (
    "packed: 550-700 TFLOPS sustained at document lengths of 8k tokens or "
    "more (the FlashAttention 3 paper reports up to 740 TFLOPS in FP16 on "
    "H100, in short runs), and 30-50% of that at 512 tokens or fewer. "
    "cached, private pages, 64 or fewer query rows: 60-80% of 3.35 TB/s of "
    "KV read. cached, shared pages: faster than private, possibly above "
    "3.35 TB/s where L2 serves repeated reads. copy: 85-95% of 3.35 TB/s."
)

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
    .env({"FLASHINFER_CUDA_ARCH_LIST": "9.0a"})
)


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


class Telemetry:
    """NVML power, SM clock, and temperature of GPU 0."""

    def __init__(self):
        import pynvml
        pynvml.nvmlInit()
        self.nvml = pynvml
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def power(self):
        """Instantaneous power in watts, or the 1 s average where unsupported."""
        value = self.nvml.nvmlDeviceGetFieldValues(self.handle, [186])[0]
        if value.nvmlReturn == 0:
            return value.value.uiVal / 1000.0
        return self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0

    def clock(self):
        """SM clock in MHz."""
        return float(self.nvml.nvmlDeviceGetClockInfo(
            self.handle, self.nvml.NVML_CLOCK_SM))

    def temp(self):
        """Die temperature in C."""
        return float(self.nvml.nvmlDeviceGetTemperature(
            self.handle, self.nvml.NVML_TEMPERATURE_GPU))

    def info(self):
        """Name, power limit, max SM clock, and driver version."""
        nvml, h = self.nvml, self.handle
        return {"name": nvml.nvmlDeviceGetName(h),
                "power_limit_w": nvml.nvmlDeviceGetPowerManagementLimit(h) / 1000,
                "max_sm_mhz": nvml.nvmlDeviceGetMaxClockInfo(h, nvml.NVML_CLOCK_SM),
                "driver": nvml.nvmlSystemGetDriverVersion()}


def settle_clock(torch, telem, seconds):
    """Run a bf16 matmul at full load until the SM clock stops dropping."""
    a = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
    c = torch.empty_like(a)
    end, prev, stable = time.time() + seconds, None, 0
    while time.time() < end and stable < 3:
        for _ in range(100):
            torch.mm(a, a, out=c)
        torch.cuda.synchronize()
        clk = telem.clock()
        stable = stable + 1 if prev is not None and clk >= prev - 5 else 0
        prev = clk
    return prev


def measure(torch, op, flops, nbytes, flush, telem, window_s, warm_s):
    """Time op over a sustained window, as mamf-finder's measure_window does.

    Args:
        torch: The torch module.
        op: The call to time.
        flops: FLOPs one call does.
        nbytes: Bytes one call must move.
        flush: Zero a 256 MiB buffer before each call, outside the timing.
        telem: The Telemetry to sample every 20 ms.
        window_s: Timed seconds.
        warm_s: Untimed seconds before the window.

    Returns:
        TFLOPS and GB/s over the window, the per-sub-window TFLOPS, the
        drop and spread in percent, whether it was steady, and the mean
        power, clock, and temperature.
    """
    l2 = torch.empty(64 * 2**20, dtype=torch.int32, device="cuda")

    def run(count, starts=None, ends=None):
        for i in range(count):
            if flush:
                l2.zero_()
            if starts is not None:
                starts[i].record()
            op()
            if ends is not None:
                ends[i].record()

    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    run(2)
    s.record()
    run(5)
    e.record()
    torch.cuda.synchronize()
    per_call = max(s.elapsed_time(e) / 5 / 1000, 1e-6)
    run(max(1, int(warm_s / per_call)))
    n = max(SUB_WINDOWS, int(window_s / per_call))
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    torch.cuda.synchronize()
    samples, stop = [], threading.Event()

    def sample():
        while not stop.is_set():
            samples.append((telem.power(), telem.clock(), telem.temp()))
            stop.wait(0.02)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        run(n, starts, ends)
        torch.cuda.synchronize()
    finally:
        stop.set()
        thread.join()
    times = [a.elapsed_time(b) / 1000 for a, b in zip(starts, ends)]
    size = n // SUB_WINDOWS
    subs = [flops * size / sum(times[i * size:(i + 1) * size]) / 1e12
            if flops else nbytes * size / sum(times[i * size:(i + 1) * size]) / 1e9
            for i in range(SUB_WINDOWS)]
    half = SUB_WINDOWS // 2
    first, second = sum(subs[:half]) / half, sum(subs[half:]) / half
    drop = (1 - second / first) * 100
    spread = (max(subs) - min(subs)) / sorted(subs)[half] * 100
    total = sum(times)

    def mean(i):
        return sum(x[i] for x in samples) / len(samples) if samples else None

    return {"tflops": flops * n / total / 1e12, "gbps": nbytes * n / total / 1e9,
            "ms_per_call": total / n * 1000, "calls": n, "subs": subs,
            "drop_pct": drop, "spread_pct": spread,
            "steady": drop <= MAX_DROP_PCT and spread <= MAX_SPREAD_PCT,
            "watts": mean(0), "mhz": mean(1), "celsius": mean(2)}


def packed_case(torch, length):
    """Inputs, FLOPs, and bytes for causal attention over equal-length documents."""
    docs = PACKED_TOKENS // length
    total = docs * length
    q = torch.randn(total, NHEADS_Q, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(total, NHEADS_KV, HEAD_DIM, dtype=torch.bfloat16,
                    device="cuda")
    v = torch.randn_like(k)
    cu = torch.arange(0, total + 1, length, dtype=torch.int32, device="cuda")
    pairs = docs * length * (length + 1) // 2
    nbytes = (2 * q.numel() + k.numel() + v.numel()) * 2
    return {"q": q, "k": k, "v": v, "cu": cu, "docs": docs,
            "pairs": pairs, "flops": FLOPS_PER_PAIR * pairs, "bytes": nbytes}


def cached_case(torch, query_rows, context, pages):
    """Inputs, FLOPs, and bytes for query rows reading paged cached KV."""
    seqs = max(CACHED_MIN_SEQUENCES,
               CACHED_KV_BYTES // (context * KV_BYTES_PER_TOKEN))
    per_seq = context // PAGE_TOKENS
    n_pages = per_seq if pages == "shared" else seqs * per_seq
    shape = (n_pages, PAGE_TOKENS, NHEADS_KV, HEAD_DIM)
    k_cache = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn_like(k_cache)
    if pages == "shared":
        table = torch.arange(per_seq, dtype=torch.int32, device="cuda").expand(
            seqs, per_seq).contiguous()
    else:
        table = torch.arange(seqs * per_seq, dtype=torch.int32,
                             device="cuda").view(seqs, per_seq)
    q = torch.randn(seqs, query_rows, NHEADS_Q, HEAD_DIM, dtype=torch.bfloat16,
                    device="cuda")
    pairs = seqs * query_rows * context
    kv_bytes = seqs * context * KV_BYTES_PER_TOKEN
    return {"q": q, "k_cache": k_cache, "v_cache": v_cache, "table": table,
            "seqs": seqs, "pairs": pairs, "flops": FLOPS_PER_PAIR * pairs,
            "kv_bytes": kv_bytes, "bytes": kv_bytes + 2 * q.numel() * 2,
            "unique_kv_bytes": n_pages * PAGE_TOKENS * KV_BYTES_PER_TOKEN}


def fa3_op(torch, point, case):
    """The FlashAttention 3 call for one point."""
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


def flashinfer_op(torch, point, case):
    """The FlashInfer call for one point, planned once."""
    import flashinfer

    workspace = torch.empty(512 * 2**20, dtype=torch.uint8, device="cuda")
    if point["kind"] == "packed":
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
        wrapper.plan(case["cu"], case["cu"], NHEADS_Q, NHEADS_KV, HEAD_DIM,
                     causal=True, q_data_type=torch.bfloat16)
        return lambda: wrapper.run(case["q"], case["k"], case["v"])
    seqs, rows = case["seqs"], point["query_rows"]
    per_seq = case["table"].shape[1]
    qo = torch.arange(0, (seqs + 1) * rows, rows, dtype=torch.int32, device="cuda")
    kv = torch.arange(0, (seqs + 1) * per_seq, per_seq, dtype=torch.int32,
                      device="cuda")
    last = torch.full((seqs,), PAGE_TOKENS, dtype=torch.int32, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(qo, kv, case["table"].flatten(), last, NHEADS_Q, NHEADS_KV,
                 HEAD_DIM, PAGE_TOKENS, causal=False, q_data_type=torch.bfloat16)
    q = case["q"].view(seqs * rows, NHEADS_Q, HEAD_DIM)
    kv_cache = (case["k_cache"], case["v_cache"])
    return lambda: wrapper.run(q, kv_cache)


def reference_error(torch, point, case, out):
    """Largest absolute difference from torch SDPA on the first two sequences."""
    import torch.nn.functional as F

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
            got = out.view(case["seqs"], -1, NHEADS_Q, HEAD_DIM)[i]
            causal = False
        want = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
            is_causal=causal, enable_gqa=True).transpose(0, 1).float()
        worst = max(worst, (got - want).abs().max().item())
    return worst


@app.function(image=image, gpu="H100!", timeout=2 * 3600,
              volumes={"/results": results_vol})
def run(stamp: str, smoke: bool) -> dict:
    """Measure every point with every library on one GPU and save the summary.

    Args:
        stamp: The UTC timestamp naming this run's directory.
        smoke: Measure one point of each kind briefly and check correctness.

    Returns:
        The summary also written to the results volume.
    """
    import torch

    run_dir = RUNS_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / ("smoke.json" if smoke else "summary.json")
    window_s, warm_s = (SMOKE_WINDOW_S, SMOKE_WARM_S) if smoke else (WINDOW_S, WARM_S)
    telem = Telemetry()
    versions = {"torch": torch.__version__, "cuda": torch.version.cuda}
    for name in ("flash_attn_3", "flashinfer-python"):
        try:
            from importlib.metadata import version
            versions[name] = version(name)
        except Exception as error:
            versions[name] = f"not installed: {error}"
    summary = {"stamp": stamp, "smoke": smoke, "gpu": telem.info(),
               "versions": versions, "prediction": PREDICTION_TEXT,
               "window_s": window_s, "points": []}
    print(json.dumps({"gpu": summary["gpu"], "versions": versions}), flush=True)
    summary["settled_mhz"] = settle_clock(torch, telem, SETTLE_S)

    for point in points(smoke):
        row = dict(point)
        if point["kind"] == "copy":
            src = torch.ones(COPY_BYTES, dtype=torch.uint8, device="cuda")
            dst = torch.empty_like(src)
            row["copy"] = measure(torch, lambda: dst.copy_(src), 0, 2 * COPY_BYTES,
                                  False, telem, window_s, warm_s)
            del src, dst
        else:
            case = (packed_case(torch, point["length"])
                    if point["kind"] == "packed"
                    else cached_case(torch, point["query_rows"], point["context"],
                                     point["pages"]))
            row.update({key: case[key] for key in case
                        if isinstance(case[key], int)})
            flush = not (point["kind"] == "cached" and point["pages"] == "shared")
            for lib, make in (("fa3", fa3_op), ("flashinfer", flashinfer_op)):
                try:
                    started = time.time()
                    op = make(torch, point, case)
                    out = op()
                    torch.cuda.synchronize()
                    row[f"{lib}_setup_s"] = time.time() - started
                    if smoke:
                        row[f"{lib}_max_abs_error"] = reference_error(
                            torch, point, case, out)
                    del out
                    row[lib] = measure(torch, op, case["flops"], case["bytes"],
                                       flush, telem, window_s, warm_s)
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
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))
        results_vol.commit()
    return summary


@app.local_entrypoint()
def main(smoke: bool = False):
    """Start the measurement and print one line per point."""
    from datetime import datetime, timezone

    print(f"prediction: {PREDICTION_TEXT}", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    call = run.spawn(stamp, smoke)
    print(f"function call id: {call.object_id}", flush=True)
    print(f"results: {RUNS_DIR / stamp}", flush=True)
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

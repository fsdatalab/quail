r"""Measure a GPU's attention and memory-bandwidth ceilings at one head layout.

Times upstream FlashAttention 3 and FlashInfer, bf16, one layer per
call, on one GPU, with calibrate.timing.measure. Each call is the size
of the calls Quail makes (AttentionSweep):

  packed  causal attention over one chunk of tokens, split into
          documents of one length; one point per length
  cached  sequences whose query rows read KV already cached in pages,
          non-causal; one point per (query rows, context). The query
          rows fill one chunk, and the sequences are capped by the KV a
          Quail arena holds
  size    one packed and one cached point repeated at a quarter and a
          half of the call size, to show how the rate depends on it
  copy    one device-to-device tensor copy

Both libraries run on the same random inputs; the call's time depends
on the shapes, not the values. FLOPs are 4 * query heads * head
dimension per query-key pair; a causal document of L tokens has
L(L+1)/2 pairs. The smoke sweep times one point of each kind briefly
and checks both libraries against torch's scaled_dot_product_attention.

On a machine with the GPU, FlashAttention 3, and FlashInfer:

    uv run python -m calibrate.attention --device h100-sxm --heads 32,8,128 \
        --root results/calibrate --smoke

--heads is query heads, KV heads, and head dimension; the call sizes
come from the quail.specs models with that layout (calibrate.targets).
Results go under <root>/<device>/attention/<stamp>/summary.json.
calibrate.modal_app runs the same code on Modal.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from calibrate.timing import Telemetry, Window, measure

LIBRARIES = ("fa3", "flashinfer")


@dataclass(frozen=True)
class AttentionSweep:
    """The calls an attention run measures, and how it times each.

    Attributes:
        chunk_tokens: Tokens per packed call, and query rows per cached
            call: the largest Quail chunk of the models with this layout.
        kv_tokens: Most cached tokens one cached call reads: the largest
            Quail KV arena of those models.
        packed_lengths: Document lengths of the packed points.
        cached_query_rows: Query rows per sequence of the cached points.
        cached_contexts: Cached tokens per sequence of the cached points.
        size_fractions: Fractions of the call size the size points repeat.
        size_length: Document length of the packed size points.
        size_read: (query rows, context) of the cached size points.
        page_tokens: Tokens per KV page.
        copy_bytes: Size of the copied tensor.
        window: How each call is settled, timed, and accepted.
    """

    chunk_tokens: int
    kv_tokens: int
    packed_lengths: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096, 8192,
                                       16384, 32768)
    cached_query_rows: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024)
    cached_contexts: tuple[int, ...] = (1024, 4096, 8192, 32768)
    size_fractions: tuple[float, ...] = (0.25, 0.5)
    size_length: int = 1024
    size_read: tuple[int, int] = (64, 8192)
    page_tokens: int = 16
    copy_bytes: int = 4 * 2**30
    window: Window = field(default_factory=Window)

    @classmethod
    def smoke(cls, chunk_tokens: int, kv_tokens: int) -> AttentionSweep:
        """Return one point of each kind, timed in one 0.5 s window."""
        return cls(chunk_tokens, kv_tokens, packed_lengths=(1024,),
                   cached_query_rows=(64,), cached_contexts=(8192,),
                   size_fractions=(0.5,),
                   window=Window(seconds=0.5, settle_min_s=0.2, settle_max_s=5.0,
                                 repeats=1, max_tries=1))

    @classmethod
    def from_dict(cls, fields: dict) -> AttentionSweep:
        """Return the sweep that asdict() turned into fields."""
        tuples = {k: tuple(v) for k, v in fields.items() if isinstance(v, list)}
        return cls(**{**fields, **tuples, "window": Window(**fields["window"])})

    def points(self) -> list[dict]:
        """Return one dict per point, naming its kind, shape, and size, copy first."""
        out = [{"kind": "copy"}]
        out += [{"kind": "packed", "length": n, "size": 1.0}
                for n in self.packed_lengths]
        out += [{"kind": "cached", "query_rows": q, "context": c, "size": 1.0}
                for c in self.cached_contexts for q in self.cached_query_rows]
        rows, context = self.size_read
        for f in self.size_fractions:
            out.append({"kind": "packed", "length": self.size_length, "size": f})
            out.append({"kind": "cached", "query_rows": rows, "context": context,
                        "size": f})
        return out

    def documents(self, point: dict) -> int:
        """Return the documents in a packed point's call."""
        return max(1, int(self.chunk_tokens * point["size"]) // point["length"])

    def sequences(self, point: dict) -> int:
        """Return the sequences in a cached point's call.

        As many as the chunk's query rows fill, but no more than the
        arena's KV holds, and at least one.
        """
        by_rows = self.chunk_tokens // point["query_rows"]
        by_kv = self.kv_tokens // point["context"]
        return max(1, int(min(by_rows, by_kv) * point["size"]))


def packed_case(torch, heads, docs, length):
    """Return inputs, FLOPs, and bytes for causal attention over equal documents."""
    n_q, n_kv, d_head = heads
    total = docs * length
    q = torch.randn(total, n_q, d_head, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(total, n_kv, d_head, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    cu = torch.arange(0, total + 1, length, dtype=torch.int32, device="cuda")
    pairs = docs * length * (length + 1) // 2
    nbytes = (2 * q.numel() + k.numel() + v.numel()) * 2
    return {"q": q, "k": k, "v": v, "cu": cu, "docs": docs, "tokens": total,
            "pairs": pairs, "flops": 4 * n_q * d_head * pairs, "bytes": nbytes}


def cached_case(torch, heads, seqs, query_rows, context, page_tokens):
    """Return inputs, FLOPs, and bytes for query rows reading paged cached KV."""
    n_q, n_kv, d_head = heads
    kv_bytes_per_token = 2 * n_kv * d_head * 2
    per_seq = context // page_tokens
    shape = (seqs * per_seq, page_tokens, n_kv, d_head)
    k_cache = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn_like(k_cache)
    table = torch.arange(seqs * per_seq, dtype=torch.int32,
                         device="cuda").view(seqs, per_seq)
    q = torch.randn(seqs, query_rows, n_q, d_head, dtype=torch.bfloat16,
                    device="cuda")
    pairs = seqs * query_rows * context
    kv_bytes = seqs * context * kv_bytes_per_token
    return {"q": q, "k_cache": k_cache, "v_cache": v_cache, "table": table,
            "seqs": seqs, "query_tokens": seqs * query_rows, "pairs": pairs,
            "flops": 4 * n_q * d_head * pairs, "kv_bytes": kv_bytes,
            "bytes": kv_bytes + 2 * q.numel() * 2}


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


MAKERS = {"fa3": fa3_op, "flashinfer": flashinfer_op}


def reference_error(torch, heads, point, case, out) -> float:
    """Return the largest absolute difference from torch SDPA on up to two sequences."""
    import torch.nn.functional as F

    n_q, _, d_head = heads
    out = (out[0] if isinstance(out, tuple) else out).float()
    worst = 0.0
    count = case["docs"] if point["kind"] == "packed" else case["seqs"]
    for i in range(min(2, count)):
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


def run(root: Path, device: str, stamp: str, heads: tuple, sweep: AttentionSweep,
        check: bool = False, prediction: str = "",
        on_save: Callable[[], None] = lambda: None) -> dict:
    """Measure every point with both libraries on GPU 0 and save the summary.

    For each point the inputs are built and both libraries called once
    (planning and any kernel compile) before either is timed. The
    library timed first alternates from point to point.

    Args:
        root: The results directory.
        device: The quail.specs device name.
        stamp: Names this run's directory.
        heads: (query heads, KV heads, head dimension).
        sweep: The calls and the timing settings.
        check: Compare each library's output with torch's.
        prediction: The prediction stated before the run.
        on_save: Called after each file is written.

    Returns:
        The summary, also written to <root>/<device>/attention/<stamp>/.
    """
    import torch

    heads = tuple(heads)
    run_dir = root / device / "attention" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    telem = Telemetry()
    versions = {"torch": torch.__version__, "cuda": torch.version.cuda}
    for name in ("flash_attn_3", "flashinfer-python"):
        try:
            from importlib.metadata import version
            versions[name] = version(name)
        except Exception as error:
            versions[name] = f"not installed: {error}"
    summary = {"stamp": stamp, "device": device, "heads": list(heads),
               "sweep": asdict(sweep), "check": check, "gpu": telem.info(),
               "versions": versions, "prediction": prediction, "points": []}
    print(json.dumps({"gpu": summary["gpu"], "versions": versions}), flush=True)

    for index, point in enumerate(sweep.points()):
        row = dict(point)
        if point["kind"] == "copy":
            src = torch.ones(sweep.copy_bytes, dtype=torch.uint8, device="cuda")
            dst = torch.empty_like(src)
            row["copy"] = measure(torch, lambda: dst.copy_(src), 0,
                                  2 * sweep.copy_bytes, False, telem, sweep.window)
            del src, dst
        else:
            case = (packed_case(torch, heads, sweep.documents(point), point["length"])
                    if point["kind"] == "packed"
                    else cached_case(torch, heads, sweep.sequences(point),
                                     point["query_rows"], point["context"],
                                     sweep.page_tokens))
            row.update({key: case[key] for key in case
                        if isinstance(case[key], int)})
            ops = {}
            for lib in LIBRARIES:
                try:
                    started = time.time()
                    ops[lib] = MAKERS[lib](torch, heads, point, case)
                    out = ops[lib]()
                    torch.cuda.synchronize()
                    row[f"{lib}_setup_s"] = time.time() - started
                    if check:
                        row[f"{lib}_max_abs_error"] = reference_error(
                            torch, heads, point, case, out)
                    del out
                except Exception as error:
                    row[lib] = {"error": f"{type(error).__name__}: {error}"[:500]}
                    traceback.print_exc()
            order = list(ops) if index % 2 else list(ops)[::-1]
            for lib in order:
                try:
                    row[lib] = measure(torch, ops[lib], case["flops"],
                                       case["bytes"], True, telem, sweep.window)
                except Exception as error:
                    row[lib] = {"error": f"{type(error).__name__}: {error}"[:500]}
                    traceback.print_exc()
            del case, ops
        torch.cuda.empty_cache()
        summary["points"].append(row)
        print(point_line(row), flush=True)
        text = json.dumps(summary, indent=2, default=str)
        (run_dir / "summary.json").write_text(text)
        on_save()
    return summary


def point_line(row: dict) -> str:
    """Return one printed line for a measured point."""
    shape = {k: row[k] for k in ("kind", "length", "query_rows", "context", "size",
                                 "docs", "seqs") if k in row}
    cells = []
    for lib in ("copy", *LIBRARIES):
        r = row.get(lib)
        if r and "error" in r:
            cells.append(f"{lib}: {r['error'][:80]}")
        elif r:
            mhz = [w["mhz"] or 0 for w in r["windows"]]
            cells.append(f"{lib}: {r['tflops']:.0f} TFLOPS {r['gbps']:.0f} GB/s "
                         f"{min(mhz):.0f}-{max(mhz):.0f} MHz "
                         f"steady {r['steady_windows']}/{len(r['windows'])}")
    return f"{json.dumps(shape)} {' | '.join(cells)}"


def parse_heads(text: str) -> tuple[int, int, int]:
    """Return (query heads, KV heads, head dimension) from text like "32,8,128"."""
    n_q, n_kv, d_head = (int(part) for part in text.split(","))
    return n_q, n_kv, d_head


def main(argv=None) -> None:
    """Measure the GPU this process sees and print where the summary went."""
    from calibrate import utc_stamp
    from calibrate.targets import call_sizes

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", required=True,
                        help="quail.specs device name, such as h100-sxm")
    parser.add_argument("--heads", type=parse_heads, required=True,
                        help="query heads, KV heads, head dimension: 32,8,128")
    parser.add_argument("--root", type=Path, default=Path("results/calibrate"),
                        help="results directory")
    parser.add_argument("--smoke", action="store_true",
                        help="one point of each kind, checked against torch")
    parser.add_argument("--prediction", default="")
    args = parser.parse_args(argv)
    chunk, kv = call_sizes(args.heads, args.device)
    sweep = (AttentionSweep.smoke(chunk, kv) if args.smoke
             else AttentionSweep(chunk, kv))
    stamp = utc_stamp()
    run(args.root, args.device, stamp, args.heads, sweep, check=args.smoke,
        prediction=args.prediction)
    print(f"summary: {args.root / args.device / 'attention' / stamp / 'summary.json'}")


if __name__ == "__main__":
    main()

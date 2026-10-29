"""Collect replay coverage and compare real chunk outputs with eager execution."""

from collections import Counter
from functools import partial
from types import SimpleNamespace


def padded_forward(graphs, chunk):
    """Run the captured bucket's same dense shapes without replaying any graph."""
    from quail.backends.quail.executor.cuda_graphs import CapturedPieces

    buffers = next(c.buffers for c in graphs.graphs
                   if c.buffers.num_tokens >= chunk.tokens)
    pieces = [SimpleNamespace(replay=partial(buffers.piece, graphs.pipeline, index))
              for index in range(len(graphs.pipeline.layers) + 1)]
    return CapturedPieces(buffers, pieces).forward(graphs.pipeline, chunk)


def attention_work(chunk):
    """Count useful QK/PV pairs from the exact full-attention chunk metadata."""
    if chunk.attention_mode != "unified":
        raise ValueError("work counting currently requires unified attention")
    unified = chunk.meta["unified"]
    if unified is None:
        bounds = chunk.meta["cu_a"].cpu().tolist()
        used = [b - a for a, b in zip(bounds, bounds[1:])]
    else:
        bounds, used = unified["cu_q_host"], unified["used_host"]

    def pairs(lo, hi):
        total = 0
        for a, b, num_keys in zip(bounds, bounds[1:], used):
            start, end = max(a, lo, b - num_keys), min(b, hi)
            if start < end:
                first, last = num_keys - b + start + 1, num_keys - b + end
                total += (end - start) * (first + last) // 2
        return int(total)

    prefix_ranges = [(start, suffix) for _, start, suffix, _ in chunk.row_spans]
    decode = chunk.decode_rows
    num_decode_tokens = 0 if decode is None else len(decode)
    shared_queries = shared_keys = []
    if chunk.meta.get("shared") is not None:
        shared_queries = [int(b - a) for a, b in zip(
            decode.shared_cu, decode.shared_cu[1:])]
        shared_keys = [int(n) for n in decode.shared_used]
    total = pairs(0, chunk.tokens)
    prefix = sum(pairs(a, b) for a, b in prefix_ranges)
    decoded = pairs(chunk.tokens - num_decode_tokens, chunk.tokens)
    return dict(
        query_tokens=[int(b - a) for a, b in zip(bounds, bounds[1:])],
        context_tokens=[int(n) for n in used],
        prefix_ranges=prefix_ranges, decode_tokens=num_decode_tokens,
        causal_pairs=total, prefix_pairs=prefix, suffix_pairs=total - prefix - decoded,
        decode_pairs=decoded, shared_queries=shared_queries, shared_keys=shared_keys,
        shared_pairs=sum(q * k for q, k in zip(shared_queries, shared_keys)))


class PiecewiseProbe:
    """Instrument piecewise dispatch for the query profiler."""

    def __init__(self):
        from quail.backends.quail.executor import cuda_graphs

        self.active = False
        self.verify = False
        self.eager = False
        self.work = False
        self.records = []
        self.checked = set()
        graph_class = getattr(cuda_graphs, "PiecewiseGraphs", None)
        if graph_class is None:
            return
        original = graph_class.forward
        probe = self

        def forward(graphs, chunk):
            if not probe.active:
                return original(graphs, chunk)
            replay = not probe.eager and graphs.supports(chunk)
            result = (graphs.pipeline.forward_chunk(chunk) if probe.eager
                      else original(graphs, chunk))
            path = ("shared" if chunk.meta.get("shared") is not None
                    else chunk.attention_mode)
            record = {"tokens": chunk.tokens, "replay": replay, "path": path}
            bucket = next((n for n in graphs.token_buckets if n >= chunk.tokens), None)
            record["dense_tokens"] = bucket if replay else chunk.tokens
            if probe.work:
                record["work"] = attention_work(chunk)
            signature = (chunk.attention_mode,
                         bool((chunk.meta.get("reads") or {}).get("nodes")),
                         bool(chunk.meta.get("shared")), bucket)
            if probe.verify and replay and len(probe.checked) < 8 and (
                    signature not in probe.checked):
                torch = graphs.pipeline.engine.torch
                expected = graphs.pipeline.forward_chunk(chunk)
                record["equal"] = torch.equal(result, expected)
                record["max_error"] = (result - expected).abs().max().item()
                record["signature"] = signature
                if not record["equal"]:
                    repeated = graphs.pipeline.forward_chunk(chunk)
                    record["eager_repeat_equal"] = torch.equal(expected, repeated)
                    expected = padded_forward(graphs, chunk)
                    record["padded_equal"] = torch.equal(result, expected)
                    record["padded_max_error"] = (result - expected).abs().max().item()
                print(f"PIECEWISE CHECK {record}", flush=True)
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                probe.checked.add(signature)
            probe.records.append(record)
            if probe.capture is None:
                probe.capture = {"seconds": graphs.capture_s,
                                 "bytes": graphs.captured_bytes,
                                 "reserved_bytes": graphs.memory_bytes,
                                 "buckets": graphs.token_buckets,
                                 "layers": len(graphs.pipeline.layers),
                                 "projection_shapes": [list(linear.weight.shape)
                                                       for linear in
                                                       graphs.pipeline.linears()]}
            return result

        graph_class.forward = forward

    def start(self, verify=False, eager=False, work=False):
        """Start collecting one execution window with the same captured KV budget."""
        self.records = []
        self.checked = set()
        self.capture = None
        self.active = True
        self.verify = verify
        self.eager = eager
        self.work = work

    def summary(self):
        """Return capture memory, pass counts, and checked output equality."""
        self.active = False
        return {"passes": len(self.records),
                "paths": dict(Counter(row["path"] for row in self.records)),
                "replays": sum(row["replay"] for row in self.records),
                "capture": self.capture,
                "work": [row for row in self.records if "work" in row],
                "checks": [row for row in self.records if "equal" in row]}

"""Collect replay coverage and compare real chunk outputs with eager execution."""

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


class PiecewiseProbe:
    """Instrument piecewise dispatch for the query profiler."""

    def __init__(self):
        from quail.backends.quail.executor import cuda_graphs

        self.active = False
        self.verify = False
        self.eager = False
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
            record = {"tokens": chunk.tokens, "replay": replay}
            bucket = next((n for n in graphs.token_buckets if n >= chunk.tokens), None)
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
            probe.capture = {"seconds": graphs.capture_s,
                             "bytes": graphs.captured_bytes,
                             "reserved_bytes": graphs.memory_bytes,
                             "buckets": graphs.token_buckets}
            return result

        graph_class.forward = forward

    def start(self, verify=False, eager=False):
        """Start collecting one execution window with the same captured KV budget."""
        self.records = []
        self.checked = set()
        self.capture = None
        self.active = True
        self.verify = verify
        self.eager = eager

    def summary(self):
        """Return capture memory, pass counts, and checked output equality."""
        self.active = False
        return {"passes": len(self.records),
                "replays": sum(row["replay"] for row in self.records),
                "capture": self.capture,
                "checks": [row for row in self.records if "equal" in row]}

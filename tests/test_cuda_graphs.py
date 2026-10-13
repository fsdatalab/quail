"""Graph padding preserves sequence boundaries and KV ownership."""

from types import SimpleNamespace

import pytest

from quail.backends.quail.executor.cuda_graphs import DecodeBuffers, DecodeGraphs

torch = pytest.importorskip("torch")


def chunk():
    buffers = DecodeBuffers(torch, 5, 16, device="cpu")
    out = buffers.chunk
    out.input_ids[:] = torch.tensor([12, 13, 14, 15, 16])
    out.positions[:] = torch.tensor([4, 5, 6, 9, 10])
    out.final_indices = torch.tensor([2, 4])
    out.meta["cu_a"] = torch.tensor([0, 3, 5], dtype=torch.int32)
    out.meta["max_a"] = 3
    out.meta["kv_src"] = torch.tensor([0, 1, 2, 3, 4])
    out.meta["kv_dst"] = torch.tensor([20, 21, 22, 41, 42])
    out.meta["reads"] = dict(
        rows=torch.arange(5), cu_q=torch.tensor([0, 3, 5], dtype=torch.int32),
        max_q=3, max_used=9, used=torch.tensor([4, 9], dtype=torch.int32),
        table=torch.tensor([[1], [2]], dtype=torch.int32),
        source=torch.arange(5, dtype=torch.int32))
    return out


def test_padding_preserves_attention_and_only_real_rows_write_kv():
    source = chunk()
    buffers = DecodeBuffers(torch, 8, 16, device="cpu")
    buffers.synthetic(7)
    buffers.copy(source)
    assert buffers.ids.tolist() == [12, 13, 14, 15, 16, 0, 0, 0]
    assert buffers.cu_a.tolist() == [0, 3, 5, 6, 7, 8, 8, 8, 8]
    assert buffers.cu_q.tolist() == buffers.cu_a.tolist()
    assert buffers.used.tolist() == [4, 9, 0, 0, 0, 0, 0, 0]
    assert buffers.source.tolist() == [0, 1, 2, 3, 4, -1, -1, -1]
    assert buffers.dst.tolist() == [20, 21, 22, 41, 42, -1, -1, -1]
    assert buffers.table[:2, :1].equal(source.meta["reads"]["table"])

    # A shorter batch must not inherit any prior sequence or write mapping.
    source.tokens = 2
    source.input_ids = source.input_ids[:2]
    source.positions = source.positions[:2]
    source.meta["cu_a"] = torch.tensor([0, 2], dtype=torch.int32)
    source.meta["kv_src"] = source.meta["kv_dst"] = None
    reads = source.meta["reads"]
    reads["rows"] = torch.tensor([1])
    reads["cu_q"] = torch.tensor([0, 1], dtype=torch.int32)
    reads["used"] = torch.tensor([4], dtype=torch.int32)
    reads["table"] = reads["table"][:1]
    reads["source"] = torch.tensor([-1, 0], dtype=torch.int32)
    buffers.copy(source)
    assert buffers.cu_a.tolist() == [0, 2, 3, 4, 5, 6, 7, 8, 8]
    assert buffers.cu_q.tolist() == list(range(9))
    assert buffers.used.tolist() == [4, 0, 0, 0, 0, 0, 0, 0]
    assert buffers.source.tolist() == [-1, 0, -1, -1, -1, -1, -1, -1]
    assert (buffers.dst == -1).all()


@pytest.mark.parametrize("unsupported", [
    "unified", "nodes", "long_context", "long_query", "many_tokens",
    "new_arena", "disabled", "canvas", "no_reads", "wide_table",
])
def test_unsupported_batches_fall_back(unsupported):
    source = chunk()
    arena = SimpleNamespace(k=[], page_tokens=16)
    calls = []
    pipeline = SimpleNamespace(
        engine=SimpleNamespace(arena=arena),
        forward_chunk=lambda x: calls.append(x) or "eager")
    graphs = DecodeGraphs(pipeline)
    graphs.storage = arena.k
    assert graphs.supports(source)
    match unsupported:
        case "unified":
            source.attention_mode = "unified"
        case "nodes":
            source.meta["reads"]["nodes"] = object()
        case "long_context":
            source.meta["reads"]["max_used"] = 65537
        case "long_query":
            source.meta["max_a"] = 33
        case "many_tokens":
            source.tokens = 513
        case "new_arena":
            arena.k = []
        case "disabled":
            graphs.enabled = False
        case "canvas":
            source.meta["canvas"] = object()
        case "no_reads":
            source.meta["reads"] = None
        case "wide_table":
            source.meta["reads"]["table"] = torch.zeros(2, 4097)
    assert graphs.forward(source) == "eager"
    assert calls == [source]


def test_replay_returns_independent_output_storage():
    source = chunk()
    arena = SimpleNamespace(k=[], page_tokens=16)
    graphs = DecodeGraphs(SimpleNamespace(engine=SimpleNamespace(arena=arena)))
    graphs.storage = arena.k
    output = torch.arange(8 * 3).reshape(8, 3)
    graphs.graphs = [SimpleNamespace(
        buffers=DecodeBuffers(torch, 8, 16, device="cpu"),
        graph=SimpleNamespace(replay=lambda: None), output=output)]
    actual = graphs.forward(source)
    expected = output[[2, 4]].clone()
    output.zero_()
    assert actual.equal(expected)
    assert graphs.replays == 1
    graphs.invalidate()
    assert not graphs.graphs
    assert graphs.storage is None


def test_worker_capture_limits_control_buffers_and_replay_admission():
    from quail.backends.quail.executor.cuda_graphs import DecodeGraphConfig

    config = DecodeGraphConfig(row_buckets=(8, 16), max_context_tokens=31,
                               max_query_tokens=3)
    buffers = DecodeBuffers(torch, 8, 16, config=config, device="cpu")
    assert buffers.table.shape == (8, 2)
    assert buffers.chunk.meta["max_a"] == 3
    assert buffers.chunk.meta["reads"]["max_used"] == 31
    source = chunk()
    arena = SimpleNamespace(k=[], page_tokens=16)
    pipeline = SimpleNamespace(engine=SimpleNamespace(arena=arena))
    graphs = DecodeGraphs(pipeline, config=config)
    other = DecodeGraphs(pipeline)
    graphs.storage = other.storage = arena.k
    assert graphs.supports(source)
    source.tokens = 17
    assert not graphs.supports(source) and other.supports(source)
    source.tokens = 5
    source.meta["reads"]["max_used"] = 32
    assert not graphs.supports(source) and other.supports(source)
    source.meta["reads"]["max_used"] = 9
    source.meta["max_a"] = 4
    assert not graphs.supports(source) and other.supports(source)

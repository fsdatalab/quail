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
    "unified", "many_tokens", "new_arena",
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
        case "many_tokens":
            source.tokens = 4097
        case "new_arena":
            arena.k = []
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


def rounds_chunk():
    # two map rounds of one document, three rows and two: each reads
    # its own pages causally, and the five rows read the document once
    source = chunk()
    meta = source.meta
    meta["cu_a"] = torch.tensor([0], dtype=torch.int32)
    meta["reads"] = None
    meta["a_rows"] = None
    meta["rounds"] = dict(
        rows=torch.arange(5), cu_q=torch.tensor([0, 3, 5], dtype=torch.int32),
        max_q=3, used=torch.tensor([7, 11], dtype=torch.int32), max_used=11,
        table=torch.tensor([[1], [2]], dtype=torch.int32),
        documents=dict(b_index=torch.arange(5),
                       cu_q=torch.tensor([0, 5], dtype=torch.int32), max_q=5,
                       used=torch.tensor([32], dtype=torch.int32), max_used=32,
                       table=torch.tensor([[3, 4]], dtype=torch.int32)))
    return source


def test_round_padding_reads_one_row_and_merges_only_into_an_unread_row():
    buffers = DecodeBuffers(torch, 8, 16, device="cpu", documents=True)
    buffers.synthetic(7)
    buffers.copy(rounds_chunk())
    rounds = buffers.chunk.meta["rounds"]
    assert rounds["cu_q"].tolist() == [0, 3, 5, 6, 7, 8, 8, 8, 8]
    # a padded round's one row reads one row and writes no KV
    assert rounds["used"].tolist() == [7, 11, 1, 1, 1, 1, 1, 1]
    assert buffers.dst.tolist() == [20, 21, 22, 41, 42, -1, -1, -1]
    documents = rounds["documents"]
    # padding merges into the last row, past the five real ones
    assert documents["b_index"].tolist() == [0, 1, 2, 3, 4, 7, 7, 7]
    assert documents["cu_q"].tolist() == [0, 5, 6, 7, 8, 8, 8, 8, 8]
    assert documents["used"].tolist() == [32, 0, 0, 0, 0, 0, 0, 0]
    assert documents["table"][:1, :2].tolist() == [[3, 4]]


def test_a_pass_of_rounds_replays_only_a_graph_with_a_spare_row():
    source = rounds_chunk()
    arena = SimpleNamespace(k=[], page_tokens=16)
    graphs = DecodeGraphs(SimpleNamespace(
        engine=SimpleNamespace(arena=arena), forward_chunk=lambda x: "eager"))
    graphs.storage = arena.k
    output = torch.arange(8 * 3).reshape(8, 3)
    replayed = []

    def captured(rows, documents):
        return SimpleNamespace(
            buffers=DecodeBuffers(torch, rows, 16, device="cpu",
                                  documents=documents),
            graph=SimpleNamespace(replay=lambda: replayed.append(rows)),
            output=output)

    graphs.graphs = [captured(8, False)]
    graphs.document_graphs = [captured(5, True), captured(8, True)]
    assert graphs.forward(source).equal(output[[2, 4]])
    # five rows need a six-row graph at least, so the eight-row one runs
    assert replayed == [8]


def test_buffers_of_every_bucket_are_views_of_one_set():
    base = DecodeBuffers(torch, 8, 16, device="cpu")
    small = DecodeBuffers(torch, 5, 16, device="cpu", base=base)
    small.copy(chunk())
    # the small bucket's inputs land in the base's first rows
    assert base.ids[:5].tolist() == [12, 13, 14, 15, 16]
    assert small.table.data_ptr() == base.table.data_ptr()
    reading = DecodeBuffers(torch, 5, 16, device="cpu", documents=True, base=base)
    assert reading.chunk.meta["rounds"]["documents"]["table"].data_ptr() == (
        base.document_table.data_ptr())

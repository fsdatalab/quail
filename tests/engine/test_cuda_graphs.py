"""Graph padding preserves sequence boundaries and KV ownership."""

from types import SimpleNamespace

import pytest

from quail.backends.quail.executor.cuda_graphs import (
    DecodeBuffers,
    DecodeGraphConfig,
    DecodeGraphs,
)

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
    assert buffers.cu_a[:9].tolist() == [0, 3, 5, 6, 7, 8, 8, 8, 8]
    assert buffers.cu_q[:9].tolist() == buffers.cu_a[:9].tolist()
    assert buffers.used[:8].tolist() == [4, 9, 0, 0, 0, 0, 0, 0]
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
    assert buffers.cu_a[:9].tolist() == [0, 2, 3, 4, 5, 6, 7, 8, 8]
    assert buffers.cu_q[:9].tolist() == list(range(9))
    assert buffers.used[:8].tolist() == [4, 0, 0, 0, 0, 0, 0, 0]
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
    config = DecodeGraphConfig(row_buckets=(8,))
    graphs = DecodeGraphs(SimpleNamespace(engine=SimpleNamespace(arena=arena)),
                          config=config)
    graphs.storage = arena.k
    graphs.buffers = DecodeBuffers(torch, 8, 16, config=config, device="cpu")
    output = torch.arange(8 * 3).reshape(8, 3)
    graphs.output = output
    graphs.graphs["tree"] = {8: SimpleNamespace(replay=lambda: None)}
    actual = graphs.forward(source)
    expected = output[[2, 4]].clone()
    output.zero_()
    assert actual.equal(expected)
    assert graphs.replays == 1
    graphs.invalidate()
    assert not graphs.graphs["tree"]
    assert graphs.storage is None


def test_unified_padding_reads_one_row_and_writes_no_kv():
    from quail.backends.quail.executor.chunk import Chunk

    source = Chunk(
        input_ids=torch.tensor([12, 13, 14, 15, 16]),
        positions=torch.tensor([4, 5, 6, 9, 10]), final_indices=torch.tensor([2, 4]),
        meta=dict(layer=0, canvas=None, gdn=None, reads=None, kv_src=None,
                  kv_dst=None, cu_a=torch.tensor([0, 3, 5], dtype=torch.int32),
                  max_a=3,
                  unified=dict(src=torch.arange(5),
                               dst=torch.tensor([20, 21, 22, 41, 42]),
                               tail_src=None, tail_dst=None,
                               used=torch.tensor([7, 11], dtype=torch.int32),
                               table=torch.tensor([[1], [2]], dtype=torch.int32),
                               max_used=11,
                               cu_q=torch.tensor([0, 3, 5], dtype=torch.int32),
                               max_q=3)),
        attention_mode="unified", tokens=5, layout=[])
    buffers = DecodeBuffers(torch, 8, 16, device="cpu")
    buffers.synthetic(7)
    buffers.copy(source)
    assert buffers.ids.tolist() == [12, 13, 14, 15, 16, 0, 0, 0]
    assert buffers.cu_q[:9].tolist() == [0, 3, 5, 6, 7, 8, 8, 8, 8]
    # a padded sequence reads one row of the scratch page and writes none
    assert buffers.used[:8].tolist() == [7, 11, 1, 1, 1, 1, 1, 1]
    assert buffers.dst.tolist() == [20, 21, 22, 41, 42, -1, -1, -1]
    assert buffers.table[:2, :1].equal(source.meta["unified"]["table"])
    assert buffers.table[2:8, 0].tolist() == [7] * 6
    view = buffers.view_unified(8)
    assert view.attention_mode == "unified"
    assert view.meta["unified"]["used"].data_ptr() == buffers.used.data_ptr()


def test_shared_padding_reads_nothing_and_no_row_merges_it():
    from quail.backends.quail.executor.chunk import Chunk

    # two rounds of two rows and one row: their unified sequences read
    # their own pages, and one stacked read covers their shared page
    unified = dict(src=torch.arange(3), dst=torch.tensor([40, 41, 60]),
                   tail_src=None, tail_dst=None,
                   used=torch.tensor([8, 11], dtype=torch.int32),
                   table=torch.tensor([[2], [3]], dtype=torch.int32), max_used=11,
                   cu_q=torch.tensor([0, 2, 3], dtype=torch.int32), max_q=2,
                   shared=dict(rows=torch.arange(3),
                               cu_q=torch.tensor([0, 3], dtype=torch.int32),
                               max_q=3, used=torch.tensor([16], dtype=torch.int32),
                               max_used=16,
                               table=torch.tensor([[1]], dtype=torch.int32),
                               source=torch.arange(3, dtype=torch.int32)))
    source = Chunk(
        input_ids=torch.tensor([12, 13, 14]), positions=torch.tensor([20, 21, 25]),
        final_indices=torch.tensor([0, 1, 2]),
        meta=dict(layer=0, canvas=None, gdn=None, reads=None, kv_src=None,
                  kv_dst=None, cu_a=torch.tensor([0, 2, 3], dtype=torch.int32),
                  max_a=2, unified=unified),
        attention_mode="unified", tokens=3, layout=[])
    buffers = DecodeBuffers(torch, 8, 16, device="cpu")
    buffers.synthetic(7)
    rows = 4
    buffers.copy(source, rows)
    shared = buffers.view_shared(rows).meta["unified"]["shared"]
    # the padding sequence reads no keys, and the padding row merges
    # nothing from the shared read
    assert shared["cu_q"].tolist() == [0, 3, 4, 4, 4]
    assert shared["used"].tolist() == [16, 0, 0, 0]
    assert shared["source"].tolist() == [0, 1, 2, -1]
    assert shared["table"][0, 0] == 1


class TinyPipeline:
    """Two layers whose attention mixes every real row of the pass."""

    num_pieces = 3

    def __init__(self):
        generator = torch.Generator().manual_seed(0)
        self.embed = torch.nn.Embedding(32, 4)
        self.qkv = [torch.randn(4, 6, generator=generator) for _ in range(2)]
        self.mlp = [torch.randn(4, 4, generator=generator) for _ in range(2)]
        self.attended = []

    def _before(self, index, hidden, residual):
        if residual is not None:
            residual.add_(hidden)
            hidden = residual
        return hidden @ self.qkv[index]

    def _after(self, index, hidden, residual):
        residual.add_(hidden)
        return torch.tanh(residual @ self.mlp[index])

    def piece_widths(self):
        return 4, 6

    def piece(self, index, ids, hidden, residual, qkv, out):
        if index == 0:
            embedded = self.embed(ids)
            residual.copy_(embedded)
            qkv.copy_(self._before(0, embedded, None))
            return
        after = self._after(index - 1, hidden, residual)
        if index < 2:
            qkv.copy_(self._before(index, after, residual))
        else:
            out.copy_(after)

    def attention_piece(self, index, qkv, chunk):
        self.attended.append(len(qkv))
        return torch.cumsum(qkv[:, :4], dim=0) / len(qkv)

    def final_piece(self, hidden, residual, final):
        return (hidden + residual).index_select(0, final)

    def forward_chunk(self, chunk):
        hidden = self.embed(chunk.input_ids)
        residual = None
        for index in range(2):
            qkv = self._before(index, hidden, residual)
            if residual is None:
                residual = hidden.clone()
            hidden = self._after(index, self.attention_piece(index, qkv, chunk),
                                 residual)
        return self.final_piece(hidden, residual, chunk.final_indices)


def test_piece_graphs_match_the_eager_pass_and_attend_only_real_rows():
    from functools import partial

    from quail.backends.quail.executor.cuda_graphs import PieceGraphs

    with torch.no_grad():
        pipeline = TinyPipeline()
        pieces = PieceGraphs(pipeline, DecodeGraphConfig(row_buckets=(8,)))
        pieces.buffers = (torch.zeros(8, dtype=torch.int64), torch.zeros(8, 4),
                          torch.zeros(8, 4), torch.zeros(8, 6), torch.zeros(8, 4))
        # a replay runs its piece over the bucket's padded rows
        views = pieces._views(8)
        pieces.graphs[8] = [SimpleNamespace(replay=partial(pipeline.piece, index,
                                                           *views))
                            for index in range(pipeline.num_pieces)]
        source = SimpleNamespace(
            tokens=5, input_ids=torch.tensor([3, 1, 4, 1, 5]),
            final_indices=torch.tensor([2, 4]), meta={})
        assert pieces.supports(source)
        expected = pipeline.forward_chunk(source)
        pipeline.attended.clear()
        assert torch.allclose(pieces.forward(source), expected)
        assert pipeline.attended == [5, 5]

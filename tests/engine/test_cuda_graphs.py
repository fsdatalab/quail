"""Piecewise padding never reaches attention or KV."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from quail.backends.quail.executor.cuda_graphs import (
    CapturedPieces,
    KVWriteBuffers,
    PiecewiseBuffers,
    PiecewiseGraphs,
)
from quail.cost.budgets import graph_memory_bytes, graph_token_buckets
from quail.specs import H100_SXM, QWEN3_4B_FP8, QWEN3_5_4B_BF16, QWEN3_32B_FP8

torch = pytest.importorskip("torch")


def chunk(num_tokens=5):
    return SimpleNamespace(
        input_ids=torch.arange(12, 12 + num_tokens),
        positions=torch.arange(num_tokens), tokens=num_tokens,
        final_indices=torch.tensor([num_tokens - 1]), attention_mode="tree",
        meta={"layer": 0, "kv_src": torch.arange(num_tokens),
              "kv_dst": torch.arange(num_tokens), "unified": None})


def pipeline():
    return SimpleNamespace(
        engine=SimpleNamespace(
            torch=torch, num_q_heads=2, num_kv_heads=1, head_dim=2,
            arena=SimpleNamespace(k=[], page_tokens=16, has_sliding=False),
            quant=lambda q: (q.clone(), torch.ones(q.shape[0], 1))),
        final_norm=SimpleNamespace(weight=torch.zeros(4)), forward_hook=None,
        forward_chunk=lambda chunk: "eager")


@pytest.mark.parametrize("model", [QWEN3_4B_FP8, QWEN3_32B_FP8])
def test_dense_qwen3_captures_the_stock_vllm_sizes_with_padding_under_256(model):
    buckets = graph_token_buckets(model, H100_SXM)
    assert len(buckets) == 81
    assert buckets[0] == 1
    assert buckets[-1] == 8_192
    for n in range(1, buckets[-1] + 1):
        assert next(b for b in buckets if b >= n) - n < 256
    assert graph_memory_bytes(model, H100_SXM) == (
        2 * buckets[-1] * model.act_per_token)


def test_hybrid_models_keep_the_eager_path_and_no_graph_reservation():
    assert PiecewiseGraphs.for_specs(None, QWEN3_5_4B_BF16, H100_SXM) is None
    assert graph_memory_bytes(QWEN3_5_4B_BF16, H100_SXM) == 0


@pytest.mark.parametrize("mode", ["tree", "unified"])
def test_padding_never_reaches_attention_or_writes_kv(mode):
    model = pipeline()
    buffers = PiecewiseBuffers(model, 8, device="cpu")
    source = chunk()
    source.attention_mode = mode
    buffers.copy(source)
    assert buffers.ids.tolist() == [12, 13, 14, 15, 16, 0, 0, 0]
    def attention(q, k, v, received, *, output):
        assert received is source
        assert q.shape[0] == k.shape[0] == v.shape[0] == source.tokens
        assert output[0].data_ptr() == buffers.o_in.data_ptr()
        assert output[1].data_ptr() == buffers.o_scale.data_ptr()
        output[0].fill_(7)
        output[1].fill_(2)

    model.engine.attention = attention
    buffers.attention(model, source)
    assert (buffers.o_in[:5] == 7).all()
    assert (buffers.o_in[5:] == 0).all()
    assert buffers.q.data_ptr() != buffers.o_in.data_ptr()
    source = chunk(2)
    buffers.copy(source)
    assert buffers.ids.tolist() == [12, 13, 0, 0, 0, 0, 0, 0]
    assert buffers.positions.tolist() == [0, 1, 0, 0, 0, 0, 0, 0]
    buffers.attention(model, source)


def test_smaller_buckets_share_the_largest_buckets_rows_but_not_its_scales():
    model = pipeline()
    largest = PiecewiseBuffers(model, 8, device="cpu")
    small = PiecewiseBuffers(model, 3, device="cpu", storage=largest)
    for name in ("ids", "positions", "residual", "output", "q", "k", "v", "o_in"):
        assert getattr(small, name).data_ptr() == getattr(largest, name).data_ptr()
        assert getattr(small, name).shape[0] == 3
    for name, mapping in small.writes.meta.items():
        assert mapping.data_ptr() == largest.writes.meta[name].data_ptr()
    assert small.o_scale.data_ptr() != largest.o_scale.data_ptr()
    small.copy(chunk(2))
    assert largest.ids[:3].tolist() == [12, 13, 0]
    with pytest.raises(ValueError, match="at least num_tokens"):
        PiecewiseBuffers(model, 9, device="cpu", storage=largest)


def test_captured_kv_mappings_mask_padding_and_keep_partial_page_order():
    from quail.backends.quail.executor.attention import Engine

    engine = Engine.__new__(Engine)
    keys = [torch.zeros(16, 1, 2), torch.zeros(16, 1, 2)]
    values = [torch.zeros_like(k) for k in keys]
    engine.arena = SimpleNamespace(
        sliding_layers={1}, layer_kv=lambda layer: (keys[layer], values[layer]))

    def scatter(k, v, src, dst, layer):
        live = dst >= 0
        keys[layer][dst[live]] = k[src[live]]
        values[layer][dst[live]] = v[src[live]]

    engine.kv_row_scatter = scatter
    full = KVWriteBuffers(torch, 3, 16, device="cpu")
    sliding = KVWriteBuffers(torch, 3, 16, device="cpu")
    mapping = dict(src=torch.tensor([0]), dst=torch.tensor([2]),
                   tail_src=torch.tensor([2]), tail_dst=torch.tensor([7]))
    full.copy(mapping)
    sliding.copy(dict(mapping, dst=torch.tensor([4]), tail_src=torch.tensor([4])))
    meta = dict(layer=0, unified=dict(full.meta, sliding=sliding.meta))
    k, v = torch.full((3, 1, 2), 5.0), torch.full((3, 1, 2), 9.0)
    for layer, row in ((0, 2), (1, 4)):
        meta["layer"] = layer
        engine.write_kv(k, v, meta)
        assert (keys[layer][[row, 7]] == 5).all()
        assert (values[layer][[row, 7]] == 9).all()
        assert keys[layer].count_nonzero() == 4
    meta["layer"] = 0
    keys[0].zero_()
    engine.write_kv(k, v, meta, copy_tail=False)
    assert (keys[0][2] == 5).all()
    assert not keys[0][7].count_nonzero()
    full.copy(None)
    assert not full.has_tails
    assert (full.meta["dst"] == -1).all()
    assert (full.meta["tail_dst"] == -1).all()
    assert not full.supports(dict(mapping, tail_dst=torch.arange(49)))
    meta.update(layer=1, kv_written=True)
    keys[1].zero_()
    engine.write_kv(k, v, meta)
    assert not keys[1].count_nonzero()


@pytest.mark.parametrize("unsupported", ["large", "empty", "disabled", "hook",
                                          "canvas", "uncaptured", "arena", "writes"])
def test_unsupported_batches_fall_back(unsupported):
    model = pipeline()
    graphs = PiecewiseGraphs(model, token_buckets=(8,), memory_bytes=1024)
    graphs.storage = model.engine.arena.k
    graphs.graphs = [CapturedPieces(PiecewiseBuffers(model, 8, device="cpu"), [])]
    source = chunk()
    assert graphs.supports(source)
    match unsupported:
        case "large":
            source.tokens = 9
        case "empty":
            source.tokens = 0
        case "disabled":
            graphs.enabled = False
        case "hook":
            model.forward_hook = lambda: None
        case "canvas":
            source.meta["canvas"] = object()
        case "uncaptured":
            graphs.graphs.clear()
        case "arena":
            model.engine.arena.k = []
        case "writes":
            source.meta["kv_dst"] = torch.arange(9)
    assert graphs.forward(source) == "eager"
    assert graphs.replays == 0


def test_shared_reads_and_prefill_chunks_are_supported():
    model = pipeline()
    graphs = PiecewiseGraphs(model, token_buckets=(8,), memory_bytes=1024)
    graphs.storage = model.engine.arena.k
    graphs.graphs = [CapturedPieces(PiecewiseBuffers(model, 8, device="cpu"), [])]
    source = chunk()
    source.meta.update(max_a=1000, reads={"nodes": object(), "max_used": 100000})
    assert graphs.supports(source)
    source.attention_mode = "unified"
    assert graphs.supports(source)


def test_replay_returns_independent_output_storage():
    model = pipeline()
    buffers = PiecewiseBuffers(model, 8, device="cpu")
    buffers.output.copy_(torch.arange(32).reshape(8, 4))
    model.engine.attention = lambda q, k, v, chunk, **kw: None
    graphs = PiecewiseGraphs(model, token_buckets=(8,), memory_bytes=1024)
    graphs.storage = model.engine.arena.k
    graphs.graphs = [CapturedPieces(buffers, [
        SimpleNamespace(replay=lambda: None), SimpleNamespace(replay=lambda: None)])]
    actual = graphs.forward(chunk())
    expected = buffers.output[[4]].clone()
    buffers.output.zero_()
    assert actual.equal(expected)
    assert graphs.replays == graphs.passes == 1


def test_passes_without_partial_pages_skip_the_tail_copy_graphs():
    model = pipeline()
    buffers = PiecewiseBuffers(model, 8, device="cpu")
    buffers.output.zero_()
    calls = []
    captured = CapturedPieces(
        buffers, [SimpleNamespace(replay=lambda: calls.append("no_tail"))],
        [SimpleNamespace(replay=lambda: calls.append("tail"))])
    source = chunk()
    source.attention_mode = "unified"
    captured.forward(model, source)
    source.meta["unified"] = dict(
        src=None, dst=None, tail_src=torch.tensor([0]), tail_dst=torch.tensor([3]))
    captured.forward(model, source)
    source.meta["unified"] = None
    captured.forward(model, source)
    assert calls == ["no_tail", "tail", "no_tail"]
    assert "kv_written" not in source.meta


@pytest.mark.parametrize("memory_bytes", [159, 160])
def test_capture_accounts_for_inactive_private_pool_storage(monkeypatch, memory_bytes):
    from quail.backends.quail.executor import cuda_graphs

    pool = (0, 1)
    allocations = iter([100, 200])
    stream = SimpleNamespace(wait_stream=lambda _: None, synchronize=lambda: None)
    cuda = SimpleNamespace(
        memory_allocated=lambda: next(allocations), graph_pool_handle=lambda: pool,
        Stream=lambda: stream, current_stream=lambda: stream,
        stream=lambda _: nullcontext(), CUDAGraph=object,
        graph=lambda *a, **kw: nullcontext(),
        memory_snapshot=lambda: [
            dict(segment_pool_id=pool, total_size=80, allocated_size=20),
            dict(segment_pool_id=(0, 0), total_size=1000, allocated_size=0)])
    model = SimpleNamespace(
        engine=SimpleNamespace(arena=SimpleNamespace(k=[]), torch=SimpleNamespace(
            cuda=cuda, inference_mode=nullcontext)), layers=[None])
    monkeypatch.setattr(
        cuda_graphs, "PiecewiseBuffers",
        lambda *a, **kw: SimpleNamespace(piece=lambda *a, **kw: None))
    graphs = PiecewiseGraphs(model, token_buckets=(1, 2), memory_bytes=memory_bytes)
    if memory_bytes < 160:
        with pytest.raises(RuntimeError, match="exceeding"):
            graphs.prepare()
        assert not graphs.graphs
    else:
        assert graphs.prepare()
        assert not graphs.prepare()
    assert graphs.captured_bytes == 160


@pytest.mark.parametrize("buckets", [(), [1], (0,), (2, 1), (1, 1), (1.5,)])
def test_invalid_buckets_are_rejected(buckets):
    with pytest.raises(ValueError, match="increasing positive integers"):
        PiecewiseGraphs(None, token_buckets=buckets, memory_bytes=0)

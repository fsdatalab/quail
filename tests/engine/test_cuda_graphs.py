"""Piecewise padding never reaches attention or KV."""

from types import SimpleNamespace

import pytest

from quail.backends.quail.executor.cuda_graphs import (
    CapturedPieces,
    PiecewiseBuffers,
    PiecewiseGraphs,
)
from quail.cost.budgets import compute_knee, graph_memory_bytes, graph_token_buckets
from quail.specs import H100_SXM, QWEN3_4B_FP8, QWEN3_5_4B_BF16, QWEN3_32B_FP8

torch = pytest.importorskip("torch")


def chunk(num_tokens=5):
    return SimpleNamespace(
        input_ids=torch.arange(12, 12 + num_tokens),
        positions=torch.arange(num_tokens), tokens=num_tokens,
        final_indices=torch.tensor([num_tokens - 1]), attention_mode="tree",
        meta={"layer": 0, "kv_dst": torch.arange(num_tokens)})


def pipeline():
    return SimpleNamespace(
        engine=SimpleNamespace(
            torch=torch, num_q_heads=2, num_kv_heads=1, head_dim=2,
            quant=lambda q: (q.clone(), torch.ones(q.shape[0], 1))),
        final_norm=SimpleNamespace(weight=torch.zeros(4)), forward_hook=None,
        forward_chunk=lambda chunk: "eager")


@pytest.mark.parametrize("model", [QWEN3_4B_FP8, QWEN3_32B_FP8])
def test_buckets_bound_padding_and_stop_at_twice_the_compute_ridge(model):
    buckets = graph_token_buckets(model, H100_SXM)
    assert buckets[0] == 1
    assert buckets[-1] - 1 < 2 * compute_knee(model, H100_SXM) <= buckets[-1]
    for n in range(1, buckets[-1] + 1):
        bucket = next(b for b in buckets if b >= n)
        assert bucket <= (3 * n + 1) // 2
    assert graph_memory_bytes(model, H100_SXM) == (
        sum(buckets) + buckets[-1]) * model.act_per_token


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
    kv = torch.zeros(8, 2, dtype=torch.bfloat16)

    def attention(q, k, v, received):
        assert received is source
        assert q.shape[0] == k.shape[0] == v.shape[0] == source.tokens
        kv[received.meta["kv_dst"]] = v
        return q, torch.ones(source.tokens, 1)

    model.engine.attention = attention
    buffers.v.fill_(7)
    buffers.attention(model, source)
    assert (kv[:5] == 7).all()
    assert (kv[5:] == 0).all()
    source = chunk(2)
    buffers.copy(source)
    assert buffers.ids.tolist() == [12, 13, 0, 0, 0, 0, 0, 0]
    assert buffers.positions.tolist() == [0, 1, 0, 0, 0, 0, 0, 0]
    buffers.attention(model, source)


@pytest.mark.parametrize("unsupported", ["large", "empty", "disabled", "hook",
                                          "canvas", "uncaptured"])
def test_unsupported_batches_fall_back(unsupported):
    model = pipeline()
    graphs = PiecewiseGraphs(model, token_buckets=(8,), memory_bytes=1024)
    graphs.graphs = [object()]
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
    assert graphs.forward(source) == "eager"
    assert graphs.replays == 0


def test_shared_reads_and_prefill_chunks_are_supported():
    model = pipeline()
    graphs = PiecewiseGraphs(model, token_buckets=(8,), memory_bytes=1024)
    graphs.graphs = [object()]
    source = chunk()
    source.meta.update(max_a=1000, reads={"nodes": object(), "max_used": 100000})
    assert graphs.supports(source)
    source.attention_mode = "unified"
    assert graphs.supports(source)


def test_replay_returns_independent_output_storage():
    model = pipeline()
    buffers = PiecewiseBuffers(model, 8, device="cpu")
    buffers.output.copy_(torch.arange(32).reshape(8, 4))
    model.engine.attention = lambda q, k, v, chunk: (q, torch.ones(len(q), 1))
    graphs = PiecewiseGraphs(model, token_buckets=(8,), memory_bytes=1024)
    graphs.graphs = [CapturedPieces(buffers, [
        SimpleNamespace(replay=lambda: None), SimpleNamespace(replay=lambda: None)])]
    actual = graphs.forward(chunk())
    expected = buffers.output[[4]].clone()
    buffers.output.zero_()
    assert actual.equal(expected)
    assert graphs.replays == graphs.passes == 1


@pytest.mark.parametrize("buckets", [(), [1], (0,), (2, 1), (1, 1), (1.5,)])
def test_invalid_buckets_are_rejected(buckets):
    with pytest.raises(ValueError, match="increasing positive integers"):
        PiecewiseGraphs(None, token_buckets=buckets, memory_bytes=0)

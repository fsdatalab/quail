"""KV scatter and paged attention over dense and mixed layer layouts."""

import math

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a CUDA device")


def test_kv_storage_scatter_and_attention():
    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.attention import Engine

    torch.manual_seed(17)
    shapes = [(2, 128), (1, 512), (2, 256)]
    arena = KVArena(
        n_layers=3, n_pages=4, page_tokens=16, n_kv=2, d_head=128,
        layer_kv=shapes, sliding_layers=(2,), sliding_window=8,
        n_sliding_pages=2)
    expected = []
    engines = []
    for layer, (heads, dim) in enumerate(shapes):
        engine = Engine(arena, n_q=heads * 2, n_kv=heads, head_dim=dim,
                        rotary=None, fp8=False)
        engines.append(engine)
        rows = 32 if layer == 2 else 64
        key = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
        value = torch.randn_like(key)
        source = torch.arange(rows, device="cuda", dtype=torch.int32)
        destination = source.flip(0)
        engine.kv_row_scatter(key, value, source, destination, layer)
        expected.append((key.flip(0), value.flip(0)))

    cu_q = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
    used = torch.tensor([23], device="cuda", dtype=torch.int32)
    table = torch.tensor([[1, 0]], device="cuda", dtype=torch.int32)
    logical_rows = torch.cat((torch.arange(16, 32, device="cuda"),
                              torch.arange(7, device="cuda")))
    query_positions = torch.arange(19, 23, device="cuda")[:, None]
    key_positions = torch.arange(23, device="cuda")[None, :]
    for layer, ((heads, dim), engine, (key, value)) in enumerate(
            zip(shapes, engines, expected)):
        stored_key, stored_value = arena.layer_kv(layer)
        torch.testing.assert_close(stored_key, key, rtol=0, atol=0)
        torch.testing.assert_close(stored_value, value, rtol=0, atol=0)
        query = torch.randn(4, heads * 2, dim, device="cuda", dtype=torch.bfloat16)
        window = (7, 0) if layer == 2 else None
        actual = engine._paged(
            query, *arena.paged_kv(layer), cu_q, 4, used, 23, table,
            causal=True, window=window)
        keys = key.index_select(0, logical_rows).repeat_interleave(2, dim=1)
        values = value.index_select(0, logical_rows).repeat_interleave(2, dim=1)
        logits = (query.float().transpose(0, 1)
                  @ keys.float().permute(1, 2, 0)) / math.sqrt(dim)
        mask = key_positions <= query_positions
        if window is not None:
            mask &= key_positions >= query_positions - window[0]
        logits.masked_fill_(~mask, float("-inf"))
        reference = (logits.softmax(-1)
                     @ values.float().transpose(0, 1)).transpose(0, 1)
        torch.testing.assert_close(actual.float(), reference, rtol=1e-2, atol=1e-2)

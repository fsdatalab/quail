"""KV tensor layouts and allocation failure behavior on CPU."""

import gc
import math
import weakref

import pytest

from quail.backends.quail.executor.arena import KVArena


@pytest.mark.parametrize("resize", [False, True])
def test_allocation_failure_keeps_no_partial_storage(monkeypatch, resize):
    torch = pytest.importorskip("torch")
    empty = torch.empty
    allocations = []
    capacity = 128

    def allocate(size, **kwargs):
        elements = size if isinstance(size, int) else math.prod(size)
        used = sum(reference().numel() for reference in allocations
                   if reference() is not None)
        if used + elements > capacity:
            raise MemoryError("allocation budget exceeded")
        tensor = empty(size, **kwargs)
        allocations.append(weakref.ref(tensor))
        return tensor

    monkeypatch.setattr(torch, "empty", allocate)

    def arena(pages):
        return KVArena(n_layers=2, n_pages=pages, page_tokens=4,
                       n_kv=2, d_head=4, device="cpu")

    if resize:
        original = arena(1)
    with pytest.raises(MemoryError, match="allocation budget exceeded") as saved:
        if resize:
            original.resize(2)
        else:
            arena(2)
    gc.collect()
    assert saved.value.__traceback__ is not None
    assert all(reference() is None for reference in allocations)

    capacity = 256
    replacement = arena(2)
    for layer in range(2):
        key, value = replacement.layer_kv(layer)
        assert key.shape == value.shape == (8, 2, 4)


@pytest.mark.parametrize("sliding", [False, True])
def test_layer_views_are_contiguous_disjoint_and_resize(sliding):
    torch = pytest.importorskip("torch")
    shapes = [(2, 8), (1, 4), (3, 8)]
    arena = KVArena(
        n_layers=3, n_pages=3, page_tokens=4, n_kv=2, d_head=8,
        layer_kv=shapes, device="cpu", sliding_layers=(1,) if sliding else (),
        sliding_window=8 if sliding else 0, n_sliding_pages=2 if sliding else 0)

    for pages, sliding_pages in ((3, 2 if sliding else 0),
                                 (5, 3 if sliding else 0)):
        arena.resize(pages, sliding_pages)
        for layer in range(3):
            key, value = arena.layer_kv(layer)
            key.fill_(10 + layer)
            value.fill_(20 + layer)
        for layer, (heads, dim) in enumerate(shapes):
            key, value = arena.layer_kv(layer)
            count = sliding_pages if sliding and layer == 1 else pages
            assert key.shape == value.shape == (count * 4, heads, dim)
            assert key.is_contiguous() and value.is_contiguous()
            assert torch.all(key == 10 + layer)
            assert torch.all(value == 20 + layer)
            paged_key, paged_value = arena.paged_kv(layer)
            assert paged_key.shape == paged_value.shape == (count, 4, heads, dim)
            paged_key[-1, -1, -1, -1] = 50 + layer
            assert key[-1, -1, -1] == 50 + layer
            assert value[-1, -1, -1] == 20 + layer

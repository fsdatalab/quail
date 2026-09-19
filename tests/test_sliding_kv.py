"""The sliding-layer KV pool: window origins, trimming, and one page currency."""

import numpy as np
import pytest

from quail.backends.quail.executor import loop
from quail.backends.quail.executor.arena import KVArena
from quail.backends.quail.executor.pack import FilterAdmission
from quail.cost import budgets
from quail.specs import DIFFUSION_GEMMA_26B_FP8, H100_SXM, QWEN3_4B_FP8

torch = pytest.importorskip("torch")

WINDOW = 32     # tokens, two pages


def cpu_arena(pages=64, sliding_pages=16):
    # layer 0 keeps every token, layer 1 slides
    return KVArena(n_layers=2, n_pages=pages, page_tokens=16, n_kv=1, d_head=2,
                   dtype=torch.float32, device="cpu",
                   layer_kv=[(1, 2), (1, 2)], sliding_layers=(1,),
                   sliding_window=WINDOW, n_sliding_pages=sliding_pages)


def test_spec_names_the_sliding_layers():
    spec = DIFFUSION_GEMMA_26B_FP8
    assert spec.sliding_window == 1024
    assert len(spec.sliding_layer_set) == 25
    assert 5 not in spec.sliding_layer_set and 6 in spec.sliding_layer_set
    assert spec.kappa_sliding == 25 * 2 * 8 * 256 * 2
    assert spec.kappa_full == 5 * 2 * 2 * 512 * 2
    assert QWEN3_4B_FP8.sliding_layer_set == frozenset()
    assert QWEN3_4B_FP8.kappa_sliding == 0


def test_arena_pages_split_follows_document_length():
    full_short, sliding_short = budgets.arena_pages(
        DIFFUSION_GEMMA_26B_FP8, H100_SXM, mean_doc_tokens=300)
    full_long, sliding_long = budgets.arena_pages(
        DIFFUSION_GEMMA_26B_FP8, H100_SXM, mean_doc_tokens=10_697)
    # short documents keep every token on both pools: equal pages
    assert full_short == sliding_short
    # long documents keep one window per document on the sliding pool
    assert full_long > 5 * full_short
    assert sliding_long < sliding_short
    assert sliding_long >= budgets.transient_sliding_pages(
        budgets.chunk_budget(DIFFUSION_GEMMA_26B_FP8, H100_SXM))
    assert budgets.arena_pages(QWEN3_4B_FP8, H100_SXM)[1] == 0


def test_window_origin_and_trim():
    arena = cpu_arena()
    assert arena.has_sliding
    assert arena.origin(20) == 0
    assert arena.origin(WINDOW) == 0
    assert arena.origin(100) == 64      # 100 - 32 = 68, down to a page
    key = ("d", 0)
    # a fresh 100-token document with 20 rows of room takes 8 pages
    # on both pools until its pass is done
    assert arena.activate(key, 100, capacity_tokens=120, base_tokens=100)
    assert len(arena.owned_pages(key)) == 8
    assert len(arena.owned_sliding_pages(key)) == 8
    assert arena.sliding_start(key) == 0
    assert arena.free_pages == min(64 - 8, int((16 - 8) * 4))
    freed = arena.trim_window(key)
    # pages before row 64 leave the sliding pool: 4 of them
    assert len(arena.owned_sliding_pages(key)) == 4
    assert arena.sliding_start(key) == 64
    assert freed == arena.free_pages - min(64 - 8, (16 - 8) * 4)
    assert arena.trim_window(key) == 0
    # rows of the sliding pool start at the origin
    rows = arena.capacity_rows_sliding(key)
    assert rows.numel() == 4 * 16
    # the every-token pool still holds all 8 pages
    assert arena.capacity_rows(key).numel() == 8 * 16
    # the key's later stages extend both pools from their own ends
    assert arena.activate(key, 110, capacity_tokens=140) is not None
    assert len(arena.owned_pages(key)) == 9
    assert len(arena.owned_sliding_pages(key)) == 5     # (140 - 64) / 16
    # retaining below the base would drop window rows
    with pytest.raises(ValueError, match="window"):
        arena.retain(key, 90)
    arena.retain(key, 100)
    assert len(arena.owned_pages(key)) == 7
    assert len(arena.owned_sliding_pages(key)) == 3     # (100 - 64) / 16
    arena.free_key(key)
    assert arena.free_pages == 64


def test_page_costs_use_the_tighter_pool():
    arena = cpu_arena(pages=64, sliding_pages=16)     # ratio 4
    # a short document is priced by the sliding pool: 2 pages x 4
    assert arena.page_cost(20) == 8
    assert arena.page_cost(20, 20) == 8
    # a long document untrimmed: 7 pages x 4; trimmed at origin 64:
    # (100 - 64) rows = 3 sliding pages x 4 = 12, above its 7 full pages
    assert arena.page_cost(100) == 28
    assert arena.page_cost(100, 100) == 12
    key = ("d", 1)
    arena.activate(key, 100, base_tokens=100)
    assert arena.held_cost(key) == 28
    arena.trim_window(key)
    assert arena.held_cost(key) == 12
    plain = KVArena(n_layers=1, n_pages=8, page_tokens=16, n_kv=1, d_head=2,
                    dtype=torch.float32, device="cpu")
    assert not plain.has_sliding
    assert plain.page_cost(100) == 7 and plain.page_cost(100, 100) == 7
    assert plain.trim_window is not None


def test_temporaries_and_resize():
    arena = cpu_arena(pages=64, sliding_pages=16)
    temp, pages = arena.alloc_temporary(40, sliding_tokens=20)
    assert len(pages) == 3
    assert len(arena.owned_sliding_pages(temp)) == 2
    arena.free_key(temp)
    with pytest.raises(RuntimeError, match="holds keys"):
        arena.activate(("d", 2), 10)
        arena.resize(32, 8)
    arena.free_key(("d", 2))
    arena.resize(32, 8)
    assert arena.n_pages == 32 and arena.n_sliding_pages == 8
    assert arena.k[1].shape[0] == 8 * 16 and arena.k[0].shape[0] == 32 * 16
    arena.resize(32, 8)     # a no-op at the same sizes


def test_filter_admission_credits_a_trim():
    sched = FilterAdmission([100], [10], 200, arena_pages=64, page_tokens=16,
                            page_cost=lambda tokens, base=None: (
                                28 if base is None else 12))
    assert sched.next_chunk() == [(0, 0, True)]
    assert sched.free_pages == 64 - 28
    sched.trim(0, 16)
    assert sched.free_pages == 64 - 12
    assert sched.resident[0] == 12
    with pytest.raises(ValueError):
        sched.trim(0, -1)
    # a kept survivor holds its trimmed cost: nothing more comes back
    sched.report(0, 0, True, release=False)
    assert sched.free_pages == 52


def _cpu_staging(monkeypatch):
    def staged(torch_, data, dtype, pinned=True):
        if isinstance(data, np.ndarray) or torch.is_tensor(data):
            return torch.as_tensor(data, dtype=dtype)
        return torch.tensor(data, dtype=dtype)

    def token_parts(torch_, sequences, total, pinned=True, staging=None):
        ids = [int(t) for seq in sequences for part in loop._token_parts(seq)
               for t in part]
        return torch.tensor(ids, dtype=torch.int64)

    monkeypatch.setattr(loop, "_staged", staged)
    monkeypatch.setattr(loop, "_staged_token_parts", token_parts)


def test_pack_chunk_builds_both_pools(monkeypatch):
    _cpu_staging(monkeypatch)
    arena = cpu_arena(pages=64, sliding_pages=32)
    key = ("d", 0)
    doc = list(range(100))
    tail = [500, 501]
    canvas = (900, 901)
    # the fresh pass: every row lands in both pools
    arena.activate(key, 100, capacity_tokens=120, base_tokens=100)
    chunk = loop.pack_chunk(
        torch, arena, [dict(key=key, prefix=doc, f=100, suffixes=[tail])],
        attention_mode="unified", canvas=canvas)
    assert chunk.fresh_keys == (key,)
    full = chunk.meta["unified"]
    sliding = full["sliding"]
    assert full["src"].tolist() == list(range(104))
    assert sliding["src"].tolist() == list(range(104))
    assert full["used"].tolist() == [104] and sliding["used"].tolist() == [104]
    assert full["table"].shape == (1, 8) and sliding["table"].shape == (1, 8)
    assert chunk.meta["canvas"]["sliding"]["used"].tolist() == [104]
    arena.trim_window(key)
    # a later stage reads the kept rows: the sliding pool from row 64
    chunk = loop.pack_chunk(
        torch, arena, [dict(key=key, prefix=None, f=100, suffixes=[[600]])],
        attention_mode="unified", canvas=canvas)
    assert chunk.fresh_keys == ()
    full = chunk.meta["unified"]
    sliding = full["sliding"]
    assert full["used"].tolist() == [103]
    assert sliding["used"].tolist() == [103 - 64]
    assert sliding["table"].shape[1] == 4
    # the suffix rows go to the key's own pages after row 100 in the
    # every-token pool and after row 36 in the sliding pool
    assert full["dst"].tolist() == arena.capacity_rows(key)[100:103].tolist()
    assert sliding["dst"].tolist() == \
        arena.capacity_rows_sliding(key)[36:39].tolist()
    assert chunk.meta["canvas"]["sliding"]["used"].tolist() == [39]
    # two suffixes take temporaries in both pools, after a copy of
    # each pool's partial last page
    chunk = loop.pack_chunk(
        torch, arena, [dict(key=key, prefix=None, f=100,
                            suffixes=[[600], [601, 602]])],
        attention_mode="unified")
    full = chunk.meta["unified"]
    sliding = full["sliding"]
    assert len(chunk.temporary_keys) == 2
    assert full["used"].tolist() == [101, 102]
    assert sliding["used"].tolist() == [37, 38]
    assert full["tail_src"].numel() == 2 * (100 % 16)
    assert sliding["tail_src"].numel() == 2 * (36 % 16)
    for temp in chunk.temporary_keys:
        arena.free_key(temp)
    arena.free_key(key)
    assert arena.free_pages == 64

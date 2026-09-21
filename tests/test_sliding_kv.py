"""The sliding-layer KV pool: window origins, trimming, and one page currency."""

import pytest
from fakes import cpu_staging

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
        budgets.chunk_budget(DIFFUSION_GEMMA_26B_FP8, H100_SXM),
        DIFFUSION_GEMMA_26B_FP8.sliding_window)
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
    # without a sliding pool a trim releases nothing
    plain.activate(key, 100, base_tokens=100)
    plain.trim_window(key)
    assert plain.held_cost(key) == 7


def test_alloc_rolls_back_when_the_sliding_pool_is_short():
    # 4 sliding pages hold 64 rows; a 100-row key needs 7 in each pool
    arena = cpu_arena(pages=64, sliding_pages=4)
    free = arena.accounting.free_pages
    assert arena.alloc(("d", 0), 100) is None
    assert arena.accounting.free_pages == free
    assert not arena.accounting.owned and not arena.sliding.owned
    assert arena.alloc(("d", 1), 40) is not None


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


def _image_group(key, blocks, pages, suffixes=([500],)):
    """One fresh document: 2 preamble rows, then (start, soft.., end) per block."""
    prefix = list(range(max(block.end for block in blocks) + 1))
    return dict(key=key, prefix=prefix, f=len(prefix), suffixes=list(suffixes),
                images=tuple(zip(blocks, pages)))


def test_pack_chunk_lists_image_blocks_with_their_kv_bounds(monkeypatch):
    from quail.pdf.prompt import ImageBlock

    cpu_staging(monkeypatch)
    arena = cpu_arena(pages=64, sliding_pages=32)
    keys = [("d", 0), ("d", 1)]
    # rows [pre pre start s s s s s end start s s s end]: blocks at 3..8 and
    # 10..13, a 14-token prefix; the second document has one 3-token block
    first = _image_group(keys[0], (ImageBlock(0, 3, 5), ImageBlock(1, 10, 3)),
                         ("p0", "p1"))
    second = _image_group(keys[1], (ImageBlock(2, 3, 3),), ("p2",))
    for group in (first, second):
        arena.activate(group["key"], group["f"], capacity_tokens=group["f"] + 4,
                       base_tokens=group["f"])
    chunk = loop.pack_chunk(torch, arena, [first, second],
                            attention_mode="unified")
    rows0 = 0
    rows1 = first["f"] + 1
    assert [(i.row0, i.rows, i.page) for i in chunk.images] == [
        (rows0 + 3, 5, "p0"), (rows0 + 10, 3, "p1"), (rows1 + 3, 3, "p2")]
    blocks = chunk.meta["blocks"]
    assert blocks["rows"].tolist() == [3, 4, 5, 6, 7, 10, 11, 12,
                                       rows1 + 3, rows1 + 4, rows1 + 5]
    assert blocks["cu_q"].tolist() == [0, 5, 8, 11]
    assert blocks["max_q"] == 5
    # each block reads KV up to its own end, in both pools on a fresh pass
    assert blocks["used"].tolist() == [8, 13, 6]
    assert blocks["max_used"] == 13
    assert blocks["sliding"]["used"].tolist() == [8, 13, 6]
    # the block sequences read their document's block table
    unified = chunk.meta["unified"]
    assert blocks["table"].tolist() == [unified["table"][0].tolist()] * 2 \
        + [unified["table"][1].tolist()]
    assert blocks["sliding"]["table"].shape == (3, unified["sliding"]["table"].shape[1])
    for key in keys:
        arena.free_key(key)


def test_pack_chunk_image_blocks_follow_a_prefix_with_two_suffixes(monkeypatch):
    """With two suffixes the prefix is its own sequence; blocks read it."""
    from quail.pdf.prompt import ImageBlock

    cpu_staging(monkeypatch)
    arena = cpu_arena(pages=64, sliding_pages=32)
    key = ("d", 0)
    group = _image_group(key, (ImageBlock(0, 3, 4),), ("p0",),
                         suffixes=([500], [501, 502]))
    arena.activate(key, group["f"], capacity_tokens=group["f"] + 4,
                   base_tokens=group["f"])
    chunk = loop.pack_chunk(torch, arena, [group], attention_mode="unified")
    unified = chunk.meta["unified"]
    # sequences: the prefix, then one per suffix
    assert unified["cu_q"].tolist() == [0, group["f"], group["f"] + 1,
                                        group["f"] + 3]
    blocks = chunk.meta["blocks"]
    assert blocks["used"].tolist() == [7]
    assert blocks["table"].tolist() == [unified["table"][0].tolist()]
    for temp in chunk.temporary_keys:
        arena.free_key(temp)
    arena.free_key(key)


def test_pack_chunk_refuses_images_off_the_paged_unified_path(monkeypatch):
    from quail.pdf.prompt import ImageBlock

    cpu_staging(monkeypatch)
    arena = cpu_arena(pages=64, sliding_pages=32)
    key = ("d", 0)
    group = _image_group(key, (ImageBlock(0, 3, 4),), ("p0",))
    # no pages: an unpaged fresh group
    with pytest.raises(ValueError, match="paged unified path"):
        loop.pack_chunk(torch, arena, [group], attention_mode="unified")
    arena.activate(key, group["f"], capacity_tokens=group["f"] + 4,
                   base_tokens=group["f"])
    with pytest.raises(ValueError, match="paged unified path"):
        loop.pack_chunk(torch, arena, [group], attention_mode="merge_quant")
    kept = dict(group, prefix=None)
    with pytest.raises(ValueError, match="paged unified path"):
        loop.pack_chunk(torch, arena, [kept], attention_mode="unified")
    past = dict(group, images=((ImageBlock(0, 3, 40), "p0"),))
    with pytest.raises(ValueError, match="runs past"):
        loop.pack_chunk(torch, arena, [past], attention_mode="unified")
    arena.free_key(key)


def test_attention_unified_runs_one_block_call_per_layer(monkeypatch):
    """Block rows get a second, non-causal call bounded at each block's end."""
    from quail.backends.quail.executor.attention import Engine
    from quail.pdf.prompt import ImageBlock

    cpu_staging(monkeypatch)
    arena = cpu_arena(pages=64, sliding_pages=32)
    key = ("d", 0)
    group = _image_group(key, (ImageBlock(0, 3, 4),), ("p0",))
    arena.activate(key, group["f"], capacity_tokens=group["f"] + 4,
                   base_tokens=group["f"])
    chunk = loop.pack_chunk(torch, arena, [group], attention_mode="unified")
    engine = Engine.__new__(Engine)
    engine.arena = arena
    engine.torch = torch
    calls = []

    def paged(q3, kp, vp, cu_q, max_q, used, max_used, table, *, causal,
              softmax_scale=None, window=None):
        calls.append((q3.shape[0], causal, window, used.tolist(), max_used))
        return torch.zeros(q3.shape[0], 1, 2)

    monkeypatch.setattr(engine, "_paged", paged)
    monkeypatch.setattr(engine, "kv_row_scatter", lambda *args: None)
    rows = chunk.tokens
    q3 = torch.zeros(rows, 1, 2)
    meta = dict(chunk.meta, layer=1)
    engine.attention_unified(q3, q3, q3, meta, window=WINDOW,
                             bidirectional_blocks=chunk.meta["blocks"])
    assert calls == [
        (rows, True, (WINDOW - 1, 0), [rows], rows),
        (4, False, (WINDOW - 1, -1), [7], 7)]
    # a layer handed no blocks runs the causal call alone
    calls.clear()
    engine.attention_unified(q3, q3, q3, dict(chunk.meta, layer=0))
    assert [call[1] for call in calls] == [True]
    arena.free_key(key)


def test_pack_chunk_builds_both_pools(monkeypatch):
    cpu_staging(monkeypatch)
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

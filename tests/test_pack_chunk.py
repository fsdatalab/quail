"""Test packed token rows, positions, and answer rows from flattened suffixes."""

import pytest
from fakes import cpu_staging
from test_sliding_kv import plain_arena

from quail.backends.quail.executor import chunk as chunk_mod
from quail.backends.quail.executor.chunk import Suffixes
from quail.cost.classify import estimate
from quail.specs import H100_SXM, QWEN3_4B_FP8

torch = pytest.importorskip("torch")


def test_pack_chunk_rows_for_many_suffixes(monkeypatch):
    cpu_staging(monkeypatch)
    arena = plain_arena()
    key = ("d", 0)
    arena.activate(key, 4, capacity_tokens=8, base_tokens=4)
    sufs = Suffixes.of([[10, 11], [12], [13, 14, 15]])
    chunk = chunk_mod.pack_chunk(
        torch, arena,
        [dict(key=key, prefix=[1, 2, 3, 4], f=4, suffixes=sufs,
              read_all_rows=True)],
        attention_mode="tree")
    assert chunk.input_ids.tolist() == [1, 2, 3, 4, 10, 11, 12, 13, 14, 15]
    # each suffix's positions restart after the prefix
    assert chunk.positions.tolist() == [0, 1, 2, 3, 4, 5, 4, 4, 5, 6]
    assert chunk.final_indices.tolist() == [4, 5, 6, 7, 8, 9]
    assert chunk.rows_per_answer == (2, 1, 3)
    assert chunk.meta["cu_a"].tolist() == [0, 4, 6, 7, 10]
    assert chunk.meta["max_a"] == 4
    assert chunk.layout == [(key, 3)]
    assert chunk.fresh_keys == (key,)
    # the prefix rows are written to the key's pages
    assert chunk.meta["kv_src"].tolist() == [0, 1, 2, 3]
    # a kept document: the last row of each suffix answers, and every
    # suffix row reads the kept prefix
    chunk = chunk_mod.pack_chunk(
        torch, arena, [dict(key=key, prefix=None, f=4, suffixes=sufs)],
        attention_mode="tree")
    assert chunk.input_ids.tolist() == [10, 11, 12, 13, 14, 15]
    assert chunk.final_indices.tolist() == [1, 2, 5]
    assert chunk.rows_per_answer == ()
    assert chunk.meta["cu_a"].tolist() == [0, 2, 3, 6]
    reads = chunk.meta["reads"]
    assert reads["rows"].tolist() == [0, 1, 2, 3, 4, 5]
    assert reads["cu_q"].tolist() == [0, 6]
    assert reads["used"].tolist() == [4]
    assert reads["source"].tolist() == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize("mode", ["tree", "unified"])
def test_decode_rounds_keep_each_fed_token_after_the_frame(monkeypatch, mode):
    cpu_staging(monkeypatch)
    arena = plain_arena()
    key = ("d", 3)
    # a three-token document, a two-token frame, the cue, and room for
    # two decoded tokens: 3 + 2 + 1 + 2 = 8 rows reserved
    arena.activate(key, 3, capacity_tokens=8, base_tokens=3)
    rows = arena.capacity_rows(key)

    def written(chunk):
        if mode == "tree":
            return chunk.meta["kv_dst"].tolist()
        return chunk.meta["unified"]["dst"].tolist()

    # round 0 packs the document, frame, and cue and keeps all six rows
    chunk = chunk_mod.pack_chunk(
        torch, arena, [dict(key=key, prefix=[7, 8, 9], f=3,
                            suffixes=[[91, 92, 93]], write_suffix_tokens=3,
                            single=True)], attention_mode=mode)
    assert written(chunk) == rows[:6].tolist()
    if mode == "tree":
        assert chunk.meta["cu_a"].tolist() == [0, 6]
        assert chunk.meta["reads"] is None
    # round 1 feeds one token at position 6, reads the six kept rows,
    # and keeps it at row 6
    chunk = chunk_mod.pack_chunk(
        torch, arena, [dict(key=key, prefix=None, f=6, suffixes=[[40]],
                            write_suffix_tokens=1, single=True)],
        attention_mode=mode)
    assert chunk.positions.tolist() == [6]
    assert written(chunk) == rows[6:7].tolist()
    if mode == "tree":
        assert chunk.meta["reads"]["used"].tolist() == [6]
    # a kept token past the key's one 16-row page is refused, not dropped
    with pytest.raises((AssertionError, ValueError)):
        chunk_mod.pack_chunk(
            torch, arena, [dict(key=key, prefix=None, f=16, suffixes=[[41]],
                                write_suffix_tokens=1, single=True)],
            attention_mode=mode)


@pytest.mark.parametrize("mode", ["tree", "unified"])
@pytest.mark.parametrize("shared", [0, 16])
@pytest.mark.parametrize("run", [1, 2])
def test_decode_cost_matches_packed_retained_kv(monkeypatch, mode, shared, run):
    cpu_staging(monkeypatch)
    arena = plain_arena()
    parent, key = ("d", 0), ("d", 1)
    arena.activate(parent, 32, capacity_tokens=40, base_tokens=32)
    arena.activate(key, 32, capacity_tokens=40, base_tokens=32,
                   borrow=(parent, shared) if shared else None)
    retained_reads = fresh_tokens = 0
    for round_ in range(2):
        first = round_ == 0
        chunk = chunk_mod.pack_chunk(
            torch, arena, [dict(
                key=key, prefix=list(range(shared, 32)) if first else None,
                f=32 if first else 35, start=shared if first else 0,
                read_key=parent if first and shared else None,
                suffixes=[[91, 92, 93]] if first else [[40, 41][:run]],
                write_suffix_tokens=3 if first else run, single=True)],
            attention_mode=mode)
        if mode == "tree":
            reads = chunk.meta["reads"]
            retained = 0 if reads is None else int(reads["used"].sum())
        else:
            pool = chunk.meta["unified"]
            retained = int((pool["used"] - pool["cu_q"].diff()).sum())
        assert retained == (shared if first else 35)
        retained_reads += retained
        fresh_tokens += len(chunk.input_ids)
        # the one label decides at the cue; the four decide at a second
        # choice that a run of one or two fed tokens reaches
        labels = ([(10,)] if first else
                  [(10, *range(11, 10 + run), 1), (10, *range(11, 10 + run), 2),
                   (20, *range(21, 20 + run), 1), (20, *range(21, 20 + run), 2)])
        cost = estimate(
            "trie_decode", 1, 0, 2, labels, lengths=(32,),
            shared=(shared,), chunk=100,
            model=QWEN3_4B_FP8, device=H100_SXM)
        assert cost.work.tokens == fresh_tokens
        assert cost.work.kv_read == retained_reads


@pytest.mark.parametrize("mode", ["tree", "unified"])
def test_warmup_runs_every_row_count_class(monkeypatch, mode):
    from quail.backends.quail.executor import loop, warmup

    cpu_staging(monkeypatch)
    arena = plain_arena(pages=256)
    shapes = []

    def forward(pipeline, arena, chunk):
        reads = chunk.meta["reads"]
        shapes.append((len(chunk.input_ids),
                       0 if reads is None else len(reads["rows"])))

    monkeypatch.setattr(loop, "_forward", forward)
    warmup._warm_row_classes(torch, arena, None, mode)
    # the first chunk writes the cached documents' KV, each with a
    # one-token question so the chunk has answer rows
    assert shapes[0][0] == 17 * (warmup.ROW_CLASS_CACHED + 1)
    if mode == "tree":
        # n rows, m of them reading cached KV, for every class pair
        assert shapes[1:] == list(warmup.ROW_CLASSES)
    else:
        assert [n for n, _ in shapes[1:]] == [n for n, _ in warmup.ROW_CLASSES]
    # every key the pass made is freed
    assert arena.accounting.free_pages == 256

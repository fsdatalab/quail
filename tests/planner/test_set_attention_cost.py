"""Set-attention costs: one pass, a join's passes, and pass groups."""


import pytest

from quail.specs import D1_2B_BF16, H100_SXM


def test_a_set_attention_model_keeps_no_arena_or_checkpoints():
    from quail.cost import budgets

    chunk = budgets.chunk_budget(D1_2B_BF16, H100_SXM)

    assert not D1_2B_BF16.reuses_kv
    assert budgets.arena_pages(D1_2B_BF16, H100_SXM) == (1, 1)
    assert budgets.checkpoint_memory_bytes(D1_2B_BF16, chunk) == 0
    assert chunk > 0


def test_a_lone_question_reads_its_whole_pass_both_ways():
    from quail.cost.work import packed, triangle

    work = packed(300, (50,), window=256)
    two = packed(300, (50, 40))

    assert work.tokens == 350
    assert work.pairs == 350 * 350
    assert work.sliding_pairs == triangle(350, 256)
    assert (work.state_written, work.state_read) == (0, 0)
    # the state reads all 390 tokens; each question, the state and itself
    assert two.pairs == 300 * 390 + 50 * 350 + 40 * 340


def test_a_set_attention_filter_costs_the_same_first_or_later():
    from quail.cost.filters import filter_cost

    first, later = (filter_cost(50, 300.0, D1_2B_BF16, H100_SXM, 8192,
                                first=flag) for flag in (True, False))

    assert first == later > 0


def test_a_set_join_costs_each_anchors_passes_with_packed():
    from quail.cost.joins import PricedJoin, packed_join_work
    from quail.cost.work import packed
    from quail.planner.statistics.tables import AliasStats

    join = PricedJoin(written_position=0, aliases=("r", "a"), anchor="r",
                      anchor_free=False, semantics="full", selectivity=None,
                      frame_tokens={"r": 3, "a": 3},
                      label_tokens={"r": 2, "a": 2}, tail_tokens=20)
    lengths = {"r": AliasStats(count=10, total=3000, squared=900_000,
                               maximum=300),
               "a": AliasStats(count=20, total=200, squared=2000, maximum=10)}

    # 20 partners per anchor: two passes of 10 questions of 32 tokens
    work = packed_join_work(join, "r", 10, 200, lengths, 5, set_size=16)
    one = packed(310, (32,) * 10)

    assert work.tokens == pytest.approx(20 * one.tokens)
    assert work.pairs == pytest.approx(20 * one.pairs)
    assert work.kv_read == pytest.approx(20 * one.kv_read)


def _groups(selectivities, questions=50, state=300.0, longest=300,
            chunk=8192, header=0):
    from quail.cost.packing import choose_pass_groups

    count = len(selectivities)
    return choose_pass_groups([questions] * count, selectivities, 5000.0, state,
                              longest, D1_2B_BF16, H100_SXM, chunk,
                              set_size=16, header=header)


def test_a_filter_that_keeps_few_rows_gets_its_own_pass_group():
    assert _groups([0.02, 0.5, 0.5]) == (0, 1, 1)


def test_filters_that_keep_most_rows_share_one_pass():
    assert _groups([0.8, 0.8, 0.8]) == (0, 0, 0)


def test_a_pass_holds_at_most_the_trained_set_size():
    groups = _groups([1.0] * 20)

    assert max(groups) >= 1
    assert all(groups.count(group) <= 16 for group in set(groups))


def test_a_pass_group_fits_one_chunk_with_the_longest_document_and_headers():
    # 300 + 50 + 3 * (50 + 5) = 515 tokens fit 520; a fifth question does not
    assert _groups([1.0] * 5, chunk=520, header=5) == (0, 0, 0, 0, 1)

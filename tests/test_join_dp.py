import itertools

from quail.planner.join_dp import optimize_joins_after_filters
from quail.planner.work import ideal_seconds
from quail.specs import H100_SXM, QWEN3_4B_FP8


def _join(left, right, *, selectivity=0.2, anchor=None, written_pos=0):
    anchor = left if anchor is None else anchor
    return {
        "aliases": [left, right],
        "anchor": anchor,
        "anchor_free": anchor == left,
        "frames": {left: [1] * 5, right: [1] * 5},
        "labels": {left: [2] * 3, right: [2] * 3},
        "tail": [3] * 2,
        "semantics": "full",
        "selectivity": selectivity,
        "written_pos": written_pos,
    }


def _search(joins, lengths, resident=()):
    docs = {
        alias: [[1] * length for length in rows]
        for alias, rows in lengths.items()
    }
    survivors = {
        alias: list(range(len(rows))) for alias, rows in lengths.items()
    }
    return optimize_joins_after_filters(
        joins, docs, survivors, set(resident), 10,
        QWEN3_4B_FP8, H100_SXM, 16_384,
    )


def test_single_join_anchors_the_long_documents():
    result = _search(
        [_join("a", "b")],
        {"a": [1_000] * 5, "b": [20] * 20},
    )

    assert result is not None
    assert result.sequence == ((0, "a"),)
    assert result.nodes[0]["anchor"] == "a"


def test_fixed_anchor_is_never_changed():
    join = _join("a", "b", anchor="b")
    join["anchor_free"] = False
    result = _search(
        [join],
        {"a": [1_000] * 5, "b": [20] * 20},
    )

    assert result is not None
    assert result.sequence == ((0, "b"),)


def test_resident_prefixes_affect_the_anchor_choice():
    lengths = {"a": [160] * 8, "b": [160] * 8}
    without_kv = _search([_join("a", "b")], lengths)
    with_b_kv = _search(
        [_join("a", "b")], lengths,
        resident={("b", row) for row in range(8)},
    )

    assert without_kv is not None
    assert with_b_kv is not None
    assert without_kv.sequence == ((0, "a"),)
    assert with_b_kv.sequence == ((0, "b"),)


def test_chain_search_executes_each_edge_once_in_a_left_deep_order():
    joins = [
        _join("a", "b", written_pos=0),
        _join("b", "c", written_pos=1),
        _join("c", "d", written_pos=2),
    ]
    result = _search(
        joins,
        {"a": [300] * 4, "b": [40] * 5,
         "c": [80] * 3, "d": [20] * 6},
    )

    assert result is not None
    assert sorted(index for index, _ in result.sequence) == [0, 1, 2]
    connected = set(joins[result.sequence[0][0]]["aliases"])
    for index, _ in result.sequence[1:]:
        edge = set(joins[index]["aliases"])
        assert edge & connected
        connected.update(edge)
    assert connected == {"a", "b", "c", "d"}


def test_dp_matches_complete_search_for_three_alias_chain():
    joins = [
        _join("a", "b", selectivity=0.1, written_pos=0),
        _join("b", "c", selectivity=0.4, written_pos=1),
    ]
    lengths = {"a": [200] * 3, "b": [80] * 5, "c": [20] * 7}
    result = _search(joins, lengths)

    assert result is not None
    candidates = []
    for first_edge, second_edge in ((0, 1), (1, 0)):
        for anchors in itertools.product(
                joins[first_edge]["aliases"], joins[second_edge]["aliases"]):
            fixed = [dict(join) for join in joins]
            for edge, anchor in zip((first_edge, second_edge), anchors):
                fixed[edge] = dict(fixed[edge], anchor=anchor,
                                   anchor_free=False)
            option = _search([fixed[first_edge], fixed[second_edge]], lengths)
            if option is not None:
                candidates.append(option)
    expected = min(
        ideal_seconds(option.work, QWEN3_4B_FP8, H100_SXM, 16_384)
        for option in candidates
    )
    actual = ideal_seconds(
        result.work, QWEN3_4B_FP8, H100_SXM, 16_384)

    assert actual == expected


def test_unsupported_join_semantics_keep_the_compile_time_plan():
    join = _join("a", "b")
    join["semantics"] = "exists"

    assert _search([join], {"a": [20], "b": [20]}) is None


def test_empty_filtered_input_needs_no_join_tuple_to_fit():
    join = _join("a", "b")
    join["tail"] = [3] * 20_000

    result = _search([join], {"a": [], "b": [20]})

    assert result is not None
    assert result.work.tokens == 0

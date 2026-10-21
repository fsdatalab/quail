from quail.planner.search.survivors import PairRelation, exact_survivors

PAIR_SETS = [
    ({("a", "b"): {(0, 0), (1, 1)}, ("b", "c"): {(0, 0)}}, (0,)),
    ({("a", "b"): {(0, 0), (1, 1)}, ("b", "c"): {(0, 0), (1, 1)},
      ("a", "c"): {(0, 1), (1, 0)}}, ()),
]


def test_survivors_for_trees_and_cycles():
    for pairs, expected in PAIR_SETS:
        survivors = exact_survivors(
            {"a": (0, 1), "b": (0, 1), "c": (0, 1)},
            tuple(PairRelation(left, right, frozenset(rows))
                  for (left, right), rows in pairs.items()),
        )
        assert survivors == {"a": expected, "b": expected, "c": expected}, pairs

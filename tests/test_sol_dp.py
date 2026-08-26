from quail.bench.sol_dp import (
    Extension,
    PairRelation,
    Work,
    exact_live_rows,
    optimize_left_deep,
)


def test_exact_live_rows_reduces_a_tree_to_exact_projections():
    live = exact_live_rows(
        {"a": (0, 1), "b": (0, 1), "c": (0, 1)},
        (
            PairRelation("a", "b", frozenset({(0, 0), (1, 1)})),
            PairRelation("b", "c", frozenset({(0, 0)})),
        ),
    )

    assert live == {"a": (0,), "b": (0,), "c": (0,)}


def test_exact_live_rows_handles_cycle_consistency():
    live = exact_live_rows(
        {"a": (0, 1), "b": (0, 1), "c": (0, 1)},
        (
            PairRelation("a", "b", frozenset({(0, 0), (1, 1)})),
            PairRelation("b", "c", frozenset({(0, 0), (1, 1)})),
            PairRelation("a", "c", frozenset({(0, 1), (1, 0)})),
        ),
    )

    assert live == {"a": (), "b": (), "c": ()}


def test_subset_dp_matches_complete_enumeration():
    edges = {frozenset(("a", "b")), frozenset(("b", "c"))}

    def extend(relations, cached, added):
        if not any(frozenset((existing, added)) in edges
                   for existing in relations):
            return ()
        options = []
        for anchor in sorted(relations | {added}):
            if not any(frozenset((anchor, other)) in edges
                       for other in (relations | {added}) - {anchor}):
                continue
            write = 0 if anchor in cached else 10
            options.append(Extension(
                Work(tokens=write + ord(anchor),
                     pairs=300 - ord(anchor),
                     kv_written=write,
                     kv_read=len(cached)),
                cached | {anchor},
                ({"added": added, "anchor": anchor},),
            ))
        return options

    def score(candidate):
        return max(
            candidate.work.tokens + candidate.work.pairs,
            candidate.work.kv_written + candidate.work.kv_read,
        )
    dynamic = optimize_left_deep(
        ("a", "b", "c"), (), Work(tokens=5), extend)

    complete = []
    all_aliases = frozenset(("a", "b", "c"))

    def visit(relations, cached, work):
        if relations == all_aliases:
            complete.append(work)
            return
        for added in all_aliases - relations:
            for extension in extend(relations, cached, added):
                visit(
                    relations | {added},
                    extension.cached,
                    work + extension.work,
                )

    for alias in all_aliases:
        visit(frozenset((alias,)), frozenset(), Work(tokens=5))

    assert min(map(score, dynamic.candidates)) == min(
        max(work.tokens + work.pairs, work.kv_written + work.kv_read)
        for work in complete
    )
    assert dynamic.state_count > 0
    assert dynamic.generated_count < len(complete) * 4

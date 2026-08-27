from quail.planner.left_deep import Extension, optimize_left_deep
from quail.planner.work import Work


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
                state_property=cached | {anchor},
                steps=({"added": added, "anchor": anchor},),
            ))
        return options

    def score(candidate):
        return max(
            candidate.work.tokens + candidate.work.pairs,
            candidate.work.kv_written + candidate.work.kv_read,
        )

    dynamic = optimize_left_deep(
        ("a", "b", "c"), frozenset(), Work(tokens=5), extend)
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
                    extension.state_property,
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

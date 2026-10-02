"""Planned anchor reuse at each execution boundary."""

from quail.planner.joins import thin


def group_sequence(seq):
    """Group consecutive full joins that share an anchor."""
    groups = []
    for spec, anchor in seq:
        if (groups and spec["semantics"] == "full"
                and groups[-1][-1][0]["semantics"] == "full"
                and groups[-1][0][1] == anchor):
            groups[-1].append((spec, anchor))
        else:
            groups.append([(spec, anchor)])
    return groups


def chained_aliases(groups, ask_aliases, barrier_aliases, workers: int) -> set:
    """Aliases whose filter chain runs in the pipeline of the join anchored on it.

    On one GPU, a chain whose first use is the join anchored on it
    streams its survivors into that join, so its KV needs no retention
    pool. A chain a join reads as a partner first, or one a barrier
    apply cuts, is not chained.
    """
    chained = set()
    partner_before = set()
    for group in groups:
        anchor = group[0][1]
        if workers == 1 and anchor in ask_aliases \
                and anchor not in chained \
                and anchor not in partner_before \
                and anchor not in barrier_aliases:
            chained.add(anchor)
        for spec, _ in group:
            for alias in spec["aliases"]:
                if alias != anchor and alias not in chained:
                    partner_before.add(alias)
    return chained


def schedule(seq, live, group_ids=None):
    """Record the next anchor use and conditional survival at each boundary.

    group_ids names the join node of each group; ``group:<index>``
    when omitted.
    """
    groups = group_sequence(seq)
    group_ids = (list(group_ids) if group_ids is not None
                 else [f"group:{i}" for i in range(len(groups))])
    counts = [dict(live)]
    for group in groups:
        after = dict(counts[-1])
        for spec, _ in group:
            thin(after, spec)
        counts.append(after)
    uses = []
    for boundary in range(len(groups) + 1):
        upcoming = {}
        for index in range(boundary, len(groups)):
            alias = groups[index][0][1]
            if alias not in upcoming:
                current = counts[boundary][alias]
                probability = (min(1.0, counts[index][alias] / current)
                               if current else 0.0)
                upcoming[alias] = [probability, index]
        uses.append(upcoming)
    return {
        "initial": uses[0],
        "before": {group_ids[i]: uses[i] for i in range(len(groups))},
        "after": {group_ids[i]: uses[i + 1] for i in range(len(groups))},
    }

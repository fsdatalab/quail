"""Planned anchor reuse at each execution boundary."""

from quail.planner.statistics import thin


def group_sequence(seq):
    """Group consecutive full joins that share an anchor."""
    groups = []
    for spec, anchor in seq:
        if (groups and spec.semantics == "full"
                and groups[-1][-1][0].semantics == "full"
                and groups[-1][0][1] == anchor):
            groups[-1].append((spec, anchor))
        else:
            groups.append([(spec, anchor)])
    return groups


def streamed_aliases(groups, ask_aliases, barrier_aliases, workers: int) -> set:
    """Return the aliases whose filter sequence feeds the join anchored on it.

    On one GPU, a sequence whose first use is the join anchored on it
    streams its survivors into that join, so its KV needs no retention
    pool. A sequence that a join reads first as a partner, or that a
    barrier apply cuts, is not streamed.

    Args:
        groups: Consecutive join groups, as from group_sequence.
        ask_aliases: Aliases that have a filter sequence.
        barrier_aliases: Aliases whose sequence a barrier apply cuts.
        workers: Number of model copies.

    Returns:
        The streamed aliases.
    """
    streamed = set()
    partner_before = set()
    for group in groups:
        anchor = group[0][1]
        if workers == 1 and anchor in ask_aliases \
                and anchor not in streamed \
                and anchor not in partner_before \
                and anchor not in barrier_aliases:
            streamed.add(anchor)
        for spec, _ in group:
            for alias in spec.aliases:
                if alias != anchor and alias not in streamed:
                    partner_before.add(alias)
    return streamed


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

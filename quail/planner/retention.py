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

"""Retention priorities for the executor: costs and future anchor uses."""

from quail.planner.joins import thin
from quail.planner.qwen3_cost import dense_params, flops_per_pair


def coefficients(model, device) -> dict:
    """Return the model's ideal prefix computation coefficients."""
    return {
        "linear_seconds": 2 * dense_params(model) / device.arithmetic_bandwidth(
            model.weight_precision),
        "pair_seconds": flops_per_pair(model) * model.layers
        / device.arithmetic_bandwidth(model.attention_precision),
    }


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


def schedule(seq, live):
    """Record the next anchor use and conditional survival at each boundary."""
    groups = group_sequence(seq)
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
        "before": {f"group:{i}": uses[i] for i in range(len(groups))},
        "after": {f"group:{i}": uses[i + 1] for i in range(len(groups))},
    }

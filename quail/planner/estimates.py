"""Price each physical node's own model work for explain."""

from quail.cost import budgets
from quail.cost.sol import speed_of_light
from quail.cost.work import Work, ask, scan
from quail.physical import AiExtract, AiFilter, AiJoin


def node_estimates(graph, *, filter_works, stage_works, live, stats, pre,
                   cap_pages, model, device, chunk) -> dict:
    """Price each node's own work, and each chain's recompute if released.

    Returns node id -> {"seconds", and for a filter chain a later join
    anchors on, "release_recompute_tokens" and
    "release_recompute_seconds"}. A node's seconds price its work
    alone; nodes do not add up to the plan's estimate because chunk
    packing shares forward passes across them. The recompute figure
    is the expected survivors past the retention pool cap times their
    prefix cost: what the join pays if the chain's KV is not pinned.
    """
    anchored = {node.anchor for node in graph.nodes if isinstance(node, AiJoin)}
    out = {}
    priced_filters = set()
    last_filters = {node.alias: node.node_id for node in graph.nodes
                    if isinstance(node, AiFilter)}
    for node in graph.nodes:
        entry = {}
        if isinstance(node, AiFilter) and node.alias in filter_works:
            operation = ask if node.alias in priced_filters else scan
            priced_filters.add(node.alias)
            work = operation(pre + stats[node.alias].mean_doc_tokens,
                             node.question_tokens, window=model.sliding_window)
            entry["seconds"] = speed_of_light(
                work * node.expected_docs, model, device, chunk).seconds
            if node.alias in anchored and last_filters[node.alias] == node.node_id:
                mean = stats[node.alias].mean_doc_tokens
                prefix = pre + mean
                pages = budgets.pages(prefix)
                fits = cap_pages // max(1, pages)
                excess = max(0.0, live.get(node.alias, 0.0) - fits)
                entry["release_recompute_tokens"] = round(excess * prefix)
                entry["release_recompute_seconds"] = speed_of_light(
                    scan(prefix, 0, window=model.sliding_window) * excess,
                    model, device, chunk).seconds
        elif isinstance(node, AiJoin):
            work = stage_works.get(node.written_pos, Work())
            entry["seconds"] = speed_of_light(work, model, device, chunk).seconds
        elif isinstance(node, AiExtract) and node.spec is not None:
            entry["seconds"] = node.spec.estimated_seconds
        # other nodes do no model work and get no seconds
        out[node.node_id] = entry
    return out

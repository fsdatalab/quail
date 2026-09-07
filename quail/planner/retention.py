"""Estimate shared KV retention from document lengths and selectivities."""

from dataclasses import replace

from quail.executor.retention import RetentionPolicy
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


def allocate(lengths, live, aliases, uses, pre, cap_pages, page_tokens, costs):
    """Allocate expected surviving prefixes in reuse priority order."""
    policy = RetentionPolicy(**costs, uses=uses)
    entries = []
    credits = {}
    for alias in sorted(set(aliases) & set(uses)):
        stats = lengths[alias]
        survival = live[alias] / stats.count if stats.count else 0
        credits[alias] = dict(pages=0.0, documents=0.0, prefix_tokens=0.0,
                              resident_count=0.0, resident_total=0.0,
                              resident_squared=0.0)
        if survival <= 0 or uses[alias][0] <= 0:
            continue
        for length, count in stats.histogram:
            pages = -(-(pre + length) // page_tokens)
            if not pages or pages > cap_pages:
                continue
            priority = policy.priority((alias, 0), pre + length, pages)
            entries.append((priority, alias, length, count, pages, survival))
    remaining = float(cap_pages)
    for _, alias, length, count, pages, survival in sorted(entries, reverse=True):
        expected = min(count * survival, remaining / pages)
        credit = credits[alias]
        credit["pages"] += expected * pages
        credit["documents"] += expected
        credit["prefix_tokens"] += expected * (pre + length)
        conditional = expected / survival
        credit["resident_count"] += conditional
        credit["resident_total"] += conditional * length
        credit["resident_squared"] += conditional * length * length
        remaining -= expected * pages
        if remaining <= 1e-8:
            break
    credited = {}
    for alias, stats in lengths.items():
        credit = credits.get(alias, {})
        credited[alias] = replace(stats, **{
            key: credit.get(key, 0.0)
            for key in ("resident_count", "resident_total", "resident_squared")
        })
    return credited, {
        alias: {key: credit[key] for key in ("pages", "documents", "prefix_tokens")}
        for alias, credit in credits.items()
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

"""Choose predicate order from prepared costs, selectivities, and reuse keys."""


from quail.logical import DEFAULT_SELECTIVITY


def default_order_rule(filters, joins) -> tuple[str, str]:
    """Return (rule, source).

    Always 'by_cost'; a predicate without a selectivity is priced with
    DEFAULT_SELECTIVITY. Pass order="as_written" to keep written order.
    """
    missing = any(p.selectivity is None for fs in filters.values()
                  for p in fs) or any(j.selectivity is None for j in joins)
    if missing:
        return "by_cost", (
            "default: by cost, with selectivity "
            f"{DEFAULT_SELECTIVITY:g} for predicates without one")
    return "by_cost", "default: by cost"


def order_filters_indexed(predicates, rule: str, *, count: float = 1,
                          resident=None):
    """Choose an order using costs at each expected survivor count.

    Try each first predicate, then rank remaining predicates by their
    cost per expected rejection. A computed reuse key costs no more model
    work. Written order breaks ties; as_written bypasses the search.

    Args:
        predicates: Each with first and later costs at a live count, a
            selectivity, and a reuse key.
        rule: "by_cost" or "as_written".
        count: The documents the first predicate reads.
        resident: resident(previous, predicate) -> whether the predicate
            reads the KV the predicate before it left; None reads it
            resident after any predicate.
    """
    indices = list(range(len(predicates)))
    if rule == "as_written" or len(indices) < 2:
        return indices
    costs = {}

    def cost(i, expected_rows, first):
        key = (i, expected_rows, first)
        if key not in costs:
            p = predicates[i]
            costs[key] = (p.first if first else p.later)(expected_rows)
        return costs[key]

    def fresh(previous, i) -> bool:
        return resident is not None and not resident(
            predicates[previous], predicates[i])

    # a predicate that rejects nothing never starts a sequence: every
    # later predicate would read as many documents, at the same price;
    # such predicates keep their written order among themselves
    starters = [i for i in indices if predicates[i].selectivity < 1]
    if not starters:
        return indices
    candidates = []
    for first in starters:
        ordered = [first]
        total = cost(first, count, True)
        expected_rows = count * predicates[first].selectivity
        scored = ({predicates[first].reuse_key}
                  if predicates[first].reuse_key is not None else set())
        remaining = [i for i in indices if i != first]
        previous = first
        while remaining:
            choices = []
            for i in remaining:
                p = predicates[i]
                seconds = (
                    0.0
                    if p.reuse_key in scored
                    else cost(
                        i,
                        expected_rows if expected_rows > 0 else 1.0,
                        fresh(previous, i),
                    )
                )
                rejected = 1.0 - p.selectivity
                rank = (seconds / rejected if rejected > 0 else
                        0.0 if seconds == 0 else float("inf"))
                choices.append((rank, i, seconds if expected_rows > 0 else 0.0))
            _, selected, seconds = min(choices)
            total += seconds
            ordered.append(selected)
            previous = selected
            expected_rows *= predicates[selected].selectivity
            if predicates[selected].reuse_key is not None:
                scored.add(predicates[selected].reuse_key)
            remaining.remove(selected)
        candidates.append((total, first, ordered))
    return min(candidates)[2]

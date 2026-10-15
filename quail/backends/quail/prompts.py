"""Bind physical predicates to model requests."""


def join_request(node) -> dict:
    """Return the bound prompt and predicate for the join driver.

    The frame includes the first partner's label, which every pair
    of the anchor shares.
    """
    labels = dict(node.label_token_ids)
    frame = tuple(node.frame_token_ids)
    if node.partners and node.partners[0] in labels:
        frame += tuple(labels[node.partners[0]])
        labels[node.partners[0]] = ()
    return {
        "anchor": node.anchor,
        "partners": list(node.partners),
        "semantics": node.semantics,
        "selectivity": node.selectivity,
        "written_pos": node.written_pos,
        "pairs_from": node.pairs_from,
        "frame": frame,
        "labels": labels,
        "tail": node.tail_token_ids,
    }

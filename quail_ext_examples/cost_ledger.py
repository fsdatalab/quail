"""Charge a query's GPU time and tokens to the nodes that used them.

A finished query carries its executed physical plan and each node's
measured metrics. This module turns them into a ledger: one row per
node with wall seconds, evaluated documents or document pairs, fresh
and cached tokens, and GPU dollars, plus the query totals. That is what
a chargeback or a query cost dashboard needs, and it needs nothing
registered:

    import quail
    from quail_ext_examples import cost_ledger

    session = quail.Session()
    session.register("reviews", quail.DocumentProvider.from_parquet(
        "reviews.parquet", id_col="id"))
    result = session.sql(
        "SELECT r.id FROM reviews r "
        "WHERE AI_FILTER(PROMPT('Is this review positive? {0}', r.body))"
    ).run()
    ledger = cost_ledger.charge(result)
    print(ledger["totals"]["usd"], ledger["totals"]["fresh_tokens"])
    print(result.explain())
"""

from __future__ import annotations

# https://modal.com/pricing, one H100
H100_USD_PER_HOUR = 3.9492

_SUMMED = (
    "evaluated_documents", "evaluated_document_pairs",
    "fresh_tokens", "cached_tokens",
)


def charge(result, *, usd_per_gpu_hour: float = H100_USD_PER_HOUR,
           gpus: int = 1) -> dict:
    """Return per node rows and totals for one finished query.

    Args:
        result: The QueryResult a query's run() returned.
        usd_per_gpu_hour: Price of one GPU hour.
        gpus: GPUs each node's wall time is charged for.
    """
    if result.plan is None:
        raise ValueError("the query has no executed physical plan")
    rows = []
    wall_s = 0.0
    for node in result.plan.topological_nodes():
        metrics = result.node_metrics[node.node_id]
        wall_s += metrics.wall_s
        rows.append({
            "node_id": node.node_id,
            "node_type": node.type_name,
            "wall_s": round(metrics.wall_s, 6),
            "input_rows": metrics.input_rows,
            "output_rows": metrics.output_rows,
            "evaluated_documents": metrics.evaluated_documents,
            "evaluated_document_pairs": metrics.evaluated_document_pairs,
            "fresh_tokens": metrics.fresh_tokens,
            "cached_tokens": metrics.cached_tokens,
            "usd": round(metrics.wall_s / 3600 * usd_per_gpu_hour * gpus, 8),
        })
    totals = {key: sum(row[key] for row in rows) for key in _SUMMED}
    totals["wall_s"] = round(wall_s, 6)
    totals["usd"] = round(wall_s / 3600 * usd_per_gpu_hour * gpus, 8)
    return {
        "usd_per_gpu_hour": usd_per_gpu_hour,
        "gpus": gpus,
        "nodes": rows,
        "totals": totals,
    }

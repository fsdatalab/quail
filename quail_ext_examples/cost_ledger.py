"""Charge each query's GPU time and tokens to the nodes that used them.

The ledger is an execution observer. After every physical node runs it
records the node's wall seconds, the documents or document pairs the
model evaluated, the fresh and cached tokens, and the GPU dollars for
that time. The report carries the per node rows and the query totals,
which is what a chargeback or a query cost dashboard needs.

Load it like any extension:

    registry = quail.ExtensionRegistry.with_built_ins()
    registry.load_extension("quail_ext_examples.cost_ledger",
                            local_python_sources=("quail_ext_examples",))
    session = quail.Session(registry=registry)
    ...
    result.report["observers"]["example.cost_ledger"]["totals"]
"""

from __future__ import annotations

# https://modal.com/pricing, one H100. The ledger prices every node's
# wall time on one GPU; a node that ran on several GPUs costs more.
H100_USD_PER_HOUR = 3.9492


class CostLedger:
    """Record time, tokens, and dollars per physical node."""

    name = "example.cost_ledger"
    usd_per_gpu_hour = H100_USD_PER_HOUR
    gpus = 1

    def __init__(self):
        self._rows = []

    def after_node(self, node, result) -> None:
        metrics = result.metrics
        usd = metrics.wall_s / 3600 * self.usd_per_gpu_hour * self.gpus
        self._rows.append({
            "node_id": node.node_id,
            "node_type": node.type_name,
            "wall_s": round(metrics.wall_s, 6),
            "input_rows": metrics.input_rows,
            "output_rows": metrics.output_rows,
            "evaluated_documents": metrics.evaluated_documents,
            "evaluated_document_pairs": metrics.evaluated_document_pairs,
            "fresh_tokens": metrics.fresh_tokens,
            "cached_tokens": metrics.cached_tokens,
            "usd": round(usd, 8),
        })

    def report(self) -> dict:
        totals = {
            key: sum(row[key] for row in self._rows)
            for key in (
                "wall_s", "evaluated_documents", "evaluated_document_pairs",
                "fresh_tokens", "cached_tokens", "usd",
            )
        }
        totals["wall_s"] = round(totals["wall_s"], 6)
        totals["usd"] = round(totals["usd"], 8)
        return {
            "usd_per_gpu_hour": self.usd_per_gpu_hour,
            "gpus": self.gpus,
            "nodes": list(self._rows),
            "totals": totals,
        }


def register_quail_extension(registry) -> None:
    """Register the cost ledger observer."""
    registry.register_observer(CostLedger.name, CostLedger)

"""The planner: budgets from the specs, decisions from token
arithmetic. No wall prediction. KV is always bf16.

The names below are the planning interface a model backend uses.
"""

from .decide import (
    balanced_shards,
    collect_operators,
    default_order_rule,
    explain,
    join_specs,
    order_filters_indexed,
    plan_quail,
    plan_query,
    preamble_tokens,
)
from .plan import CorpusStats, EngineConfig, PhysicalPlan, Refusal

__all__ = [
    "CorpusStats",
    "EngineConfig",
    "PhysicalPlan",
    "Refusal",
    "balanced_shards",
    "collect_operators",
    "default_order_rule",
    "explain",
    "join_specs",
    "order_filters_indexed",
    "plan_quail",
    "plan_query",
    "preamble_tokens",
]

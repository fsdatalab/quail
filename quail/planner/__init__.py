"""The planner: every transformation of a plan.

Logical optimizer rules, the physical optimizer protocol, and physical
plan assembly from the cost model's numbers.

No wall prediction. KV is always bf16.

The names below are the planning interface a model backend uses.
"""

from .decide import (
    balanced_shards,
    build_physical_plan,
    explain,
    hash_join_nodes,
    plan_query,
    refine_plan,
)
from .filters import default_order_rule, order_filters_indexed
from .plan import CorpusStats, EngineConfig, PhysicalPlan, Refusal
from .statistics import join_specs, preamble_tokens

__all__ = [
    "CorpusStats",
    "EngineConfig",
    "PhysicalPlan",
    "Refusal",
    "balanced_shards",
    "build_physical_plan",
    "default_order_rule",
    "hash_join_nodes",
    "explain",
    "join_specs",
    "order_filters_indexed",
    "plan_query",
    "preamble_tokens",
    "refine_plan",
]

"""The planner: every transformation of a plan.

Logical and physical rules, candidate search, and executable
plan assembly from the cost model's numbers.

No wall prediction. KV is always bf16.

The names below are the planning interface a model backend uses.
"""

from .graph.builder import build_physical_plan, hash_join_nodes
from .graph.shards import balanced_shards
from .operators.common import preamble_tokens
from .operators.join import join_specs
from .optimize import plan_query, refine_plan
from .plan import CorpusStats, EngineConfig, PhysicalPlan, Refusal
from .search.filter_order import default_order_rule, order_filters_indexed

__all__ = [
    "CorpusStats",
    "EngineConfig",
    "PhysicalPlan",
    "Refusal",
    "balanced_shards",
    "build_physical_plan",
    "default_order_rule",
    "hash_join_nodes",
    "join_specs",
    "order_filters_indexed",
    "plan_query",
    "preamble_tokens",
    "refine_plan",
]

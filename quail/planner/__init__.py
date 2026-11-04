"""The planner: Quail's Rust planner and the plan types it returns.

The Rust planner in ``_quail_native`` applies the logical and physical
rules, searches the alternatives, and prices each plan;
``quail.planner.native`` is its bridge. KV is always bf16.
"""

from .optimize import plan_query, refine_plan
from .plan import CorpusStats, EngineConfig, PhysicalPlan, Refusal

__all__ = [
    "CorpusStats",
    "EngineConfig",
    "PhysicalPlan",
    "Refusal",
    "plan_query",
    "refine_plan",
]

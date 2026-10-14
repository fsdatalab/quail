"""Price logical alternatives through the planner's complete physical search."""

from quail.planner.api import search_plan
from quail.planner.optimizer.logical import LogicalPlanningContext
from quail.planner.plan import Refusal


def physical_plan(logical, context, registry):
    """Return the cheapest refined physical candidate for a logical plan."""
    if isinstance(context, LogicalPlanningContext):
        context = context.physical_context(logical)
    return search_plan(logical, context, registry)


def estimated_seconds(logical, context, registry) -> float | None:
    """Return the refined plan's estimated seconds, or None when refused."""
    physical = physical_plan(logical, context, registry)
    return None if isinstance(physical, Refusal) else physical.estimated_seconds

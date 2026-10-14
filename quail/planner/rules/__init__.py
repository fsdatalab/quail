"""Built in universal and cost-based rules in dependency order."""

from quail.planner.rules.cost_based import (
    ChooseClassifyImplementation,
    LiftMaps,
    PrefixSharing,
    TreeAttention,
)
from quail.planner.rules.universal import (
    DistinctElimination,
    DistinctPushdown,
    FilterPushdown,
    KvRetention,
    LimitPushdown,
    ProjectionPushdown,
)


def built_in_rules() -> tuple:
    """Return rules in dependency order across both categories.

    Classification implementation determines pipeline compatibility for
    KV retention. Prefix sharing prices the writes retention requires.
    Tree attention uses the selected shared prefixes.
    """
    return (
        DistinctElimination(), DistinctPushdown(), ProjectionPushdown(),
        FilterPushdown(), LiftMaps(), LimitPushdown(),
        ChooseClassifyImplementation(), KvRetention(), PrefixSharing(),
        TreeAttention(),
    )

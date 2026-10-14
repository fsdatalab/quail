"""Built in universal and cost-based rules in dependency order."""

from quail.planner.rules.cost_based import (
    ChooseAttentionAlgorithm,
    ChooseClassifyImplementation,
    ShouldLiftMaps,
    ShouldSharePrefixes,
)
from quail.planner.rules.universal import (
    ShouldEliminateDistinct,
    ShouldPushDownDistinct,
    ShouldPushDownFilters,
    ShouldPushDownLimit,
    ShouldPushDownProjection,
    ShouldRetainKv,
)


def built_in_rules() -> tuple:
    """Return rules in dependency order across both categories.

    Classification implementation determines pipeline compatibility for
    KV retention. Prefix sharing prices the writes retention requires.
    Tree attention uses the selected shared prefixes.
    """
    return (
        ShouldEliminateDistinct(), ShouldPushDownDistinct(), ShouldPushDownProjection(),
        ShouldPushDownFilters(), ShouldLiftMaps(), ShouldPushDownLimit(),
        ChooseClassifyImplementation(), ShouldRetainKv(), ShouldSharePrefixes(),
        ChooseAttentionAlgorithm(),
    )

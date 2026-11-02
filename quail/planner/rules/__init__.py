"""Built in logical and physical rules in execution order."""

from quail.planner.rules.logical import (
    ShouldEliminateDistinct,
    ShouldLiftColumnOperators,
    ShouldPushDownDistinct,
    ShouldPushDownFilters,
    ShouldPushDownProjection,
)
from quail.planner.rules.physical import (
    ChooseAttentionPath,
    ChooseClassifyImplementation,
    JoinDocumentEnds,
    PackSetPasses,
    ShouldDeduplicate,
    ShouldEliminateWindowSort,
    ShouldPushDownLimit,
    ShouldPushDownLimitThroughWindow,
    ShouldRetainKv,
    ShouldSharePrefixes,
)


def built_in_rules() -> tuple:
    """Return logical rules followed by physical rules in dependency order.

    Classification implementation determines pipeline compatibility for
    KV retention and for packing set-attention passes. Prefix sharing
    prices the writes retention requires. Tree attention uses the
    selected shared prefixes. Deduplication reads the plan every other
    rule settled.
    """
    return (
        ShouldEliminateDistinct(), ShouldPushDownDistinct(), ShouldPushDownProjection(),
        ShouldPushDownFilters(), ShouldLiftColumnOperators(),
        ShouldEliminateWindowSort(),
        ShouldPushDownLimitThroughWindow(), ShouldPushDownLimit(),
        ChooseClassifyImplementation(), PackSetPasses(), JoinDocumentEnds(),
        ShouldRetainKv(),
        ShouldSharePrefixes(),
        ChooseAttentionPath(), ShouldDeduplicate(),
    )

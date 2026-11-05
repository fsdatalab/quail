"""Built-in node runtimes, keyed by the physical node they execute."""

from quail.execution.runner import NodeRuntime
from quail.execution.runtimes.dataflow import (
    BarrierRuntime,
    ExchangeRuntime,
    HashJoinRuntime,
    ModelNodeRuntime,
    RecombineRuntime,
)
from quail.execution.runtimes.filter import FilterRuntime
from quail.execution.runtimes.relational import (
    LimitRuntime,
    ProjectRuntime,
    ScanRuntime,
    SortRuntime,
)
from quail.execution.runtimes.stage import QueryStageRuntime
from quail.execution.runtimes.unnest import UnnestRuntime
from quail.physical import (
    Barrier,
    Exchange,
    Filter,
    HashJoin,
    Limit,
    Project,
    QueryStage,
    Recombine,
    Scan,
    Sort,
    Unnest,
)

__all__ = [
    "BarrierRuntime",
    "ExchangeRuntime",
    "FilterRuntime",
    "HashJoinRuntime",
    "LimitRuntime",
    "UnnestRuntime",
    "ModelNodeRuntime",
    "ProjectRuntime",
    "RecombineRuntime",
    "ScanRuntime",
    "SortRuntime",
    "built_in_runtimes",
]


def built_in_runtimes() -> dict[str, NodeRuntime]:
    """Return runtimes for the backend independent physical nodes."""
    return {
        Scan.runtime_key: ScanRuntime(),
        QueryStage.runtime_key: QueryStageRuntime(),
        Barrier.runtime_key: BarrierRuntime(),
        Exchange.runtime_key: ExchangeRuntime(),
        HashJoin.runtime_key: HashJoinRuntime(),
        Recombine.runtime_key: RecombineRuntime(),
        Project.runtime_key: ProjectRuntime(),
        Sort.runtime_key: SortRuntime(),
        Limit.runtime_key: LimitRuntime(),
        Unnest.runtime_key: UnnestRuntime(),
        Filter.runtime_key: FilterRuntime(),
    }

"""Built-in node runtimes, keyed by the physical node they execute."""

from quail.execution.runner import NodeRuntime
from quail.execution.runtimes.dataflow import (
    BarrierRuntime,
    ExchangeRuntime,
    ForeignRuntime,
    HashJoinRuntime,
    ModelNodeRuntime,
    RecombineRuntime,
)
from quail.execution.runtimes.relational import (
    AggregateRuntime,
    LimitRuntime,
    ProjectRuntime,
    ResultFilterRuntime,
    ScanRuntime,
    SortRuntime,
    WindowRuntime,
)
from quail.execution.runtimes.unnest import UnnestRuntime
from quail.physical import (
    Aggregate,
    Barrier,
    Exchange,
    Foreign,
    HashJoin,
    Limit,
    Project,
    Recombine,
    ResultFilter,
    Scan,
    Sort,
    Unnest,
    Window,
)

__all__ = [
    "AggregateRuntime",
    "BarrierRuntime",
    "ExchangeRuntime",
    "ForeignRuntime",
    "HashJoinRuntime",
    "LimitRuntime",
    "UnnestRuntime",
    "ModelNodeRuntime",
    "ProjectRuntime",
    "RecombineRuntime",
    "ResultFilterRuntime",
    "ScanRuntime",
    "SortRuntime",
    "WindowRuntime",
    "built_in_runtimes",
]


def built_in_runtimes() -> dict[str, NodeRuntime]:
    """Return runtimes for the backend independent physical nodes."""
    return {
        Scan.runtime_key: ScanRuntime(),
        Barrier.runtime_key: BarrierRuntime(),
        Exchange.runtime_key: ExchangeRuntime(),
        Foreign.runtime_key: ForeignRuntime(),
        HashJoin.runtime_key: HashJoinRuntime(),
        Recombine.runtime_key: RecombineRuntime(),
        Project.runtime_key: ProjectRuntime(),
        Aggregate.runtime_key: AggregateRuntime(),
        Sort.runtime_key: SortRuntime(),
        Window.runtime_key: WindowRuntime(),
        Limit.runtime_key: LimitRuntime(),
        Unnest.runtime_key: UnnestRuntime(),
        ResultFilter.runtime_key: ResultFilterRuntime(),
    }

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
    ScanRuntime,
    SortRuntime,
)
from quail.physical import (
    Aggregate,
    Barrier,
    Exchange,
    Foreign,
    HashJoin,
    Limit,
    Project,
    Recombine,
    Scan,
    Sort,
)

__all__ = [
    "AggregateRuntime",
    "BarrierRuntime",
    "ExchangeRuntime",
    "ForeignRuntime",
    "HashJoinRuntime",
    "LimitRuntime",
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
        Barrier.runtime_key: BarrierRuntime(),
        Exchange.runtime_key: ExchangeRuntime(),
        Foreign.runtime_key: ForeignRuntime(),
        HashJoin.runtime_key: HashJoinRuntime(),
        Recombine.runtime_key: RecombineRuntime(),
        Project.runtime_key: ProjectRuntime(),
        Aggregate.runtime_key: AggregateRuntime(),
        Sort.runtime_key: SortRuntime(),
        Limit.runtime_key: LimitRuntime(),
    }

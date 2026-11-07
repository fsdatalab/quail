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
)
from quail.physical import (
    Barrier,
    Exchange,
    Filter,
    HashJoin,
    Limit,
    Project,
    Recombine,
    Scan,
)

__all__ = [
    "BarrierRuntime",
    "ExchangeRuntime",
    "FilterRuntime",
    "HashJoinRuntime",
    "LimitRuntime",
    "ModelNodeRuntime",
    "ProjectRuntime",
    "RecombineRuntime",
    "ScanRuntime",
    "built_in_runtimes",
]


def built_in_runtimes() -> dict[str, NodeRuntime]:
    """Return runtimes for the backend independent physical nodes."""
    return {
        Scan.runtime_key: ScanRuntime(),
        Barrier.runtime_key: BarrierRuntime(),
        Exchange.runtime_key: ExchangeRuntime(),
        HashJoin.runtime_key: HashJoinRuntime(),
        Recombine.runtime_key: RecombineRuntime(),
        Project.runtime_key: ProjectRuntime(),
        Limit.runtime_key: LimitRuntime(),
        Filter.runtime_key: FilterRuntime(),
    }

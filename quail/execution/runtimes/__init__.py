"""Built-in node runtimes, keyed by the physical node they execute."""

from quail.execution.runner import NodeRuntime
from quail.execution.runtimes.dataflow import (
    BarrierRuntime,
    HashJoinRuntime,
    ModelNodeRuntime,
)
from quail.execution.runtimes.filter import FilterRuntime
from quail.execution.runtimes.relational import (
    LimitRuntime,
    ScanRuntime,
)
from quail.physical import (
    Barrier,
    Filter,
    HashJoin,
    Limit,
    Scan,
)

__all__ = [
    "BarrierRuntime",
    "FilterRuntime",
    "HashJoinRuntime",
    "LimitRuntime",
    "ModelNodeRuntime",
    "ScanRuntime",
    "built_in_runtimes",
]


def built_in_runtimes() -> dict[str, NodeRuntime]:
    """Return runtimes for the backend independent physical nodes."""
    return {
        Scan.runtime_key: ScanRuntime(),
        Barrier.runtime_key: BarrierRuntime(),
        HashJoin.runtime_key: HashJoinRuntime(),
        Limit.runtime_key: LimitRuntime(),
        Filter.runtime_key: FilterRuntime(),
    }

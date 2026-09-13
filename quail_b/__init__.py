"""QUAIL-B document tables, query definitions, and reference labels."""

from quail_b.benchmark import load_benchmark, select_queries
from quail_b.data import load_table
from quail_b.labels import load_ground_truth, load_ground_truth_workload
from quail_b.queries import QuerySpec, get_query
from quail_b.reporting import report
from quail_b.run import run
from quail_b.scoring import RunOutput

__version__ = "0.5.0"

__all__ = [
    "QuerySpec",
    "RunOutput",
    "get_query",
    "load_benchmark",
    "load_ground_truth",
    "load_ground_truth_workload",
    "load_table",
    "report",
    "run",
    "select_queries",
]

"""QUAIL-B document tables, query definitions, and reference labels."""

from quail_b.data import load_table
from quail_b.labels import load_ground_truth, load_ground_truth_workload
from quail_b.queries import get_query

__all__ = ["load_table", "get_query", "load_ground_truth", "load_ground_truth_workload"]

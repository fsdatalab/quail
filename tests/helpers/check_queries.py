"""The check queries' SQL and the schemas of the tables they read."""

from __future__ import annotations

from google.protobuf.json_format import MessageToDict
from substrait import plan_pb2

from bench.substrait import query_sql
from quail_b.queries import queries

CHECK_QUERIES = (
    "IMDB-1", "IMDB-2", "IMDB-4", "FEV-9", "IMDB-11", "AGENT-5", "IMDB-15",
    "CUAD-1", "CUAD-4", "CUAD-10", "CUAD-15", "CUAD-11", "SALES-5", "IMDB-24",
    "CUAD-20",
)


def read_tables(node, tables: dict) -> None:
    """Add the schema of every named table a plan reads to tables.

    Args:
        node: A Substrait plan, or part of one, as a dict.
        tables: Table name -> list of (column, type name) pairs.
    """
    if isinstance(node, dict):
        read = node.get("read", {})
        if "namedTable" in read:
            schema = read["baseSchema"]
            types = [next(iter(kind)) for kind in schema["struct"]["types"]]
            name = ".".join(read["namedTable"]["names"])
            tables[name] = list(zip(schema["names"], types))
        for value in node.values():
            read_tables(value, tables)
    elif isinstance(node, list):
        for value in node:
            read_tables(value, tables)


def check_query_entries(query_ids=CHECK_QUERIES) -> list[dict]:
    """Return each query's id, SQL, and the schemas of the tables it reads.

    Args:
        query_ids: The quail-b query ids to return.
    """
    specs = queries(include_draft=True)
    entries = []
    for query_id in query_ids:
        spec = specs[query_id]
        plan = plan_pb2.Plan()
        plan.ParseFromString(spec.plan_bytes)
        tables = {}
        read_tables(MessageToDict(plan), tables)
        entries.append(
            {"id": query_id, "sql": query_sql(spec.info), "tables": tables})
    return entries

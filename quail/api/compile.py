"""Compile SQL with DataFusion and Quail's Rust binder.

DataFusion parses and binds the statement, Quail's analyzer rule claims
its model region, and the Rust binder reads the region into Quail's
logical plan: one plan per statement, with any CTE that runs as its own
stage in a WithCTE root.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum

import pyarrow as pa

from quail.common.expressions import CompileError
from quail.logical import LogicalPlan
from quail.physical.codec import decode_value


class SQLDialect(StrEnum):
    """The SQL dialects Quail accepts."""

    SNOWFLAKE = "snowflake"
    BQ = "bq"


# a dialect -> DataFusion's name for it
DATAFUSION_DIALECTS = {"snowflake": "Snowflake", "bq": "BigQuery"}

# the start of a DataFusion error, which names its kind
_ERROR_KINDS = ("Error during planning: ", "SQL error: ", "Schema error: ",
                "External error: ")


def _schema_bytes(schema: pa.Schema) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema) as writer:
        writer.write_table(schema.empty_table())
    return sink.getvalue().to_pybytes()


# what WHERE tests besides an AI call, in the refusal of anything else
REGULAR_PREDICATES = (
    "=, <>, <, <=, >, >=, IN (literals), IS NULL, IS NOT NULL, LIKE, ILIKE, "
    "CONTAINS, STARTSWITH, ENDSWITH, REGEXP_LIKE, or a boolean column or scalar "
    "function of columns")


def _sql_text(expression: str) -> str:
    """A DataFusion expression as SQL writes it: literals quoted, AI calls dotted."""
    expression = re.sub(r'Utf8\("((?:[^"\\]|\\.)*)"\)', r"'\1'", expression)
    expression = re.sub(r"Int64\((-?\d+)\)", r"\1", expression)
    return re.sub(r"\bai_(\w+)\(", lambda match: f"AI_{match[1].upper()}(", expression)


def _message(error: Exception, sql: str) -> str:
    """A DataFusion error's message, in the front end's words where it has them."""
    text = str(error).splitlines()[0]
    for kind in _ERROR_KINDS:
        if kind in text:
            text = text.split(kind, 1)[1]
    text = text.removeprefix("CompileError: ")
    if found := re.match(r"Invalid function '(\w+)'", text):
        name = found[1]
        if name.lower() == "mod":
            call = re.search(r"\bMOD\s*\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)", sql,
                             re.IGNORECASE)
            shown = f"{call[1]} % {call[2]}" if call else "MOD"
            return (f"{shown}: Arrow compute has no remainder or integer "
                    f"division function")
        return (f"unknown function {name!r}; a scalar function of Arrow compute "
                f"is called by its name, such as abs or utf8_reverse")
    if found := re.search(r'column "([^"]+)" must appear in the GROUP BY clause', text):
        return f"{found[1]} is in SELECT but not in GROUP BY and not aggregated"
    non_boolean = r"Cannot create filter with non-boolean predicate '(.*)' returning"
    if found := re.match(non_boolean, text):
        return (f"{_sql_text(found[1])} is not supported; a regular predicate is "
                f"{REGULAR_PREDICATES}")
    return text


def compile_with_datafusion(sql: str, catalog, model,
                            dialect: str = "snowflake"):
    """Compile AI SQL text into one logical plan.

    Args:
        sql: The statement.
        catalog: Registered tables and their schemas.
        model: The model whose chat turn and prompt layout the prompts
            are bound in.
        dialect: "snowflake" or "bq".

    Raises:
        CompileError: The query is invalid or uses unsupported SQL.
    """
    import _quail_native

    def refuse(columns):
        raise RuntimeError("compiling a query scans no table")

    def run_region(kind, payload):
        raise RuntimeError("compiling a query runs no model region")

    context = _quail_native.NativeContext(run_region)
    for name, provider in catalog.providers.items():
        context.register(name, _schema_bytes(provider.schema()), refuse)
    try:
        regions, whole = context.logical_plans(
            sql, tuple(model.turn), model.prompt_layout,
            DATAFUSION_DIALECTS[str(dialect).lower()])
    except RuntimeError as error:
        raise CompileError(_message(error, sql)) from None
    if not regions:
        raise CompileError("the query has no AI predicate")
    if len(regions) != 1 or not whole:
        raise CompileError(
            "Quail plans a query that is one model region; this one also needs "
            "SQL outside it, which quail.SessionContext runs")
    plan, message = regions[0]
    if message is not None:
        raise CompileError(message)
    return decode_value(LogicalPlan, json.loads(plan))

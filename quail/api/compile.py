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


# the end of a refusal of a region shape the binder does not read; such
# a region reaches Quail as SQL text
READ_AS_SQL = "is outside what Quail binds from DataFusion"

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
    lines = str(error).splitlines()
    # an optimizer rule's error names the rule, then its cause
    causes = [index for index, line in enumerate(lines) if line == "caused by"]
    text = lines[causes[-1] + 1] if causes and causes[-1] + 1 < len(lines) else lines[0]
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
                            dialect: str = "snowflake",
                            rules=None) -> LogicalPlan | None:
    """Compile AI SQL text into its model region's logical plan.

    Returns:
        The plan of the query's one model region, or None when the query
        has more than one (see ``DataFusionQuery``).

    Raises:
        CompileError: The query is invalid or uses unsupported SQL.
    """
    compiled = compile_region(sql, catalog, model, dialect, rules)
    return None if compiled is None else compiled[0]


def compile_region(sql: str, catalog, model, dialect: str = "snowflake",
                   rules=None) -> tuple[LogicalPlan, bool] | None:
    """Compile AI SQL text into the logical plan of its one model region.

    Args:
        sql: The statement.
        catalog: Registered tables and their schemas.
        model: The model whose chat turn and prompt layout the prompts
            are bound in.
        dialect: "snowflake" or "bq".
        rules: The names of the optimizer rules each model region runs,
            in order, or None for the built-in rules.

    Returns:
        The region's plan and whether the region is the whole query, or
        None when the query has more than one model region.

    Raises:
        CompileError: The query is invalid or uses unsupported SQL.
    """
    import _quail_native

    def refuse(columns):
        raise RuntimeError("compiling a query scans no table")

    def run_region(kind, payload, tables=None):
        raise RuntimeError("compiling a query runs no model region")

    context = _quail_native.NativeContext(
        run_region, rules=None if rules is None else list(rules))
    for name, provider in catalog.providers.items():
        context.register(name, _schema_bytes(provider.schema()), refuse,
                         getattr(provider, "id_col", None) or None)
    try:
        regions, whole = context.logical_plans(
            sql, tuple(model.turn), model.prompt_layout,
            DATAFUSION_DIALECTS[str(dialect).lower()])
    except RuntimeError as error:
        raise CompileError(_message(error, sql)) from None
    if not regions:
        raise CompileError("the query has no AI predicate")
    if len(regions) != 1:
        for _, message in regions:
            # a region the binder does not read runs as SQL of its own
            if message is not None and READ_AS_SQL not in message:
                raise CompileError(message)
        return None
    plan, message = regions[0]
    if message is not None:
        if not whole and READ_AS_SQL in message:
            return None
        raise CompileError(message)
    return decode_value(LogicalPlan, json.loads(plan)), whole

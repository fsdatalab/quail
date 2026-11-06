"""Compile SQL with DataFusion and Quail's Rust binder.

DataFusion parses and binds the statement, Quail's analyzer rule claims
its model region, and the Rust binder reads the region into Quail's
logical plan. A CTE or FROM subquery the binder does not merge is its
own region, whose rows the reading region takes as an input.
"""

from __future__ import annotations

import re
from enum import StrEnum

import pyarrow as pa


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


def compile_message(error: Exception, sql: str) -> str:
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
            return f"MOD is not a DataFusion function; write {shown}"
        return f"unknown function {name!r}"
    if found := re.search(r'column "([^"]+)" must appear in the GROUP BY clause', text):
        return f"{found[1]} is in SELECT but not in GROUP BY and not aggregated"
    non_boolean = r"Cannot create filter with non-boolean predicate '(.*)' returning"
    if found := re.match(non_boolean, text):
        return (f"{_sql_text(found[1])} is not supported; a regular predicate is "
                f"{REGULAR_PREDICATES}")
    return text

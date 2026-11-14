"""Compile SQL with DataFusion and Quail's rules.

DataFusion parses and binds the statement, Quail's last optimizer rule
claims its model regions, and Rust derives the region's operators from
DataFusion's plan of it. A CTE or FROM subquery the region does not
read through is its own region, whose rows the reading region takes as
an input.
"""

from __future__ import annotations

import re
from enum import StrEnum


class SQLDialect(StrEnum):
    """The SQL dialects Quail accepts."""

    SNOWFLAKE = "snowflake"
    BQ = "bq"


# a dialect -> DataFusion's name for it
DATAFUSION_DIALECTS = {"snowflake": "Snowflake", "bq": "BigQuery"}

# the start of a DataFusion error, which names its kind
_ERROR_KINDS = ("Error during planning: ", "SQL error: ", "Schema error: ",
                "External error: ", "CompileError: ")


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
    text = str(error).strip()
    # DataFusion wraps rule failures (including lower_ai_calls) in a cause chain.
    text = re.split(r"\n\s*caused by\s*\n", text)[-1].strip()
    while prefix := next(
        (kind for kind in _ERROR_KINDS if text.startswith(kind)), None
    ):
        text = text.removeprefix(prefix)
    if 'List(Struct("feature"' in text:
        if "character_length" in text:
            return ("LENGTH needs text, so use array_length for a feature list "
                    "or UNNEST to test its fields.")
        if "Cannot infer common argument type" in text:
            return ("AI.SAE_FEATURES returns a list, so use UNNEST "
                    "before testing its fields.")
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

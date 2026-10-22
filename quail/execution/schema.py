"""The Arrow schema of a query's result, computed without running it.

A CTE's result is the table a later query reads, so that query is
compiled against the schema before anything runs. The types follow
the result the executor builds: a score is a float, a label is text,
an extraction is a struct of answer fields beside a struct of their
character spans, and aggregates and windows are typed by the kernels
that compute them.
"""

from types import SimpleNamespace

import pyarrow as pa

from quail.catalog import Catalog
from quail.execution.runtimes.relational import aggregate_table, window_table
from quail.logical import (
    PROBABILITIES_SUFFIX,
    SPAN_SUFFIX,
    Aggregate,
    LogicalPlan,
    Window,
    column_name,
    cte_column_names,
)
from quail.logical.expressions import (
    aliased_call,
    column_references,
    evaluate_expression,
)

SPAN_TYPE = pa.struct([("start", pa.int32()), ("end", pa.int32())])
PROBABILITIES_TYPE = pa.map_(pa.string(), pa.float64())


def _projected_columns(plan: LogicalPlan, catalog: Catalog) -> dict:
    """Return one empty array per column the projection returns, unpacked."""
    empty = {}
    columns = {}
    for column in plan.projection.columns:
        name = column_name(column)
        call = aliased_call(column)
        if call is None:
            (ref, *_) = column_references(column)
            table = empty.setdefault(
                ref.provider, catalog.get(ref.provider).schema().empty_table())
            columns[name] = evaluate_expression(column, table)
        elif call.kind == "score":
            columns[name] = pa.array([], pa.float64())
        elif call.kind == "label":
            columns[name] = pa.array([], pa.string())
            if call.probabilities:
                columns[name + PROBABILITIES_SUFFIX] = pa.array(
                    [], PROBABILITIES_TYPE)
        else:
            columns[name] = pa.array([], pa.string())
            columns[name + SPAN_SUFFIX] = pa.array([], SPAN_TYPE)
    return columns


def _pack(columns: dict, packing) -> dict:
    """Pack extraction members into struct columns, as the result does."""
    owner = {member: name for name, members in packing for _, member in members}
    packed = {}
    for name, values in columns.items():
        struct = owner.get(name)
        if struct is None:
            packed[name] = values
        elif struct not in packed:
            packed[struct] = pa.array([], pa.struct([
                (field, columns[member].type)
                for field, member in dict(packing)[struct]]))
    return packed


def result_schema(plan: LogicalPlan, catalog: Catalog) -> pa.Schema:
    """Return the schema of a plan's result, named as a CTE's columns.

    Args:
        plan: The compiled query.
        catalog: The tables the plan scans, for their column types.

    Returns:
        The Arrow schema of the result, with each column named as
        ``cte_column_names`` names it.
    """
    table = pa.table(_pack(_projected_columns(plan, catalog),
                           plan.projection.packing()))
    for node in plan.walk():
        if isinstance(node, Aggregate):
            table = aggregate_table(table, SimpleNamespace(
                keys=node.keys,
                aggregates=tuple((call.name, call.function, call.argument)
                                 for call in node.aggregates),
                having=tuple((test.aggregate.name, test.comparison, test.value)
                             for test in node.having),
                columns=node.output))
        elif isinstance(node, Window):
            table = window_table(table, SimpleNamespace(
                calls=node.calls, columns=node.output))
    return table.rename_columns(list(cte_column_names(plan))).schema


def cte_table(result: pa.Table, schema: pa.Schema) -> pa.Table:
    """Return a stage's result as the table the next stage reads.

    Label columns are decoded from dictionaries to text, and the
    columns take the CTE's names.

    Raises:
        ValueError: The result's types differ from the schema the next
            stage was compiled against.
    """
    columns = [
        column.cast(column.type.value_type)
        if pa.types.is_dictionary(column.type) else column
        for column in result.columns]
    if len(columns) != len(schema) or [
            column.type for column in columns] != [field.type for field in schema]:
        raise ValueError(
            f"the CTE's result has types {[c.type for c in columns]}, but "
            f"the query reading it was compiled for {schema.types}")
    return pa.table(dict(zip(schema.names, columns)))

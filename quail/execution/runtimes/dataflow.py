"""Runtimes for the dataflow nodes: exchange, barrier, recombine, and join."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from quail.common.pairs import columns_key, pair_table
from quail.execution.ids import document_ids
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.result import (
    IndexRelation,
    build_result_declaration,
    joined_survivors,
    true_answer_rows,
)
from quail.execution.rows import Rows, as_table
from quail.physical import (
    Barrier,
    Exchange,
    HashJoin,
    Recombine,
    ValueType,
)


class ExchangeRuntime:
    """Pass anchor ids through; several GPUs route them to their KV."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Exchange):
            raise TypeError(type(node).__name__)
        if len(inputs) != 1:
            raise ValueError("Exchange takes one survivor input")
        return NodeResult({f"rows:{node.anchor}": next(iter(inputs.values()))})


class BarrierRuntime:
    """Prune survivor IDs using completed join answers."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Barrier):
            raise TypeError(type(node).__name__)
        from quail.execution.types import _join_answers_table

        survivors = {}
        tables = []
        for port in node.inputs:
            value = inputs[port.name]
            if port.value_type is ValueType.ROWS:
                alias = port.source.port.split(":", 1)[1]
                survivors[alias] = document_ids(value)
            elif port.value_type is ValueType.JOIN_ANSWERS:
                tables.append(value if isinstance(value, pa.Table)
                              else _join_answers_table(value))
        survivors = joined_survivors(
            [true_answer_rows(table) for table in tables], survivors)
        return NodeResult({f"rows:{alias}": Rows({alias: survivors[alias]})
                           for alias in node.aliases})


def relation_values(columns, expression):
    """Return one key column's values from the alias's value table."""
    for ref in expression.references:
        if columns is None or ref.column not in columns.column_names:
            raise KeyError(f"column {ref.alias}.{ref.column} is not among the "
                           f"request's relations")
    return expression.evaluate(columns)


class RecombineRuntime:
    """Run the configured exact answer relation join."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Recombine):
            raise TypeError(type(node).__name__)
        if context.recombine is None:
            answer_tables = []
            survivors = {}
            for input_port in node.inputs:
                value = as_table(inputs[input_port.name])
                if not isinstance(value, pa.Table):
                    raise TypeError(
                        "Recombine inputs must be Arrow tables"
                    )
                if input_port.value_type is ValueType.JOIN_ANSWERS:
                    answer_tables.append(true_answer_rows(value))
                elif input_port.value_type is ValueType.ROWS:
                    alias = input_port.source.port.split(":", 1)[1]
                    survivors[alias] = value.column(alias).combine_chunks()
                else:
                    raise TypeError(
                        f"Recombine cannot read {input_port.value_type.value}"
                    )
            declaration, schema = build_result_declaration(
                answer_tables,
                survivors,
                node.alias_order[0],
            )
            value = IndexRelation(declaration, schema)
        else:
            value = context.recombine(node, inputs)
        return NodeResult({"tuples": value})


class ModelNodeRuntime:
    """Delegate one model node to the shared model execution object."""

    def execute(self, node, inputs, context) -> NodeResult:
        if context.model_execution is None:
            raise RuntimeError("model node has no model execution object")
        if context.model_inputs is not None:
            inputs = context.model_inputs(node, inputs, context)
        result = context.model_execution.execute(node, inputs)
        if not isinstance(result, NodeResult):
            raise TypeError("model execution must return NodeResult")
        if context.model_result is not None:
            context.model_result(node, result, context)
        return result


class HashJoinRuntime:
    """Pair the rows of two tables whose key columns are equal."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, HashJoin):
            raise TypeError(type(node).__name__)
        sides = {}
        for alias, keys in ((node.left, [left for left, _ in node.on]),
                            (node.right, [right for _, right in node.on])):
            port = next(port for port in node.inputs
                        if port.source.port == f"rows:{alias}")
            ids = document_ids(inputs[port.name])
            columns = context.sources.get(columns_key(alias))
            # a key column may carry the alias's name: the ids stay apart
            sides[alias] = (ids, [relation_values(columns, key).take(ids)
                                  for key in keys])
        positions = pair_table(node.left, sides[node.left][1],
                               node.right, sides[node.right][1])
        # pair_table pairs row positions; map them back to the ids read
        pairs = pa.table({
            alias: pc.take(sides[alias][0], positions.column(alias))
            for alias in (node.left, node.right)})
        return NodeResult(
            {f"pairs:{node.written_position}": pairs},
            NodeMetrics(
                input_rows=sum(len(ids) for ids, _ in sides.values()),
                output_rows=pairs.num_rows))

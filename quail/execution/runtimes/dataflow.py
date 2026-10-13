"""Runtimes for the dataflow nodes: exchange, barrier, foreign, recombine, and join."""

from __future__ import annotations

from quail.catalog.pairs import columns_key, pair_ids_table, pair_table
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.result import (
    IndexRelation,
    build_result_declaration,
    true_answer_rows,
)
from quail.physical import (
    Barrier,
    Exchange,
    Foreign,
    GraphValidationError,
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
        return NodeResult({f"ids:{node.anchor}": next(iter(inputs.values()))})


class BarrierRuntime:
    """Prune survivor IDs using completed join answers."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Barrier):
            raise TypeError(type(node).__name__)
        import pyarrow as pa
        import pyarrow.compute as pc

        from quail.execution.types import _join_answers_table

        survivors = {}
        tables = []
        arrow_inputs = set()
        for port in node.inputs:
            value = inputs[port.name]
            if port.value_type is ValueType.DOCUMENT_IDS:
                alias = port.source.port.split(":", 1)[1]
                if isinstance(value, pa.Table):
                    arrow_inputs.add(alias)
                    value = value.column(alias)
                else:
                    value = pa.array(value, type=pa.int32())
                survivors[alias] = value
            elif port.value_type is ValueType.JOIN_ANSWERS:
                tables.append(value if isinstance(value, pa.Table)
                              else _join_answers_table(value))
        for table in tables:
            table = true_answer_rows(table)
            for alias in table.column_names:
                if alias in survivors:
                    table = table.filter(pc.is_in(
                        table[alias], value_set=survivors[alias].cast(table[alias].type)
                    ))
            for alias in table.column_names:
                if alias in survivors:
                    ids = survivors[alias]
                    survivors[alias] = pc.filter(ids, pc.is_in(
                        ids, value_set=pc.unique(table[alias]).cast(ids.type)))
        return NodeResult({
            f"ids:{alias}": (
                pa.table({alias: survivors[alias]})
                if alias in arrow_inputs else survivors[alias].to_pylist()
            )
            for alias in node.aliases
        })


def _ids_of(value) -> list:
    """Ids from a list, an Arrow array, or a one-column Arrow table."""
    import pyarrow as pa

    if isinstance(value, pa.Table):
        if value.num_columns != 1:
            raise ValueError("an apply() returning ids must return one "
                             "column, an array, or a list of ids")
        value = value.column(0)
    if isinstance(value, (pa.Array, pa.ChunkedArray)):
        value = value.to_pylist()
    ids = []
    for document in value:
        if document is None:
            raise ValueError(
                "an apply() function returned a null id; an outer join "
                "leaves nulls, use join_type='inner'")
        ids.append(int(document))
    return ids


def _pairs_of(value, left: str, right: str) -> list:
    """(left, right) id pairs from an Arrow table with both alias columns."""
    import pyarrow as pa

    if not isinstance(value, pa.Table) or not {left, right} <= set(
            value.column_names):
        raise ValueError(
            f"an apply() returning pairs must return an Arrow table with "
            f"columns {left!r} and {right!r}")
    return list(zip(_ids_of(value.column(left)), _ids_of(value.column(right))))


def alias_table(alias: str, ids, columns, names):
    """The Arrow table an apply() function sees for one alias.

    Its first column is the alias's row indices under the alias
    name; the rest are the requested value columns for those rows.
    """
    import pyarrow as pa

    ids = [int(document) for document in ids]
    arrays = {alias: pa.array(ids, type=pa.int32())}
    for name in names:
        if columns is None or name not in columns.column_names:
            raise KeyError(
                f"column {alias}.{name} is not among the request's relations")
        arrays[name] = columns.column(name).take(pa.array(ids, pa.int64()))
    return pa.table(arrays)


def foreign_call(node, values, context):
    """Bind a user function to its input columns and collect call metrics.

    Args:
        node: The Foreign node.
        values: alias -> the ids, or id table, the node was given.
        context: The execution context holding the functions and
            column sources.

    Returns:
        (call, metrics): call(ids by alias) -> kept ids or pairs, as the
        node's ``ids`` says, checked against what it was given;
        metrics() -> the NodeMetrics of every call so far.
    """
    if not isinstance(node, Foreign):
        raise TypeError(type(node).__name__)
    function = context.functions.get(node.function)
    if function is None:
        raise KeyError(
            f"apply() function {node.function!r} is not registered on "
            f"this session")
    missing = [alias for alias in node.aliases if alias not in values]
    if missing:
        raise GraphValidationError(
            f"{node.node_id!r} has no id input for {missing}")
    columns = {alias: context.sources.get(columns_key(alias))
               for alias in node.aliases}
    names = {alias: [column for owner, column in node.columns
                     if owner == alias] for alias in node.aliases}
    counters = dict(calls=0, input_rows=0, output_rows=0)

    def call(ids_by_alias):
        tables = {alias: alias_table(alias, ids, columns[alias],
                                     names[alias])
                  for alias, ids in ids_by_alias.items()}
        counters["calls"] += 1
        counters["input_rows"] += sum(
            len(ids) for ids in ids_by_alias.values())
        result = function(tables)
        if node.ids == "pairs":
            left, right = node.aliases
            pairs = _pairs_of(result, left, right)
            allowed = {alias: set(ids) for alias, ids in ids_by_alias.items()}
            for a, b in pairs:
                if a not in allowed[left] or b not in allowed[right]:
                    raise ValueError(
                        f"apply() {node.function!r} returned pair "
                        f"({a}, {b}) outside its input ids")
            counters["output_rows"] += len(pairs)
            return pairs
        (alias,) = node.aliases
        kept = _ids_of(result)
        given = set(ids_by_alias[alias])
        if not set(kept) <= given:
            raise ValueError(
                f"apply() {node.function!r} returned ids it was not "
                f"given; a function never invents an id")
        if node.ids == "preserve" and set(kept) != given:
            raise ValueError(
                f"apply() {node.function!r} preserves ids but dropped "
                f"{len(given) - len(set(kept))}")
        counters["output_rows"] += len(kept)
        return kept

    def metrics():
        return NodeMetrics(
            input_rows=counters["input_rows"],
            output_rows=counters["output_rows"],
            extension={"calls": counters["calls"]})

    return call, metrics


def foreign_ids(node, inputs) -> dict:
    """Return the input IDs for a Foreign node, indexed by table alias."""
    values = {}
    for port in node.inputs:
        if port.source.port.startswith("ids:"):
            values[port.source.port.split(":", 1)[1]] = inputs[port.name]
    return values


def foreign_outputs(node, result) -> dict:
    """Build a Foreign node's output ports from its function result."""
    if node.ids == "pairs":
        return {f"pairs:{node.written_pos}":
                pair_ids_table(*node.aliases, result)}
    return {f"ids:{node.aliases[0]}": result}


class ForeignRuntime:
    """Call a user function once over every id it was given.

    Inside a pipeline, the executor calls a per-batch Foreign on each
    document.
    """

    def execute(self, node, inputs, context) -> NodeResult:
        import pyarrow as pa

        values = foreign_ids(node, inputs)
        call, metrics = foreign_call(node, values, context)
        ids_by_alias = {}
        for alias in node.aliases:
            value = values[alias]
            if isinstance(value, pa.Table):
                value = value.column(alias).to_pylist()
            ids_by_alias[alias] = list(value)
        return NodeResult(foreign_outputs(node, call(ids_by_alias)), metrics())


class RecombineRuntime:
    """Run the configured exact answer relation join."""

    def execute(self, node, inputs, context) -> NodeResult:
        if not isinstance(node, Recombine):
            raise TypeError(type(node).__name__)
        if context.recombine is None:
            import pyarrow as pa


            answer_tables = []
            survivors = {}
            for input_port in node.inputs:
                value = inputs[input_port.name]
                if not isinstance(value, pa.Table):
                    raise TypeError(
                        "Recombine inputs must be Arrow tables"
                    )
                if input_port.value_type is ValueType.JOIN_ANSWERS:
                    answer_tables.append(true_answer_rows(value))
                elif input_port.value_type is ValueType.DOCUMENT_IDS:
                    if len(value.column_names) != 1:
                        raise ValueError(
                            "a survivor relation needs one alias column"
                        )
                    alias = value.column_names[0]
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
        import pyarrow as pa
        import pyarrow.compute as pc

        if not isinstance(node, HashJoin):
            raise TypeError(type(node).__name__)
        sides = {}
        for alias, names in ((node.left, [left for left, _ in node.on]),
                             (node.right, [right for _, right in node.on])):
            port = next(port for port in node.inputs
                        if port.source.port == f"ids:{alias}")
            table = alias_table(alias, inputs[port.name],
                                context.sources.get(columns_key(alias)), names)
            sides[alias] = (table.column(alias),
                            [table.column(name) for name in names])
        positions = pair_table(node.left, sides[node.left][1],
                               node.right, sides[node.right][1])
        # pair_table pairs row positions; map them back to the ids read
        pairs = pa.table({
            alias: pc.take(sides[alias][0], positions.column(alias))
            for alias in (node.left, node.right)})
        return NodeResult(
            {f"pairs:{node.written_pos}": pairs},
            NodeMetrics(
                input_rows=sum(len(ids) for ids, _ in sides.values()),
                output_rows=pairs.num_rows))

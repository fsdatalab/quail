"""The result nodes a physical plan ends with."""

from quail.logical import Aggregate as LogicalAggregate
from quail.logical import Filter as LogicalFilter
from quail.logical import Result as LogicalResult
from quail.logical import Unnest as LogicalUnnest
from quail.logical import Window as LogicalWindow
from quail.logical import packed_columns
from quail.physical import (
    Aggregate,
    Limit,
    PortRef,
    Project,
    ResultFilter,
    Sort,
    Unnest,
    Window,
)
from quail.physical.base import input_ports


def result_nodes(root: LogicalResult, inputs: tuple[PortRef, ...],
                 columns: tuple[str, ...], structs: tuple = ()) -> tuple:
    """Return the Project, its unnests, and the nodes ending a plan.

    A sort key over a source column the projection does not return is
    projected as a hidden column, which the Sort drops. Each logical
    Unnest and each Filter above it becomes an Unnest or a ResultFilter
    after the Project, in plan order; the last of them returns the
    result columns in SELECT order. A GROUP BY adds
    an Aggregate over the projected columns, whose output the later
    nodes read, and window calls a Window after it. A plan with an
    ORDER BY, DISTINCT, or OFFSET ends in a Sort carrying the limit; a
    plain LIMIT ends in a Limit.

    Args:
        root: The result modifiers above a projection or aggregation.
        inputs: The ports the projection reads, the relation first.
        columns: The result column names, in the projection's order.
        structs: The struct columns the Project packs, as
            Project.structs names them.
    """
    window = root.input if isinstance(root.input, LogicalWindow) else None
    below = root.input.input if window is not None else root.input
    chain = []  # the logical Unnest and Filter nodes, lowest first
    node = below.input if isinstance(below, LogicalAggregate) else below
    while isinstance(node, (LogicalUnnest, LogicalFilter)):
        chain.insert(0, node)
        node = node.input
    returned = chain[-1].result_columns() if chain else ()
    # with an aggregation the sort keys name its outputs, not inputs
    hidden = () if isinstance(below, LogicalAggregate) else tuple(dict.fromkeys(
        key.name for key in root.order
        if key.name not in columns and key.name not in returned))
    nodes = [Project(
        node_id="project",
        inputs=input_ports(tuple(inputs)),
        columns=tuple(columns) + hidden,
        structs=tuple(structs),
    )]
    rows = (PortRef("project", "rows"),)
    result = packed_columns(columns, structs)
    for index, logical in enumerate(chain):
        result = logical.result_columns()
        last = index == len(chain) - 1
        if last and root.output:
            # a column only ORDER BY reads passes through to the Sort
            result = tuple(root.output) + tuple(dict.fromkeys(
                key.name for key in root.order
                if key.name in result and key.name not in root.output))
        if isinstance(logical, LogicalUnnest):
            node_id = f"unnest_{index}"
            nodes.append(Unnest(node_id=node_id, inputs=input_ports(rows),
                                lists=tuple(logical.columns),
                                columns=result + hidden))
        else:
            node_id = f"result_filter_{index}"
            nodes.append(ResultFilter(node_id=node_id, inputs=input_ports(rows),
                                      condition=logical.condition,
                                      columns=result + hidden))
        rows = (PortRef(node_id, "rows"),)
        if last and root.output:
            result = tuple(root.output)
    if isinstance(below, LogicalAggregate):
        aggregation = below
        nodes.append(Aggregate(
            node_id="aggregate",
            inputs=input_ports(rows),
            keys=tuple(aggregation.keys),
            aggregates=tuple((a.name, a.function, a.argument)
                             for a in aggregation.aggregates),
            having=tuple((t.aggregate.name, t.comparison, t.value)
                         for t in aggregation.having),
            columns=tuple(aggregation.output),
        ))
        rows = (PortRef("aggregate", "rows"),)
        result = tuple(aggregation.output)
    if window is not None:
        # the sort keys no window output names pass through to the Sort
        result = tuple(window.output) + tuple(
            key.name for key in root.order if key.name not in window.output)
        nodes.append(Window(
            node_id="window",
            inputs=input_ports(rows),
            calls=tuple(window.calls),
            columns=result,
        ))
        rows = (PortRef("window", "rows"),)
        result = tuple(window.output)
    if root.order or root.distinct or root.offset:
        nodes.append(Sort(
            node_id="sort",
            inputs=input_ports(rows),
            keys=tuple((key.name, key.descending, key.nulls_first)
                       for key in root.order),
            columns=result,
            distinct=root.distinct,
            offset=root.offset,
            fetch=root.limit,
        ))
    elif root.limit is not None:
        nodes.append(Limit(
            node_id="limit",
            inputs=input_ports(rows),
            count=root.limit,
        ))
    return tuple(nodes)

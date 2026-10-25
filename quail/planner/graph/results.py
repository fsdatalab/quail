"""The result nodes a physical plan ends with."""

from quail.logical import Aggregate as LogicalAggregate
from quail.logical import Filter as LogicalFilter
from quail.logical import Project as LogicalProject
from quail.logical import Result as LogicalResult
from quail.logical import Unnest as LogicalUnnest
from quail.logical import Window as LogicalWindow
from quail.logical import column_name, packed_columns
from quail.logical.nodes.filter import above_unnest
from quail.physical import (
    Aggregate,
    Filter,
    Limit,
    PortRef,
    Project,
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
    Unnest and each Filter above it becomes an Unnest or a Filter
    after the source Project, in plan order. A later Project selects
    result columns and applies aliases. A GROUP BY adds
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
    chain = []
    node = below.input if isinstance(below, LogicalAggregate) else below
    while isinstance(node, (LogicalUnnest, LogicalFilter, LogicalProject)):
        if isinstance(node, LogicalProject) and not above_unnest(node.input):
            break
        chain.insert(0, node)
        node = node.input
    # with an aggregation the sort keys name its outputs, not inputs
    hidden = () if chain or isinstance(below, LogicalAggregate) else tuple(
        dict.fromkeys(key.name for key in root.order if key.name not in columns))
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
        if isinstance(logical, LogicalUnnest):
            node_id = f"unnest_{index}"
            nodes.append(Unnest(node_id=node_id, inputs=input_ports(rows),
                                lists=tuple(logical.columns),
                                outer=logical.outer,
                                columns=result + hidden))
        elif isinstance(logical, LogicalProject):
            node_id = f"project_{index}"
            expressions = logical.columns
            if not isinstance(below, LogicalAggregate):
                available = logical.input.result_columns()
                window_names = {call.name for call in window.calls} if window else set()
                expressions += tuple(
                    key.name if key.name in available else key.expression
                    for key in root.order
                    if key.name not in result and key.name not in window_names)
            nodes.append(Project(
                node_id=node_id, inputs=input_ports(rows),
                columns=tuple(column_name(column) for column in expressions),
                expressions=expressions))
        else:
            node_id = f"filter_{index}"
            nodes.append(Filter(node_id=node_id, inputs=input_ports(rows),
                                predicate=logical.condition,
                                columns=result + hidden))
        rows = (PortRef(node_id, "rows"),)
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

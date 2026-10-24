"""The result nodes a physical plan ends with."""

from quail.logical import Aggregate as LogicalAggregate
from quail.logical import Result as LogicalResult
from quail.logical import Window as LogicalWindow
from quail.logical import packed_columns
from quail.physical import Aggregate, Limit, PortRef, Project, Sort, Unnest, Window
from quail.physical.base import input_ports


def result_nodes(root: LogicalResult, inputs: tuple[PortRef, ...],
                 columns: tuple[str, ...], structs: tuple = (),
                 unnest=None) -> tuple:
    """Return the Project, Unnest, Aggregate, Window, and Sort or Limit ending a plan.

    A sort key over a source column the projection does not return is
    projected as a hidden column, which the Sort drops. A GROUP BY adds
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
        unnest: The logical Unnest of one feature list column, or None.
            Its node follows the Project, so the later nodes read one
            row per element.
    """
    window = root.input if isinstance(root.input, LogicalWindow) else None
    below = root.input.input if window is not None else root.input
    # with an aggregation the sort keys name its outputs, not inputs
    unnested = () if unnest is None else unnest.fields
    hidden = () if isinstance(below, LogicalAggregate) else tuple(dict.fromkeys(
        key.name for key in root.order
        if key.name not in columns and key.name not in unnested))
    nodes = [Project(
        node_id="project",
        inputs=input_ports(tuple(inputs)),
        columns=tuple(columns) + hidden,
        structs=tuple(structs),
    )]
    rows = (PortRef("project", "rows"),)
    result = packed_columns(columns, structs)
    if unnest is not None:
        result = tuple(unnest.output) or tuple(
            name for column in result for name in (
                unnested if column == unnest.column else (column,)))
        # a field only ORDER BY reads passes through to the Sort
        sorted_fields = tuple(dict.fromkeys(
            key.name for key in root.order
            if key.name in unnested and key.name not in result))
        nodes.append(Unnest(
            node_id="unnest",
            inputs=input_ports(rows),
            column=unnest.column,
            tests=tuple(unnest.tests),
            columns=result + sorted_fields + hidden,
        ))
        rows = (PortRef("unnest", "rows"),)
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

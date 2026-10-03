"""The result nodes a physical plan ends with."""

from quail.logical import Project as LogicalProject
from quail.physical import Aggregate, Limit, PortRef, Project, Sort
from quail.physical.base import input_ports


def result_nodes(root: LogicalProject, inputs: tuple[PortRef, ...],
                 columns: tuple[str, ...]) -> tuple:
    """Return the Project, Aggregate, and Sort or Limit that end a plan.

    A sort key over a source column the projection does not return is
    projected as a hidden column, which the Sort drops. A GROUP BY adds
    an Aggregate over the projected columns, whose output the later
    nodes read. A plan with an ORDER BY, DISTINCT, or OFFSET ends in a
    Sort carrying the limit; a plain LIMIT ends in a Limit.

    Args:
        root: The logical projection.
        inputs: The ports the projection reads, the relation first.
        columns: The result column names, in the projection's order.
    """
    # with an aggregation the sort keys name its outputs, not inputs
    hidden = () if root.aggregation is not None else tuple(dict.fromkeys(
        key.name for key in root.order if key.name not in columns))
    nodes = [Project(
        node_id="project",
        inputs=input_ports(tuple(inputs)),
        columns=tuple(columns) + hidden,
    )]
    rows = (PortRef("project", "rows"),)
    result = tuple(columns)
    if root.aggregation is not None:
        aggregation = root.aggregation
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

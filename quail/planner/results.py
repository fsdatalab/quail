"""The result nodes a physical plan ends with."""

from quail.logical import Project as LogicalProject
from quail.physical import Limit, PortRef, Project, Sort
from quail.physical.base import input_ports


def result_nodes(root: LogicalProject, inputs: tuple[PortRef, ...],
                 columns: tuple[str, ...]) -> tuple:
    """Return the Project and the Sort or Limit that end a plan.

    A sort key over a source column the projection does not return is
    projected as a hidden column, which the Sort drops. A plan with an
    ORDER BY, DISTINCT, or OFFSET ends in a Sort carrying the limit;
    a plain LIMIT ends in a Limit.

    Args:
        root: The logical projection.
        inputs: The ports the projection reads, the relation first.
        columns: The result column names, in the projection's order.
    """
    hidden = tuple(dict.fromkeys(
        key.name for key in root.order if key.name not in columns))
    nodes = [Project(
        node_id="project",
        inputs=input_ports(tuple(inputs)),
        columns=tuple(columns) + hidden,
    )]
    rows = (PortRef("project", "rows"),)
    if root.order or root.distinct or root.offset:
        nodes.append(Sort(
            node_id="sort",
            inputs=input_ports(rows),
            keys=tuple((key.name, key.descending, key.nulls_first)
                       for key in root.order),
            columns=tuple(columns),
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

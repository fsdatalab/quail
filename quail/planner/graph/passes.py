"""Set-attention passes: which operators of a plan share a pass over each document.

Under set attention, a pass reads a document and then up to the
model's set size of questions; nothing is kept between passes. The
executor packs a pipeline's operators of one kind into the same
passes: a table's filters by their stage, and consecutive
classifications or extractions in blocks of the set size. pass_groups
gives those groups so the plan prices, and explain() shows, the
passes the executor runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

from quail.physical import AiClassify, AiExtract, AiFilter, PhysicalGraph
from quail.physical.pipelines import build_pipelines

PACKED_TYPES = (AiFilter, AiClassify, AiExtract)


@dataclass(frozen=True)
class PassGroup:
    """Operators whose questions share each pass over a table's documents.

    Attributes:
        alias: The table.
        nodes: The operators, in question order.
        rows: The documents the passes are expected to read.
    """

    alias: str
    nodes: tuple
    rows: float


def expected_documents(node) -> float:
    """The documents an operator is expected to be asked about."""
    if isinstance(node, AiFilter):
        return node.expected_rows
    return node.spec.expected_rows


def pass_groups(graph: PhysicalGraph, set_size: int) -> list[PassGroup]:
    """The plan's set-attention passes, in execution order.

    Args:
        graph: The physical graph.
        set_size: The most questions in one pass.
    """
    pipelines = build_pipelines(graph)
    groups, seen = [], set()
    for node in graph.topological_nodes():
        pipeline = pipelines.get(node.node_id)
        if pipeline is None or id(pipeline) in seen:
            continue
        seen.add(id(pipeline))
        packed = [member for member in pipeline if isinstance(member, PACKED_TYPES)]
        for _, run in groupby(packed, key=type):
            run = list(run)
            if isinstance(run[0], AiFilter):
                blocks = [list(stage) for _, stage in
                          groupby(run, key=lambda member: member.stage)]
            else:
                blocks = [run[start:start + set_size]
                          for start in range(0, len(run), set_size)]
            alias = run[0].document_aliases[0]
            groups.extend(PassGroup(alias, tuple(block),
                                    expected_documents(block[0]))
                          for block in blocks)
    return groups

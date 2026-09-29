"""Pipelines: chains of per-document operators a graph runs as one.

A pipeline is the longest chain of operators that each take one
table's documents one at a time: its AI.IF filter chain, its
classification, the filters on the classification's labels and the
per-batch applies on its documents, and the join anchored on it. The
executor runs the whole chain over each document while its KV is
resident, as DuckDB runs a vector through every operator of a pipeline
before the next vector.

A chain ends at a breaker: a node that needs every document at once
(Barrier, Exchange, a barrier apply, the recombination, a projection),
a join that reads the documents as partners, a second consumer of the
same documents, or a join, which settles each anchor's KV itself,
unless a classification of the rows it kept follows it.
"""

from __future__ import annotations

from dataclasses import dataclass

from quail.physical import (
    AiClassify,
    AiFilter,
    AiJoin,
    Filter,
    Foreign,
    PhysicalGraph,
    PhysicalNode,
)


@dataclass(frozen=True)
class Pipeline:
    """One table's chain of per-document operators, in graph order."""

    alias: str
    members: tuple[PhysicalNode, ...]

    @property
    def sink(self) -> PhysicalNode:
        """The last member; the pipeline runs when its inputs are ready."""
        return self.members[-1]

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(member.node_id for member in self.members)


def operator_aliases(node: PhysicalNode) -> tuple[str, ...]:
    """The aliases whose documents a per-document operator can take.

    Empty for a node that needs every document at once. A per-batch
    apply returning pairs takes either of its two aliases' documents.
    """
    if isinstance(node, AiFilter):
        return (node.alias,)
    if isinstance(node, AiClassify):
        # a classification of joined rows takes the join's anchors,
        # with the partners the join kept for each
        if node.spec is None:
            return ()
        return (node.spec.anchor,)
    if isinstance(node, Filter):
        # a filter over one table's rows; it joins a chain only after
        # the classification whose labels its predicate reads
        return tuple(node.aliases) if len(node.aliases) == 1 else ()
    if isinstance(node, Foreign):
        if node.kind != "per_batch":
            return ()
        return tuple(node.aliases) if node.ids == "pairs" else node.aliases[:1]
    if isinstance(node, AiJoin):
        return (node.anchor,)
    return ()


def _document_ports(node: PhysicalNode, alias: str) -> tuple[str, ...]:
    """The output ports on which the operator hands the alias's documents on."""
    if isinstance(node, (AiClassify, Filter)):
        return ("scores", f"ids:{alias}")
    if isinstance(node, Foreign) and node.ids == "pairs":
        return (f"pairs:{node.written_pos}",)
    if isinstance(node, AiJoin):
        # a classification of joined rows reads the join's answers
        return tuple(f"join_answers:{stage.written_pos}"
                     for stage in node.stages) + (f"ids:{alias}",)
    return (f"ids:{alias}",)


def _classifies_rows_of(node: PhysicalNode, join: AiJoin) -> bool:
    """Whether the node classifies the rows the join keeps, on its anchor."""
    if not isinstance(node, AiClassify) or node.spec is None:
        return False
    spec = node.spec
    return (spec.partner is not None and spec.anchor == join.anchor
            and any(set(spec.aliases) == {stage.anchor, *stage.partners}
                    for stage in join.stages))


def build_pipelines(graph: PhysicalGraph) -> dict[str, Pipeline]:
    """Group the graph's per-document operators into pipelines.

    Returns:
        Member node id -> its pipeline, for every member.
    """
    consumers: dict[tuple[str, str], list[PhysicalNode]] = {}
    for node in graph.nodes:
        for port in node.inputs:
            consumers.setdefault(
                (port.source.node_id, port.source.port), []).append(node)
    order = {node.node_id: index
             for index, node in enumerate(graph.topological_nodes())}

    def takes(node: PhysicalNode, alias: str) -> bool:
        return alias in operator_aliases(node)

    def reads(consumer: PhysicalNode, producer: PhysicalNode) -> bool:
        return any(port.source.node_id == producer.node_id
                   for port in consumer.inputs)

    def continuation(node: PhysicalNode, alias: str) -> list[PhysicalNode]:
        """The operators that take the documents on from node, in order.

        One consumer continues the chain. Several continue it only
        when each reads the one before it, as a per-batch apply
        returning pairs and the join reading those pairs do; any
        other fan-out needs the documents at once and ends the chain.
        A join settles each anchor's KV itself and ends the chain,
        unless a classification of the rows it kept follows it.
        """
        following = []
        for port in _document_ports(node, alias):
            for consumer in consumers.get((node.node_id, port), ()):
                if isinstance(node, AiJoin) and not _classifies_rows_of(consumer, node):
                    continue
                if takes(consumer, alias) and consumer not in following:
                    following.append(consumer)
        following.sort(key=lambda consumer: order[consumer.node_id])
        for earlier, later in zip(following, following[1:]):
            if not reads(later, earlier):
                return []
        return following

    pipelines: dict[str, Pipeline] = {}
    followed: set[str] = set()
    for node in graph.topological_nodes():
        aliases = operator_aliases(node)
        if not aliases or node.node_id in followed:
            continue
        # a filter or apply left on its own reads a table
        if not isinstance(node, (AiFilter, AiClassify, AiJoin)):
            continue
        (alias,) = aliases
        members = [node]
        followed.add(node.node_id)
        while True:
            after = [consumer for consumer in continuation(members[-1], alias)
                     if consumer.node_id not in followed]
            if not after:
                break
            members.extend(after)
            followed.update(consumer.node_id for consumer in after)
        pipeline = Pipeline(alias, tuple(members))
        for member in members:
            pipelines[member.node_id] = pipeline
    return pipelines


def external_inputs(pipeline: Pipeline, member: PhysicalNode) -> tuple:
    """The member's input ports fed from outside the pipeline."""
    inside = set(pipeline.node_ids)
    return tuple(port for port in member.inputs
                 if port.source.node_id not in inside)

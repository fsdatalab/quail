"""Group compatible per-document operators into execution pipelines.

A pipeline processes each document through its filters, classifications,
and per-batch functions while its KV remains available. Materializing
operators, incompatible prompt heads, and independent consumers end a
pipeline. A join can continue into classification of the pairs it keeps.
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
        """Return the last operator in the pipeline."""
        return self.members[-1]

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(member.node_id for member in self.members)


def operator_aliases(node: PhysicalNode) -> tuple[str, ...]:
    """Return the document aliases an operator can process incrementally.

    Args:
        node: Physical operator.

    Returns:
        Document aliases, or an empty tuple for an operator requiring
        materialized input. A per-batch function returning pairs can process
        either of its two aliases.
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
    """Return output ports carrying documents for the given alias."""
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
    """Return whether the node classifies the pairs produced by the join."""
    if not isinstance(node, AiClassify) or node.spec is None:
        return False
    spec = node.spec
    return (spec.partner is not None and spec.anchor == join.anchor
            and any(set(spec.aliases) == {stage.anchor, *stage.partners}
                    for stage in join.stages))


def build_pipelines(graph: PhysicalGraph) -> dict[str, Pipeline]:
    """Group compatible per-document operators into pipelines.

    Args:
        graph: Physical operator graph.

    Returns:
        A mapping from each member node ID to its Pipeline.
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
        """Find consumers that can continue this operator's pipeline.

        One consumer continues the chain. Several continue it only
        when each reads the one before it, as a per-batch apply
        returning pairs and the join reading those pairs do; any
        other fan-out needs the documents at once and ends the chain.
        A join manages each anchor's KV itself and ends the chain,
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
        # only an AI filter, classification, or join starts a pipeline
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
    """Return member input ports whose sources are outside the pipeline."""
    inside = set(pipeline.node_ids)
    return tuple(port for port in member.inputs
                 if port.source.node_id not in inside)

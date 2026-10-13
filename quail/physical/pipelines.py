"""Find graph paths whose operators can advance documents with resident KV."""

from quail.physical import AiClassify, AiJoin, PhysicalGraph, PhysicalNode


def build_pipelines(
    graph: PhysicalGraph, workers: int = 1, completed: tuple[str, ...] = (),
) -> dict[str, tuple[PhysicalNode, ...]]:
    """Group compatible per-document operators into pipelines.

    Args:
        graph: Physical operator graph.
        workers: Number of model workers.
        completed: Nodes whose outputs are already available.

    Returns:
        A mapping from each node ID to its compatible graph path.
    """
    consumers: dict[tuple[str, str], list[PhysicalNode]] = {}
    for node in graph.nodes:
        for port in node.inputs:
            consumers.setdefault(
                (port.source.node_id, port.source.port), []).append(node)
    order = {node.node_id: index
             for index, node in enumerate(graph.topological_nodes())}

    def takes(node: PhysicalNode, alias: str) -> bool:
        return alias in node.document_aliases

    def reads(consumer: PhysicalNode, producer: PhysicalNode) -> bool:
        return any(port.source.node_id == producer.node_id
                   for port in consumer.inputs)

    def continuation(node: PhysicalNode, alias: str,
                     first: PhysicalNode) -> list[PhysicalNode]:
        """Find consumers that can continue this operator's pipeline.

        One consumer continues the sequence. Several continue it only
        when each reads the one before it, as a per-batch apply
        returning pairs and the join reading those pairs do; any
        other fan-out needs the documents at once and ends the sequence.
        Full joins on the same anchor share one scheduler run. A
        classification of the rows they keep can follow them.
        """
        following = []
        for port in node.document_outputs(alias):
            for consumer in consumers.get((node.node_id, port), ()):
                if consumer.node_id in completed:
                    continue
                if (not node.can_precede(consumer)
                        or not consumer.can_follow(node, first)):
                    continue
                if workers > 1 and (not node.worker_pipelining
                                    or type(consumer) is not type(node)):
                    continue
                if takes(consumer, alias) and consumer not in following:
                    following.append(consumer)
        if isinstance(node, AiJoin):
            next_joins = [consumer for consumer in following
                          if isinstance(consumer, AiJoin)]
            if next_joins:
                following = next_joins
        following.sort(key=lambda consumer: order[consumer.node_id])
        for earlier, later in zip(following, following[1:]):
            if not reads(later, earlier):
                return []
        return following

    pipelines: dict[str, tuple[PhysicalNode, ...]] = {}
    followed: set[str] = set(completed)
    for node in graph.topological_nodes():
        aliases = node.document_aliases
        if not aliases or node.node_id in followed:
            continue
        # only an AI filter, classification, extraction, or join
        # starts a pipeline
        if node.backend is None:
            continue
        (alias,) = aliases
        members = [node]
        followed.add(node.node_id)
        while True:
            after = [consumer
                     for consumer in continuation(members[-1], alias, members[0])
                     if consumer.node_id not in followed]
            if not after:
                break
            members.extend(after)
            followed.update(consumer.node_id for consumer in after)
        joins = [member for member in members if isinstance(member, AiJoin)]
        for join in joins if workers == 1 else ():
            candidates = [consumer for consumer in consumers.get(
                (join.node_id, f"join_answers:{join.written_pos}"), ())
                if isinstance(consumer, AiClassify) and consumer.classifies_join(join)]
            if len(candidates) == 1 and candidates[0].node_id not in followed:
                members.append(candidates[0])
                followed.add(candidates[0].node_id)
        pipeline = tuple(members)
        for member in members:
            pipelines[member.node_id] = pipeline
    return pipelines

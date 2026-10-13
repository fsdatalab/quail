"""Set document KV lifetimes from physical graph consumers."""

from dataclasses import replace

from quail.execution.pipelines import build_pipelines
from quail.physical import AiFilter, AiJoin, PhysicalGraph


def plan_kv_retention(graph: PhysicalGraph, workers: int = 1,
                      shared_head: tuple[int, ...] = ()) -> PhysicalGraph:
    """Set operators' KV retention from the next compatible document consumer.

    KV between operators on one path stays active in the scheduler. It stays
    available after the path finishes when the next path reads the same
    documents under the same prompt head. Worker boundaries use the same
    paths as execution.
    """
    def head(node):
        return shared_head if node.document_head is None else node.document_head

    paths = build_pipelines(graph, workers)
    ordered = graph.topological_nodes()
    starts = [node for node in ordered
              if node.node_id in paths and paths[node.node_id][0] is node]
    replacements = {}
    for index, first in enumerate(starts):
        members = paths[first.node_id]
        alias = first.document_aliases[0]
        following = next((later for later in starts[index + 1:]
                          if later.document_aliases == (alias,)), None)
        keep = (following is not None
                and head(first) == head(following)
                and (workers == 1 or first.worker_pipelining
                     and following.worker_pipelining))
        model_nodes = [node for node in members if node.backend is not None]
        # Joins settle anchors after any classification of their surviving pairs.
        settling = next((node for node in reversed(model_nodes)
                         if isinstance(node, AiJoin)), model_nodes[-1])
        for position, node in enumerate(model_nodes):
            retain = bool(keep and node is settling)
            if isinstance(node, AiJoin):
                retain |= any(isinstance(later, AiJoin)
                              for later in model_nodes[position + 1:])
            fields = {"keep_kv": retain}
            if isinstance(node, AiFilter):
                fields["arena_writes"] = (node.arena_writes or retain
                                          or position < len(model_nodes) - 1)
            replacements[node.node_id] = replace(node, **fields)
    return PhysicalGraph(tuple(replacements.get(node.node_id, node)
                               for node in graph.nodes), graph.root)

"""Partition graph inputs and merge typed outputs across GPU workers."""

from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from quail.execution.ids import document_ids, id_positions
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.rows import Rows
from quail.physical import ValueType
from quail.planner import balanced_shards


def keyed_shards(groups, doc_tokens, k: int) -> tuple:
    """Shard documents by stop key group, balanced by token count.

    Every document of a group lands on one worker, so the worker's
    per-key stop sees the whole group.
    """
    tokens = {}
    for group, length in zip(groups, doc_tokens):
        tokens[group] = tokens.get(group, 0) + length
    ordered = list(tokens)
    shards, _ = balanced_shards([tokens[group] for group in ordered], k)
    placed = {ordered[index]: worker
              for worker, shard in enumerate(shards) for index in shard}
    out = [[] for _ in range(k)]
    for document, group in enumerate(groups):
        out[placed[group]].append(document)
    return tuple(tuple(shard) for shard in out)


def shard_input(port, value, alias, ids):
    """Select one worker's anchor rows, keeping its partners unchanged."""
    ids = document_ids(ids)
    if port.value_type is ValueType.DOCUMENT_IDS and port.source.port == f"ids:{alias}":
        values = document_ids(value)
        return pc.filter(values, pc.is_in(values, value_set=ids))
    if isinstance(value, Rows) and alias in value.ids:
        return value.filter(pc.is_in(value.ids[alias], value_set=ids))
    if port.value_type is ValueType.PAIRS and alias in value.column_names:
        return value.filter(pc.is_in(value[alias], value_set=ids))
    return value


def merge_results(nodes, outputs, alias, ids, wall):
    """Merge each output according to its port type in the input's row order."""
    position = {int(document): index for index, document in enumerate(ids)}

    def order(documents):
        return np.argsort([position[int(document)] for document in
                           id_positions(documents)], kind="stable")

    def merge(port, values):
        if isinstance(values[0], pa.Table):
            joined = pa.concat_tables(values)
            return joined.take(order(joined[alias]))
        if port.value_type is ValueType.DOCUMENT_IDS:
            joined = pa.concat_arrays([document_ids(value) for value in values])
            return joined.take(order(joined))
        if port.value_type is ValueType.ROWS:
            joined = Rows.concat(values)
            return joined.take(order(joined.ids[alias]))
        if port.value_type is ValueType.FILTER_ANSWERS:
            return {document: answer for value in values
                    for document, answer in value.items()}
        if port.value_type is ValueType.JOIN_ANSWERS:
            return merge_join_answers(values)
        raise TypeError(f"cannot merge worker output {port.value_type.value!r}")

    results = {}
    for index, node in enumerate(nodes):
        parts = [output["results"][node.node_id] for output in outputs]
        metrics = sum((part.metrics for part in parts), NodeMetrics())
        results[node.node_id] = NodeResult(
            {port.name: merge(port, [part.outputs[port.name] for part in parts])
             for port in node.outputs},
            replace(metrics, wall_s=wall if index == 0 else 0.0))
    return results


def merge_join_answers(parts: list) -> dict:
    """Merge answer rows while translating each worker's local anchor indices."""
    rows, anchor_index = {}, []
    members = None
    for part in parts:
        base = len(anchor_index)
        anchor_index.extend(part["anchor_index"])
        rows.update((base + int(local), row) for local, row in part["rows"].items())
        if part.get("anchor_partners") is not None:
            if members is None:
                members = {}
            members.update((base + int(local), streamed)
                           for local, streamed in part["anchor_partners"].items())
    return dict(parts[0], rows=rows, anchor_index=anchor_index,
                anchor_partners=members)

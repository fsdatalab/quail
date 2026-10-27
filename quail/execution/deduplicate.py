"""Deduplicate a group of model operators' inputs, and answer every copy."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pyarrow as pa

from quail.catalog.pairs import columns_key, pair_ids_table, pair_keys
from quail.catalog.tokens import FIRST_DOCUMENT
from quail.execution.ids import document_ids
from quail.execution.metrics import NodeResult
from quail.execution.result import true_answer_rows
from quail.execution.rows import Rows, rows_of
from quail.execution.types import _join_answers_table
from quail.physical import AiFilter, AiInference, AiJoin, Foreign, ValueType

MODEL_NODES = (AiFilter, AiJoin, AiInference)


@dataclass
class Deduplicated:
    """One group's inputs over key documents, and the expansion of its results.

    Attributes:
        inputs: Node id -> port name -> value, the group's outside
            inputs with one document, or pair, per key.
        keys: Alias -> each document's key document, as int64.
        every: Alias -> the documents the group read from outside.
        pairs: Node id -> the pairs the node read from outside.
        inside: The ids of the group's nodes.
        nodes: The group's nodes, in the order they run.
    """

    inputs: dict
    keys: dict
    every: dict
    pairs: dict
    inside: set
    nodes: tuple

    def expanded(self, results: dict) -> dict:
        """Give every document, or pair, the answers of its key documents.

        Args:
            results: Node id -> NodeResult over the key documents.
        """
        current = dict(self.every)
        expanded = {}
        # node id, port name -> the expanded output of an earlier member
        earlier_outputs = {}
        for node in self.nodes:
            result = results[node.node_id]
            pairs = self._pairs_read(node, earlier_outputs)
            # every port stands for the documents the node read, before
            # its survivors become what the next node reads
            read = dict(current)
            outputs = {}
            ports = sorted(node.outputs,
                           key=lambda port: port.value_type is not
                           ValueType.JOIN_ANSWERS)
            for port in ports:
                value = result.outputs[port.name]
                aliases = self._aliases(node, port, value)
                if not any(alias in self.keys for alias in aliases):
                    outputs[port.name] = value
                    continue
                keys = self._keys(aliases, pairs)
                if port.value_type is ValueType.JOIN_ANSWERS:
                    table = (value if isinstance(value, pa.Table)
                             else _join_answers_table(value))
                    outputs[port.name] = _expanded(table, aliases, read, keys, pairs)
                elif port.value_type is ValueType.FILTER_ANSWERS:
                    outputs[port.name] = _expanded(value, aliases, read, keys, None)
                elif isinstance(node, AiJoin):
                    # an AI join's anchor rows come from its expanded answers
                    joined = next(table for table in outputs.values()
                                  if isinstance(table, pa.Table))
                    (anchor,) = aliases
                    outputs[port.name] = Rows({anchor: _join_survivors(
                        joined, anchor, node.semantics, read[anchor],
                        len(keys[anchor]))})
                else:
                    rows = rows_of(value, aliases[0])
                    table = _expanded(rows.table(), aliases, read, keys,
                                      pairs if len(aliases) == 2 else None)
                    outputs[port.name] = Rows(
                        {alias: table.column(alias) for alias in aliases},
                        table.drop_columns(list(aliases)))
                if port.value_type is ValueType.ROWS and len(aliases) == 1:
                    current[aliases[0]] = document_ids(
                        outputs[port.name]).to_numpy()
                earlier_outputs[(node.node_id, port.name)] = outputs[port.name]
            metrics = result.metrics
            read = self._rows_read(node, pairs)
            if read is not None:
                metrics = replace(metrics, input_rows=read)
            rows = next((value for value in outputs.values()
                         if isinstance(value, Rows)), None)
            if rows is not None:
                metrics = replace(metrics, output_rows=rows.num_rows)
            expanded[node.node_id] = NodeResult(outputs, metrics)
        return expanded

    def _keys(self, aliases, pairs) -> dict:
        """Each alias's key documents; a table without copies is its own key."""
        keys = {}
        for alias in aliases:
            if alias in self.keys:
                keys[alias] = self.keys[alias]
                continue
            # cover every id the answers or the pairs can name
            known = [self.every.get(alias, np.zeros(0, dtype=np.int64))]
            if pairs is not None and alias in pairs.column_names:
                known.append(pairs.column(alias).to_numpy())
            size = max((int(ids.max()) + 1 for ids in known if len(ids)),
                       default=0)
            keys[alias] = np.arange(size, dtype=np.int64)
        return keys

    def _pairs_read(self, node, earlier_outputs):
        """The pairs a node stands for: read from outside, or a join's inside."""
        if node.node_id in self.pairs:
            return self.pairs[node.node_id]
        for port in node.inputs:
            if (port.value_type is ValueType.JOIN_ANSWERS
                    and port.source.node_id in self.inside):
                return true_answer_rows(earlier_outputs[
                    (port.source.node_id, port.source.port)])
        return None

    def _rows_read(self, node, pairs):
        if pairs is not None:
            return pairs.num_rows
        for port in node.inputs:
            if port.value_type is ValueType.ROWS:
                alias = port.source.port.split(":", 1)[1]
                if alias in self.every:
                    return len(self.every[alias])
        return None

    @staticmethod
    def _aliases(node, port, value) -> tuple[str, ...]:
        if isinstance(value, Rows):
            return value.aliases
        if port.value_type is ValueType.ROWS:
            return (port.name.split(":", 1)[1],)
        # a model operator's answers cover its anchor and partner; a
        # condition's cover the tables it tests
        if isinstance(node, AiJoin):
            return (node.anchor, *node.partners)
        if isinstance(node, AiFilter):
            return (node.alias,)
        return tuple(node.aliases)


def deduplicate(graph, nodes, inputs, context) -> Deduplicated | None:
    """Reduce a group's outside inputs to one document, or pair, per key.

    Args:
        graph: The physical graph the group belongs to.
        nodes: The group's nodes, a pipeline or one node.
        inputs: Node id -> port name -> value, the group's outside inputs.
        context: The execution context; its ``deduplicate`` names, per
            alias, the row columns documents are grouped by beside their
            tokens.

    Returns:
        The reduced inputs and their expansion, or None when the group
        reads no deduplicated table, or reads an input the
        expansion cannot give back to every copy.
    """
    if not context.deduplicate or not any(
            isinstance(node, MODEL_NODES) for node in nodes):
        return None
    if any(isinstance(node, Foreign) for node in nodes) or any(
            isinstance(node, AiJoin) and len(node.partners) != 1
            for node in nodes):
        return None
    inside = {node.node_id for node in nodes}
    keys = {}
    every = {}
    pairs = {}
    reduced = {node.node_id: dict(inputs.get(node.node_id, {})) for node in nodes}
    outside = [(node, port) for node in nodes for port in node.inputs
               if port.source.node_id not in inside]
    # the rows ports say which documents the group read; a pair input
    # read beside them is filtered to those documents when expanded
    for node, port in sorted(outside, key=lambda item: _is_pairs(
            item[1], inputs[item[0].node_id][item[1].name])):
        value = inputs[node.node_id][port.name]
        if _is_pairs(port, value):
            if isinstance(graph.node(port.source.node_id), Foreign):
                return None
            aliases = (value.aliases if isinstance(value, Rows)
                       else tuple(value.column_names))
            table = value.table() if isinstance(value, Rows) else value
            # a pair input may carry only its two id columns: other
            # values could differ between copies
            if len(aliases) != 2 or table.num_columns != 2:
                return None
            _read_pairs(context, aliases, table, keys, every, pairs, node,
                        port, reduced)
        elif port.value_type is ValueType.ROWS:
            alias = (value.aliases[0] if isinstance(value, Rows)
                     else port.source.port.split(":", 1)[1])
            ids = document_ids(value).to_numpy()
            if alias in every and not np.array_equal(every[alias], ids):
                return None
            every[alias] = ids
            if _key(context, alias, keys):
                value = _reduced_rows(value, alias, ids, keys[alias])
                if value is None:
                    return None
                reduced[node.node_id][port.name] = value
        else:
            return None
    if not keys:
        return None
    return Deduplicated(reduced, keys, every, pairs, inside, tuple(nodes))


def _is_pairs(port, value) -> bool:
    """Whether an input holds pairs: a pair table, or rows over two tables."""
    return port.value_type is ValueType.PAIRS or (
        isinstance(value, Rows) and len(value.aliases) == 2)


def _key(context, alias: str, keys: dict) -> bool:
    """Load an alias's key documents into keys; False when it has none."""
    if alias not in context.deduplicate:
        return False
    if alias not in keys:
        columns = context.sources.get(columns_key(alias))
        if columns is None or FIRST_DOCUMENT not in columns.column_names:
            raise KeyError(f"{alias!r} needs its {FIRST_DOCUMENT!r} column")
        keys[alias] = columns.column(FIRST_DOCUMENT).to_numpy().astype(np.int64)
    return True


def _reduced_rows(value, alias: str, ids, keys):
    """One row per key document, or None when copies carry different values.

    Every copy gets its key document's row back, so a value column is
    only kept when each copy holds the same value as its key's row.
    Otherwise the group runs without deduplication.
    """
    kept, first, inverse = np.unique(keys[ids], return_index=True,
                                     return_inverse=True)
    if isinstance(value, Rows) and value.values.num_columns:
        if not value.values.take(first[inverse]).equals(value.values):
            return None
        rows = value.take(first)
        return Rows({alias: pa.array(kept, pa.int32())}, rows.values)
    kept = pa.array(kept, pa.int32())
    return Rows({alias: kept}) if isinstance(value, Rows) else kept


def _read_pairs(context, aliases, table, keys, every, pairs, node, port,
                reduced) -> None:
    """Record a pair input and reduce it to one pair of key documents each."""
    left, right = aliases
    for alias in aliases:
        if alias not in every:
            every[alias] = np.unique(table.column(alias).to_numpy())
    if not any(_key(context, alias, keys) for alias in aliases):
        return
    pairs[node.node_id] = table
    sides = []
    for alias in aliases:
        documents = table.column(alias).to_numpy()
        mapped = keys[alias][documents] if alias in keys else documents
        sides.append(mapped)
    kept = np.unique(_pair_keys(*sides))
    value = pair_ids_table(left, right, np.column_stack(
        (kept >> 32, kept & 0xFFFFFFFF)))
    if port.value_type is ValueType.ROWS:
        value = Rows({alias: value.column(alias) for alias in aliases})
    reduced[node.node_id][port.name] = value


def _expanded(table: pa.Table, aliases, every, keys, pairs) -> pa.Table:
    """Give each answer row to every document whose key document it names.

    Args:
        table: Answer rows, one int32 id column per alias.
        aliases: The id columns to expand.
        every: alias -> the documents the answers stand for.
        keys: alias -> each document's key document.
        pairs: The pairs the answers stand for when they cover two
            tables, or None for every combination of their documents.
    """
    answered = {alias: table.column(alias).to_numpy() for alias in aliases}
    if pairs is not None and len(aliases) == 2:
        kept = np.ones(pairs.num_rows, dtype=bool)
        for alias in aliases:
            member = np.zeros(len(keys[alias]), dtype=bool)
            member[every[alias]] = True
            kept &= member[pairs.column(alias).to_numpy()]
        found = {alias: pairs.column(alias).to_numpy()[kept] for alias in aliases}
        index, rows = _matches(
            _pair_keys(*(keys[alias][found[alias]] for alias in aliases)),
            _pair_keys(*answered.values()))
        found = {alias: documents[index] for alias, documents in found.items()}
    else:
        rows = np.arange(table.num_rows)
        found = {}
        for alias in aliases:
            index, at = _matches(keys[alias][every[alias]], answered[alias][rows])
            rows = rows[at]
            found = {name: documents[at] for name, documents in found.items()}
            found[alias] = every[alias][index]
    table = table.take(rows)
    for alias, documents in found.items():
        table = table.set_column(
            table.schema.get_field_index(alias), table.schema.field(alias),
            pa.array(documents, pa.int32()))
    return table


def _pair_keys(anchors, partners) -> np.ndarray:
    """One int64 per pair: the anchor's id, then the partner's."""
    return pair_keys(pa.array(anchors, pa.int64()),
                     pa.array(partners, pa.int64())).to_numpy()


def _matches(wanted, answered) -> tuple[np.ndarray, np.ndarray]:
    """Return positions (i, j) with wanted[i] == answered[j], ordered by i."""
    if not len(answered) or not len(wanted):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if answered.dtype.kind == "i" and answered.min() >= 0:
        size = int(max(answered.max(), wanted.max(initial=-1))) + 1
        # a position table only for document ids, not for pair keys,
        # whose range is far larger than their count
        if (size <= 4 * (len(answered) + len(wanted))
                and np.bincount(answered, minlength=size).max() == 1):
            # one answer row per key: look each key up by position
            row_of = np.full(size, -1, dtype=np.int64)
            row_of[answered] = np.arange(len(answered))
            rows = row_of[wanted]
            index = np.flatnonzero(rows >= 0)
            return index, rows[index]
    common = np.result_type(wanted, answered)
    joined = pa.table({"key": wanted.astype(common),
                       "i": np.arange(len(wanted))}).join(
        pa.table({"key": answered.astype(common),
                  "j": np.arange(len(answered))}), keys="key")
    index, rows = (joined.column(name).to_numpy() for name in ("i", "j"))
    order = np.argsort(index, kind="stable")
    return index[order], rows[order]


def _join_survivors(joined: pa.Table, anchor, semantics, reached,
                    num_documents: int) -> np.ndarray:
    """The anchors an AI join keeps, from its answers over every pair."""
    matched = np.zeros(num_documents, dtype=bool)
    matched[true_answer_rows(joined).column(anchor).to_numpy()] = True
    if semantics == "anti":
        return np.sort(reached[~matched[reached]]).astype(np.int32)
    return np.flatnonzero(matched).astype(np.int32)

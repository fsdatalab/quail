"""The rows a model region returns, sent on while the region runs.

The runner runs a region's operators in steps: one operator, or one
pipeline. A row of the region is final once the last step has produced
it. The last step's outputs for some rows, beside the whole outputs of
the steps before it, give the region's rows for those rows, as Rust
builds them from the whole outputs. A ``RegionStream`` sends those
outputs on as the last step's operator reports them, then the rest
once the step finishes, so each row is sent once.

The runner and the operators read the stream from module state: the
region runs on one thread, and the operators sit several calls below
the code that starts it, as with ``quail.progress.answer_sink``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from quail.execution.rows import Rows
from quail.execution.types import output_table

_STREAM = None


def region_stream() -> RegionStream | None:
    """Return the stream of the region running on this process, or None."""
    return _STREAM


@contextlib.contextmanager
def streaming(stream: RegionStream | None):
    """Install a region's stream while the block runs; None installs none."""
    global _STREAM
    previous, _STREAM = _STREAM, stream
    try:
        yield stream
    finally:
        _STREAM = previous


class RegionStream:
    """Sends a region's finished rows on as its last step produces them.

    Args:
        on_outputs: Callable(tables) run with outputs that give some of
            the region's rows: each relation by (node id, port), the
            whole ones of the steps before the last and the last step's
            for the rows finished since the last call.

    Attributes:
        finished: Whether the last step finished and its rest was sent.
    """

    def __init__(self, on_outputs: Callable[[dict], None]):
        self.on_outputs = on_outputs
        self.finished = False
        # the last step's operator whose rows are sent as they finish
        self._node = None
        self._before = {}
        # the aliases naming a sent row, and each sent row's key
        self._aliases = None
        self._sent = []

    def begin(self, graph, results: Mapping, node) -> None:
        """Start the last step.

        Args:
            graph: The graph the runner runs.
            results: Each finished node's NodeResult, by node id.
            node: The step's one operator, whose finished rows the
                executor reports, or None to send the step's rows
                when it finishes.
        """
        self._before = _tables(graph, results)
        self._node = node

    def wants(self, node) -> bool:
        """Whether the node's finished rows go to ``partial``."""
        return self._node is not None and node.node_id == self._node.node_id

    def partial(self, node, outputs: Mapping) -> None:
        """Send the region's rows for some finished rows of the last step.

        Args:
            node: The operator reporting them.
            outputs: Its output values for those rows alone, by port.
        """
        if not self.wants(node):
            return
        tables = {}
        ports = {port.name: port for port in node.outputs}
        for name, value in outputs.items():
            tables[(node.node_id, name)] = output_table(node, ports[name], value)
            if isinstance(value, Rows):
                self._aliases = value.aliases
                self._sent.append(_keys(value.table(), value.aliases))
        self.on_outputs({**self._before, **tables})

    def finish(self, graph, results: Mapping, members) -> None:
        """Send the region's rows of the last step not yet sent.

        Args:
            graph: The graph the runner runs.
            results: Each node's NodeResult, by node id.
            members: The last step's nodes.
        """
        tables = _tables(graph, {member.node_id: results[member.node_id]
                                 for member in members})
        if self._node is not None and self._sent:
            sent = np.concatenate(self._sent)
            for key, table in tables.items():
                if key[0] == self._node.node_id and all(
                        alias in table.column_names for alias in self._aliases):
                    keys = pa.array(_keys(table, self._aliases))
                    tables[key] = table.filter(pc.invert(
                        pc.is_in(keys, value_set=pa.array(sent))))
            rest = [table.num_rows for key, table in tables.items()
                    if key[0] == self._node.node_id]
            if not any(rest):
                self.finished = True
                return
        self.finished = True
        self.on_outputs({**self._before, **tables})


def _tables(graph, results: Mapping) -> dict:
    """Each output relation of the given node results, by (node id, port)."""
    tables = {}
    for node in graph.nodes:
        if node.node_id not in results:
            continue
        ports = {port.name: port for port in node.outputs}
        for name, value in results[node.node_id].outputs.items():
            tables[(node.node_id, name)] = output_table(node, ports[name], value)
    return tables


def _keys(table: pa.Table, aliases) -> np.ndarray:
    """One int64 key per row from its document ids, one or two aliases."""
    columns = [table.column(alias).to_numpy().astype(np.int64) for alias in aliases]
    if len(columns) == 1:
        return columns[0]
    first, second = columns[:2]
    return (first << 32) | second

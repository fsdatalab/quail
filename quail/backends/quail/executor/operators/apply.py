"""Execute per-batch apply calls within a model pipeline."""

import numpy as np

from quail.backends.quail.executor.parts import SingleResultPart
from quail.backends.quail.executor.stages import Stage
from quail.execution.ids import id_positions
from quail.execution.metrics import NodeResult
from quail.execution.runtimes.dataflow import foreign_call, foreign_outputs


class ApplyGate(SingleResultPart):
    """User function applied as each document reaches this pipeline position."""

    stages = ()

    @classmethod
    def prepare(cls, state, nodes, ids, parts, inputs, context):
        """Bind an apply call to its document and partner inputs."""
        node = nodes[0]
        values = {}
        internal = {member.node_id for part in parts for member in part.nodes}
        for port in node.inputs:
            if not port.source.port.startswith("ids:"):
                continue
            port_alias = port.source.port.split(":", 1)[1]
            values[port_alias] = (
                ids if port.source.node_id in internal
                else inputs[node.node_id][port.name])
        call, metrics = foreign_call(node, values, context)
        return cls(node, call, metrics,
                   {alias: id_positions(value) for alias, value in values.items()})

    def __init__(self, node, call, metrics, values):
        self.node = node
        self.call = call
        self.metrics = metrics
        self.values = values
        self.produced = []      # one array of kept ids or pair rows per call
        self.rows = {}          # anchor -> its partners, for pairs

    def gate(self, key):
        document = key[1]
        node = self.node
        if node.ids == "pairs":
            partner = next(alias for alias in node.aliases
                           if alias != key[0])
            produced = self.call({key[0]: [document],
                                  partner: self.values[partner]})
            if len(produced):
                self.rows[document] = np.where(
                    produced[:, 0] == document, produced[:, 1], produced[:, 0])
        else:
            produced = self.call({key[0]: [document]})
        self.produced.append(produced)
        return None if len(produced) else Stage.DROP

    def finish(self, every):
        return None

    def result(self, tokens, gpu_s, chunks, stats) -> NodeResult:
        produced = (np.concatenate(self.produced) if self.produced
                    else np.zeros((0, 2) if self.node.ids == "pairs" else 0,
                                  dtype=np.int64))
        return NodeResult(foreign_outputs(self.node, produced), self.metrics())

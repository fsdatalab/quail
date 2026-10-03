"""Execute per-batch apply calls within a model pipeline."""

from quail.backends.quail.executor.stages import Stage
from quail.execution.runner import NodeResult, foreign_call, foreign_outputs


def apply_part(node, ids, parts, inputs, context):
    """Bind an apply call to its document and partner inputs."""
    values = {}
    for port in node.inputs:
        if not port.source.port.startswith("ids:"):
            continue
        port_alias = port.source.port.split(":", 1)[1]
        values[port_alias] = (
            ids if port.source.node_id in {
                part.node.node_id for part in parts}
            else inputs[node.node_id][port.name])
    call, metrics = foreign_call(node, values, context)
    return ApplyGate(node, call, metrics, values)


class ApplyGate:
    """User function applied as each document reaches this pipeline position."""

    stages = ()
    document_done = None
    result_value = None

    def __init__(self, node, call, metrics, values):
        self.node = node
        self.call = call
        self.metrics = metrics
        self.values = values
        self.produced = []
        self.rows = {}          # anchor -> partners, for pairs

    def gate(self, key):
        document = key[1]
        node = self.node
        if node.ids == "pairs":
            partner = next(alias for alias in node.aliases
                           if alias != key[0])
            partners = list(self.values[partner])
            pairs = self.call({key[0]: [document], partner: partners})
            self.produced.extend(pairs)
            mine = [b if a == document else a for a, b in pairs]
            if not mine:
                return Stage.DROP
            self.rows[document] = mine
            return None
        kept = self.call({key[0]: [document]})
        self.produced.extend(kept)
        return None if kept else Stage.DROP

    def finish(self, every):
        return None

    def result(self, tokens, gpu_s, chunks, stats) -> NodeResult:
        return NodeResult(foreign_outputs(self.node, self.produced),
                          self.metrics())

"""AI.MAP branches through the shared document scheduler.

The document runs the first map after the earlier stages keep it.
Dependent anchors run the other maps at the same stage, borrowing the
document's whole pages. Each branch decodes one row per pass. The
document's stage is reported only after every branch's answer ends.
"""

import time
from dataclasses import replace

import numpy as np
import pyarrow as pa

from quail.backends.quail.executor.extract import load_vocabulary
from quail.backends.quail.executor.parts import (
    ExecutionPart,
    document_inputs,
    execute_parts,
    execution_prefix_tree,
    gpu_seconds,
    join_prior_rows,
    prior_inputs,
    stage_spans,
)
from quail.backends.quail.executor.stages import Stage, StageRequests
from quail.catalog.tokens import DocumentKeys, DocumentPrefixes, PrefixTree
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.rows import Rows


class NoAnswers:
    """A readout for a stage whose answer comes from elsewhere: one 0 per row."""

    dtype = None
    trailing_rows = 1
    normalize = False

    @staticmethod
    def submit(normed, rows_per_answer=None):
        return normed.shape[0]

    @staticmethod
    def result(handle):
        return [0] * handle


def execute_maps(state, nodes, inputs) -> dict:
    """Run a pipeline of maps alone as branches, one per document and map.

    The first node is charged the documents' tokens and the wall time.

    Args:
        state: The loaded model and current query state.
        nodes: The AiMap nodes, in pipeline order.
        inputs: The documents' ids and the tokenized documents by alias.

    Returns:
        Node id -> NodeResult with the node's answer table.
    """
    started = time.perf_counter()
    state.gpu_timing = bool(inputs.get("gpu_timing", False))
    ids = np.asarray(inputs["document_ids"], dtype=np.int64)
    (alias,) = nodes[0].spec.aliases
    part = MapPart(state, nodes, ids, 0)
    prefixes = DocumentPrefixes(nodes[0].document_head,
                                inputs["documents"][alias], ids)
    results = execute_parts(
        state, [part], prefixes, DocumentKeys(alias, ids),
        keep_kv=nodes[-1].keep_kv,
        prefix_tree=execution_prefix_tree(nodes[0], prefixes,
                                          state.loaded_model.arena),
        limit=inputs.get("limit"),
        label=f"map {', '.join(node.spec.name for node in nodes)}")
    first = nodes[0].node_id
    results[first] = replace(results[first], metrics=replace(
        results[first].metrics, wall_s=time.perf_counter() - started))
    return results


class MapPart(ExecutionPart):
    """One pipeline's consecutive maps, decoded as branches at one stage."""

    combine = True
    attention = "unified"
    execute_group = staticmethod(execute_maps)

    @staticmethod
    def model_inputs(node, inputs, context, sequence=None, nodes=None):
        return {**document_inputs(node, inputs, context, sequence, nodes),
                "limit": context.state.get("filter_limit"),
                "gpu_timing": context.state.get("gpu_timing", False)}

    @classmethod
    def prepare(cls, state, nodes, ids, parts, inputs, context):
        priors, prior_tables = prior_inputs(nodes[0], parts,
                                            inputs[nodes[0].node_id])
        first = sum(len(part.answer_limits) for part in parts)
        return cls(state, nodes, ids, first, priors, prior_tables)

    def __init__(self, state, nodes, ids, first, priors=(), prior_tables=()):
        self.state = state
        self.nodes = list(nodes)
        self.node = self.nodes[0]
        self.ids = np.asarray(ids, dtype=np.int64)
        self.first = first          # the pipeline's index of the first map
        self.priors = priors
        self.prior_tables = prior_tables
        self._decoder = None
        self.asked = [np.zeros(len(self.ids), dtype=bool) for _ in self.nodes]
        self.slots = np.arange(len(self.ids), dtype=np.int64)[None, :]
        self._results = {}
        tails = []
        for node in self.nodes:
            head, tail = node.spec.prompt_token_parts
            if len(tail) < 2:
                raise ValueError("an AI.MAP tail ends in its answer cue")
            tails.append(list(tail))
        def requests(key):
            q = key[0][1] - self.first if isinstance(key[0], tuple) else 0
            return StageRequests([tails[q]])

        def decide(anchor, row):
            q, document = self.branch_of[anchor]
            self.asked[q][document] = True
            return True

        self.branch_of = {a: (0, a) for a in range(len(self.ids))}
        self.stages = [Stage(
            suffixes=requests, suffix_tokens=max(map(len, tails)),
            rewind=True, append=True, single=True, readout=NoAnswers(),
            decide=decide, answer_tokens=max(self.answer_limits),
            answer_index=self.first,
            label=f"maps {', '.join(n.spec.name for n in self.nodes)}")]

    def branch_inputs(self, prefixes, keys, tree, page_tokens, stage):
        """Append dependent anchors and their direct document borrows."""
        num_anchors = len(prefixes)
        num_documents = len(self.ids)
        num_dependents = num_documents * (len(self.nodes) - 1)
        self.slots = np.vstack((
            np.arange(num_documents, dtype=np.int64),
            np.arange(num_anchors, num_anchors + num_dependents, dtype=np.int64
                      ).reshape(num_documents, len(self.nodes) - 1).T))
        dependencies = {}
        keys = list(keys)
        parents = [None] * num_anchors if tree is None else list(tree.parent)
        shared = [0] * num_anchors if tree is None else list(tree.shared)
        order = list(range(num_anchors) if tree is None else tree.order)
        positions = list(range(num_anchors))
        for document in range(num_documents):
            alias, document_id = keys[document]
            for q in range(1, len(self.nodes)):
                anchor = int(self.slots[q, document])
                self.branch_of[anchor] = (q, document)
                dependencies[anchor] = (document, stage)
                keys.append(((alias, self.first + q), document_id))
                positions.append(document)
                parents.append(document)
                shared.append(max(0, len(prefixes[document]) - 1)
                              // page_tokens * page_tokens)
        order = [a for parent in order
                 for a in (parent, *(
                     self.slots[1:, parent].tolist()
                     if parent < num_documents else ()))]
        return (DocumentPrefixes((), prefixes, positions), keys,
                PrefixTree(order, parents, shared), dependencies)

    @property
    def decoder(self):
        """The pipeline's MapDecoder, which writes the part's answers."""
        return self._decoder

    @decoder.setter
    def decoder(self, decoder):
        self._decoder = decoder
        decoder.slot_maps[self.first] = {
            anchor: self.first + q for anchor, (q, _) in self.branch_of.items()}
        decoder.share |= self.nodes[0].attention == "tree"

    def _slots(self, q: int, documents) -> np.ndarray:
        """The decoder slots of map q's answers over the documents."""
        documents = np.asarray(documents, dtype=np.int64)
        return self.slots[q, documents]

    @property
    def answer_limits(self) -> tuple:
        return tuple(node.spec.max_answer_tokens for node in self.nodes)

    def finish(self, every) -> int:
        """Return the tokens the maps fed: their tails and their decode rows."""
        tails = sum(int(asked.sum()) * len(node.spec.prompt_token_parts[1])
                    for asked, node in zip(self.asked, self.nodes))
        rows = self.decoder.rows_fed[self.first:self.first + len(self.nodes)]
        self.fed = [int(n) for n in rows]
        return tails + sum(self.fed)

    def complete(self, fresh, spans, stats, offset):
        """Build each map's answer table and attribute its GPU work."""
        tokenizer = load_vocabulary(self.state).tokenizer
        own = [int(asked.sum()) * len(node.spec.prompt_token_parts[1]) + fed
               for asked, node, fed in zip(self.asked, self.nodes, self.fed)]
        for q, node in enumerate(self.nodes):
            spec = node.spec
            asked = np.flatnonzero(self.asked[q])
            tokens = self.decoder.answers(self.first + q, self._slots(q, asked))
            answers = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            (alias,) = spec.aliases
            rows = Rows({alias: self.ids[asked].astype(np.int32)},
                        pa.table({spec.name: pa.array(answers, pa.string())}))
            rows = join_prior_rows(rows, alias, self.priors, self.prior_tables)
            selected = stage_spans(spans, offset, offset + 1)
            fresh_tokens = fresh - sum(own[1:]) if q == 0 else own[q]
            tail = len(spec.prompt_token_parts[1])
            self._results[node.node_id] = NodeResult(
                {node.rows_port: rows},
                NodeMetrics(
                    input_rows=len(asked), output_rows=len(asked),
                    evaluated_documents=len(asked),
                    fresh_tokens=fresh_tokens,
                    gpu_s=gpu_seconds(selected) * fresh_tokens / max(1, fresh),
                    chunks=len(selected),
                    extension={
                        "output": spec.name, "aliases": list(spec.aliases),
                        "input_rows": len(asked),
                        "answer_tokens": int(sum(map(len, tokens))),
                        "decode_rows": self.fed[q],
                        "suffix_tokens": tail,
                        "borrowed_prefix_tokens": (
                            stats.get("borrowed_tokens", 0) if q == 0 else 0),
                        "pack_s": (round(stats.get("pack_s", 0.0), 3)
                                   if q == 0 else 0.0)}))
        self.result_value = self._results[self.nodes[0].node_id]

    def results(self) -> dict:
        return self._results

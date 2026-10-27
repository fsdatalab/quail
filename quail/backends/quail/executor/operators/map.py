"""AI.MAP through the shared document scheduler.

Each map is one stage of its pipeline: the stage writes the map's
instruction after the document, in place of what the operator before
it appended, and its last row chooses the answer's first token. The
pipeline's MapDecoder writes the rest of the answer one row per pass,
and the stage's row is reported once the answer ends, so a limit
counts only documents with a whole answer.

Maps that run on their own over one table run as branches instead:
each (document, map) pair is its own sequence, the branches of a
document borrow its prompt's whole pages from the first branch, and
every branch decodes in the same passes.
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
    gpu_seconds,
    join_prior_rows,
    prior_inputs,
    shared_prefix_tree,
    stage_spans,
)
from quail.backends.quail.executor.stages import Stage, StageRequests
from quail.catalog.tokens import DocumentPrefixes
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
    part = MapPart(state, nodes, ids, 0, branches=True)
    count = len(nodes)
    prefixes = DocumentPrefixes(nodes[0].document_head,
                                inputs["documents"][alias], np.repeat(ids, count))
    keys = [((alias, q), int(document_id))
            for document_id in ids for q in range(count)]
    results = execute_parts(
        state, [part], prefixes, keys, keep_kv=nodes[-1].keep_kv,
        prefix_tree=(shared_prefix_tree(nodes[0].spec.name, prefixes,
                                        state.loaded_model.arena)
                     if count > 1 or nodes[0].spec.share_prefixes else None),
        label=f"map {', '.join(node.spec.name for node in nodes)}")
    first = nodes[0].node_id
    results[first] = replace(results[first], metrics=replace(
        results[first].metrics, wall_s=time.perf_counter() - started))
    return results


class MapPart(ExecutionPart):
    """One pipeline's consecutive maps and their answer tables.

    With branches, the part's sequences are (document, map) pairs,
    document-major, and its one stage feeds each pair its own map's
    tail; otherwise its sequences are the documents and each map is a
    stage.
    """

    combine = True
    attention = "unified"
    execute_group = staticmethod(execute_maps)

    model_inputs = staticmethod(document_inputs)

    @classmethod
    def prepare(cls, state, nodes, ids, parts, inputs, context):
        priors, prior_tables = prior_inputs(nodes[0], parts,
                                            inputs[nodes[0].node_id])
        first = sum(len(part.answer_limits) for part in parts)
        return cls(state, nodes, ids, first, priors, prior_tables)

    def __init__(self, state, nodes, ids, first, priors=(), prior_tables=(),
                 branches=False):
        self.state = state
        self.nodes = list(nodes)
        self.ids = np.asarray(ids, dtype=np.int64)
        self.first = first          # the pipeline's index of the first map
        self.priors = priors
        self.prior_tables = prior_tables
        self._decoder = None
        self.asked = [np.zeros(len(self.ids), dtype=bool) for _ in self.nodes]
        self.branches = branches
        tails = []
        for node in self.nodes:
            head, tail = node.spec.prompt_token_parts
            if len(tail) < 2:
                raise ValueError("an AI.MAP tail ends in its answer cue")
            tails.append(list(tail))
        if branches:
            count = len(self.nodes)

            def requests(key):
                return StageRequests([tails[key[0][1]]])

            def decide(anchor, row):
                self.asked[anchor % count][anchor // count] = True
                return True

            # each branch's tail is its own suffix; the cue's row
            # chooses the answer's first token
            self.stages = [Stage(
                suffixes=requests, suffix_tokens=max(map(len, tails)),
                append=True, single=True, readout=NoAnswers(), decide=decide,
                answer_tokens=max(self.answer_limits),
                label=f"maps {', '.join(n.spec.name for n in self.nodes)}")]
            return
        self.stages = []
        for q, (node, tail) in enumerate(zip(self.nodes, tails)):

            def decide(anchor, row, q=q):
                self.asked[q][anchor] = True
                return True

            # the tail but its cue is the frame; the cue's row chooses
            # the answer's first token
            self.stages.append(Stage(
                suffixes=[[tail[-1]]], frame=list(tail[:-1]), rewind=True,
                append=True, single=True, readout=NoAnswers(), decide=decide,
                answer_tokens=node.spec.max_answer_tokens,
                label=f"map {node.spec.name}"))
        self._results = {}

    @property
    def decoder(self):
        """The pipeline's MapDecoder, which writes the part's answers."""
        return self._decoder

    @decoder.setter
    def decoder(self, decoder):
        self._decoder = decoder
        if self.branches:
            # a branch writes the answer of its own map
            decoder.slot_maps = np.tile(
                np.arange(self.first, self.first + len(self.nodes)), len(self.ids))

    def _slots(self, q: int, documents) -> np.ndarray:
        """The decoder slots of map q's answers over the documents."""
        documents = np.asarray(documents, dtype=np.int64)
        return documents * len(self.nodes) + q if self.branches else documents

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
            selected = stage_spans(spans, offset + q, offset + q + 1)
            tail = len(spec.prompt_token_parts[1])
            self._results[node.node_id] = NodeResult(
                {node.rows_port: rows},
                NodeMetrics(
                    input_rows=len(asked), output_rows=len(asked),
                    evaluated_documents=len(asked),
                    fresh_tokens=fresh - sum(own[1:]) if q == 0 else own[q],
                    gpu_s=gpu_seconds(selected), chunks=len(selected),
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

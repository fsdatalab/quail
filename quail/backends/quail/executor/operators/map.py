"""AI.MAP through the shared document scheduler.

Each map is one stage of its pipeline: the stage writes the map's
instruction after the document, in place of what the operator before
it appended, and its last row chooses the answer's first token. The
pipeline's MapDecoder writes the rest of the answer one row per pass,
and the stage's row is reported once the answer ends, so a limit
counts only documents with a whole answer.
"""

import numpy as np
import pyarrow as pa

from quail.backends.quail.executor.extract import load_vocabulary
from quail.backends.quail.executor.parts import (
    ExecutionPart,
    gpu_seconds,
    join_prior_rows,
    prior_inputs,
    stage_spans,
)
from quail.backends.quail.executor.stages import Stage
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


class MapPart(ExecutionPart):
    """One pipeline's consecutive maps and their answer tables."""

    combine = True
    attention = "unified"

    @staticmethod
    def model_inputs(node, inputs, context, sequence=None, nodes=None):
        if nodes is not None:
            inputs = inputs[node.node_id]
        # the one input carries the documents' ids, or the rows of an
        # earlier operator over them
        (value,) = inputs.values()
        ids = (value.ids[node.spec.aliases[0]].to_numpy()
               if isinstance(value, Rows) else np.asarray(value, dtype=np.int64))
        return {"document_ids": ids}

    @classmethod
    def prepare(cls, state, nodes, ids, parts, inputs, context):
        priors, prior_tables = prior_inputs(nodes[0], parts,
                                            inputs[nodes[0].node_id])
        first = sum(len(part.answer_limits) for part in parts)
        return cls(state, nodes, ids, first, priors, prior_tables)

    def __init__(self, state, nodes, ids, first, priors=(), prior_tables=()):
        self.state = state
        self.nodes = list(nodes)
        self.ids = np.asarray(ids, dtype=np.int64)
        self.first = first          # the pipeline's index of the first map
        self.priors = priors
        self.prior_tables = prior_tables
        self.decoder = None         # the pipeline's MapDecoder, set to run
        self.asked = [np.zeros(len(self.ids), dtype=bool) for _ in self.nodes]
        self.stages = []
        for q, node in enumerate(self.nodes):
            head, tail = node.spec.prompt_token_parts
            if len(tail) < 2:
                raise ValueError("an AI.MAP tail ends in its answer cue")

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
            answers = tokenizer.batch_decode(
                self.decoder.answers(self.first + q, asked),
                skip_special_tokens=True)
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
                        "answer_tokens": int(sum(map(len, self.decoder.answers(
                            self.first + q, asked)))),
                        "decode_rows": self.fed[q],
                        "suffix_tokens": tail,
                        "borrowed_prefix_tokens": (
                            stats.get("borrowed_tokens", 0) if q == 0 else 0),
                        "pack_s": (round(stats.get("pack_s", 0.0), 3)
                                   if q == 0 else 0.0)}))
        self.result_value = self._results[self.nodes[0].node_id]

    def results(self) -> dict:
        return self._results

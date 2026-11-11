"""AI.MAP branches in a region's GPU loop.

The document runs the first map after the earlier stages keep it.
Dependent anchors run the other maps at the same stage, borrowing the
document's whole pages. Each branch decodes one row per pass. The
document's stage is reported only after every branch's answer ends.
"""

from dataclasses import replace
from itertools import groupby

import numpy as np
import pyarrow as pa

from quail.backends.quail.executor.model_operator import (
    ColumnOperator,
    ModelOperator,
    gpu_seconds,
    join_prior_rows,
    operator_label,
    prior_inputs,
    stage_spans,
)
from quail.backends.quail.executor.stages import Stage, StageRequests
from quail.common.tokens import DocumentPrefixes, PrefixTree
from quail.execution.finished import finished_rows, finished_table
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


class MapOperator(ColumnOperator, ModelOperator):
    """A chain's consecutive maps, decoded as branches at one stage."""

    combine = True
    attention = "unified"
    # the word naming the operator's nodes in a progress label
    word = "map"

    @classmethod
    def build(cls, region, nodes, chain):
        """Build the maps; maps alone return their answers without input columns."""
        priors, prior_tables = ((), ()) if region.one_family else prior_inputs(
            nodes[0], chain, region.node_inputs(nodes[0]))
        first = sum(len(operator.answer_limits) for operator in chain)
        return cls(region.query, nodes, region.document_ids(), first, priors,
                   prior_tables)

    @classmethod
    def alone(cls, region, operators, table, settings):
        nodes = [node for operator in operators for node in operator.nodes]
        return replace(settings, label=operator_label(cls.word, nodes))

    def __init__(self, state, nodes, ids, first, priors=(), prior_tables=()):
        self.state = state
        self.nodes = list(nodes)
        self.node = self.nodes[0]
        self.ids = np.asarray(ids, dtype=np.int64)
        self.first = first          # the chain's index of the first map
        self.priors = priors
        self.prior_tables = prior_tables
        self._decoder = None
        self.asked = [np.zeros(len(self.ids), dtype=bool) for _ in self.nodes]
        self.slots = np.arange(len(self.ids), dtype=np.int64)[None, :]
        self._results = {}
        self.finished = []      # (map, document) answers ended since the last take
        tails = []
        for node in self.nodes:
            head, tail = node.spec.prompt_token_parts
            if len(tail) < 2:
                raise ValueError("an AI.MAP tail ends in its answer cue")
            tails.append(list(tail))
        def requests(key):
            branch = isinstance(key[0], tuple) and isinstance(key[0][1], int)
            q = key[0][1] - self.first if branch else 0
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
            label=operator_label("maps", self.nodes))]

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
        """The region's MapDecoder, which writes the operator's answers."""
        return self._decoder

    @decoder.setter
    def decoder(self, decoder):
        self.bind_decoder(decoder)
        decoder.slot_maps[self.first] = {
            anchor: self.first + q for anchor, (q, _) in self.branch_of.items()}

    def bind_decoder(self, decoder):
        """Attach answer storage after the branch slots have been assigned."""
        self._decoder = decoder
        for q, slots in enumerate(self.slots):
            decoder.answer_slots[self.first + q] = slots
        decoder.share |= any(getattr(node, "attention", "") == "tree"
                             for node in self.nodes)

    def _slots(self, q: int, documents) -> np.ndarray:
        """The decoder slots of map q's answers over the documents."""
        documents = np.asarray(documents, dtype=np.int64)
        return self.slots[q, documents]

    @property
    def answer_limits(self) -> tuple:
        return tuple(node.spec.max_answer_tokens for node in self.nodes)

    @property
    def expected_answer_tokens(self) -> tuple:
        return tuple(node.spec.expected_answer_tokens for node in self.nodes)

    def settle(self, document, survived, row):
        """Record that a branch's answer ended: one map's answer for a document."""
        q, position = self.branch_of.get(document, (None, None))
        if q is not None and self.asked[q][position]:
            self.finished.append((q, position))

    def take_finished(self):
        """Return each map's rows whose answers ended since the last call."""
        finished, self.finished = self.finished, []
        if not finished or finished_rows() is None:
            return []
        tables = []
        for q in sorted({q for q, _ in finished}):
            node = self.nodes[q]
            positions = sorted(position for found, position in finished if found == q)
            tables.append(finished_table(node.node_id, node.rows_port,
                                         self.answer_rows(q, positions).table()))
        return tables

    def answer_rows(self, q, positions):
        """The Rows of map q's answers over documents, by position."""
        spec = self.nodes[q].spec
        (values,) = self.column_values(q, positions)
        (alias,) = spec.aliases
        return Rows({alias: self.ids[positions].astype(np.int32)},
                    pa.table({spec.name: values}))

    def read_answers(self, every) -> int:
        """Return the tokens the maps fed: their tails and their decode rows."""
        tails = sum(int(asked.sum()) * len(node.spec.prompt_token_parts[1])
                    for asked, node in zip(self.asked, self.nodes))
        rows = self.decoder.rows_fed[self.first:self.first + len(self.nodes)]
        self.fed = [int(n) for n in rows]
        return tails + sum(self.fed)

    def column_values(self, index, positions) -> tuple:
        tokens = self.decoder.answers(self.first + index, self._slots(index, positions))
        tokenizer = self.state.loaded_model.tokenizer
        # an answer is the model's text without the whitespace around it:
        # the first token of a turn starts with a space
        texts = [text.strip() for text in tokenizer.decode_batch(tokens)]
        return (pa.array(texts, pa.string()),)

    def close(self, fresh, spans, stats, offset) -> dict:
        """Build each map's answer table and attribute its GPU work."""
        own = [int(asked.sum()) * len(node.spec.prompt_token_parts[1]) + fed
               for asked, node, fed in zip(self.asked, self.nodes, self.fed)]
        for q, node in enumerate(self.nodes):
            spec = node.spec
            asked = np.flatnonzero(self.asked[q])
            tokens = self.decoder.answers(self.first + q, self._slots(q, asked))
            rows = join_prior_rows(self.answer_rows(q, asked), spec.aliases[0],
                                   self.priors, self.prior_tables)
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
                        "answers_stopped_early": int(
                            self.decoder.stopped_early[self.first + q]),
                        "suffix_tokens": tail,
                        "borrowed_prefix_tokens": (
                            stats.get("borrowed_tokens", 0) if q == 0 else 0),
                        "pack_s": (round(stats.get("pack_s", 0.0), 3)
                                   if q == 0 else 0.0),
                        "answer_admission": stats.get("answer_admission"),
                        "answer_evictions": (stats.get("num_evictions", 0)
                                             if q == 0 else 0),
                        "recomputed_tokens": (stats.get("recomputed_tokens", 0)
                                              if q == 0 else 0),
                        **(admission_metrics(stats) if q == 0 else {})}))
        return self._results

    def results(self) -> dict:
        return self._results


def admission_metrics(stats) -> dict:
    """Headroom and per-pass work metrics of the map's run, for its first map.

    headroom_granted_pages is the mean headroom an admitted answer was
    given; headroom_granted_share the share given any;
    headroom_running_pages the mean headroom running answers held when
    the curves were rebuilt; and headroom_partial_admissions the
    admissions that committed only some of a document's waiting answers.
    The pass_ metrics count rounds and the tokens, attention pairs, and KV
    tokens read in them, with their SoL seconds.
    """
    granted = np.asarray(stats.get("headroom_granted", []), dtype=float)
    trace = stats.get("headroom_trace", [])
    answers = sum(entry[3] for entry in trace)
    metrics = {
        "headroom_granted_pages": round(float(granted.mean()), 3) if len(granted)
        else 0.0,
        "headroom_granted_share": round(float((granted > 0).mean()), 3)
        if len(granted) else 0.0,
        "headroom_running_pages": round(
            sum(entry[3] * entry[4] for entry in trace) / answers, 3)
        if answers else 0.0,
        "headroom_partial_admissions": stats.get("headroom_partial_admissions", 0),
    }
    metrics["admission_s"] = round(stats.get("admission_s", 0.0), 3)
    for name, value in stats.get("pass_work", {}).items():
        metrics[f"pass_{name}"] = round(value, 3) if isinstance(value, float) else value
    return metrics


class AnswerBranches(MapOperator):
    """Adjacent map and extraction fields sharing one branching stage."""

    def __init__(self, operators):
        self.operators = operators
        first = operators[0]
        super().__init__(first.state,
                         [node for operator in operators for node in operator.nodes],
                         first.ids, first.first)
        self.asked = [asked for operator in operators for asked in operator.asked]

    def settle(self, document, survived, row):
        """Record a branch's ended answer in the operator whose map it is."""
        q, position = self.branch_of.get(document, (None, None))
        if q is None or not self.asked[q][position]:
            return
        for operator in self.operators:
            start = operator.first - self.first
            if start <= q < start + len(operator.nodes):
                operator.finished.append((q - start, position))

    def take_finished(self):
        return [table for operator in self.operators
                for table in operator.take_finished()]

    def branch_inputs(self, prefixes, keys, tree, page_tokens, stage):
        result = super().branch_inputs(prefixes, keys, tree, page_tokens, stage)
        for operator in self.operators:
            start = operator.first - self.first
            operator.slots = self.slots[start:start + len(operator.nodes)]
        return result

    def bind_decoder(self, decoder):
        super().bind_decoder(decoder)
        for operator in self.operators:
            operator.bind_decoder(decoder)

    def read_answers(self, every):
        self.own = [operator.read_answers(every) for operator in self.operators]
        return sum(self.own)

    def close(self, fresh, spans, stats, offset) -> dict:
        for index, operator in enumerate(self.operators):
            tokens = fresh - sum(self.own[1:]) if index == 0 else self.own[index]
            share = tokens / max(1, fresh)
            selected = [({j: n * share for j, n in rows.items()}, total, start, end)
                        for rows, total, start, end in spans]
            operator.close(tokens, selected, stats if index == 0 else {}, offset)
        return self.results()

    def results(self):
        return {key: value for operator in self.operators
                for key, value in operator.results().items()}


def branch_operators(operators):
    """Coalesce adjacent answer operators without crossing dependent conditions."""
    combined = []
    for answers, group in groupby(operators,
                                  lambda operator: isinstance(operator, MapOperator)):
        members = list(group)
        combined.extend([AnswerBranches(members)] if answers and len(members) > 1
                        else members)
    return combined

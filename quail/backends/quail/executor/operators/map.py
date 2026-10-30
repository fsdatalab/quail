"""AI.MAP branches through the shared document scheduler.

The document runs the first map after the earlier stages keep it.
Dependent anchors run the other maps at the same stage, borrowing the
document's whole pages. Each branch decodes one row per pass. The
document's stage is reported only after every branch's answer ends.
"""

import time
from dataclasses import replace
from itertools import groupby

import numpy as np
import pyarrow as pa

from quail.backends.quail.executor.extract import load_vocabulary
from quail.backends.quail.executor.parts import (
    ColumnPart,
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


class MapPart(ColumnPart, ExecutionPart):
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

    def finish(self, every) -> int:
        """Return the tokens the maps fed: their tails and their decode rows."""
        tails = sum(int(asked.sum()) * len(node.spec.prompt_token_parts[1])
                    for asked, node in zip(self.asked, self.nodes))
        rows = self.decoder.rows_fed[self.first:self.first + len(self.nodes)]
        self.fed = [int(n) for n in rows]
        return tails + sum(self.fed)

    def column_values(self, index, positions) -> tuple:
        tokens = self.decoder.answers(self.first + index, self._slots(index, positions))
        tokenizer = load_vocabulary(self.state).tokenizer
        return (pa.array(tokenizer.batch_decode(tokens, skip_special_tokens=True),
                         pa.string()),)

    def complete(self, fresh, spans, stats, offset):
        """Build each map's answer table and attribute its GPU work."""
        own = [int(asked.sum()) * len(node.spec.prompt_token_parts[1]) + fed
               for asked, node, fed in zip(self.asked, self.nodes, self.fed)]
        for q, node in enumerate(self.nodes):
            spec = node.spec
            asked = np.flatnonzero(self.asked[q])
            tokens = self.decoder.answers(self.first + q, self._slots(q, asked))
            (values,) = self.column_values(q, asked)
            (alias,) = spec.aliases
            rows = Rows({alias: self.ids[asked].astype(np.int32)},
                        pa.table({spec.name: values}))
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
                                   if q == 0 else 0.0),
                        "answer_admission": stats.get("answer_admission"),
                        "answer_evictions": (stats.get("num_evictions", 0)
                                             if q == 0 else 0),
                        "recomputed_tokens": (stats.get("recomputed_tokens", 0)
                                              if q == 0 else 0),
                        **(admission_metrics(stats) if q == 0 else {})}))
        self.result_value = self._results[self.nodes[0].node_id]

    def results(self) -> dict:
        return self._results


def admission_metrics(stats) -> dict:
    """Headroom and per-pass work metrics of the map's run, for its first map.

    headroom_granted_pages is the mean headroom an admitted answer was
    given; headroom_granted_share the share given any; and
    headroom_running_pages the mean headroom running answers held when
    the curves were rebuilt. The pass_ metrics count rounds and the
    tokens, attention pairs, and KV tokens read in them, with their SoL
    seconds.
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
    }
    for name, value in stats.get("pass_work", {}).items():
        metrics[f"pass_{name}"] = round(value, 3) if isinstance(value, float) else value
    return metrics


class AnswerBranches(MapPart):
    """Adjacent map and extraction fields sharing one branching stage."""

    def __init__(self, parts):
        self.parts = parts
        first = parts[0]
        super().__init__(first.state, [node for part in parts for node in part.nodes],
                         first.ids, first.first)
        self.asked = [asked for part in parts for asked in part.asked]

    def branch_inputs(self, prefixes, keys, tree, page_tokens, stage):
        result = super().branch_inputs(prefixes, keys, tree, page_tokens, stage)
        for part in self.parts:
            start = part.first - self.first
            part.slots = self.slots[start:start + len(part.nodes)]
        return result

    def bind_decoder(self, decoder):
        super().bind_decoder(decoder)
        for part in self.parts:
            part.bind_decoder(decoder)

    def finish(self, every):
        self.own = [part.finish(every) for part in self.parts]
        return sum(self.own)

    def complete(self, fresh, spans, stats, offset):
        for index, part in enumerate(self.parts):
            tokens = fresh - sum(self.own[1:]) if index == 0 else self.own[index]
            share = tokens / max(1, fresh)
            selected = [({j: n * share for j, n in rows.items()}, total, start, end)
                        for rows, total, start, end in spans]
            part.complete(tokens, selected, stats if index == 0 else {}, offset)

    def results(self):
        return {key: value for part in self.parts
                for key, value in part.results().items()}


def branch_parts(parts):
    """Coalesce adjacent answer parts without crossing dependent conditions."""
    combined = []
    for answers, group in groupby(parts, lambda part: isinstance(part, MapPart)):
        members = list(group)
        combined.extend([AnswerBranches(members)] if answers and len(members) > 1
                        else members)
    return combined

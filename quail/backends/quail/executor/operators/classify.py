"""Classify documents, or the pairs a join kept, and build their label rows."""

import numpy as np
import pyarrow as pa

from quail.backends.quail.executor.label_stages import ClassifyStages, JoinPartners
from quail.backends.quail.executor.model_operator import (
    ColumnOperator,
    SingleNodeOperator,
    join_prior_rows,
    prior_inputs,
)
from quail.backends.quail.reranker import (
    classify_outputs,
    probability_column,
    score_rows,
    scored_batch,
)
from quail.common.expressions import PROBABILITIES_SUFFIX
from quail.common.selection import document_positions
from quail.execution.finished import finished_rows, finished_table
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.progress import answer_sink


class ClassifyOperator(ColumnOperator, SingleNodeOperator):
    """A classification's stages, its streamed labels, and its result table.

    Args:
        node: The AiClassify.
        ids: The document ids it reads.
        plan: Its ClassifyStages.
        priors: (operator, source port) of each rows input from the chain.
        kept: KeptPairs, for the pairs a join kept.
        prior_tables: The rows inputs from outside the chain.
    """

    @classmethod
    def build(cls, region, nodes, chain):
        """Build labels for documents or for the pairs retained by a join."""
        from quail.backends.quail.executor.operators.join import JoinOperator

        node, spec, ids = nodes[0], nodes[0].spec, region.document_ids()
        position = document_positions(ids)
        if spec.partner is None:
            plan = ClassifyStages(region.query, spec, len(ids),
                                  lambda key: position[key[1]])
            plan.seeds = ids
            priors, prior_tables = prior_inputs(node, chain, region.node_inputs(node))
            return cls(node, ids, plan, priors, prior_tables=prior_tables)
        join = next(operator for operator in reversed(chain)
                    if isinstance(operator, JoinOperator))
        stage = next(stage for stage in join.nodes
                     if set(spec.aliases) == {stage.anchor, *stage.partners})
        # the join's partner tuples, one partner alias each
        partner_ids = [int(entry[0]) if isinstance(entry, (tuple, list))
                       else int(entry)
                       for entry in join.inputs["partner_indices"][
                           stage.written_position]]
        documents = region.context.state["documents"][spec.partner]
        kept = KeptPairs(join.nodes.index(stage))
        partners = JoinPartners(
            ids=partner_ids,
            documents=[documents[partner] for partner in partner_ids],
            kept=kept.of)
        plan = ClassifyStages(region.query, spec, len(ids),
                              lambda key: position[key[1]], partners=partners)
        join.add_reader(kept)
        return cls(node, ids, plan, kept=kept)

    def __init__(self, node, ids, plan: ClassifyStages, priors=(), kept=None,
                 prior_tables=()):
        self.node = node
        self.ids = ids
        self.plan = plan
        self.stages = plan.stages
        self.priors = priors
        self.prior_tables = prior_tables
        self.kept = kept
        self.finished = []      # documents settled since the last take
        self.sent = set()       # documents whose labels went to the sink
        self.reported = set()   # documents whose label rows were taken

    def settle(self, document, survived, row):
        self.finished.append(document)

    def take_finished(self):
        """Send the settled documents' labels to the answer sink; return their rows.

        A settled document's label is final once it has one; a document
        whose answer names no label has no row. A settled anchor's pairs
        are labeled once it settles.
        """
        finished, self.finished = self.finished, []
        documents = sorted(set(finished))
        self._send(documents)
        if finished_rows() is None:
            return []
        plan = self.plan
        ready = [document for document in documents
                 if document not in self.reported
                 and (plan.partners is not None or plan.labels[document] is not None)]
        self.reported.update(ready)
        if plan.partners is not None:
            pairs = [(anchor, partner) for anchor in ready
                     for partner in plan.partners.kept(anchor)
                     if (anchor, partner) in plan.pair_labels]
            rows = self._pair_rows(pairs) if pairs else None
        else:
            rows = self._label_rows(ready) if ready else None
        if rows is None:
            return []
        return [finished_table(self.node.node_id, self.node.rows_port, rows.table())]

    def _pair_ids(self, pairs) -> np.ndarray:
        """The (anchor, partner) document ids of pairs given by position."""
        return np.asarray([[self.ids[anchor], self.plan.partners.ids[partner]]
                           for anchor, partner in pairs],
                          dtype=np.int32).reshape(-1, 2)

    def _pair_rows(self, pairs):
        """The label rows of (anchor, partner) pairs given by position."""
        spec, plan = self.node.spec, self.plan
        return score_rows(self._pair_ids(pairs), spec.aliases, spec.name,
                          [plan.pair_labels[pair] for pair in pairs], pa.string())

    def _label_rows(self, documents):
        """The label rows of documents that have a label, with probabilities."""
        spec, plan = self.node.spec, self.plan
        rows = np.asarray([[self.ids[index]] for index in documents],
                          dtype=np.int32).reshape(-1, 1)
        scored = score_rows(rows, spec.aliases, spec.name,
                            [plan.labels[index] for index in documents],
                            pa.string())
        if plan.probabilities is not None:
            scored = scored.with_column(
                spec.name + PROBABILITIES_SUFFIX,
                probability_column(spec.labels, plan.probabilities[documents]))
        return scored

    def _send(self, documents) -> None:
        """Send the labels of documents that have one and were not sent."""
        sink = answer_sink()
        if sink is None or self.plan.partners is not None:
            return
        ready = [document for document in documents
                 if document not in self.sent
                 and self.plan.labels[document] is not None]
        if ready:
            self.sent.update(ready)
            rows = np.asarray(self.ids).reshape(-1, 1)[np.asarray(ready)]
            sink(scored_batch(self.node, rows,
                              [self.plan.labels[document] for document in ready]))

    def read_answers(self, every) -> int:
        """Finalize labels and count tokens processed after the documents."""
        self.reached = len(every[0]) if every else 0
        self.suffix_tokens, self.streamed_tokens = self.plan.finish(
            every, frame_tokens=self.frame_tokens)
        self._send(range(len(self.plan.labels)))
        return self.streamed_tokens

    def column_values(self, index, positions) -> tuple:
        return (pa.array([self.plan.labels[position] for position in positions],
                         pa.string()),)

    def result(self, tokens, gpu_s, chunks, stats) -> NodeResult:
        spec = self.node.spec
        plan = self.plan
        if plan.partners is not None:
            pairs = sorted(plan.pair_labels)
            scored = self._pair_rows(pairs)
            sink = answer_sink()
            if sink is not None and pairs:
                sink(scored_batch(self.node, self._pair_ids(pairs),
                                  scored.column(spec.name).to_pylist()))
            return NodeResult(classify_outputs(self.node, scored), NodeMetrics(
                input_rows=len(pairs), output_rows=len(pairs),
                evaluated_document_pairs=len(pairs), fresh_tokens=tokens,
                gpu_s=gpu_s, chunks=chunks,
                extension={"output": spec.name, "aliases": list(spec.aliases),
                           "input_rows": len(pairs),
                           "suffix_tokens": self.suffix_tokens}))
        # a document whose answer named no label has no row
        labeled = [index for index, label in enumerate(plan.labels)
                   if label is not None]
        scored = join_prior_rows(self._label_rows(labeled), spec.aliases[0],
                                 self.priors, self.prior_tables)
        return NodeResult(classify_outputs(self.node, scored), NodeMetrics(
            input_rows=self.reached, output_rows=len(labeled),
            evaluated_documents=self.reached, fresh_tokens=tokens,
            gpu_s=gpu_s, chunks=chunks,
            extension={
                "output": spec.name, "aliases": list(spec.aliases),
                "input_rows": self.reached,
                "suffix_tokens": self.suffix_tokens,
                "borrowed_prefix_tokens": stats.get("borrowed_tokens", 0),
                "pack_s": round(stats.get("pack_s", 0.0), 3),
            }))


class KeptPairs:
    """Partner indices retained by the join for each anchor document."""

    def __init__(self, stage: int):
        self.stage = stage
        self.partners = {}

    def record(self, anchor, row, indices) -> None:
        """Record the anchor's kept partners from its row of answers."""
        self.partners[anchor] = [
            position if indices is None else indices[position]
            for position, bit in enumerate(row) if bit]

    def of(self, anchor) -> list:
        return self.partners.get(anchor, [])

"""Prepare classification stages and materialize their labels."""

import numpy as np
import pyarrow as pa

from quail.backends.quail.executor.classify import ClassifyStages, JoinPartners
from quail.backends.quail.executor.state import QueryExecutionState
from quail.execution.reranker import (
    _score_table,
    attach_prior_columns,
    classify_outputs,
    scored_batch,
)
from quail.execution.runner import NodeMetrics, NodeResult
from quail.physical import AiClassify, ValueType
from quail.progress import answer_sink


def classification_part(state: QueryExecutionState, node, ids, parts, inputs):
    """Prepare a classification with the labels its inputs provide."""
    position = {document: index for index, document in enumerate(ids)}
    plan = ClassifyStages(state, node.spec, len(ids),
                          lambda key: position[key[1]])
    plan.seeds = ids
    by_node = {part.node.node_id: part for part in parts}
    priors = [(by_node[port.source.node_id], port.source.port)
              for port in node.inputs
              if port.value_type is ValueType.SCORES
              and port.source.node_id in by_node]
    prior_tables = [inputs[port.name] for port in node.inputs
                    if port.value_type is ValueType.SCORES
                    and port.source.node_id not in by_node]
    return ClassifyPart(node, ids, plan, priors, prior_tables=prior_tables)


def joined_classification(state: QueryExecutionState, node, join, ids,
                          join_inputs, context):
    """The part classifying the rows the join keeps, on its anchors."""
    if not isinstance(node, AiClassify) or node.spec.partner is None:
        raise TypeError(
            f"{node.type_name!r} cannot follow a join in its pipeline")
    spec = node.spec
    stage = next(stage for stage in join.stages
                 if set(spec.aliases) == {stage.anchor, *stage.partners})
    # the join's partner tuples, one partner alias each
    partner_ids = [int(entry[0]) if isinstance(entry, (tuple, list))
                   else int(entry)
                   for entry in join_inputs["partner_indices"][
                       stage.written_pos]]
    documents = context.state["docs"][spec.partner]
    kept = KeptPairs(join.stages.index(stage))
    partners = JoinPartners(
        ids=partner_ids,
        documents=[documents[partner] for partner in partner_ids],
        kept=kept.of)
    position = {document: index for index, document in enumerate(ids)}
    plan = ClassifyStages(state, spec, len(ids),
                          lambda key: position[key[1]], partners=partners)
    return ClassifyPart(node, ids, plan, kept=kept)


class ClassifyPart:
    """A classification's stages inside a pipeline."""

    gate = None
    document_done = None
    result_value = None

    def __init__(self, node, ids, plan: ClassifyStages, priors=(), kept=None,
                 prior_tables=()):
        self.node = node
        self.ids = ids
        self.plan = plan
        self.priors = priors      # (part, port) of each label table read
        self.prior_tables = prior_tables
        self.kept = kept          # KeptPairs, for joined rows
        self.stages = plan.stages
        self.reached = 0
        self.suffix_tokens = 0

    def label(self, document) -> str | None:
        """The document's label so far, by its id."""
        return self.plan.labels[self.ids.index(document)] \
            if not hasattr(self, "_position") else \
            self.plan.labels[self._position[document]]

    def finish(self, every) -> int:
        """Label the documents; returns the frame and suffix tokens packed."""
        self.reached = len(every[0]) if every else 0
        self.suffix_tokens, streamed = self.plan.finish(every)
        return streamed

    def result(self, tokens, gpu_s, chunks, stats) -> NodeResult:
        spec = self.node.spec
        plan = self.plan
        if plan.partners is not None:
            pairs = sorted(plan.pair_labels)
            rows = np.asarray(
                [[self.ids[anchor], plan.partners.ids[partner]]
                 for anchor, partner in pairs], dtype=np.int32).reshape(-1, 2)
            table = _score_table(rows, spec.aliases, spec.name,
                                 [plan.pair_labels[pair] for pair in pairs],
                                 pa.string())
            sink = answer_sink()
            if sink is not None and len(rows):
                sink(scored_batch(self.node, rows, table))
            return NodeResult(classify_outputs(self.node, table), NodeMetrics(
                input_rows=len(pairs), output_rows=len(pairs),
                evaluated_document_pairs=len(pairs), fresh_tokens=tokens,
                gpu_s=gpu_s, chunks=chunks,
                extension={"output": spec.name, "aliases": list(spec.aliases),
                           "input_rows": len(pairs),
                           "suffix_tokens": self.suffix_tokens}))
        # a document whose answer named no label has no row
        labeled = [index for index, label in enumerate(plan.labels)
                   if label is not None]
        rows = np.asarray([[self.ids[index]] for index in labeled],
                          dtype=np.int32).reshape(-1, 1)
        table = _score_table(rows, spec.aliases, spec.name,
                             [plan.labels[index] for index in labeled],
                             pa.string())
        # the labels the parts before it gave these documents, as a
        # classify node on its own carries them from its scores input
        priors = {spec.aliases[0]: prior for prior in self.prior_tables}
        priors.update({spec.aliases[0]: part.result_value.outputs[port]
                       for part, port in self.priors})
        table = attach_prior_columns(table, priors)
        sink = answer_sink()
        if sink is not None and len(rows):
            sink(scored_batch(self.node, rows, table))
        return NodeResult(classify_outputs(self.node, table), NodeMetrics(
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
    """The partners the classification's join stage kept for each anchor."""

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

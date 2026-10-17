"""Prepare classification stages and materialize their labels."""

import numpy as np
import pyarrow as pa

from quail.backends.quail.executor.label_stages import (
    ClassifyStages,
    JoinPartners,
    document_prefixes,
)
from quail.backends.quail.executor.parts import (
    AnswerStream,
    SingleResultPart,
    execute_parts,
    shared_prefix_tree,
)
from quail.backends.quail.executor.state import QueryExecutionState
from quail.backends.quail.reranker import (
    RerankerBatch,
    classify_outputs,
    probability_column,
    score_rows,
    scored_batch,
)
from quail.catalog.selection import document_positions
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.logical import PROBABILITIES_SUFFIX
from quail.physical import ValueType
from quail.progress import answer_sink


class ClassifyBatch(SingleResultPart):
    """Classification stages, answer streaming, and batch results."""

    def __init__(self, plan, on_answers=None, prefix_tokens=0):
        self.plan = plan
        self.stages = plan.stages
        self.prefix_tokens = prefix_tokens
        self.stream = AnswerStream(on_answers)
        self.document_done = self.stream.chunk(lambda anchor: plan.labels[anchor])

    def finish(self, every) -> int:
        """Finalize labels and count tokens processed after the documents."""
        self.reached = len(every[0]) if every else 0
        self.suffix_tokens, self.streamed_tokens = self.plan.finish(every)
        self.stream.finish(range(len(self.plan.labels)),
                           lambda anchor: self.plan.labels[anchor])
        return self.streamed_tokens

    def result(self, tokens, gpu_s, chunks, stats) -> RerankerBatch:
        return RerankerBatch(
            self.plan.labels, fresh_tokens=tokens,
            cached_tokens=self.prefix_tokens + self.streamed_tokens - tokens,
            suffix_tokens=self.suffix_tokens,
            borrowed_tokens=stats.get("borrowed_tokens", 0),
            pack_s=stats.get("pack_s", 0.0), gpu_s=gpu_s, chunks=chunks,
            probabilities=self.plan.probabilities)

    def results(self) -> dict:
        return {self.plan.spec.name: self.result_value}


class QuailClassifier:
    """Classify document rows with Quail's token admission and KV arena."""

    def __init__(self, state: QueryExecutionState):
        self.state = state

    def classify(self, spec, rows, documents, on_answers=None) -> RerankerBatch:
        """Classify document rows using the shared scheduler and KV arena.

        Args:
            spec: Classification specification.
            rows: Document IDs to classify, in input order.
            documents: Tokenized documents indexed by table alias and document ID.
            on_answers: Optional callable(row positions, labels) run with each
                chunk's newly labeled rows.

        Returns:
            A RerankerBatch containing labels, optional label probabilities,
            token counts, and execution metrics.
        """
        state = self.state
        rows = np.asarray(rows, dtype=np.int32).reshape(-1)
        prefixes = document_prefixes(spec, documents, rows)
        keys = [("classify", spec.name, index) for index in range(len(rows))]
        tree = (shared_prefix_tree(spec.name, prefixes, state.loaded_model.arena)
                if spec.share_prefixes else None)
        plan = ClassifyStages(state, spec, len(rows), lambda key: key[2])
        # a canvas is drawn from the document's row, so an answer is
        # the same in any batch
        plan.seeds = rows
        part = ClassifyBatch(plan, on_answers, sum(map(len, prefixes)))
        return execute_parts(
            state, [part], prefixes, keys, keep_kv=False, tree=tree,
            label=f"classify {spec.name}")[spec.name]

def classification_part(state: QueryExecutionState, node, ids, parts, inputs):
    """Prepare a classification with the labels its inputs provide."""
    position = document_positions(ids)
    plan = ClassifyStages(state, node.spec, len(ids),
                          lambda key: position[key[1]])
    plan.seeds = ids
    by_node = {part.node.node_id: part for part in parts}
    priors = [(by_node[port.source.node_id], port.source.port)
              for port in node.inputs
              if port.value_type is ValueType.ROWS
              and port.source.node_id in by_node]
    prior_tables = [inputs[port.name] for port in node.inputs
                    if port.value_type is ValueType.ROWS
                    and port.source.node_id not in by_node]
    return ClassifyPart(node, ids, plan, priors, prior_tables=prior_tables)


def joined_classification(state: QueryExecutionState, node, join, ids,
                          join_inputs, context):
    """Build classification stages for the pairs retained by a join."""
    spec = node.spec
    stage = next(stage for stage in join
                 if set(spec.aliases) == {stage.anchor, *stage.partners})
    # the join's partner tuples, one partner alias each
    partner_ids = [int(entry[0]) if isinstance(entry, (tuple, list))
                   else int(entry)
                   for entry in join_inputs["partner_indices"][
                       stage.written_pos]]
    documents = context.state["docs"][spec.partner]
    kept = KeptPairs(join.index(stage))
    partners = JoinPartners(
        ids=partner_ids,
        documents=[documents[partner] for partner in partner_ids],
        kept=kept.of)
    position = document_positions(ids)
    plan = ClassifyStages(state, spec, len(ids),
                          lambda key: position[key[1]], partners=partners)
    return ClassifyPart(node, ids, plan, kept=kept)


class ClassifyPart(ClassifyBatch):
    """Classification stages and result tables within a pipeline."""

    @staticmethod
    def model_inputs(node, inputs, context, sequence=None, nodes=None):
        # scored through the reranker readout over the documents
        return {"score_inputs": inputs, "documents": context.state["docs"]}

    @classmethod
    def prepare(cls, state, nodes, ids, parts, inputs, context):
        """Prepare labels for documents or for the pairs retained by a join."""
        from quail.backends.quail.executor.operators.join import JoinPart

        node = nodes[0]
        if node.spec.partner is None:
            return classification_part(state, node, ids, parts, inputs[node.node_id])
        join = next(part for part in reversed(parts) if isinstance(part, JoinPart))
        part = joined_classification(state, node, join.nodes, ids, join.inputs, context)
        join.add_reader(part.kept)
        return part

    def __init__(self, node, ids, plan: ClassifyStages, priors=(), kept=None,
                 prior_tables=()):
        self.node = node
        self.ids = ids
        self.priors = priors      # (part, port) of each label table read
        self.prior_tables = prior_tables
        self.kept = kept          # KeptPairs, for joined rows
        sink = answer_sink()
        rows = np.asarray(ids).reshape(-1, 1) if sink is not None else None
        super().__init__(plan,
            (lambda positions, values: sink(scored_batch(
                node, rows[positions], values)))
            if sink is not None and plan.partners is None else None)

    def results(self) -> dict:
        return {self.node.node_id: self.result_value}

    @property
    def tested_columns(self) -> tuple[str, ...]:
        """The columns a Filter in the pipeline may test."""
        return (self.node.spec.name,)

    def tested_values(self, column: str, positions) -> pa.Array:
        """Return the labels decoded so far of the documents at the positions."""
        return pa.array([self.plan.labels[position] for position in positions],
                        pa.string())

    def result(self, tokens, gpu_s, chunks, stats) -> NodeResult:
        spec = self.node.spec
        plan = self.plan
        if plan.partners is not None:
            pairs = sorted(plan.pair_labels)
            rows = np.asarray(
                [[self.ids[anchor], plan.partners.ids[partner]]
                 for anchor, partner in pairs], dtype=np.int32).reshape(-1, 2)
            table = score_rows(rows, spec.aliases, spec.name,
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
        table = score_rows(rows, spec.aliases, spec.name,
                           [plan.labels[index] for index in labeled],
                           pa.string())
        if plan.probabilities is not None:
            table = table.with_column(
                spec.name + PROBABILITIES_SUFFIX,
                probability_column(spec.labels, plan.probabilities[labeled]))
        # the labels the parts before it gave these documents, as a
        # classify node on its own carries them from its scores input
        for prior in (*self.prior_tables, *(part.result_value.outputs[port]
                                             for part, port in self.priors)):
            table = table.joined(prior, spec.aliases[0])
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

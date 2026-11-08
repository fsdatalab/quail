"""AI.CLASSIFY under set attention: option-code questions packed in passes."""

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from quail.backends.quail.executor.model_loading import full_output_head
from quail.backends.quail.executor.operators.join import (
    JoinQuestions,
    join_passes,
)
from quail.backends.quail.executor.pass_groups import (
    option_columns,
    option_probabilities,
)
from quail.backends.quail.executor.readout import AsyncPassReadout
from quail.backends.quail.executor.set_passes import own_stage, run_passes
from quail.backends.quail.reranker import (
    classify_outputs,
    probability_column,
    score_rows,
    scored_batch,
)
from quail.common.expressions import PROBABILITIES_SUFFIX
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.progress import answer_sink


@dataclass
class SetLabels:
    """One classification's labels from set-attention passes.

    Attributes:
        labels: Per row, its label, or None for a row no pass reached.
        probabilities: Per row, each label's probability.
        tokens: The fresh tokens of the passes this classification led.
        gpu_s: Their forward-pass GPU seconds.
        chunks: Their forward passes.
        suffix_tokens: The tokens of this classification's question, summed
            over the rows it labeled.
    """

    labels: np.ndarray
    probabilities: np.ndarray
    tokens: int = 0
    gpu_s: float = 0.0
    chunks: int = 0
    suffix_tokens: int = 0


def classify_result(node, ids, labels: SetLabels, priors) -> NodeResult:
    """A classification's NodeResult: the labeled rows joined with its inputs.

    Args:
        node: The AiClassify.
        ids: The document ids it was asked about.
        labels: Its SetLabels.
        priors: The Rows of each of its rows inputs.
    """
    spec = node.spec
    labeled = [index for index, label in enumerate(labels.labels)
               if label is not None]
    rows = np.asarray([[ids[index]] for index in labeled],
                      dtype=np.int32).reshape(-1, 1)
    values = [labels.labels[index] for index in labeled]
    scored = score_rows(rows, spec.aliases, spec.name, values, pa.string())
    if spec.probabilities:
        scored = scored.with_column(
            spec.name + PROBABILITIES_SUFFIX,
            probability_column(spec.labels, labels.probabilities[labeled]))
    for prior in priors:
        scored = scored.joined(prior, spec.aliases[0])
    sink = answer_sink()
    if sink is not None and len(rows):
        sink(scored_batch(node, rows, values))
    return NodeResult(classify_outputs(node, scored), NodeMetrics(
        input_rows=len(ids), output_rows=len(labeled),
        evaluated_documents=len(labeled), fresh_tokens=labels.tokens,
        gpu_s=labels.gpu_s, chunks=labels.chunks,
        extension={"output": spec.name, "aliases": list(spec.aliases),
                   "input_rows": len(ids), "suffix_tokens": labels.suffix_tokens}))


def score_result(node, ids, scores, priors, cost, suffix_tokens) -> NodeResult:
    """A score's NodeResult: the scored rows joined with its inputs.

    Args:
        node: The AiScore.
        ids: The document ids it scored.
        scores: Their scores, in order.
        priors: The Rows of each of its rows inputs.
        cost: The fresh tokens, GPU seconds, and forward passes it led.
        suffix_tokens: Its question's tokens, summed over the rows.
    """
    spec = node.spec
    rows = np.asarray(ids, dtype=np.int32).reshape(-1, 1)
    scored = score_rows(rows, spec.aliases, spec.name, np.asarray(scores))
    for prior in priors:
        scored = scored.joined(prior, spec.aliases[0])
    sink = answer_sink()
    if sink is not None and len(rows):
        sink(scored_batch(node, rows, scores))
    return NodeResult(classify_outputs(node, scored), NodeMetrics(
        input_rows=len(ids), output_rows=len(ids), evaluated_documents=len(ids),
        fresh_tokens=cost["tokens"], gpu_s=cost["gpu_s"], chunks=cost["chunks"],
        extension={"output": spec.name, "aliases": list(spec.aliases),
                   "input_rows": len(ids), "suffix_tokens": suffix_tokens}))


def joined_labels(state, node, join) -> NodeResult:
    """Classify the pairs a join kept, as option-code questions in each anchor's passes.

    Each anchor document is the state again; each partner the join kept
    for it is one question: the partner's label and document, then the
    classification's question and options.

    Args:
        state: The query's execution state.
        node: The AiClassify over the join's pairs.
        join: The JoinOperator that ran the join, under set attention.
    """
    spec = node.spec
    (join_node,) = [member for member in join.nodes if node.classifies_join(member)]
    answer = join.results()[join_node.node_id].outputs[
        f"join_answers:{join_node.written_position}"]
    position_of = {int(document): index for index, document in enumerate(join.ids)}
    kept = {}
    for local, row in answer["rows"].items():
        members = (None if answer["anchor_partners"] is None
                   else answer["anchor_partners"][local])
        members = range(len(row)) if members is None else members
        chosen = [int(member) for member, bit in zip(members, row) if bit]
        if chosen:
            kept[position_of[int(answer["anchor_index"][local])]] = chosen
    _, tail = spec.prompt_token_parts
    questions = JoinQuestions(
        partners=(spec.partner,),
        label_token_ids=((spec.partner, spec.join_layout[1]),),
        tail_token_ids=tuple(tail), separator_tokens=spec.separator_tokens,
        frame_token_ids=spec.join_layout[0],
        question_header_token_ids=spec.question_header_token_ids,
        anchor=spec.anchor, document_ends=node.document_ends)
    targets = sorted({token for ids in spec.label_token_ids for token in ids})
    torch = state.torch
    model_spec = state.loaded_model.model_spec
    passes = join_passes(questions, join.inputs["prefixes"], sorted(kept), kept.get,
                         join.inputs["partner_documents"], model_spec.set_size,
                         state.chunk_tokens)
    passes.stages = [0] * len(passes.states)
    readout = AsyncPassReadout(torch, torch.nn.functional,
                               full_output_head(state.loaded_model.model), targets,
                               None)
    label = f"classify {spec.name}"
    run_passes(state, [own_stage(state, readout, label, passes)], passes,
               label=label)
    rows = np.zeros((0, 2), dtype=np.int32)
    values = np.zeros(0, dtype=object)
    if passes.states:
        # every pass's question rows, in pass order, and their pairs: an
        # anchor's passes cover its kept partners in order
        logits = np.concatenate([np.asarray(value, dtype=np.float64)
                                 for value in passes.values])
        chosen = option_probabilities(
            logits, option_columns(spec.label_token_ids, targets),
            model_spec.answer_temperature).argmax(-1)
        anchors = sorted(kept)
        counts = np.array([len(kept[anchor]) for anchor in anchors])
        tuples = answer["partner_index"]
        partners = np.array([int(tuples[member][0]) for anchor in anchors
                             for member in kept[anchor]], dtype=np.int64)
        rows = np.stack((np.repeat(np.asarray(join.ids, dtype=np.int64)[anchors],
                                   counts), partners), -1).astype(np.int32)
        values = np.asarray(spec.labels, dtype=object)[chosen]
    pairs = rows
    scored = score_rows(rows, spec.aliases, spec.name, list(values), pa.string())
    sink = answer_sink()
    if sink is not None and len(rows):
        sink(scored_batch(node, rows, list(values)))
    return NodeResult(classify_outputs(node, scored), NodeMetrics(
        input_rows=len(pairs), output_rows=len(pairs),
        evaluated_document_pairs=len(pairs), fresh_tokens=passes.tokens,
        gpu_s=passes.gpu_s, chunks=passes.chunks,
        extension={"output": spec.name, "aliases": list(spec.aliases),
                   "input_rows": len(pairs)}))

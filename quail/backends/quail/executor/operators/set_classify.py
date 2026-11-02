"""AI.CLASSIFY under set attention: option-code questions packed in passes."""

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from quail.backends.quail.executor.model_loading import full_output_head
from quail.backends.quail.executor.operators.join import (
    JoinQuestions,
    join_pass_stage,
)
from quail.backends.quail.executor.parts import gpu_seconds, input_staging
from quail.backends.quail.executor.readout import AsyncSlotLogits
from quail.backends.quail.executor.set_passes import (
    group_passes,
    spec_questions,
    state_tails,
)
from quail.backends.quail.executor.stages import Stage, run_stages
from quail.backends.quail.reranker import (
    RerankerBatch,
    classify_outputs,
    probability_column,
    score_rows,
    scored_batch,
)
from quail.catalog.tokens import concat_tokens
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.logical import PROBABILITIES_SUFFIX
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


def option_probabilities(logits, columns, spellings, temperature) -> np.ndarray:
    """Each option's probability: its best spelling's logit, over the temperature.

    Args:
        logits: One question row's logits of the readout's targets.
        columns: Token id -> its column in logits.
        spellings: Per option, its token ids.
        temperature: The model's calibrated answer temperature.
    """
    scores = np.array([max(logits[columns[token]] for token in ids)
                       for ids in spellings]) / temperature
    scores = np.exp(scores - scores.max())
    return scores / scores.sum()


def classify_passes(state, specs, rows, documents) -> list[SetLabels]:
    """Label rows for every classification, their questions packed in passes.

    Each pass reads a document and the state tail, then up to the
    model's set size of the classifications' questions; each question
    is read at its last row.

    Args:
        state: The query's execution state.
        specs: set_choice ClassifySpecs over one table.
        rows: The document ids to label.
        documents: Tokenized documents by alias and document id.
    """
    first = specs[0]
    (alias,) = first.aliases
    head, state_tail, questions = spec_questions(specs)
    tails = state_tails(state)
    prefixes = [tails.prefix(concat_tokens(head, documents[alias][row]), state_tail)
                for row in rows]
    model_spec = state.loaded_model.model_spec
    passes = group_passes(prefixes, questions, first.question_header_token_ids,
                         model_spec.set_size, state.chunk_tokens)
    targets = sorted({token for spec in specs for ids in spec.label_token_ids
                      for token in ids})
    columns = {token: column for column, token in enumerate(targets)}
    torch = state.torch
    readout = AsyncSlotLogits(torch, torch.nn.functional,
                              full_output_head(state.loaded_model.model), targets)
    out = [SetLabels(np.full(len(rows), None, dtype=object),
                     np.full((len(rows), len(spec.labels)), np.nan))
           for spec in specs]
    for members, suffix, shape in passes:
        stage = Stage(suffixes=[suffix], readout=readout, questions=(shape,),
                      decide=lambda _, row: True,
                      label=f"classify pass ({len(members)} questions)")
        answers, spans, tokens = run_stages(
            torch, state.loaded_model.arena, state.loaded_model.pipeline, [stage],
            prefixes, state.chunk_tokens, staging=input_staging(state),
            gpu_timing=state.gpu_timing, paged=False, label=stage.label)
        for position, answer in answers[0].items():
            logits = np.asarray(answer[0], dtype=float)
            for row, member in enumerate(members):
                spec, labels = specs[member], out[member]
                labels.probabilities[position] = option_probabilities(
                    logits[row], columns, spec.label_token_ids,
                    model_spec.answer_temperature)
                labels.labels[position] = spec.labels[
                    int(np.argmax(labels.probabilities[position]))]
                labels.suffix_tokens += shape[row]
        lead = out[members[0]]
        lead.tokens += tokens
        lead.gpu_s += gpu_seconds(spans)
        lead.chunks += len(spans)
    return out


def set_classify_batch(state, spec, rows, documents) -> RerankerBatch:
    """One classification's RerankerBatch under set attention."""
    (labels,) = classify_passes(state, [spec], rows, documents)
    return RerankerBatch(
        labels.labels, fresh_tokens=labels.tokens, cached_tokens=0,
        suffix_tokens=labels.suffix_tokens, gpu_s=labels.gpu_s,
        chunks=labels.chunks,
        probabilities=labels.probabilities if spec.probabilities else None)


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


def joined_label_passes(state, node, join) -> NodeResult:
    """Classify the pairs a join kept, as option-code questions in each anchor's passes.

    Each anchor document is the state again; each partner the join kept
    for it is one question: the partner's label and document, then the
    classification's question and options.

    Args:
        state: The query's execution state.
        node: The AiClassify over the join's pairs.
        join: The JoinPart that ran the join, under set attention.
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
        question_header_token_ids=spec.question_header_token_ids)
    targets = sorted({token for ids in spec.label_token_ids for token in ids})
    columns = {token: column for column, token in enumerate(targets)}
    torch = state.torch
    model_spec = state.loaded_model.model_spec
    stage, states, owners = join_pass_stage(
        questions, join.inputs["prefixes"], sorted(kept), kept.get,
        join.inputs["partner_documents"], state_tails(state),
        AsyncSlotLogits(torch, torch.nn.functional,
                        full_output_head(state.loaded_model.model), targets),
        model_spec.set_size, state.chunk_tokens)
    labels, tokens, spans = {}, 0, []
    if stage is not None:
        answers, spans, tokens = run_stages(
            torch, state.loaded_model.arena, state.loaded_model.pipeline, [stage],
            states, state.chunk_tokens, staging=input_staging(state),
            gpu_timing=state.gpu_timing, paged=False, label=f"classify {spec.name}")
        for index, anchor in enumerate(owners):
            for logits in np.asarray(answers[0][index][0], dtype=float):
                probabilities = option_probabilities(
                    logits, columns, spec.label_token_ids,
                    model_spec.answer_temperature)
                labels.setdefault(anchor, []).append(
                    spec.labels[int(np.argmax(probabilities))])
    tuples = answer["partner_index"]
    pairs = [(int(join.ids[anchor]), int(tuples[member][0]), label)
             for anchor in sorted(labels)
             for member, label in zip(kept[anchor], labels[anchor])]
    rows = np.asarray([pair[:2] for pair in pairs], dtype=np.int32).reshape(-1, 2)
    values = [pair[2] for pair in pairs]
    scored = score_rows(rows, spec.aliases, spec.name, values, pa.string())
    sink = answer_sink()
    if sink is not None and len(rows):
        sink(scored_batch(node, rows, values))
    return NodeResult(classify_outputs(node, scored), NodeMetrics(
        input_rows=len(pairs), output_rows=len(pairs),
        evaluated_document_pairs=len(pairs), fresh_tokens=tokens,
        gpu_s=gpu_seconds(spans), chunks=len(spans),
        extension={"output": spec.name, "aliases": list(spec.aliases),
                   "input_rows": len(pairs)}))

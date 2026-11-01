"""AI.CLASSIFY under set attention: option-code questions packed in passes."""

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from quail.backends.quail.executor.model_loading import full_output_head
from quail.backends.quail.executor.operators.filter import (
    execute_set_filters,
    filter_inputs,
    state_tails,
)
from quail.backends.quail.executor.operators.join import split_passes
from quail.backends.quail.executor.parts import gpu_seconds, input_staging
from quail.backends.quail.executor.readout import AsyncSlotLogits
from quail.backends.quail.executor.stages import Stage, run_stages
from quail.backends.quail.reranker import (
    RerankerBatch,
    classify_outputs,
    probability_column,
    score_rows,
    scored_batch,
)
from quail.catalog.tokens import concat_tokens
from quail.execution.ids import id_positions
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.rows import Rows
from quail.logical import PROBABILITIES_SUFFIX
from quail.physical import AiClassify, AiFilter
from quail.physical.base import ValueType
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

    Raises:
        ValueError: The classifications differ in prompt head or state tail.
    """
    first = specs[0]
    (alias,) = first.aliases
    head, tail = first.prompt_token_parts
    state_tail = list(tail[:first.state_tail_tokens])
    questions = []
    for spec in specs:
        spec_head, spec_tail = spec.prompt_token_parts
        if (tuple(spec_head) != tuple(head)
                or list(spec_tail[:spec.state_tail_tokens]) != state_tail):
            raise ValueError("the classifications of one pass share their "
                             "prompt head and state tail")
        questions.append(list(spec_tail[spec.state_tail_tokens:]))
    tails = state_tails(state)
    prefixes = [tails.prefix(concat_tokens(head, documents[alias][row]), state_tail)
                for row in rows]
    model_spec = state.loaded_model.model_spec
    header = list(first.question_header_token_ids)
    blocks = split_passes(max(map(len, prefixes), default=0), range(len(specs)),
                          [len(question) for question in questions], len(header),
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
    for block in blocks:
        suffix, shape = [], []
        for index, member in enumerate(block):
            question = (header if index else []) + questions[member]
            suffix += question
            shape.append(len(question))
        stage = Stage(suffixes=[suffix], readout=readout, questions=(tuple(shape),),
                      decide=lambda _, row: True,
                      label=f"classify pass ({len(block)} questions)")
        answers, spans, tokens = run_stages(
            torch, state.loaded_model.arena, state.loaded_model.pipeline, [stage],
            prefixes, state.chunk_tokens, staging=input_staging(state),
            gpu_timing=state.gpu_timing, paged=False, label=stage.label)
        for position, answer in answers[0].items():
            logits = np.asarray(answer[0], dtype=float)
            for row, member in enumerate(block):
                spec, labels = specs[member], out[member]
                labels.probabilities[position] = option_probabilities(
                    logits[row], columns, spec.label_token_ids,
                    model_spec.answer_temperature)
                labels.labels[position] = spec.labels[
                    int(np.argmax(labels.probabilities[position]))]
                labels.suffix_tokens += shape[row]
        lead = out[block[0]]
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


def pipeline_ids(first, inputs) -> np.ndarray:
    """The document ids a pipeline's first operator reads."""
    alias = first.document_aliases[0]
    anchor_port = next((port.name for port in first.inputs
                        if port.source.port == f"rows:{alias}"), None)
    source = (inputs[first.node_id][anchor_port] if anchor_port is not None
              else next(iter(inputs[first.node_id].values())))
    return id_positions(source.ids[alias] if isinstance(source, Rows) else source)


def execute_set_pipeline(state, nodes, inputs, context) -> dict:
    """Run a pipeline of AI filters and classifications under set attention.

    Consecutive filters share their stages' passes, and consecutive
    classifications share theirs; each run reads the rows the run
    before it kept.

    Raises:
        ValueError: The pipeline holds another operator.
    """
    alias = nodes[0].document_aliases[0]
    ids = pipeline_ids(nodes[0], inputs)
    runs = []
    for node in nodes:
        if not isinstance(node, (AiFilter, AiClassify)):
            raise ValueError(f"a set-attention pipeline runs AI filters and "
                             f"classifications, not {node.type_name!r}")
        if runs and type(runs[-1][-1]) is type(node):
            runs[-1].append(node)
        else:
            runs.append([node])
    results = {}
    for index, run in enumerate(runs):
        if isinstance(run[0], AiFilter):
            run_inputs = filter_inputs(context.state, run[0], ids)
            if index < len(runs) - 1:
                run_inputs["limit"] = None
            results.update(execute_set_filters(state, run, run_inputs))
            ids = id_positions(results[run[-1].node_id].outputs[
                f"rows:{alias}"].ids[alias])
            continue
        every = classify_passes(state, [node.spec for node in run], ids,
                                context.state["documents"])
        for node, labels in zip(run, every):
            priors = [
                results[port.source.node_id].outputs[port.source.port]
                if port.source.node_id in results
                else inputs[node.node_id][port.name]
                for port in node.inputs if port.value_type is ValueType.ROWS]
            results[node.node_id] = classify_result(node, ids, labels, priors)
    return results

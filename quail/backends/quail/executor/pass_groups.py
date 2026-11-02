"""One set-attention pass group: questions of any kind asked of each document.

A pass group (PackSetPasses) is the operators whose questions share a
pass over each document: AI filters (yes/no), classifications (option
codes), and extractions (pointer reads). run_pass_group asks them all
in one pass per document and decodes the answers for every document
at once.
"""

from dataclasses import dataclass, field

import numpy as np

from quail.backends.quail.executor.model_loading import full_output_head
from quail.backends.quail.executor.parts import gpu_seconds, input_staging
from quail.backends.quail.executor.readout import AsyncPassReadout
from quail.backends.quail.executor.set_passes import group_passes, state_tails
from quail.backends.quail.executor.stages import Stage, run_stages
from quail.physical import AiFilter, ClassifySpec, ExtractSpec


@dataclass(frozen=True)
class SetQuestion:
    """One operator's question in a pass group.

    Attributes:
        kind: "yes_no" for an AI filter, "choice" for a classification,
            or "read" for an extraction.
        head: The prompt tokens before each document.
        state_tail: The tokens a pass reads after the document.
        tokens: The question's tokens after the state tail.
        header: The tokens that open the question after another.
        spellings: For a choice, per option, its token ids.
        slots: The rows read, counted back from the question's last row.
    """

    kind: str
    head: tuple
    state_tail: tuple
    tokens: tuple
    header: tuple
    spellings: tuple = ()
    slots: tuple = (0,)


def node_question(node) -> SetQuestion:
    """The question an AiFilter, or the spec of a classification or extraction, asks."""
    if isinstance(node, AiFilter):
        cut, ids = node.state_tail_tokens, tuple(node.question_token_ids)
        return SetQuestion("yes_no", tuple(node.document_head), ids[:cut],
                           ids[cut:], tuple(node.question_header_token_ids))
    spec = getattr(node, "spec", node)
    head, tail = (tuple(part) for part in spec.prompt_token_parts)
    cut = spec.state_tail_tokens
    if isinstance(spec, ClassifySpec):
        return SetQuestion("choice", head, tail[:cut], tail[cut:],
                           tuple(spec.question_header_token_ids),
                           spellings=tuple(spec.label_token_ids))
    if isinstance(spec, ExtractSpec):
        return SetQuestion("read", head, tail[:cut], tail[cut:],
                           tuple(spec.question_header_token_ids),
                           slots=tuple(spec.read_slots))
    raise TypeError(f"no set-attention question for {type(node).__name__}")


@dataclass
class GroupAnswers:
    """A pass group's answers, for the documents its passes reached.

    Attributes:
        positions: The answered documents, as positions in the documents
            asked, in order.
        answers: Per question: a yes/no question's yes-minus-no margins
            (documents,); a choice's option probabilities (documents,
            options); a read's first row, last row, and raw P(found)
            (documents, 3), its rows counted from the document's first
            token plus one, 0 for nothing.
        suffix_tokens: Per question, its tokens summed over the answered
            documents.
        tokens: The fresh tokens of the group's passes.
        gpu_s: Their forward-pass GPU seconds.
        chunks: Their forward passes.
    """

    positions: np.ndarray
    answers: list
    suffix_tokens: list = field(default_factory=list)
    tokens: int = 0
    gpu_s: float = 0.0
    chunks: int = 0


def option_columns(spellings, targets) -> np.ndarray:
    """Per option, the target columns of its spellings, -1 past its last."""
    column = {token: index for index, token in enumerate(targets)}
    matrix = np.full((len(spellings), max(len(ids) for ids in spellings)), -1,
                     dtype=np.int64)
    for option, ids in enumerate(spellings):
        matrix[option, :len(ids)] = [column[token] for token in ids]
    return matrix


def option_probabilities(logits, matrix, temperature) -> np.ndarray:
    """Each option's probability, from its best spelling, for every row at once.

    Args:
        logits: (rows, targets) target logits at each question's slot.
        matrix: option_columns of the options.
        temperature: The model's calibrated answer temperature.

    Returns:
        (rows, options) probabilities.
    """
    scores = np.where(matrix >= 0, logits[:, np.maximum(matrix, 0)],
                      -np.inf).max(-1) / temperature
    scores = np.exp(scores - scores.max(-1, keepdims=True))
    return scores / scores.sum(-1, keepdims=True)


def _layout(questions, members, yes, no, targets):
    """A pass's slot kinds and, per member, where its answer sits in a row."""
    column = {token: index for index, token in enumerate(targets)}
    kinds, places, code = [], [], 0
    reads = sum(questions[member].kind == "read" for member in members)
    width = len(targets)
    read = 0
    for member in members:
        question = questions[member]
        if question.kind == "read":
            kinds += ["start", "end"]
            places.append(("read", read))
            read += 1
            continue
        kinds.append("code")
        if question.kind == "yes_no":
            places.append(("yes_no", code, [column[t] for t in yes],
                           [column[t] for t in no]))
        else:
            places.append(("choice", code,
                           option_columns(question.spellings, targets)))
        code += 1
    return kinds, places, code * width, reads


def _decode(values, places, code_width, width, temperature):
    """Each member's answers from the (documents, row) values of one pass."""
    documents = values.shape[0]
    code = values[:, :code_width].reshape(documents, -1, width) if code_width else None
    spans = values[:, code_width:].reshape(documents, -1, 3)
    out = []
    for place in places:
        if place[0] == "read":
            out.append(spans[:, place[1]])
        elif place[0] == "yes_no":
            _, slot, yes, no = place
            out.append(code[:, slot, yes].max(-1) - code[:, slot, no].max(-1))
        else:
            _, slot, matrix = place
            out.append(option_probabilities(code[:, slot], matrix, temperature))
    return out


def run_pass_group(state, questions, prefixes, *, limit=None, stop_groups=None,
                   label="pass group") -> GroupAnswers:
    """Ask every question of a pass group of each document.

    Each document's pass reads its prompt head and document, the state
    tail, then the questions, each after the first opened by its header.
    The planner sized the group to fit one chunk; a document longer than
    it planned for splits the group into more passes.

    Args:
        state: The query's execution state.
        questions: The group's SetQuestions, in plan order.
        prefixes: Per document, its prompt head and document tokens.
        limit: Stop admitting documents once this many passed every
            yes/no question; None asks every document.
        stop_groups: Per document, its stop key group, or None.
        label: The progress label.

    Raises:
        ValueError: The questions differ in prompt head or state tail.
    """
    first = questions[0]
    if any(q.head != first.head or q.state_tail != first.state_tail
           for q in questions):
        raise ValueError("the questions of one pass group share their prompt "
                         "head and state tail")
    tails = state_tails(state)
    states = [tails.prefix(prefix, first.state_tail) for prefix in prefixes]
    head = len(first.head)
    reads_any = any(q.kind == "read" for q in questions)
    # a read's document rows: the first row, which stands for nothing,
    # then each document row the state kept
    state_rows = ([np.r_[0, np.arange(head, min(len(prefix), len(kept)))]
                   for prefix, kept in zip(prefixes, states)] if reads_any else None)
    rows = state.answer_rows
    yes = [rows.allowed[i] for i in rows.true_cols.tolist()]
    no = [rows.allowed[i] for i in rows.false_cols.tolist()]
    targets = sorted({token for q in questions if q.kind != "read"
                      for token in (yes + no if q.kind == "yes_no"
                                    else [t for ids in q.spellings for t in ids])})
    model_spec = state.loaded_model.model_spec
    torch = state.torch
    head_weights = full_output_head(state.loaded_model.model)
    answers = [None] * len(questions)
    suffix_tokens = [0] * len(questions)
    group = GroupAnswers(np.zeros(0, dtype=np.int64), answers, suffix_tokens)
    passes = group_passes(states, [q.tokens for q in questions], first.header,
                          model_spec.set_size, state.chunk_tokens)
    if len(passes) > 1:
        # every pass of a split group asks every document, so their
        # answers line up
        limit, stop_groups = None, None
    for members, suffix, shape in passes:
        kinds, places, code_width, reads = _layout(questions, members, yes, no,
                                                   targets)
        yes_no = [(place[1], place[2], place[3]) for place in places
                  if place[0] == "yes_no"]
        width = len(targets)

        def decide(_, row, yes_no=yes_no, width=width):
            values = np.asarray(row[0]).reshape(-1)
            return all(values[slot * width + np.asarray(y)].max()
                       > values[slot * width + np.asarray(n)].max()
                       for slot, y, n in yes_no)

        stage = Stage(
            suffixes=[suffix],
            readout=AsyncPassReadout(torch, torch.nn.functional, head_weights,
                                     targets, kinds, model_spec.read_span_tokens),
            questions=(shape,),
            question_slots=tuple(questions[member].slots for member in members),
            state_reads=(None if not reads else (lambda anchor: state_rows[anchor])),
            decide=decide, label=label)
        found, spans, tokens = run_stages(
            torch, state.loaded_model.arena, state.loaded_model.pipeline, [stage],
            states, state.chunk_tokens, staging=input_staging(state),
            gpu_timing=state.gpu_timing, paged=False, label=label,
            limit=limit, stop_groups=stop_groups)
        positions = np.fromiter(sorted(found[0]), dtype=np.int64,
                                count=len(found[0]))
        values = (np.stack([np.asarray(found[0][int(p)][0], dtype=np.float64)
                            .reshape(-1) for p in positions])
                  if len(positions) else np.zeros((0, code_width + 3 * reads)))
        for member, answer, length in zip(members, _decode(
                values, places, code_width, width, model_spec.answer_temperature),
                shape):
            answers[member] = answer
            suffix_tokens[member] = length * len(positions)
        group.positions = positions
        group.tokens += tokens
        group.gpu_s += gpu_seconds(spans)
        group.chunks += len(spans)
    return group

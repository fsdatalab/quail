"""One set-attention pass group: questions of any kind asked of each document.

A pass group (PackSetPasses) is the operators whose questions share a
pass over each document: AI filters (yes/no), scores (yes/no, read as
a probability), classifications (option codes), and extractions
(pointer reads). A PassGroup asks them all in one pass per document,
in one GPU loop, and decodes the answers for every document at once.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from quail.backends.quail.executor.model_loading import full_output_head
from quail.backends.quail.executor.readout import AsyncPassReadout
from quail.backends.quail.executor.set_passes import Passes, group_passes, run_passes
from quail.backends.quail.executor.stages import Stage
from quail.common.document_ends import ends_for, joined_state
from quail.physical import AiFilter, ClassifySpec, ExtractSpec, ScoreSpec


@dataclass(frozen=True)
class SetQuestion:
    """One operator's question in a pass group.

    Attributes:
        kind: "yes_no" for an AI filter, "score" for a score, "choice"
            for a classification, or "read" for an extraction.
        head: The prompt tokens before each document.
        state_tail: The tokens a pass reads after the document.
        tokens: The question's tokens after the state tail.
        header: The tokens that open the question after another.
        spellings: For a choice, per option, its token ids.
        slots: The rows read, counted back from the question's last row.
        ends: The plan's joins of each document's last token with the
            state tail, as (last token, tokens) pairs (JoinDocumentEnds);
            empty joins nothing.
    """

    kind: str
    head: tuple
    state_tail: tuple
    tokens: tuple
    header: tuple
    spellings: tuple = ()
    slots: tuple = (0,)
    ends: tuple = ()


def node_question(node) -> SetQuestion:
    """The question an AiFilter, or a score, classification, or extraction, asks."""
    spec = node.spec if hasattr(node, "spec") else None
    if isinstance(node, AiFilter):
        cut, ids = node.state_tail_tokens, tuple(node.question_token_ids)
        alias, head = node.alias, tuple(node.document_head)
        question = SetQuestion("yes_no", head, ids[:cut], ids[cut:],
                               tuple(node.question_header_token_ids))
    else:
        head, tail = (tuple(part) for part in spec.prompt_token_parts)
        cut, alias = spec.state_tail_tokens, spec.aliases[0]
        header = tuple(spec.question_header_token_ids)
        if isinstance(spec, ClassifySpec):
            question = SetQuestion("choice", head, tail[:cut], tail[cut:], header,
                                   spellings=tuple(spec.label_token_ids))
        elif isinstance(spec, ScoreSpec):
            question = SetQuestion("score", head, tail[:cut], tail[cut:], header)
        elif isinstance(spec, ExtractSpec):
            question = SetQuestion("read", head, tail[:cut], tail[cut:], header,
                                   slots=tuple(spec.read_slots))
        else:
            raise TypeError(f"no set-attention question for {type(node).__name__}")
    ends = ends_for(node.document_ends, alias, question.state_tail)
    return question if ends is None else replace(question, ends=tuple(ends.items()))


@dataclass
class GroupAnswers:
    """A pass group's answers, for the documents its passes reached.

    Attributes:
        positions: The answered documents, as positions in the documents
            asked, in order.
        answers: Per question: a yes/no question's yes-minus-no margins
            (documents,); a score's probabilities (documents,); a
            choice's option probabilities (documents, options); a read's
            first row, last row, and raw P(found) (documents, 3), its rows
            counted from the document's first token plus one, 0 for
            nothing.
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
        if question.kind in ("yes_no", "score"):
            places.append((question.kind, code, [column[t] for t in yes],
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
        elif place[0] in ("yes_no", "score"):
            kind, slot, yes, no = place
            margin = code[:, slot, yes].max(-1) - code[:, slot, no].max(-1)
            # AsyncScores' probability: the margin over the temperature
            out.append(margin if kind == "yes_no"
                       else 1 / (1 + np.exp(-margin / temperature)))
        else:
            _, slot, matrix = place
            out.append(option_probabilities(code[:, slot], matrix, temperature))
    return out


class PassGroup:
    """A pass group's stages in a set-attention run, and its answers.

    Each document's pass reads its prompt head and document, the state
    tail, then the questions, each after the first opened by its header.
    The planner sized the group to fit one chunk; a document longer than
    it planned for splits the group into more passes, one stage each.

    Args:
        state: The query's execution state.
        questions: The group's SetQuestions, in plan order.
        prefixes: Per document the run may ask, its prompt head and
            document tokens; the longest state sizes the passes.
        label: The progress label.

    Raises:
        ValueError: The questions differ in prompt head or state tail.
    """

    def __init__(self, state, questions, prefixes, label="pass group"):
        first = questions[0]
        if any(q.head != first.head or q.state_tail != first.state_tail
               for q in questions):
            raise ValueError("the questions of one pass group share their prompt "
                             "head and state tail")
        self.state = state
        self.label = label
        self.questions = questions
        self.ends = dict(first.ends) if first.ends else None
        self.reads_any = any(q.kind == "read" for q in questions)
        self.reads = []         # per pass asked, the state rows a read scores
        rows = state.answer_rows
        yes = [rows.allowed[i] for i in rows.true_cols.tolist()]
        no = [rows.allowed[i] for i in rows.false_cols.tolist()]
        targets = sorted({token for q in questions if q.kind != "read"
                          for token in (yes + no if q.kind in ("yes_no", "score")
                                        else [t for ids in q.spellings for t in ids])})
        self.width = len(targets)
        model_spec = state.loaded_model.model_spec
        self.temperature = model_spec.answer_temperature
        torch = state.torch
        head_weights = full_output_head(state.loaded_model.model)
        longest = max((len(self.state_of(prefix)) for prefix in prefixes), default=0)
        self.passes = []
        self.stages = []
        for members, suffix, shape in group_passes(
                longest, [q.tokens for q in questions], first.header,
                model_spec.set_size, state.chunk_tokens):
            kinds, places, code_width, reads = _layout(questions, members, yes, no,
                                                       targets)
            yes_no = [(place[1], place[2], place[3]) for place in places
                      if place[0] == "yes_no"]

            def decide(_, row, yes_no=yes_no, width=self.width):
                values = np.asarray(row[0]).reshape(-1)
                return all(values[slot * width + np.asarray(y)].max()
                           > values[slot * width + np.asarray(n)].max()
                           for slot, y, n in yes_no)

            self.passes.append((members, places, code_width, reads, shape))
            self.stages.append(Stage(
                suffixes=[suffix],
                readout=AsyncPassReadout(torch, torch.nn.functional, head_weights,
                                         targets, kinds, model_spec.read_span_tokens),
                questions=(shape,),
                question_slots=tuple(questions[member].slots for member in members),
                state_reads=self.reads.__getitem__ if reads else None,
                decide=decide, label=label))

    def state_of(self, prefix):
        """A document's state: its prefix, joined with the state tail."""
        return joined_state(prefix, self.questions[0].state_tail, self.ends)

    def ask(self, prefixes, stop_groups=None, limit=None,
            num_key_survivors=1) -> GroupAnswers:
        """Ask every question of the group of each document, in one GPU loop.

        Args:
            prefixes: Per document, its prompt head and document tokens.
            stop_groups: Per document, its stop key group, or None. The
                groups apply to a group of one pass only, so the answers
                of split passes line up.
            limit: Stop once this many documents passed every yes/no
                question, or None.
            num_key_survivors: How many documents of a stop key group
                must pass before the rest of the group is skipped.
        """
        states = [self.state_of(prefix) for prefix in prefixes]
        head = len(self.questions[0].head)
        # a read's document rows: the first row, which stands for nothing,
        # then each document row the state kept
        reads = ([np.r_[0, np.arange(head, min(len(prefix), len(kept)))]
                  for prefix, kept in zip(prefixes, states)]
                 if self.reads_any else [])
        passes = Passes(stop_groups=(None if stop_groups is None
                                     or len(self.passes) > 1 else list(stop_groups)))
        self.reads[:] = []
        for stage, (*_, shape) in enumerate(self.passes):
            passes.stages += [stage] * len(states)
            passes.states += states
            passes.suffix_tokens += [sum(shape)] * len(states)
            passes.owners += range(len(states))
            self.reads += reads
        return self.answers(run_passes(self.state, self.stages, passes,
                                       label=self.label, limit=limit,
                                       num_key_survivors=num_key_survivors),
                            len(states))

    def answers(self, passes, count) -> GroupAnswers:
        """Decode the group's answers from its answered passes.

        Args:
            passes: The group's answered Passes, pass by pass, each over
                every document in order.
            count: The documents asked.
        """
        questions = self.questions
        answers = [None] * len(questions)
        suffix_tokens = [0] * len(questions)
        group = GroupAnswers(np.zeros(0, dtype=np.int64), answers, suffix_tokens,
                             tokens=passes.tokens, gpu_s=passes.gpu_s,
                             chunks=passes.chunks)
        for index, (members, places, code_width, reads, shape) in enumerate(
                self.passes):
            found = passes.values[index * count:(index + 1) * count]
            positions = np.asarray([position for position, value in enumerate(found)
                                    if value is not None], dtype=np.int64)
            values = (np.stack([np.asarray(found[p], dtype=np.float64).reshape(-1)
                                for p in positions])
                      if len(positions) else np.zeros((0, code_width + 3 * reads)))
            for member, answer, length in zip(members, _decode(
                    values, places, code_width, self.width, self.temperature),
                    shape):
                answers[member] = answer
                suffix_tokens[member] = length * len(positions)
            group.positions = positions
        return group

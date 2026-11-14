"""Schedule LLM operator stages while retaining each document's KV.

Filters, joins, and classifications describe their work as Stage objects.
The shared admission scheduler combines documents at different stages in
each forward pass and releases KV when a document finishes or is removed.

A stage list can also write after the document, for an operator that
decodes an answer over several stages. An append stage's request
tokens join the document's KV after its frame, so the next append
stage reads them instead of feeding them again; a rewinding stage
puts its request right after the frame in place of what earlier
stages appended, as the first stage of a new question does; a stage
whose request discards tokens replaces the last tokens an earlier stage
appended, as a stage that verified only part of what the one before
fed does. Admission reserves pages for the frame and for what the
append stages leave in KV, bounded by appended_tokens when the stages
replace some of their own tokens. A stage's suffixes can be each
document's own, chosen when the document enters the stage from what
earlier stages answered, and a stage can read every row of a request
through a ragged readout.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from quail.backends.quail.executor.admission import DROP, SKIP


@dataclass
class StageRequests:
    """Token sequences and retention for one document at one stage.

    Attributes:
        suffixes: Token sequences processed after the document and frame.
        write_suffix_tokens: Leading tokens to retain from the first suffix.
            None uses the stage's append setting.
        read_rows: Trailing output rows to read per suffix. None uses the
            stage's readout settings.
        discard: Tokens to remove from the previously appended KV before
            processing these suffixes.
        questions: Under set attention, per suffix, the tokens of each
            question in it, as Stage.questions gives them for a stage's
            own suffixes.
    """

    suffixes: Any
    write_suffix_tokens: int | None = None
    read_rows: list[int] | None = None
    discard: int = 0
    questions: tuple = ()

    def __post_init__(self):
        from quail.backends.quail.executor.chunk import Suffixes

        self.suffixes = Suffixes.of(self.suffixes)


@dataclass
class Stage:
    """Token requests, answer readout, and advancement condition for one step.

    Attributes:
        suffixes: Token sequences processed after each document and frame,
            or a callback from document key to StageRequests,
            asked when the document enters the stage, so a
            stage can feed what an earlier stage's answer chose for it.
            The callback may return DROP or SKIP as requests does; it
            takes the place of requests, and suffix_tokens bounds it.
        suffix_tokens: The most tokens one sequence of a callback
            suffixes may have; admission and page capacity count it.
        readout: Converts selected hidden states to Boolean answers, scores,
            or category token values. Its dtype determines the answer array.
        frame: Tokens retained after the document. Consecutive identical
            frames are written only once.
        requests: Optional callback from document key to suffix indices.
            None selects all suffixes; DROP removes the document; SKIP
            advances it without a request.
        prepare: Optional callback from a list of document keys, run
            before those documents enter the stage, so requests can
            decide them together and answer each from what it saved.
        decide: Optional callback from document index and answer row to a
            Boolean. True advances the document or retains it at the final
            stage. Without a callback, any true answer is sufficient.
        read_all_rows: Whether to read every suffix row rather than its last.
        read_rows: Optional count of trailing rows to read per suffix,
            overriding read_all_rows. None takes the readout's
            trailing_rows for every suffix when it has more than one.
        single: Whether each document has one request that can be packed
            with its prefix and frame as one entry.
        append: Whether the request's tokens join the document's KV after
            the frame and any tokens earlier append stages added, so the
            next append stage reads them instead of feeding them again.
        rewind: Whether the request's tokens follow the frame directly,
            in place of the tokens earlier append stages added, as the
            first stage of a new question over the same document does.
        label: Name displayed in progress messages.
        paths: Optional trie paths and ancestor references from trie_paths().
        read_mask: Optional mask of the rows read in the sole paths
            suffix, in path order; None reads every row.
        canvas: Optional callback from document index to diffusion canvas
            tokens, shared by its suffixes or supplied per suffix. None uses
            the pipeline's default canvas.
        canvas_rows: Number of rows in each canvas returned by canvas.
        answer_tokens: Limit on tokens a decoder writes after the stage's
            request, as AI.MAP's answer. Admission reserves headroom for each answer.
        answer_index: Index of this stage's first map in the decoder.
        answer_dependencies: Decoder fields this stage's requests read.
        questions: Under set attention, per suffix, the tokens of each
            question in it, in order, after the state that the prefix
            holds; each document requests one suffix, and each
            question's question_slots rows are read. Empty when the
            whole sequence is one question.
        question_slots: The rows read in each question, counted back
            from its last row; or, per question, its own such rows.
        state_reads: Optional callback from document index to the rows
            of its prefix to read before the questions' rows, as a
            pointer read scores them.
    """

    DROP = DROP
    SKIP = SKIP

    suffixes: Any
    readout: Any
    frame: list = field(default_factory=list)
    requests: Callable | None = None
    prepare: Callable | None = None
    decide: Callable | None = None
    read_all_rows: bool = False
    read_rows: Any = None
    single: bool = False
    rewind: bool = False
    append: bool = False
    label: str = ""
    paths: list | None = None
    read_mask: Any = None
    canvas: Callable | None = None
    canvas_rows: int = 0
    suffix_tokens: int = 0
    answer_tokens: int = 0
    answer_index: int = 0
    answer_dependencies: tuple[int, ...] = ()
    questions: tuple = ()
    question_slots: tuple = (0,)
    state_reads: Callable | None = None

    def __post_init__(self):
        if self.own_suffixes:
            if self.suffix_tokens < 1:
                raise ValueError("a stage whose suffixes come from a callback "
                                 "names its most tokens in suffix_tokens")
            if (self.requests is not None or self.paths is not None
                    or self.read_rows is not None or self.canvas is not None):
                raise ValueError("a stage whose suffixes come from a callback "
                                 "takes no requests, paths, read_rows, or canvas")
            return
        trailing = 1 if self.readout is None else self.readout.trailing_rows
        if self.read_rows is None and trailing > 1:
            self.read_rows = [trailing] * len(self.suffixes)
            self.read_all_rows = True

    @property
    def own_suffixes(self) -> bool:
        """Whether each document's suffixes come from the callback."""
        return callable(self.suffixes)


def shared_preamble_tokens(question_ids) -> int:
    """Return the length of the token prefix shared by multiple questions."""
    if len(question_ids) < 2:
        return 0
    p = 0
    while all(len(q) > p and q[p] == question_ids[0][p]
              for q in question_ids):
        p += 1
    return p


def filter_stages(question_ids, readout) -> list:
    """Build one filter stage per question with a shared prompt frame.

    The preamble common to every question is written into the
    document's KV once, at the first stage; each stage's suffix is its
    question past the preamble.

    Raises:
        ValueError: A question has no tokens past the shared preamble.
    """
    p = shared_preamble_tokens(question_ids)
    frame = list(question_ids[0][:p])
    stages = []
    for index, question in enumerate(question_ids):
        tail = list(question[p:])
        if not tail:
            raise ValueError(
                f"stage {index} question has no tokens beyond the shared "
                f"preamble ({p} tokens)")
        stages.append(Stage(suffixes=[tail], readout=readout, frame=frame,
                            single=True, label=f"filter stage {index}"))
    return stages


def frame_writes(stages) -> list:
    """Return which stages need to write a frame not retained by the prior stage."""
    writes = []
    previous = None
    for stage in stages:
        frame = list(stage.frame)
        writes.append(bool(frame) and frame != previous)
        previous = frame
    return writes


def _advance_stage(stages, a, j, row):
    """Apply the stage's decision after its complete row has returned."""
    stage = stages[j]
    if stage.decide is None:
        return bool(any(row) if stage.readout.dtype is None else np.any(row))
    return bool(stage.decide(a, row))


def question_ends(questions, slots=(0,)) -> np.ndarray:
    """The mask of each question's read rows in a suffix of these questions.

    Args:
        questions: Each question's tokens, in order.
        slots: Each read row, counted back from its question's last row;
            or, per question, its own such rows.
    """
    mask = np.zeros(int(sum(questions)), dtype=bool)
    ends = np.cumsum(questions) - 1
    if slots and isinstance(slots[0], tuple):
        for end, own in zip(ends, slots):
            mask[end + np.asarray(own, dtype=np.int64)] = True
        return mask
    for slot in slots:
        mask[ends + slot] = True
    return mask


def state_share_points(parents, shared, limit=None):
    """The token counts at which each document's borrowers start.

    Args:
        parents: Per document, the document it borrows from or None.
        shared: Per document, the tokens it borrows from its parent.
        limit: Most counts to keep per document, or None for all. The
            counts that save the most tokens (borrowers times length)
            are kept; a borrower at another count computes its prefix.

    Returns:
        Per document, the sorted distinct counts its borrowers start
        from; an empty tuple for a document nobody borrows from.
    """
    counts = [{} for _ in parents]
    for parent, count in zip(parents, shared):
        if parent is not None and count:
            counts[parent][count] = counts[parent].get(count, 0) + 1
    points = []
    for found in counts:
        keep = sorted(found, key=lambda n: (-found[n] * n, n))
        points.append(tuple(sorted(keep if limit is None else keep[:limit])))
    return points


@dataclass
class PendingAnswers:
    """One launched chunk's readout and progress through its document decisions."""

    groups: list
    handles: list
    values: list | None = None
    positions: np.ndarray | None = None
    at: int = 0
    step: int = 0

"""Set-attention passes: splitting questions into passes, and running them.

Under set attention a pass is a state, a document and the text after
it, then up to the model's set size of questions. Each operator builds
its questions; these helpers split questions into passes. The state
ends each document as the plan joined it (JoinDocumentEnds).

run_passes asks one step's passes in one GPU loop: each pass is an
anchor of one StageExecutor, which asks it at its stage only.
"""

from dataclasses import dataclass, field

from quail.backends.quail.executor.model_operator import gpu_seconds, input_staging
from quail.backends.quail.executor.stages import Stage


def split_passes(state: int, members, lengths, header: int, set_size: int,
                 budget: int) -> list[list]:
    """Split an anchor's members into passes, in order.

    A pass holds at most set_size questions and fits the chunk budget
    with the state; every question after a pass's first carries the
    header.

    Raises:
        ValueError: One question does not fit a chunk with the state.
    """
    passes, current, used = [], [], state
    for member in members:
        tokens = int(lengths[member])
        if state + tokens > budget:
            raise ValueError(
                f"a join question of {tokens} tokens does not fit one "
                f"{budget}-token chunk with its {state}-token state")
        if current and (len(current) == set_size
                        or used + header + tokens > budget):
            passes.append(current)
            current, used = [], state
        used += tokens + (header if current else 0)
        current.append(member)
    if current:
        passes.append(current)
    return passes


def group_passes(state: int, questions, header, set_size: int, budget: int) -> list:
    """Split a pass group's questions, asked of every document, into passes.

    Every document is asked the same questions, so the passes are the
    same for each; the longest state decides what fits a chunk.

    Args:
        state: The tokens of the longest document's state.
        questions: Each question's tokens, in order.
        header: The tokens that open a question after another.
        set_size: The most questions in one pass.
        budget: The chunk token budget.

    Returns:
        Per pass: its question indices, its suffix tokens, and each
        question's length in the suffix.
    """
    passes = split_passes(state, range(len(questions)),
                          [len(question) for question in questions], len(header),
                          set_size, budget)
    out = []
    for members in passes:
        suffix, shape = [], []
        for index, member in enumerate(members):
            question = (list(header) if index else []) + list(questions[member])
            suffix += question
            shape.append(len(question))
        out.append((members, suffix, tuple(shape)))
    return out


@dataclass
class Passes:
    """The passes one step asks, and their answers once they ran.

    Attributes:
        stages: Per pass, its stage in the step's stage list.
        states: Per pass, its state tokens.
        suffix_tokens: Per pass, the tokens of its questions.
        owners: Per pass, the document or anchor it asks about.
        requests: Per pass, its StageRequests at a stage whose suffixes
            are each pass's own (own_stage), or None.
        reads: Per pass, the state rows a pointer read scores, or None.
        stop_groups: Per pass, its stop key group, or None.
        values: Per pass, its answer, or None for a pass that never ran.
        tokens: The fresh tokens of the passes that ran.
        gpu_s: Their forward-pass GPU seconds.
        chunks: Their forward passes.
        spans: Their forward passes' timing spans.
    """

    stages: list = field(default_factory=list)
    states: list = field(default_factory=list)
    suffix_tokens: list = field(default_factory=list)
    owners: list = field(default_factory=list)
    requests: list | None = None
    reads: list | None = None
    stop_groups: list | None = None
    values: list = field(default_factory=list)
    tokens: int = 0
    gpu_s: float = 0.0
    chunks: int = 0
    spans: list = field(default_factory=list)


def own_stage(state, readout, label: str, passes: Passes) -> Stage:
    """A stage whose one request is each pass's own, from passes.requests."""
    return Stage(suffixes=passes.requests.__getitem__,
                 suffix_tokens=state.chunk_tokens, readout=readout,
                 decide=lambda _, row: True, label=label)


def run_passes(state, stages, passes: Passes, *, label: str, limit=None,
               num_key_survivors=1) -> Passes:
    """Ask a step's passes in one GPU loop; fill in their answers and costs.

    Each pass is one anchor that asks only its stage, settling there.

    Args:
        state: The query's execution state.
        stages: The step's stages; passes.stages indexes them.
        passes: The step's Passes.
        label: The progress label.
        limit: Stop once this many passes survived the last stage, or None.
        num_key_survivors: How many passes of a stop key group must
            survive before the rest of the group is skipped.
    """
    from quail.backends.quail.executor.stage_executor import StageExecutor

    passes.values = [None] * len(passes.states)
    if not passes.states:
        return passes
    loaded = state.loaded_model
    answers, spans, _ = StageExecutor(
        state.torch, loaded.arena, loaded.pipeline, stages, passes.states,
        state.chunk_tokens, staging=input_staging(state), limit=limit,
        paged=False, unit="passes", label=label, default_attention="unified",
        stop_groups=passes.stop_groups, gpu_timing=state.gpu_timing,
        num_key_survivors=num_key_survivors, start_stages=passes.stages).run()
    for anchor, (j, length) in enumerate(zip(passes.stages, passes.suffix_tokens)):
        row = answers[j].get(anchor)
        if row is not None:
            passes.values[anchor] = row[0]
            passes.tokens += len(passes.states[anchor]) + length
    passes.spans = spans
    passes.gpu_s, passes.chunks = gpu_seconds(spans), len(spans)
    return passes

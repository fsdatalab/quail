"""Set-attention passes: splitting questions into passes, and one loop to run them.

Under set attention a pass is a state, a document and the text after
it, then up to the model's set size of questions. Each operator builds
its questions; these helpers split questions into passes. The state
ends each document as the plan joined it (JoinDocumentEnds).

SetPasses runs every step of a chain in one GPU loop: each pass is an
anchor of one StageExecutor, which asks it at its step's stage only.
"""

from dataclasses import dataclass, field

from quail.backends.quail.executor.model_operator import (
    gpu_seconds,
    input_staging,
    stage_spans,
)
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
        stages: Per pass, its stage in the run's stage list.
        states: Per pass, its state tokens.
        suffix_tokens: Per pass, the tokens of its questions.
        owners: Per pass, the document or anchor it asks about.
        requests: Per pass, its StageRequests at a stage whose suffixes
            are each pass's own, or None for the stage's one suffix.
        reads: Per pass, the state rows a pointer read scores, or None.
        stop_groups: Per pass, its stop key group, or None.
        values: Per pass, its answer, or None for a pass that never ran.
        tokens: The fresh tokens of the passes that ran.
        gpu_s: Their forward-pass GPU seconds.
        chunks: Their forward passes.
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


class SetPasses:
    """One GPU loop over the set-attention passes of a chain's steps.

    Each step's operator adds its stages before the run, then a
    generator of steps yields each step's Passes and is sent them
    back answered. A step's passes join the loop once every pass of
    the step before it has answered or been skipped, since a step reads
    what the steps before it kept.

    Args:
        state: The query's execution state.
    """

    def __init__(self, state):
        self.state = state
        self.stages = []
        self.executor = None
        self._requests = {}     # anchor -> its StageRequests at an own stage
        self._reads = {}        # anchor -> the state rows a pointer read scores

    def add_stage(self, stage: Stage) -> int:
        """Add a stage to the run; return its index."""
        self.stages.append(stage)
        return len(self.stages) - 1

    def own_stage(self, readout, label: str) -> int:
        """Add a stage whose one request is each pass's own; return its index."""
        return self.add_stage(Stage(
            suffixes=self._requests.pop, suffix_tokens=self.state.chunk_tokens,
            readout=readout, decide=lambda _, row: True, label=label))

    def state_reads(self, anchor):
        """The state rows a pointer read of the anchor's pass scores."""
        return self._reads[anchor]

    def run(self, steps, *, label: str, limit=None, stop_keys=False,
            num_key_survivors=1):
        """Run a generator's steps of passes; return what the generator returns.

        Args:
            steps: A generator that yields Passes and takes them back
                answered.
            label: The progress label.
            limit: Stop once this many passes survived the last stage,
                or None.
            stop_keys: Whether passes may carry stop key groups.
            num_key_survivors: How many passes of a stop key group must
                survive before the rest of the group is skipped.
        """
        from quail.backends.quail.executor.stage_executor import StageExecutor

        state = self.state
        loaded = state.loaded_model
        # the step waiting for its passes: (Passes, its first anchor), and
        # the first of them still running
        current = [None, 0, 0]
        result = []

        def take(passes):
            """Send answered passes to the generator; queue the next step's."""
            try:
                passes = steps.send(passes)
            except StopIteration as stop:
                result.append(stop.value)
                return None
            return passes

        def arrivals(executor):
            while True:
                passes, first, at = current
                if passes is not None:
                    end = first + len(passes.states)
                    while at < end and executor.sched.ended(at):
                        at += 1
                    current[2] = at
                    if at < end:
                        return
                    passes = take(self._answered(passes, first))
                else:
                    passes = take(None)
                if passes is None:
                    current[0] = None
                    executor.input_ended()
                    return
                first = len(executor.keys)
                self._queue(executor, passes, first, stop_keys)
                current[:] = [passes, first, first]

        self.executor = StageExecutor(
            state.torch, loaded.arena, loaded.pipeline, self.stages, [],
            state.chunk_tokens, anchor_keys=[], on_settled=None,
            staging=input_staging(state), attention_mode=None, prefix_tree=None,
            stats=None, limit=limit, paged=False, unit="passes",
            count_answers=False, label=label, default_attention="unified",
            on_chunk=None, on_answers=None,
            stop_groups=[] if stop_keys else None, gpu_timing=state.gpu_timing,
            num_key_survivors=num_key_survivors, arrivals=arrivals)
        self.executor.run()
        # a limit ends the run with a step still waiting: its passes that
        # never ran have no answers
        passes, first, _ = current
        while not result:
            passes = take(None if passes is None else self._answered(passes, first))
            first = None
        return result[0]

    def _queue(self, executor, passes, first, stop_keys) -> None:
        """Add a step's passes to the run as anchors at their stages."""
        for index in range(len(passes.states)):
            if passes.requests is not None and passes.requests[index] is not None:
                self._requests[first + index] = passes.requests[index]
            if passes.reads is not None:
                self._reads[first + index] = passes.reads[index]
        stop_groups = None
        if stop_keys:
            stop_groups = (passes.stop_groups if passes.stop_groups is not None
                           else [None] * len(passes.states))
        executor.add_documents(passes.states, stop_groups=stop_groups,
                               start_stages=passes.stages)

    def _answered(self, passes, first) -> Passes:
        """Fill in the answers and costs of a step's passes."""
        if first is None or not passes.states:
            passes.values = [None] * len(passes.states)
            return passes
        executor = self.executor
        answers = executor.sched.answers
        values = []
        for index, (j, length) in enumerate(zip(passes.stages,
                                                passes.suffix_tokens)):
            anchor = first + index
            self._reads.pop(anchor, None)
            row = answers[j].pop(anchor, None)
            values.append(None if row is None else row[0])
            if row is not None:
                passes.tokens += len(passes.states[index]) + length
        passes.values = values
        spans = stage_spans(executor.spans, min(passes.stages),
                            max(passes.stages) + 1)
        passes.gpu_s, passes.chunks = gpu_seconds(spans), len(spans)
        return passes

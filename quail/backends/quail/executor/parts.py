"""Compose operator stages and collect their execution metrics."""

import time

from quail.backends.quail.executor.chunk import InputStaging
from quail.backends.quail.executor.stages import Stage
from quail.backends.quail.executor.state import QueryExecutionState
from quail.execution.tokens import prefix_tree
from quail.physical import AiClassify
from quail.progress import logger


def input_staging(state: QueryExecutionState):
    """Return reusable transfer buffers with an empty fixed-token cache."""
    if state.loaded_model.input_staging is None:
        state.loaded_model.input_staging = InputStaging(state.torch)
    state.loaded_model.input_staging.fixed_tokens.clear()
    return state.loaded_model.input_staging


class StagesPart:
    """Additional stages contributed by a pipeline's final operator."""

    node = None
    gate = None
    document_done = None
    result_value = None

    def __init__(self, stages):
        self.stages = stages

    def finish(self, every):
        return None

    def result(self, tokens, gpu_s, chunks, stats):
        return None


def gated_stages(parts) -> list:
    """Combine operator stages and attach intervening conditions.

    Args:
        parts: Pipeline parts in execution order. Parts without stages
            contribute a condition checked before the next stage.

    Returns:
        The combined Stage list, with conditions attached to stage requests.

    Raises:
        ValueError: A trailing condition has no following stage.
    """
    stages = []
    pending = []
    for part in parts:
        if not part.stages:
            pending.append(part.gate)
            continue
        first = part.stages[0]
        if pending:
            gates = list(pending)
            asked = first.requests

            def guarded(key, gates=gates, asked=asked):
                for gate in gates:
                    if gate(key) is Stage.DROP:
                        return Stage.DROP
                return None if asked is None else asked(key)

            first.requests = guarded
            pending = []
        stages.extend(part.stages)
    if pending:
        raise ValueError("a pipeline ends with a gate and no stage after it")
    return stages


def _part_slices(parts) -> list:
    offset = 0
    slices = []
    for part in parts:
        slices.append((offset, offset + len(part.stages)))
        offset += len(part.stages)
    return slices


def report_chain_transitions(parts, transitions) -> None:
    """Notify filter parts when documents pass or fail their final stage.

    Args:
        parts: Pipeline parts in execution order.
        transitions: (document index, stage index, passed) completion records.
            Intermediate successful stages do not complete a filter part.
    """
    for part, (low, high) in zip(parts, _part_slices(parts)):
        if part.document_done is None:
            continue
        finished = [(anchor, stage - low, passed)
                    for anchor, stage, passed in transitions
                    if low <= stage < high
                    and (not passed or stage == high - 1)]
        if finished:
            part.document_done(finished)


def complete_chain(parts, every, spans, tokens, stats, torch, inputs):
    """Finalize pipeline parts and store each operator's result.

    The first part with stages receives document token costs, prefix-sharing
    counts, and packing time. Later parts receive their own frame and suffix
    token costs. GPU time is apportioned by the rows each part processed.

    Args:
        parts: Pipeline parts in execution order.
        every: Answer mappings for all stages.
        spans: Per-chunk row counts and timing events.
        tokens: Fresh token count excluding the final consumer's own work.
        stats: Packing time and prefix-sharing counts.
        torch: Torch module used to synchronize timing events.
        inputs: Execution settings, including whether GPU timing is enabled.
    """
    slices = _part_slices(parts)
    own = [part.finish(every[low:high])
           for part, (low, high) in zip(parts, slices)]
    staged = [index for index, part in enumerate(parts) if part.stages]
    remainder = tokens - sum(own[index] or 0 for index in staged[1:])
    for index, (part, (low, high)) in enumerate(zip(parts, slices)):
        part_spans = stage_spans(spans, low, high)
        part.result_value = part.result(
            remainder if staged and index == staged[0] else (own[index] or 0),
            gpu_seconds(torch, part_spans, inputs),
            chunks(part_spans, inputs), stats if staged and index == staged[0]
            else {})


def execution_prefix_tree(node, documents, arena):
    """Build the requested prefix tree, or return None if sharing is disabled."""
    shares = (node.spec.share_prefixes if isinstance(node, AiClassify)
              else node.share_prefixes)
    if not shares:
        return None
    started = time.perf_counter()
    tree = prefix_tree(documents, arena.page_tokens)
    logger.info(
        "prefix sharing on %s: %s documents borrow %s tokens "
        "(tree built in %.2f s)",
        node.spec.aliases[0] if isinstance(node, AiClassify)
        else getattr(node, "alias", None) or node.anchor,
        len(documents), tree.shared_tokens, time.perf_counter() - started)
    return tree


def stage_spans(spans, low, high) -> list:
    """Select chunk timing spans covering a range of stages.

    Args:
        spans: Tuples of stage row counts, chunk tokens, and CUDA events.
        low: First included stage index.
        high: First excluded stage index.

    Returns:
        Spans with only the selected stage row counts. Total chunk token
        counts remain unchanged for proportional GPU time accounting.
    """
    narrowed = []
    for by_stage, tokens, start, end in spans:
        rows = {j: n for j, n in by_stage.items() if low <= j < high}
        if rows:
            narrowed.append((rows, tokens, start, end))
    return narrowed


def gpu_seconds(torch, spans, inputs) -> float:
    """Calculate GPU seconds attributable to the selected stage rows.

    Args:
        torch: Torch module used to synchronize CUDA events.
        spans: Selected stage row counts and timing events per chunk.
        inputs: Execution settings, including gpu_timing.

    Returns:
        GPU seconds weighted by each span's fraction of chunk rows, or 0.0
        if GPU timing is disabled.
    """
    if not inputs.get("gpu_timing"):
        return 0.0
    # every chunk's answers were read, so its end event has completed
    torch.cuda.synchronize()
    return sum(
        start.elapsed_time(end) * min(1.0, sum(rows.values()) / tokens)
        for rows, tokens, start, end in spans if tokens) / 1000.0


def chunks(spans, inputs) -> int:
    """Return the number of timed forward chunks, or zero if timing is disabled."""
    return len(spans) if inputs.get("gpu_timing") else 0

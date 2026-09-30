"""Compose operator stages and collect their execution metrics."""

import time

from quail.backends.quail.executor import loop
from quail.backends.quail.executor.stages import Stage
from quail.execution.tokens import prefix_tree
from quail.physical import AiClassify
from quail.progress import logger


def input_staging(state):
    """The node's reusable input transfer buffers."""
    if "input_staging" not in state:
        state["input_staging"] = loop.InputStaging(state["torch"])
    state["input_staging"].fixed_tokens.clear()
    return state["input_staging"]


class StagesPart:
    """Stages the sink of a pipeline adds after the parts before it."""

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
    """The parts' stages in order; each gate guards the next stage."""
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
    """Hand each filter part the documents that settled at its stages."""
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
    """Finish every part of a driven chain and store its result.

    tokens are the run's fresh tokens less the consumer's own. The
    first part with stages, which packed the documents, takes them
    less the frames and suffixes of the parts after it; prefix
    borrowing and packing time go to it as well.
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
    """The node's prefix tree, or None when the plan did not ask for one."""
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
    """The chunks that packed rows of stages low to high, with those rows.

    A chunk's tokens stay whole, so a part's share of the chunk's time
    is its rows over the chunk's tokens.
    """
    narrowed = []
    for by_stage, tokens, start, end in spans:
        rows = {j: n for j, n in by_stage.items() if low <= j < high}
        if rows:
            narrowed.append((rows, tokens, start, end))
    return narrowed


def gpu_seconds(torch, spans, inputs) -> float:
    """Seconds the chunks ran on the GPU, by the spans' share of rows.

    0.0 unless timing was asked for.
    """
    if not inputs.get("gpu_timing"):
        return 0.0
    # every chunk's answers were read, so its end event has completed
    torch.cuda.synchronize()
    return sum(
        start.elapsed_time(end) * min(1.0, sum(rows.values()) / tokens)
        for rows, tokens, start, end in spans if tokens) / 1000.0


def chunks(spans, inputs) -> int:
    """Forward chunks the spans' rows ran in; 0 unless timing was asked for."""
    return len(spans) if inputs.get("gpu_timing") else 0


def require_execution_state(state) -> None:
    """Check that the model and query have been attached."""
    missing = {
        "torch", "async_answers", "chunk_tokens",
        "arena", "pipeline",
    } - set(state)
    if missing:
        raise RuntimeError(
            f"Quail model execution is missing state {sorted(missing)}")

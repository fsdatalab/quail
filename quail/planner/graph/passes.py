"""Set-attention passes: which operators of a plan share a pass over each document.

Under set attention, a pass reads a document and then up to the
model's set size of questions; nothing is kept between passes. In each
pipeline, a run of consecutive AI filters, of classifications of one
document, or of extractions can share passes. set_pass_groups splits
each run into pass groups with choose_pass_groups; the PackSetPasses
rule writes them on the plan, and pricing, explain(), and the executor
read them back with pass_members.
"""

from __future__ import annotations

from itertools import groupby

from quail.cost import budgets
from quail.cost.packing import choose_pass_groups
from quail.logical import effective_selectivity
from quail.logical.prompts import PROMPT_LAYOUTS
from quail.physical import AiClassify, AiExtract, AiFilter, PhysicalGraph
from quail.physical.pipelines import build_pipelines
from quail.planner.statistics.tables import summarize_alias


def packed_kind(node) -> str | None:
    """The kind of question a node asks in a set-attention pass, or None."""
    if isinstance(node, AiFilter):
        return "filter"
    if isinstance(node, AiClassify) and node.spec.partner is None:
        return "label"
    if isinstance(node, AiExtract):
        return "extract"
    return None


def expected_documents(node) -> float:
    """The documents an operator is expected to be asked about."""
    if isinstance(node, AiFilter):
        return node.expected_rows
    return node.spec.expected_rows


def head_tokens(node) -> int:
    """The prompt tokens before each document."""
    if isinstance(node, AiFilter):
        return node.preamble_tokens
    return len(node.spec.prompt_token_parts[0])


def question_tokens(node, state_tail: int) -> int:
    """An operator's question tokens after the state tail."""
    if isinstance(node, AiFilter):
        return max(1, node.question_tokens - state_tail)
    return len(node.spec.prompt_token_parts[1]) - node.spec.state_tail_tokens


def set_pass_groups(graph: PhysicalGraph, context) -> dict[str, int]:
    """Each packable operator's pass group, numbered across the plan.

    Args:
        graph: The physical graph.
        context: The planning context, with the model, device,
            tokenizer, and document tokens.
    """
    model = context.model
    layout = PROMPT_LAYOUTS[model.prompt_layout]
    state_tail = len(context.tokenizer(layout.question_separator
                                       + layout.question_label))
    header = len(context.tokenizer(layout.question_label))
    chunk = budgets.chunk_budget(model, context.device)
    pipelines = build_pipelines(graph)
    groups, seen, first = {}, set(), 0
    for node in graph.topological_nodes():
        pipeline = pipelines.get(node.node_id)
        if pipeline is None or id(pipeline) in seen:
            continue
        seen.add(id(pipeline))
        for kind, run in groupby(pipeline, key=packed_kind):
            if kind is None:
                continue
            run = list(run)
            summary = summarize_alias(
                context.document_tokens[run[0].document_aliases[0]],
                model.sliding_window, memo=context.memo)
            state = head_tokens(run[0]) + state_tail
            chosen = choose_pass_groups(
                [question_tokens(member, state_tail) for member in run],
                [effective_selectivity(member.selectivity) if kind == "filter"
                 else 1.0 for member in run],
                expected_documents(run[0]), state + summary.mean,
                state + summary.maximum, model, context.device, chunk,
                set_size=model.set_size, header=header)
            groups.update((member.node_id, first + group)
                          for member, group in zip(run, chosen))
            first += max(chosen) + 1
    return groups


def pass_members(graph: PhysicalGraph) -> list[tuple]:
    """The operators of each pass group, in execution order."""
    members = {}
    for node in graph.topological_nodes():
        if getattr(node, "pass_group", -1) >= 0:
            members.setdefault(node.pass_group, []).append(node)
    return [tuple(nodes) for nodes in members.values()]

"""Set-attention passes: which operators of a plan share a pass over each document.

Under set attention, a pass reads a document and then up to the
model's set size of questions; nothing is kept between passes. In each
pipeline, a run of AI filters, classifications of one document,
extractions, and scores of one document, with any conditions on their
outputs between them, can share passes. set_pass_groups splits each
run into pass groups with choose_pass_groups; the PackSetPasses rule
writes them on the plan, and pricing, explain(), and the executor read
them back.
"""

from __future__ import annotations

from quail.cost import budgets
from quail.cost.packing import choose_pass_groups
from quail.logical.prompts import PROMPT_LAYOUTS
from quail.physical import (
    AiClassify,
    AiExtract,
    AiFilter,
    AiJoin,
    AiScore,
    Filter,
    PhysicalGraph,
)
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
    if isinstance(node, AiScore) and node.document_aliases:
        return "score"
    return None


def document_tails(node) -> tuple:
    """The tables a set-attention node reads and the tokens after their documents.

    Returns:
        Per table and tail, (alias, the prompt head before each document,
        the tail): a pass's state tail for a packable operator, the frame
        after each anchor and the separator after each partner for a join
        or a classification of its pairs; empty for any other node.
    """
    if isinstance(node, AiFilter):
        return ((node.alias, tuple(node.document_head),
                 tuple(node.question_token_ids[:node.state_tail_tokens])),)
    if isinstance(node, AiJoin):
        separator = tuple(node.tail_token_ids[:node.separator_tokens])
        return ((node.anchor, tuple(node.document_head),
                 tuple(node.frame_token_ids)),
                *((partner, (), separator) for partner in node.partners))
    if isinstance(node, AiClassify) and node.spec.partner is not None:
        spec = node.spec
        head, tail = spec.prompt_token_parts
        return ((spec.anchor, tuple(head), tuple(spec.join_layout[0])),
                (spec.partner, (), tuple(tail[:spec.separator_tokens])))
    if packed_kind(node):
        head, tail = node.spec.prompt_token_parts
        return ((node.spec.aliases[0], tuple(head),
                 tuple(tail[:node.spec.state_tail_tokens])),)
    return ()


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
        for chain in chains(pipeline):
            summary = summarize_alias(
                context.document_tokens[chain[0].document_aliases[0]],
                model.sliding_window, memo=context.memo)
            state = head_tokens(chain[0]) + state_tail
            rows = [expected_documents(member) for member in chain]
            # the share of rows that reach the next operator: a filter's
            # selectivity, and a condition's between them
            kept = [later / earlier if earlier else 1.0
                    for earlier, later in zip(rows, rows[1:])] + [1.0]
            chosen = choose_pass_groups(
                [question_tokens(member, state_tail) for member in chain], kept,
                rows[0], state + summary.mean, state + summary.maximum, model,
                context.device, chunk, set_size=model.set_size, header=header)
            groups.update((member.node_id, first + group)
                          for member, group in zip(chain, chosen))
            first += max(chosen) + 1
    return groups


def chains(pipeline) -> list[list]:
    """The pipeline's runs of packable operators, in order.

    A condition on their outputs stays inside a run, since it asks the
    model nothing; any other operator, such as a join, ends it.
    """
    out, current = [], []
    for node in pipeline:
        if packed_kind(node):
            current.append(node)
        elif not (isinstance(node, Filter) and len(node.aliases) == 1):
            if current:
                out.append(current)
            current = []
    if current:
        out.append(current)
    return out


def pass_members(graph: PhysicalGraph) -> list[tuple]:
    """The operators of each pass group, in execution order."""
    members = {}
    for node in graph.topological_nodes():
        if getattr(node, "pass_group", -1) >= 0:
            members.setdefault(node.pass_group, []).append(node)
    return [tuple(nodes) for nodes in members.values()]

"""Lower one logical alternative for the Quail engine: the Quail backend's planner."""

from dataclasses import replace

from quail.common.prompts import PROMPT_LAYOUTS
from quail.logical import has_label, has_score
from quail.physical import AiFilter, AiJoin
from quail.physical.lowering import answer_ids
from quail.planner.graph.builder import build_physical_plan
from quail.planner.optimizer.candidates import PhysicalCandidate
from quail.planner.plan import Refusal
from quail.planner.validation import joined_classification_refusal


def plan_region(region, context) -> tuple[PhysicalCandidate, ...]:
    """Build a Quail candidate without applying optimizer rules."""
    if has_label(region.logical_plan):
        if context.model.role == "reranker":
            return (PhysicalCandidate(None, Refusal(
                reasons=("a reranker model cannot run AI.CLASSIFY",),
                constraint="reranker_only_scores", needed=1, available=0,
                unit="AI.CLASSIFY expressions"), float("inf")),)
    scored = has_score(region.logical_plan)
    if context.model.role == "reranker":
        if not scored:
            refusal = Refusal(
                reasons=("a reranker model can only be used with AI.SCORE",),
                constraint="reranker_only_scores",
                needed=1,
                available=0,
                unit="AI.SCORE expressions",
            )
            return (PhysicalCandidate(None, refusal, float("inf")),)
    logical = region.logical_plan
    plan = build_physical_plan(
        logical, model=context.model, device=context.device,
        document_tokens=context.document_tokens, gpus=context.gpus,
        order=context.order, pair_fractions=context.pair_fractions,
        context=context)
    if isinstance(plan, Refusal):
        return (PhysicalCandidate(None, plan, float("inf"), logical),)
    refusal = joined_classification_refusal(plan.graph)
    if refusal is not None:
        return (PhysicalCandidate(None, refusal, float("inf"), logical),)
    plan = bind_runtime_data(plan, region, context)
    return (PhysicalCandidate(plan.graph, plan, plan.estimated_seconds, logical),)


def set_question_parts(lowered, model, tokenizer) -> dict:
    """A set-attention filter's state tail length and later-question header.

    The layout tokenizes the separator and question label as the first
    tail segment; a pass reads them with the document and opens each
    later question with the label alone.

    Raises:
        ValueError: The model's layout does not tokenize the label alone.
    """
    layout = PROMPT_LAYOUTS[model.prompt_layout]
    if not layout.label_segment:
        raise ValueError(f"{model.name!r}: a set-attention layout tokenizes "
                         f"its question label as its own segment")
    first = lowered.prompt.tail_segments[0]
    return dict(state_tail_tokens=len(tokenizer(first)),
                question_header_token_ids=tuple(tokenizer(layout.question_label)))


def set_join_parts(lowered, model, tokenizer) -> dict:
    """A set-attention join's partner separator length and question header.

    Raises:
        ValueError: The model's layout does not label join records.
    """
    layout = PROMPT_LAYOUTS[model.prompt_layout]
    if layout.record_label is None:
        raise ValueError(f"{model.name!r}: a set-attention layout labels "
                         f"each join record with its table")
    first = lowered.prompt.tail_segments[0]
    return dict(separator_tokens=len(tokenizer(first)),
                question_header_token_ids=tuple(tokenizer(layout.question_label)))


def bind_runtime_data(plan, region, context):
    """Put tokenized prompts and answer tokens in the physical plan."""
    tokenizer = context.tokenizer
    operators = region.logical_plan.operators()
    filters, joins = operators.filters, operators.joins
    encoded_nodes = []
    for node in plan.nodes:
        if isinstance(node, AiFilter):
            predicates = filters[node.alias]
            lowered = context.lowered(predicates[node.written_position].prompt)
            question = tuple(lowered.tail_token_ids)
            if not question:
                raise ValueError("filter prompts have no token ids")
            node = replace(node, question_token_ids=question)
            if context.model.attention_mask == "set":
                node = replace(node, **set_question_parts(
                    lowered, context.model, tokenizer))
        elif isinstance(node, AiJoin):
            prompt = context.lowered(joins[node.written_position].prompt)
            runtime_ids = {
                alias: (tuple(label), tuple(frame))
                for alias, label, frame in prompt.label_token_ids
            }
            node = replace(
                node, frame_token_ids=runtime_ids[node.anchor][1],
                label_token_ids=tuple((alias, runtime_ids[alias][0])
                                      for alias in node.partners),
                tail_token_ids=tuple(prompt.tail_token_ids))
            if context.model.attention_mask == "set":
                node = replace(node, **set_join_parts(
                    prompt, context.model, tokenizer))
        encoded_nodes.append(node)

    true_ids, false_ids = (
        answer_ids(context.model, tokenizer) if tokenizer is not None
        else ([], []))
    return replace(
        plan,
        nodes=tuple(encoded_nodes),
        root=plan.root,
        settings={
            **plan.settings,
            "true_ids": true_ids,
            "false_ids": false_ids,
        },
    )

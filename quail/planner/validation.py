"""Execution constraints checked during planning and graph construction."""

from quail.execution.pipelines import build_pipelines
from quail.logical import (
    Alias,
    Apply,
    SemanticClassify,
    SemanticExtract,
    classified_above_joins,
    is_score,
)
from quail.logical.prompts import prompt_aliases
from quail.physical import AiClassify, AiJoin
from quail.planner.plan import Refusal


def _refusal(reason: str, constraint: str = "unsupported_score_query"):
    return Refusal(
        reasons=(reason,),
        constraint=constraint,
        needed=1,
        available=0,
        unit="queries",
    )


class ScoreRefusedError(Exception):
    """Carries a Refusal out of the plan builder."""

    def __init__(self, refusal: Refusal):
        super().__init__(refusal.reasons[0])
        self.refusal = refusal


def score_projections(logical) -> tuple[Alias, ...]:
    return tuple(
        expression
        for expression in logical.projection.columns
        if isinstance(expression, Alias) and expression.expression.kind == "score"
    )


def score_refusal(logical, context) -> Refusal | None:
    """Check the execution constraints of score operators in a plan."""
    operators = logical.operators()
    projected = score_projections(logical)
    predicates = [p.expression for values in operators.filters.values()
                  for p in values] + [j.predicate for j in operators.joins]
    if context is None or context.tokenizer is None:
        return _refusal("AI.SCORE planning needs the model's tokenizer")
    prompts = [p.prompt for values in operators.filters.values() for p in values]
    prompts.extend(j.prompt for j in operators.joins)
    prompts.extend(item.expression.prompt for item in projected)
    if any(len(prompt_aliases(prompt)) > 2 for prompt in prompts):
        return _refusal("AI.SCORE supports one document or one document pair")
    if any(isinstance(node, Apply) for node in logical.walk()):
        return _refusal("AI.SCORE cannot be mixed with apply()")
    if not all(is_score(expression) for expression in predicates):
        return _refusal("AI.SCORE cannot be mixed with generative AI predicates")
    if any(join.semantics != "full" for join in operators.joins):
        return _refusal("AI.SCORE supports full joins only")
    if any(len(call.aliases()) != 1 for call, _ in operators.labels.calls):
        return _refusal(
            "AI.CLASSIFY over a document pair needs its AI join pipeline "
            "on one GPU; AI.SCORE plans cannot execute that classification",
            "joined_classify_pipeline")
    if classified_above_joins(logical.root) and operators.joins:
        return _refusal("classification after a score join is unsupported")
    pairs = [j.prompt for j in operators.joins] + [
        item.expression.prompt for item in projected
        if len(item.expression.aliases()) == 2]
    if len(set(pairs)) > 1:
        return _refusal("AI.SCORE currently supports one document pair expression")
    if any(join.anchor not in (None, prompt_aliases(join.prompt)[0])
           for join in operators.joins):
        return _refusal("AI.SCORE anchors on its first prompt document")
    return None


def classification_refusal(context) -> Refusal | None:
    """Check whether the model supports classification planning.

    Args:
        context: Planning context with the tokenizer and model specification.

    Returns:
        A Refusal describing missing support, or None if planning can proceed.
    """
    model = context.model
    if context.tokenizer is None:
        reason = "AI.CLASSIFY planning needs the model's tokenizer"
    elif model.canvas_tokens and model.answer_canvas is None:
        reason = f"{model.name!r} names no answer canvas to read labels on"
    else:
        return None
    return Refusal(reasons=(reason,), constraint="unsupported_classify_query",
                   needed=1, available=0, unit="queries")


def joined_classification_refusal(graph, workers: int) -> Refusal | None:
    """Check that each pair classification can run in its join's pipeline.

    Args:
        graph: Physical operator graph.
        workers: Number of model workers.

    Returns:
        A Refusal if a pair classification cannot share its source join's
        pipeline on one worker, or None if every pair classification can run.
    """
    pipelines = build_pipelines(graph) if workers == 1 else {}
    for node in graph.nodes:
        if not isinstance(node, AiClassify) or node.spec.partner is None:
            continue
        pipeline = pipelines.get(node.node_id)
        if pipeline is not None and any(
                isinstance(member, AiJoin)
                and any(port.source.node_id == member.node_id for port in node.inputs)
                for member in pipeline.members):
            continue
        return Refusal(
            reasons=(f"AI.CLASSIFY {node.spec.name!r} over a document pair "
                     "must run in its join's pipeline on one GPU; this plan "
                     "requires unsupported standalone pair classification",),
            constraint="joined_classify_pipeline", needed=1, available=0,
            unit="supported join pipelines")
    return None


def has_label(logical) -> bool:
    """Return whether a logical plan classifies documents or joined rows."""
    return any(isinstance(node, SemanticClassify) for node in logical.walk())


def has_extract(logical) -> bool:
    """Return whether a logical plan copies answers from documents."""
    return any(isinstance(node, SemanticExtract) for node in logical.walk())


def extraction_refusal(context) -> Refusal | None:
    """Check whether the model supports extraction planning.

    An extraction reads the model's next-token probabilities over the
    document's own tokens, which a reranker, a decision model, and a
    diffusion model with an answer canvas do not give.

    Args:
        context: Planning context with the tokenizer and model specification.

    Returns:
        A Refusal describing missing support, or None if planning can proceed.
    """
    model = context.model
    if context.tokenizer is None:
        reason = "AI.EXTRACT planning needs the model's tokenizer"
    elif model.role in ("reranker", "decision") or model.canvas_tokens:
        reason = (f"AI.EXTRACT reads next-token probabilities over the "
                  f"document's tokens, which {model.name!r} does not give")
    else:
        return None
    return Refusal(reasons=(reason,), constraint="unsupported_extract_query",
                   needed=1, available=0, unit="queries")


class ClassifyRefusedError(Exception):
    """A classification cannot run within the selected execution constraints."""

    def __init__(self, reason, needed, available):
        super().__init__(reason)
        self.reason, self.needed, self.available = reason, needed, available

    def refusal(self) -> Refusal:
        """Convert this exception to a planning Refusal."""
        return Refusal(reasons=(self.reason,), constraint="suffix_over_chunk",
                       needed=self.needed, available=self.available, unit="tokens")


def anchor_candidates(spec: dict, honor_forced: bool = True) -> list:
    """Candidate anchors for one join.

    Gates keep their outer table, a forced full anchor is honored, a
    free full join offers every table.
    """
    if spec["semantics"] != "full":
        return [spec["anchor"]]
    if honor_forced and not spec.get("anchor_free"):
        return [spec["anchor"]]
    return list(spec["aliases"])


def join_input_tokens(spec, anchor: str, maximum: dict, pre: int) -> int:
    """Count tokens for the longest input under one join orientation."""
    return (pre + maximum[anchor] + spec["frame_tokens"][anchor]
            + spec["tail_tokens"]
            + sum(spec["label_tokens"][partner] + maximum[partner]
                  for partner in spec["aliases"] if partner != anchor))


def score_input_tokens(spec, context) -> tuple[int, int]:
    """Return the prefix and suffix token counts of the longest input.

    The prefix is what stays resident while suffixes stream past it:
    the fixed prompt head, plus the first document for a pair.
    """
    parts = spec.prompt_token_parts
    canvas = context.model.canvas_tokens
    longest = [
        max(context.document_tokens[alias], default=0) for alias in spec.aliases
    ]
    if len(spec.aliases) == 1:
        return len(parts[0]), longest[0] + len(parts[1]) + canvas
    prefix = len(parts[0]) + longest[0] + len(parts[1])
    return prefix, longest[1] + len(parts[2]) + canvas

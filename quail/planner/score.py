"""Prompt preparation, costs, and physical operators for AI.SCORE."""

import math
from dataclasses import dataclass

from quail.cost import budgets
from quail.cost.budgets import PAGE_TOKENS
from quail.cost.sol import speed_of_light
from quail.cost.work import Work, triangle
from quail.logical import (
    Alias,
    Apply,
    classified_above_joins,
    is_score,
)
from quail.logical.prompts import (
    bind_join_prompt,
    bind_prompt,
    true_false_token_ids,
)
from quail.physical import (
    AiScore,
    Comparison,
    Filter,
    PortRef,
    ScoreSpec,
)
from quail.physical.base import input_ports
from quail.planner.plan import Refusal
from quail.reranker import render_qwen3_reranker_input


def _refusal(reason: str, constraint: str = "unsupported_score_query"):
    return Refusal(
        reasons=(reason,),
        constraint=constraint,
        needed=1,
        available=0,
        unit="queries",
    )


def _prompt_aliases(prompt) -> tuple[str, ...]:
    return tuple(dict.fromkeys(argument.alias for argument in prompt.args))


class ScoreRefusedError(Exception):
    """Carries a Refusal out of the plan builder."""

    def __init__(self, refusal: Refusal):
        super().__init__(refusal.reasons[0])
        self.refusal = refusal


def _without_document(template: str, marker: str) -> str:
    """Return the query text with the scored document's placeholder gone.

    The reranker gives the document its own field, so a placeholder
    standing alone at either end of the template only marks where the
    document goes and is dropped. A placeholder inside a sentence
    becomes "this document".
    """
    text = template.strip()
    if text.startswith(marker):
        text = text[len(marker):].lstrip()
    if text.endswith(marker):
        text = text[:-len(marker)].rstrip()
    return text.replace(marker, "this document")


def _query_template(prompt) -> str:
    aliases = _prompt_aliases(prompt)
    if len(aliases) == 1:
        return _without_document(prompt.template, "{0}")
    if len(aliases) == 2 and len(prompt.args) == 2:
        return _without_document(prompt.template, "{1}")
    raise ValueError("AI.SCORE supports one document or one document pair")


def _pages(tokens: int) -> int:
    return -(-tokens // PAGE_TOKENS)


def _longest_input(spec, context) -> tuple[int, int]:
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


def score_fixed_tokens(token_parts, canvas_tokens: int, draws: int) -> int:
    """Return the tokens a score adds to each input besides its documents.

    Args:
        token_parts: The prompt's tokenized pieces around the documents.
        canvas_tokens: The model's canvas rows per answer; 0 without one.
        draws: The noise draws averaged per answer.

    Returns:
        The prompt pieces, the canvas, and each later draw's cue and canvas.
    """
    return (sum(map(len, token_parts)) + canvas_tokens
            + (draws - 1) * (1 + canvas_tokens))


def _score_work(count, mean_tokens, fixed_tokens, *, variance=0.0,
                prefix_tokens=0.0, prefix_variance=0.0, groups=0.0,
                shared_tokens=0, copies=1) -> Work:
    if count <= 0:
        return Work()
    length = mean_tokens + fixed_tokens
    work = Work(
        tokens=length,
        pairs=triangle(length) + variance / 2,
        kv_written=length,
    ) * count
    reused = max(0.0, count - groups)
    work += Work(
        tokens=-prefix_tokens,
        pairs=-triangle(prefix_tokens) - prefix_variance / 2,
        kv_written=-prefix_tokens,
        kv_read=prefix_tokens,
    ) * reused
    shared_reuses = max(0.0, groups - copies)
    return work + Work(
        tokens=-shared_tokens,
        pairs=-triangle(shared_tokens),
        kv_written=-shared_tokens,
        kv_read=shared_tokens,
    ) * shared_reuses


def _variance(lengths) -> float:
    mean = sum(lengths) / max(1, len(lengths))
    return sum((length - mean) ** 2 for length in lengths) / max(1, len(lengths))


def _token_parts(prompt, context) -> tuple[tuple[int, ...], ...]:
    """Tokenize the fixed pieces of the rendered prompt around the documents."""
    if context.model.role != "reranker":
        return _answer_token_parts(prompt, context)
    query_template = _query_template(prompt)
    rendered = render_qwen3_reranker_input(query_template, "{document}")
    before, after = rendered.split("{document}")
    parts = (before, after)
    if len(_prompt_aliases(prompt)) == 2:
        first, middle = before.split("{0}")
        parts = (first, middle, after)
    # Documents are tokenized separately; tokens cannot span these boundaries.
    # These ids can differ from tokenizing the complete prompt string.
    return tuple(tuple(context.tokenizer(part)) for part in parts)


def _answer_token_parts(prompt, context) -> tuple[tuple[int, ...], ...]:
    """Tokenize the AI.IF layout around the documents, for a generative model.

    A document gets the AI.IF filter prompt and a pair the AI.IF join
    prompt anchored on its first document.
    """
    tokenizer, turn = context.tokenizer, context.model.turn
    layout = context.model.prompt_layout
    aliases = _prompt_aliases(prompt)
    if len(aliases) == 1:
        bound = bind_prompt(prompt.template, prompt.args, tokenizer, turn,
                            layout)
        return tuple(bound.preamble_token_ids), tuple(bound.tail_token_ids)
    if len(prompt.args) != 2:
        raise ValueError("AI.SCORE supports one document or one document pair")
    bound = bind_join_prompt(prompt.template, prompt.args, tokenizer, turn,
                             layout)
    pieces = {alias: (label, frame) for alias, label, frame in bound.label_token_ids}
    left, right = aliases
    return (tuple(bound.preamble_token_ids),
            tuple(pieces[left][1]) + tuple(pieces[right][0]),
            tuple(bound.tail_token_ids))


def answer_ids(model, tokenizer) -> tuple[list[int], list[int]]:
    """The token ids a score compares: yes and no, or TRUE and FALSE.

    Args:
        model: The ModelSpec; a reranker answers yes or no.
        tokenizer: Callable text -> token ids.
    """
    if model.role == "reranker":
        return list(tokenizer("yes")), list(tokenizer("no"))
    return true_false_token_ids(tokenizer)


def score_spec(
    prompt,
    *,
    name: str,
    expected_inputs: float,
    mean_tokens: float,
    context,
    chunk_tokens: int,
    pair_fraction: float = 1.0,
    prefix_groups: float | None = None,
    token_parts_by_prompt: dict | None = None,
):
    if token_parts_by_prompt is None:
        token_parts_by_prompt = {}
    token_parts = token_parts_by_prompt.get(prompt)
    if token_parts is None:
        token_parts = _token_parts(prompt, context)
        token_parts_by_prompt[prompt] = token_parts
    aliases = _prompt_aliases(prompt)
    lengths = [context.document_tokens[alias] for alias in aliases]
    canvas = context.model.canvas_tokens
    # a one-table score on a canvas model may average noise draws, each
    # the cue and its canvas after the document's KV; every draw is priced
    draws = context.canvas_draws if canvas and len(aliases) == 1 else 1
    fixed_tokens = score_fixed_tokens(token_parts, canvas, draws)
    shared = len(token_parts[0])
    prefix = float(shared)
    prefix_variance = 0.0
    groups = min(float(context.gpu_count), expected_inputs)
    capacity = budgets.arena_tokens(context.model, context.device, chunk_tokens)
    if len(aliases) == 2:
        prefix += sum(lengths[0]) / max(1, len(lengths[0])) + len(token_parts[1])
        prefix_variance = _variance(lengths[0])
        groups = min(
            expected_inputs,
            len(lengths[0]) if prefix_groups is None else prefix_groups,
        )
        longest = sum(max(values, default=0) for values in lengths) + fixed_tokens
        if longest > capacity:
            prefix = float(shared)
            prefix_variance = 0.0
            groups = min(float(context.gpu_count), expected_inputs)
    work = _score_work(
        expected_inputs, mean_tokens, fixed_tokens,
        variance=sum(_variance(values) for values in lengths),
        prefix_tokens=prefix, prefix_variance=prefix_variance,
        groups=groups, shared_tokens=shared, copies=context.gpu_count,
    )
    estimate = speed_of_light(
        work * (1 / max(1, min(context.gpu_count, expected_inputs))),
        context.model, context.device, chunk_tokens,
    ).seconds
    return ScoreSpec(
        name=name,
        aliases=aliases,
        query_template=_query_template(prompt),
        arguments=tuple(
            (argument.alias, argument.column) for argument in prompt.args
        ),
        expected_inputs=expected_inputs,
        estimated_seconds=estimate,
        pair_fraction=pair_fraction,
        prompt_token_parts=token_parts,
        draws=draws,
    ), work


def score_projections(logical) -> tuple[Alias, ...]:
    return tuple(
        expression
        for expression in logical.projection.columns
        if isinstance(expression, Alias) and expression.expression.kind == "score"
    )


def score_name(prompt, projected, fallback: str) -> str:
    # the front end rejects one prompt projected under two names
    return next(
        (item.name for item in projected if item.expression.prompt == prompt),
        fallback,
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
    if any(len(_prompt_aliases(prompt)) > 2 for prompt in prompts):
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
    if any(join.anchor not in (None, _prompt_aliases(join.prompt)[0])
           for join in operators.joins):
        return _refusal("AI.SCORE anchors on its first prompt document")
    return None


@dataclass(frozen=True)
class ScoreJoinCost:
    """Price a score join at the survivor counts supplied by join search."""

    prompt: object
    context: object
    chunk: int
    pair_fraction: float

    def work(self, live) -> Work:
        left, right = _prompt_aliases(self.prompt)
        expected, groups = pair_counts(live, (left, right), self.pair_fraction)
        means = [sum(self.context.document_tokens[alias])
                 / max(1, len(self.context.document_tokens[alias]))
                 for alias in (left, right)]
        return score_spec(
            self.prompt, name="", expected_inputs=expected,
            mean_tokens=sum(means), context=self.context,
            chunk_tokens=self.chunk, pair_fraction=self.pair_fraction,
            prefix_groups=groups)[1]


def pair_counts(live, aliases, fraction) -> tuple[float, float]:
    """Return expected pairs and first documents with at least one pair."""
    left, right = aliases
    touched = (1.0 if fraction >= 1 else
               -math.expm1(live[right] * math.log1p(-fraction)))
    return live[left] * live[right] * fraction, live[left] * touched


class ScoreLowering:
    """Append score and comparison operators to a shared physical graph."""

    def __init__(self, context, nodes, projected, chunk):
        self.context = context
        self.nodes = nodes
        self.projected = projected
        self.chunk = chunk
        self.capacity = budgets.arena_tokens(context.model, context.device, chunk)
        self.names = {}
        self.token_parts = {}
        self.work = Work()
        self.index = 0

    def score(self, prompt, fallback, inputs, expected, mean,
              *, pair_fraction=1.0, prefix_groups=None) -> PortRef:
        name = score_name(prompt, self.projected, fallback)
        spec, work = score_spec(
            prompt, name=name, expected_inputs=expected, mean_tokens=mean,
            context=self.context, chunk_tokens=self.chunk,
            pair_fraction=pair_fraction, prefix_groups=prefix_groups,
            token_parts_by_prompt=self.token_parts)
        prefix, suffix = _longest_input(spec, self.context)
        what = (f"a document in {spec.aliases[0]!r}" if len(spec.aliases) == 1
                else f"one pair of {spec.aliases[0]!r} and {spec.aliases[1]!r}")
        needed, available, unit = prefix + suffix, self.chunk, "tokens"
        if needed <= available:
            needed = _pages(prefix) + _pages(suffix)
            available, unit = self.capacity // PAGE_TOKENS, "pages"
        if needed > available:
            raise ScoreRefusedError(Refusal(
                reasons=(f"{what} needs {needed} {unit} with its prompt, "
                         f"but the execution budget is {available} {unit}",),
                constraint="suffix_over_chunk", needed=needed,
                available=available, unit=unit))
        node = AiScore(
            node_id=f"ai-score:{self.index}", inputs=input_ports(tuple(inputs)),
            backend_name=self.context.backend, model=self.context.model.name,
            spec=spec)
        self.index += 1
        self.nodes.append(node)
        self.names[prompt] = name
        self.work += work
        return PortRef(node.node_id, "scores")

    def comparison(self, predicate, source, aliases, position) -> PortRef:
        alias = aliases[0] if len(aliases) == 1 else "join"
        expression = (predicate.expression if len(aliases) == 1
                      else predicate.predicate)
        node = Filter(
            node_id=f"filter:{alias}:{position}",
            inputs=input_ports((source,)),
            predicate=Comparison(self.names[predicate.prompt],
                                 expression.comparison, expression.threshold),
            aliases=tuple(aliases), selectivity=predicate.selectivity,
            written_pos=position)
        self.nodes.append(node)
        return PortRef(node.node_id, "scores")

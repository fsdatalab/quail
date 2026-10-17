"""AI.SCORE: its statistics, cost, spec, and plan nodes."""

from dataclasses import dataclass

from quail.cost import budgets
from quail.cost.score import ScoreCost
from quail.cost.work import Work
from quail.logical import ColumnRef, RegularPredicate, effective_selectivity
from quail.logical.prompts import (
    prompt_aliases,
    score_query_template,
)
from quail.physical import (
    AiScore,
    Filter,
    PortRef,
    ScoreSpec,
)
from quail.physical.base import input_ports
from quail.planner.operators.common import bind_inference, inference_node
from quail.planner.operators.ports import Ports
from quail.planner.plan import Refusal
from quail.planner.prompt_lowering import score_token_parts
from quail.planner.statistics.memory import arena_capacity
from quail.planner.statistics.tables import pair_counts, summarize_alias
from quail.planner.validation import (
    ScoreRefusedError,
    score_input_tokens,
)


@dataclass(frozen=True)
class ScoreJoinCost:
    """Supply score work to join search at the given survivor counts."""

    aliases: tuple[str, str]
    cost: ScoreCost
    pair_fraction: float

    def work(self, expected_rows) -> Work:
        expected, groups = pair_counts(expected_rows, self.aliases, self.pair_fraction)
        return self.cost.estimate(expected, prefix_groups=groups)[0]


def score_statistics(prompt, context, chunk, *, mean_tokens=None):
    """Prepare a score's prompt tokens and numeric cost inputs once per query."""
    key = ("score_prompt", prompt)
    if key not in context.memo:
        context.memo[key] = score_token_parts(prompt, context.model, context.tokenizer)
    parts = context.memo[key]
    aliases = prompt_aliases(prompt)
    summaries = tuple(summarize_alias(
        context.document_tokens[alias], context.model.sliding_window,
        memo=context.memo) for alias in aliases)
    if mean_tokens is None:
        mean_tokens = sum(summary.mean for summary in summaries)
    cost = ScoreCost(
        token_lengths=tuple(map(len, parts)), summaries=summaries,
        mean_tokens=mean_tokens, model=context.model, device=context.device,
        chunk_tokens=chunk,
        capacity=arena_capacity(context, chunk),
        gpus=context.gpus,
        draws=context.canvas_draws if context.model.canvas_tokens
        and len(aliases) == 1 else 1)
    return parts, cost


def score_name(prompt, projected, fallback: str) -> str:
    # the front end rejects one prompt projected under two names
    return next(
        (item.name for item in projected if item.expression.prompt == prompt),
        fallback,
    )


def score_spec(prompt, *, name, expected_rows, mean_tokens, context,
               chunk_tokens, pair_fraction=1.0, prefix_groups=None):
    """Build a score specification from prepared prompt tokens and costs."""
    parts, cost = score_statistics(
        prompt, context, chunk_tokens, mean_tokens=mean_tokens)
    work, seconds = cost.estimate(expected_rows, prefix_groups=prefix_groups)
    return bind_inference(
        ScoreSpec, prompt, name=name,
        query_template=score_query_template(prompt),
        expected_rows=expected_rows, estimated_seconds=seconds,
        pair_fraction=pair_fraction, prompt_token_parts=parts, draws=cost.draws), work


class ScoreLowering:
    """Append score and comparison operators to a shared physical graph."""

    def __init__(self, context, nodes, projected, chunk):
        self.context = context
        self.nodes = nodes
        self.projected = projected
        self.chunk = chunk
        self.capacity = arena_capacity(context, chunk)
        self.names = {}
        self.work = Work()
        self.index = 0

    def score(self, prompt, fallback, inputs, expected_rows, mean_tokens,
              *, pair_fraction=1.0, prefix_groups=None) -> PortRef:
        name = score_name(prompt, self.projected, fallback)
        spec, work = score_spec(
            prompt, name=name, expected_rows=expected_rows, mean_tokens=mean_tokens,
            context=self.context, chunk_tokens=self.chunk,
            pair_fraction=pair_fraction, prefix_groups=prefix_groups)
        prefix, suffix = score_input_tokens(spec, self.context)
        what = (f"a document in {spec.aliases[0]!r}" if len(spec.aliases) == 1
                else f"one pair of {spec.aliases[0]!r} and {spec.aliases[1]!r}")
        needed, available, unit = prefix + suffix, self.chunk, "tokens"
        if needed <= available:
            needed = budgets.pages(prefix) + budgets.pages(suffix)
            available, unit = self.capacity // budgets.PAGE_TOKENS, "pages"
        if needed > available:
            raise ScoreRefusedError(Refusal(
                reasons=(f"{what} needs {needed} {unit} with its prompt, "
                         f"but the execution budget is {available} {unit}",),
                constraint="suffix_over_chunk", needed=needed,
                available=available, unit=unit))
        node = inference_node(
            AiScore, spec, inputs, index=self.index,
            backend=self.context.backend, model=self.context.model.name)
        self.index += 1
        self.nodes.append(node)
        self.names[prompt] = name
        self.work += work
        return PortRef(node.node_id, "rows")

    def comparison(self, predicate, source, aliases, position) -> PortRef:
        alias = aliases[0] if len(aliases) == 1 else "join"
        expression = (predicate.expression if len(aliases) == 1
                      else predicate.predicate)
        node = Filter(
            node_id=f"filter:{alias}:{position}",
            inputs=input_ports((source,)),
            predicate=RegularPredicate(
                ColumnRef(alias, "", self.names[predicate.prompt]),
                expression.comparison, expression.threshold),
            aliases=tuple(aliases), selectivity=predicate.selectivity,
            written_position=position)
        self.nodes.append(node)
        return PortRef(node.node_id, "rows")


def emit_scores(builder, alias, positions, expected_rows, ports) -> Ports:
    """Append a run of the alias's score predicates in their chosen order.

    Args:
        builder: The graph builder.
        alias: The table.
        positions: The written positions to emit, in run order.
        expected_rows: The documents the run reads.
        ports: Where the documents stand before the run.

    Returns:
        Where the documents stand after the run: the last comparison's
        rows and ids.
    """
    stats = builder.operator_order.statistics.stats
    for position in positions:
        predicate = builder.operators.filters[alias][position]
        source = ports.source
        if predicate.prompt not in builder.scoring.names:
            source = builder.scoring.score(
                predicate.prompt, f"__score_{alias}_{position}", (source,),
                expected_rows, stats[alias].mean_document_tokens)
        rows = builder.scoring.comparison(predicate, source, (alias,), position)
        ports = Ports(PortRef(rows.node_id, f"ids:{alias}"), rows)
        expected_rows *= effective_selectivity(predicate.selectivity)
    return ports


def emit_projected_scores(builder, alias, expected_rows, ports) -> Ports:
    """Append the alias's scores returned in SELECT, over the live documents.

    A score keeps every document it reads, so the documents' ids stay
    the ones before it.
    """
    stats = builder.operator_order.statistics.stats
    for expression in builder.projected_scores:
        call = expression.expression
        if call.aliases() != (alias,) or call.prompt in builder.scoring.names:
            continue
        rows = builder.scoring.score(
            call.prompt, expression.name, (ports.source,), expected_rows,
            stats[alias].mean_document_tokens)
        ports = Ports(ports.ids, rows)
    return ports


def emit_score_pair(builder, prompt, position, pair_fraction, pair_inputs,
                    ports_by_alias) -> PortRef:
    """Append the score of one pair prompt and return its rows port.

    Args:
        builder: The graph builder.
        prompt: The pair prompt.
        position: The join's written position.
        pair_fraction: The pair_fraction of the cross product the pairs keep.
        pair_inputs: The ports supplying the pairs, if any.
        ports_by_alias: alias -> where its documents stand.
    """
    stats = builder.operator_order.statistics.stats
    aliases = tuple(dict.fromkeys(ref.alias for ref in prompt.args))
    expected, prefixes = pair_counts(
        builder.operator_order.expected_rows, aliases, pair_fraction
    )
    return builder.scoring.score(
        prompt, f"__score_join_{position}",
        tuple(ports_by_alias[a].source for a in aliases) + pair_inputs,
        expected, sum(stats[a].mean_document_tokens for a in aliases),
        pair_fraction=pair_fraction, prefix_groups=prefixes)


def emit_score_joins(builder, group, ports_by_alias, score_pairs) -> PortRef:
    """Append the scores and comparisons of one group of score joins.

    Args:
        builder: The graph builder.
        group: The group's (PricedJoin, step) pairs.
        ports_by_alias: alias -> where its documents stand.
        score_pairs: The rows of the pairs scored so far, or None.

    Returns:
        The rows of the pairs the group keeps.
    """
    for spec, _ in group:
        position = spec.written_position
        join = builder.operators.joins[position]
        pair_inputs = ()
        if position in builder.pairs_src:
            producer = builder.pairs_src[position]
            pair_inputs = (PortRef(producer.node_id, f"pairs:{position}"),)
        if score_pairs is None:
            score_pairs = emit_score_pair(
                builder, join.prompt, position, spec.pair_fraction, pair_inputs,
                ports_by_alias)
        score_pairs = builder.scoring.comparison(
            join, score_pairs, spec.aliases, position)
    return score_pairs

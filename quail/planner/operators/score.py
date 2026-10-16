"""AI.SCORE: its statistics, cost, spec, and plan nodes."""

from dataclasses import dataclass

from quail.cost import budgets
from quail.cost.score import ScoreCost
from quail.cost.work import Work
from quail.logical import effective_selectivity
from quail.logical.prompts import (
    prompt_aliases,
    score_query_template,
)
from quail.physical import (
    AiScore,
    Condition,
    Filter,
    PortRef,
    ScoreSpec,
)
from quail.physical.base import input_ports
from quail.planner.operators.common import bind_inference, inference_node
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

    def work(self, live) -> Work:
        expected, groups = pair_counts(live, self.aliases, self.pair_fraction)
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
        workers=context.gpu_count,
        draws=context.canvas_draws if context.model.canvas_tokens
        and len(aliases) == 1 else 1)
    return parts, cost


def score_name(prompt, projected, fallback: str) -> str:
    # the front end rejects one prompt projected under two names
    return next(
        (item.name for item in projected if item.expression.prompt == prompt),
        fallback,
    )


def score_spec(prompt, *, name, expected_inputs, mean_tokens, context,
               chunk_tokens, pair_fraction=1.0, prefix_groups=None):
    """Build a score specification from prepared prompt tokens and costs."""
    parts, cost = score_statistics(
        prompt, context, chunk_tokens, mean_tokens=mean_tokens)
    work, seconds = cost.estimate(expected_inputs, prefix_groups=prefix_groups)
    return bind_inference(
        ScoreSpec, prompt, name=name,
        query_template=score_query_template(prompt),
        expected_inputs=expected_inputs, estimated_seconds=seconds,
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

    def score(self, prompt, fallback, inputs, expected, mean,
              *, pair_fraction=1.0, prefix_groups=None) -> PortRef:
        name = score_name(prompt, self.projected, fallback)
        spec, work = score_spec(
            prompt, name=name, expected_inputs=expected, mean_tokens=mean,
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
            predicate=Condition(self.names[predicate.prompt],
                                expression.comparison, expression.threshold),
            aliases=tuple(aliases), selectivity=predicate.selectivity,
            written_pos=position)
        self.nodes.append(node)
        return PortRef(node.node_id, "rows")


def emit_scores(builder, alias):
    """Append the alias's score predicates and projected scores."""
    stats = builder.operator_order.statistics.stats
    orders = builder.operator_order.filter_orders
    live = float(stats[alias].n_docs)
    for position in orders.get(alias, ()):
        predicate = builder.operators.filters[alias][position]
        source = builder.rows_src.get(alias, builder.ids_src[alias])
        if predicate.prompt not in builder.scoring.names:
            source = builder.scoring.score(
                predicate.prompt, f"__score_{alias}_{position}", (source,),
                live, stats[alias].mean_doc_tokens)
        source = builder.scoring.comparison(predicate, source, (alias,), position)
        builder.rows_src[alias] = source
        builder.ids_src[alias] = PortRef(source.node_id, f"ids:{alias}")
        live *= effective_selectivity(predicate.selectivity)
    for expression in builder.projected_scores:
        call = expression.expression
        if call.aliases() != (alias,) or call.prompt in builder.scoring.names:
            continue
        builder.rows_src[alias] = builder.scoring.score(
            call.prompt, expression.name,
            (builder.rows_src.get(alias, builder.ids_src[alias]),), live,
            stats[alias].mean_doc_tokens)



def emit_score_pair(builder, prompt, position, fraction, pair_inputs):
    """Append the score of one pair prompt and return its rows port."""
    stats = builder.operator_order.statistics.stats
    aliases = tuple(dict.fromkeys(ref.alias for ref in prompt.args))
    expected, prefixes = pair_counts(builder.operator_order.live, aliases, fraction)
    return builder.scoring.score(
        prompt, f"__score_join_{position}",
        tuple(builder.rows_src.get(a, builder.ids_src[a]) for a in aliases)
        + pair_inputs,
        expected, sum(stats[a].mean_doc_tokens for a in aliases),
        pair_fraction=fraction, prefix_groups=prefixes)


def emit_score_joins(builder, group):
    """Append the scores and comparisons of one group of score joins."""
    for spec, _ in group:
        position = spec.written_pos
        join = builder.operators.joins[position]
        pair_inputs = ()
        if position in builder.pairs_src:
            producer = builder.pairs_src[position]
            pair_inputs = (PortRef(producer.node_id,
                                  f"pairs:{position}"),)
        if builder.score_pairs is None:
            builder.score_pairs = emit_score_pair(
                builder, join.prompt, position, spec.pair_fraction, pair_inputs)
        builder.score_pairs = builder.scoring.comparison(
            join, builder.score_pairs, spec.aliases, position)

"""The numbers the cost-based rules and the physical planner read.

One PlanStatistics holds what the cost model knows about a logical
plan before any decision: each table's document lengths, the model's
token budgets, the preamble, and the joins as the search prices them.
The functions after it read the decisions the logical rules recorded
on the plan's nodes: each table's filter order and the join sequence.
"""

import math
from collections import Counter
from dataclasses import dataclass
from functools import partial
from operator import mul

from quail.cost import budgets
from quail.cost.filters import filter_chain_work, filter_cost
from quail.cost.retention import retention_pages
from quail.cost.score import ScoreCost
from quail.cost.work import Work, triangle
from quail.logical import (
    LogicalPlan,
    SemanticFilter,
    classified_above_joins,
    effective_selectivity,
    is_score,
    join_conditions,
    model_call,
)
from quail.logical.prompts import prompt_aliases, score_token_parts
from quail.planner.filter_order import PredicateCost
from quail.planner.plan import CorpusStats
from quail.specs import DeviceSpec, ModelSpec


@dataclass(frozen=True)
class AliasStats:
    """Length sums and counts below the attention window for one alias."""

    count: int
    total: int
    squared: int
    maximum: int
    window: int = 0
    short_counts: tuple[tuple[int, int], ...] = ()

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def window_pairs(self, offset: float) -> float:
        """Sum sliding pairs for document lengths plus an offset."""
        window = self.window
        linear = window * (self.total + offset * self.count)
        linear -= self.count * window * (window - 1) / 2
        return linear + sum(
            count * (triangle(length + offset)
                     - window * (length + offset) + window * (window - 1) / 2)
            for length, count in self.short_counts if length + offset < window)

    def window_reads(self, offset: float) -> float:
        """Sum retained prefix keys visible to the first suffix token."""
        cap = self.window - 1
        return self.count * cap - sum(
            count * (cap - length - offset)
            for length, count in self.short_counts if length + offset < cap)


def summarize_alias(lengths, window: int = 0) -> AliasStats:
    """Summarize document lengths in one pass."""
    count = total = squared = maximum = 0
    short_counts = Counter()
    for raw in lengths:
        length = int(raw)
        count += 1
        total += length
        squared += length * length
        maximum = max(maximum, length)
        if length < window:
            short_counts[length] += 1
    return AliasStats(
        count=count,
        total=total,
        squared=squared,
        maximum=maximum,
        window=window,
        short_counts=tuple(sorted(short_counts.items())),
    )


def scale_alias(stats: AliasStats, fraction: float) -> AliasStats:
    """Return the summary of the fraction of the documents expected to remain."""
    return AliasStats(
        count=round(stats.count * fraction),
        total=round(stats.total * fraction),
        squared=round(stats.squared * fraction),
        maximum=stats.maximum,
        window=stats.window,
        short_counts=tuple((length, round(count * fraction))
                           for length, count in stats.short_counts),
    )


def alias_stats(lengths: dict, window: int = 0) -> dict[str, AliasStats]:
    """Normalize raw length lists or accept summaries from a caller."""
    return {
        alias: (values if isinstance(values, AliasStats)
                else summarize_alias(values, window))
        for alias, values in lengths.items()
    }


def surviving_docs(n_docs: float, n_partners: float,
                   tuple_selectivity) -> float:
    """Expected distinct documents with at least one matching tuple."""
    tuple_selectivity = effective_selectivity(tuple_selectivity)
    return n_docs * (1.0 - (1.0 - tuple_selectivity)
                     ** max(1.0, n_partners))


def thin(live: dict, spec: dict) -> None:
    """Update live document counts after one join's selectivity."""
    sel = spec["selectivity"]
    aliases = spec["aliases"]

    def others(x):
        out = 1.0
        for a in aliases:
            if a != x:
                out *= live[a]
        return out

    if spec["semantics"] == "full":
        new = {a: surviving_docs(live[a], others(a), sel)
               for a in aliases}
        live.update(new)
    else:
        anchor = spec["anchor"]
        matched = surviving_docs(live[anchor], others(anchor), sel)
        live[anchor] = (matched if spec["semantics"] == "exists"
                        else live[anchor] - matched)


def cross_tuples(spec: dict, live: dict) -> float:
    """Expected tuples of one join.

    The live cross product, or the fraction of it the join's equality
    conditions keep.
    """
    tuples = 1.0
    for a in spec["aliases"]:
        tuples *= live[a]
    return tuples * spec.get("pair_fraction", 1.0)


def question_tokens(prompt, canvas: int = 0) -> int:
    """Token count of the prompt's per-evaluation tail.

    canvas is the rows a diffusion model appends to every evaluation
    to answer on; a decoder answers on the tail's last row.
    """
    if prompt.tail_tokens is None or prompt.preamble_tokens is None:
        raise ValueError(
            "prompts were bound without a tokenizer; the planner "
            "needs token counts (pass one to compile_sql / docs)")
    return prompt.tail_tokens + canvas


def preamble_tokens(filters, joins) -> int:
    """Return the engine preamble's token count from any bound prompt."""
    for fs in filters.values():
        for p in fs:
            if p.prompt.preamble_tokens is not None:
                return p.prompt.preamble_tokens
    for j in joins:
        if j.prompt.preamble_tokens is not None:
            return j.prompt.preamble_tokens
    return 0


def _join_aliases(join) -> list:
    """Return the join's table aliases in placeholder order."""
    return [r.alias for r in join.prompt.args]


def _label_counts(join) -> dict:
    """Return alias -> (block_label_tokens, anchor_frame_tokens)."""
    out = {a: (lt, nt) for a, lt, nt in join.prompt.labels}
    if any(lt is None for lt, _ in out.values()):
        raise ValueError(
            "join prompts were bound without a tokenizer; the planner "
            "needs token counts (pass one to compile_sql / docs)")
    return out


def join_specs(joins, pair_fractions=None, canvas: int = 0, *,
               context=None, chunk: int = 0) -> list:
    """The joins as the search's spec dicts, in written order.

    pair_fractions maps a written position to the fraction of the
    cross product its equality conditions keep; a join with
    conditions but no entry is priced as the full cross product.
    canvas is the rows a diffusion model appends to every pair.
    """
    pair_fractions = pair_fractions or {}
    out = []
    for i, j in enumerate(joins):
        conditions = join_conditions(j)
        fraction = pair_fractions.get(i, 1.0) if conditions else 1.0
        if is_score(j.predicate):
            parts, cost = score_statistics(j.prompt, context, chunk)
            left, right = prompt_aliases(j.prompt)
            head, middle, tail = map(len, parts)
            out.append(dict(
                written_pos=i, aliases=[left, right], anchor=left,
                anchor_free=False, semantics=j.semantics,
                selectivity=j.selectivity,
                frame_tokens={left: head + middle, right: 0},
                label_tokens={left: 0, right: 0}, tail_tokens=tail + canvas,
                on=[(c.left.alias, c.left.column, c.right.alias, c.right.column)
                    for c in conditions], pair_fraction=fraction,
                cost=ScoreJoinCost((left, right), cost, fraction)))
            continue
        labels = _label_counts(j)
        out.append(dict(
            written_pos=i, aliases=_join_aliases(j), anchor=j.anchor,
            anchor_free=(j.anchor is None and j.semantics == "full"),
            semantics=j.semantics, selectivity=j.selectivity,
            frame_tokens={a: nt for a, (lt, nt) in labels.items()},
            label_tokens={a: lt for a, (lt, nt) in labels.items()},
            tail_tokens=question_tokens(j.prompt, canvas),
            on=[(c.left.alias, c.left.column, c.right.alias, c.right.column)
                for c in conditions],
            pair_fraction=fraction))
    return out


@dataclass(frozen=True)
class PlanStatistics:
    """What the cost model knows about one plan before any decision.

    Attributes:
        stats: alias -> CorpusStats of its documents.
        lengths: alias -> AliasStats, the length sums the join search
            and the stage pricing read.
        chunk: The forward pass token budget.
        arena_pages: The KV arena's page split.
        admission: The admission budget in tokens.
        cap_pages: Pages the retention pool may hold.
        pre: The engine preamble's token count.
        specs: The joins as the search's spec dicts, in written order.
        asks: alias -> written positions of its AI.IF predicates.
    """

    stats: dict
    lengths: dict
    chunk: int
    arena_pages: tuple
    admission: int
    cap_pages: int
    pre: int
    specs: tuple
    asks: dict


def ask_positions(filters) -> dict:
    """Return alias -> written positions of its AI.IF predicates."""
    return {alias: [i for i, p in enumerate(predicates)
                   if not is_score(p.expression)]
            for alias, predicates in filters.items()}


def plan_statistics(plan: LogicalPlan, *, model: ModelSpec,
                    device: DeviceSpec, doc_tokens: dict,
                    pair_fractions=None, scan_fractions=None,
                    context=None) -> PlanStatistics:
    """Gather the statistics of one logical plan.

    Args:
        plan: The logical plan.
        model: Model spec.
        device: Device spec.
        doc_tokens: alias -> per-document token counts, of every
            document in the table.
        pair_fractions: join written position -> the fraction of the
            cross product its equality conditions keep.
        scan_fractions: alias -> the fraction of its documents the
            regular predicates are expected to keep; the alias's counts and
            sums are scaled by it.
        context: Prompt preparation and model settings for score operators.

    Raises:
        ValueError: A scanned alias has no document token counts.
    """
    operators = plan.operators()
    asks = ask_positions(operators.filters)
    ask_filters = {alias: [operators.filters[alias][position]
                           for position in positions]
                   for alias, positions in asks.items() if positions}
    lengths = {a: summarize_alias(t, model.sliding_window)
               for a, t in doc_tokens.items()}
    for alias, fraction in (scan_fractions or {}).items():
        if alias in lengths:
            lengths[alias] = scale_alias(lengths[alias], fraction)
    stats = {
        a: CorpusStats(n_docs=s.count, total_tokens=s.total,
                       max_doc_tokens=s.maximum)
        for a, s in lengths.items()
    }
    for s in operators.scans:
        if s.alias not in stats:
            raise ValueError(f"no doc_tokens for alias {s.alias!r}")
    chunk = budgets.chunk_budget(model, device)
    # the longest documents bind the sliding-pool split
    longest_mean = max(
        (st.mean_doc_tokens for st in stats.values()), default=None)
    arena_split = budgets.arena_pages(model, device, chunk, longest_mean)
    admission = arena_split[0] * budgets.PAGE_TOKENS
    return PlanStatistics(
        stats=stats, lengths=lengths, chunk=chunk,
        arena_pages=tuple(arena_split), admission=admission,
        cap_pages=retention_pages(admission, chunk, budgets.PAGE_TOKENS),
        pre=preamble_tokens(ask_filters, operators.joins),
        specs=tuple(join_specs(operators.joins, pair_fractions,
                               model.canvas_tokens, context=context,
                               chunk=chunk)),
        asks=asks)


def cached_statistics(plan: LogicalPlan, memo: dict, *, model: ModelSpec,
                      device: DeviceSpec, doc_tokens: dict,
                      pair_fractions=None, scan_fractions=None,
                      context=None) -> PlanStatistics:
    """Return plan_statistics, computed once per memo and plan root.

    Every logical rule and physical planner of one query that prices
    the same plan share one summary of each table's lengths.
    """
    key = ("statistics", plan.root)
    if key not in memo:
        memo[key] = plan_statistics(
            plan, model=model, device=device, doc_tokens=doc_tokens,
            pair_fractions=pair_fractions, scan_fractions=scan_fractions,
            context=context)
    return memo[key]


def filter_stop_keys(plan: LogicalPlan) -> dict:
    """Return alias -> the stop key the distinct_pushdown rule put on its filters."""
    keys = {}
    for node in plan.walk():
        if isinstance(node, SemanticFilter) and node.stop_key:
            (alias,) = model_call(node.predicates[0].expression).aliases()
            keys[alias] = tuple(node.stop_key)
    return keys


def live_after_filters(plan: LogicalPlan, statistics: PlanStatistics) -> dict:
    """Return the expected live documents per alias after its filters run.

    A filter on a label of a table classified after the joins thins
    nothing before them.
    """
    after_joins = classified_above_joins(plan.root)
    operators = plan.operators()
    live = {a: float(st.n_docs) for a, st in statistics.stats.items()}
    for alias in operators.all_filters():
        survival = 1.0
        for predicate in operators.filters.get(alias, ()):
            survival *= effective_selectivity(predicate.selectivity)
        if alias not in after_joins:
            for test in operators.label_filters.get(alias, ()):
                survival *= effective_selectivity(test.selectivity)
        live[alias] *= survival
    return live


def filter_alias_work(preds, stats, order, pre: int,
                      canvas: int = 0, window: int = 0) -> Work:
    """Prepare a filter chain's counts and price its recorded order."""
    predicates = [preds[index] for index in order]
    return filter_chain_work(
        [question_tokens(predicate.prompt, canvas) for predicate in predicates],
        [effective_selectivity(predicate.selectivity) for predicate in predicates],
        pre + stats.mean_doc_tokens, float(stats.n_docs), window)


def filter_works(plan: LogicalPlan, statistics: PlanStatistics,
                 model: ModelSpec, orders: dict) -> dict:
    """Return the expected Work of each table's filter chain in run order.

    Args:
        plan: The logical plan.
        statistics: The plan's statistics.
        model: Model spec.
        orders: alias -> written positions of its model predicates in
            run order.
    """
    return {
        alias: filter_alias_work(
            predicates, statistics.stats[alias],
            [i for i in orders.get(alias, []) if i in statistics.asks[alias]],
            statistics.pre, model.canvas_tokens, model.sliding_window)
        for alias, predicates in plan.operators().filters.items()
    }


@dataclass(frozen=True)
class ScoreJoinCost:
    """Supply score work to join search at the given survivor counts."""

    aliases: tuple[str, str]
    cost: ScoreCost
    pair_fraction: float

    def work(self, live) -> Work:
        expected, groups = pair_counts(live, self.aliases, self.pair_fraction)
        return self.cost.estimate(expected, prefix_groups=groups)[0]


def pair_counts(live, aliases, fraction) -> tuple[float, float]:
    """Return expected pairs and first documents with at least one pair."""
    left, right = aliases
    touched = (1.0 if fraction >= 1 else
               -math.expm1(live[right] * math.log1p(-fraction)))
    return live[left] * live[right] * fraction, live[left] * touched


def score_statistics(prompt, context, chunk, *, mean_tokens=None):
    """Prepare a score's prompt tokens and numeric cost inputs once per query."""
    key = ("score_prompt", prompt)
    if key not in context.memo:
        context.memo[key] = score_token_parts(prompt, context.model, context.tokenizer)
    parts = context.memo[key]
    aliases = prompt_aliases(prompt)
    lengths = tuple(context.document_tokens[alias] for alias in aliases)
    if mean_tokens is None:
        mean_tokens = sum(sum(values) / max(1, len(values)) for values in lengths)
    cost = ScoreCost(
        token_lengths=tuple(map(len, parts)), lengths=lengths,
        mean_tokens=mean_tokens, model=context.model, device=context.device,
        chunk_tokens=chunk,
        capacity=budgets.arena_tokens(context.model, context.device, chunk),
        workers=context.gpu_count,
        draws=context.canvas_draws if context.model.canvas_tokens
        and len(aliases) == 1 else 1)
    return parts, cost


def prepare_filter_costs(predicates, *, prefix_tokens, model, device,
                         chunk_tokens, context=None) -> tuple[PredicateCost, ...]:
    """Prepare costs and reusable result identities for predicate ordering."""
    prepared = []
    for predicate in predicates:
        if is_score(predicate.expression):
            _, cost = score_statistics(
                predicate.prompt, context, chunk_tokens, mean_tokens=prefix_tokens)
            first = later = cost.seconds
            reuse_key = predicate.prompt
        else:
            q = question_tokens(predicate.prompt, model.canvas_tokens)
            first = partial(mul, filter_cost(
                q, prefix_tokens, model, device, chunk_tokens, first=True))
            later = partial(mul, filter_cost(
                q, prefix_tokens, model, device, chunk_tokens, first=False))
            reuse_key = None
        prepared.append(PredicateCost(
            effective_selectivity(predicate.selectivity), first, later, reuse_key))
    return tuple(prepared)


def classify_statistics(context, alias: str, shared=()) -> "ClassifyStatistics":
    """Gather document lengths and execution limits for one classification table.

    Args:
        context: The PlanningContext.
        alias: The table's alias.
        shared: Per document, the prefix tokens an earlier document
            also has, when the plan shares prefixes; else empty.

    Returns:
        The table's document lengths, budgets, and model settings.
    """
    lengths = [int(length) for length in context.document_tokens[alias]]
    count = len(lengths)
    total = sum(lengths)
    chunk = budgets.chunk_budget(context.model, context.device)
    capacity = budgets.arena_tokens(context.model, context.device, chunk)
    return ClassifyStatistics(
        alias=alias, mean=total / max(1, count),
        longest=max(lengths, default=0), budget=min(chunk, capacity),
        chunk=chunk,
        draws=context.canvas_draws if context.model.answer_canvas else 1,
        model=context.model,
        device=context.device,
        capacity=capacity, lengths=tuple(lengths), shared=tuple(shared),
        tree=(budgets.tree_attention_allowed(context.model)
              and getattr(context, "attention", None) != "unified"))


@dataclass(frozen=True)
class ClassifyStatistics:
    """Document lengths, model settings, and execution budgets for one table.

    Attributes:
        alias: Table alias in the query.
        mean: Mean document length in tokens.
        longest: Maximum document length in tokens.
        budget: Smaller of the forward-pass token budget and KV capacity.
        chunk: Maximum tokens per forward pass.
        model: Model specification.
        device: Device specification.
        capacity: KV arena capacity in tokens.
        draws: Maximum diffusion draws per document; one for causal models.
        lengths: Document lengths in tokens.
        shared: Shared prefix length per document, or an empty tuple.
        tree: Whether the tree attention path, which trie_tree needs, is
            available: an fp8 model without a canvas whose attention is not
            forced to unified.
    """

    alias: str
    mean: float
    longest: int
    budget: int
    chunk: int
    model: object
    device: object
    capacity: int = 0
    draws: int = 1
    lengths: tuple = ()
    shared: tuple = ()
    tree: bool = False

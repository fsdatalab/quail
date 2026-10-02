"""The numbers the cost-based rules and the physical planner read.

One PlanStatistics holds what the cost model knows about a logical
plan before any decision: each table's document lengths, the model's
token budgets, the preamble, and the joins as the search prices them.
The functions after it read the decisions the logical rules recorded
on the plan's nodes: each table's filter order and the join sequence.
"""

from dataclasses import dataclass, replace

from quail.cost import budgets
from quail.cost.retention import retention_pages
from quail.cost.work import Work, ask, scan
from quail.logical import (
    LabelIn,
    LogicalPlan,
    SemanticFilter,
    SemanticJoin,
    classified_above_joins,
    effective_selectivity,
    join_conditions,
    model_call,
)
from quail.planner import joins as joinsearch
from quail.planner.plan import CorpusStats
from quail.specs import DeviceSpec, ModelSpec


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


def join_specs(joins, pair_fractions=None, canvas: int = 0) -> list:
    """The joins as the search's spec dicts, in written order.

    pair_fractions maps a written position to the fraction of the
    cross product its equality conditions keep; a join with
    conditions but no entry is priced as the full cross product.
    canvas is the rows a diffusion model appends to every pair.
    """
    pair_fractions = pair_fractions or {}
    out = []
    for i, j in enumerate(joins):
        labels = _label_counts(j)
        conditions = join_conditions(j)
        out.append(dict(
            written_pos=i, aliases=_join_aliases(j), anchor=j.anchor,
            anchor_free=(j.anchor is None and j.semantics == "full"),
            semantics=j.semantics, selectivity=j.selectivity,
            frame_tokens={a: nt for a, (lt, nt) in labels.items()},
            label_tokens={a: lt for a, (lt, nt) in labels.items()},
            tail_tokens=question_tokens(j.prompt, canvas),
            on=[(c.left.alias, c.left.column, c.right.alias, c.right.column)
                for c in conditions],
            pair_fraction=(pair_fractions.get(i, 1.0) if conditions
                           else 1.0)))
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
    """Return alias -> written positions of its AI.IF predicates.

    Filters on labels and projected labels are classification work
    after the AI.IF chain.
    """
    return {alias: [position for position, predicate in enumerate(predicates)
                    if not isinstance(predicate.expression, LabelIn)]
            for alias, predicates in filters.items()}


def plan_statistics(plan: LogicalPlan, *, model: ModelSpec,
                    device: DeviceSpec, doc_tokens: dict,
                    pair_fractions=None) -> PlanStatistics:
    """Gather the statistics of one logical plan.

    Args:
        plan: The logical plan.
        model: Model spec.
        device: Device spec.
        doc_tokens: alias -> per-document token counts.
        pair_fractions: join written position -> the fraction of the
            cross product its equality conditions keep.

    Raises:
        ValueError: A scanned alias has no document token counts.
    """
    operators = plan.operators()
    asks = ask_positions(operators.filters)
    ask_filters = {alias: [operators.filters[alias][position]
                           for position in positions]
                   for alias, positions in asks.items() if positions}
    lengths = {a: joinsearch.summarize_alias(t, model.sliding_window)
               for a, t in doc_tokens.items()}
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
                               model.canvas_tokens)),
        asks=asks)


def undecided(root) -> object:
    """Return the plan root with the filter_order and join_order decisions cleared."""

    def visit(node):
        children = tuple(visit(child) for child in node.children())
        if children != node.children():
            node = node.with_children(children)
        if isinstance(node, SemanticFilter) and node.order:
            return replace(node, order=())
        if isinstance(node, SemanticJoin) and node.exec_idx is not None:
            return replace(node, exec_idx=None, exec_anchor=None)
        return node

    return visit(root)


def cached_statistics(plan: LogicalPlan, memo: dict, *, model: ModelSpec,
                      device: DeviceSpec, doc_tokens: dict,
                      pair_fractions=None) -> PlanStatistics:
    """Return plan_statistics, computed once per memo and undecided plan root.

    The decisions the rules record do not change the statistics, so
    every rule and the physical planner of one query share one
    summary of each table's lengths.
    """
    key = ("statistics", undecided(plan.root))
    if key not in memo:
        memo[key] = plan_statistics(
            plan, model=model, device=device, doc_tokens=doc_tokens,
            pair_fractions=pair_fractions)
    return memo[key]


def filter_orders(plan: LogicalPlan) -> dict:
    """Return alias -> written positions of its AI.IF predicates in execution order.

    Reads the ``order`` the filter_order rule recorded on each
    SemanticFilter; a node without one runs its predicates as written.
    A written position counts the alias's predicates over every
    SemanticFilter, lowest node first, as Operators lists them.
    """
    counted = {}
    orders = {}
    for node in plan.walk():
        if not isinstance(node, SemanticFilter):
            continue
        positions = []
        for predicate in node.predicates:
            (alias,) = model_call(predicate.expression).aliases()
            positions.append((alias, counted.get(alias, 0)))
            counted[alias] = counted[alias] + 1 if alias in counted else 1
        for index in node.order or range(len(node.predicates)):
            alias, position = positions[index]
            if not isinstance(node.predicates[index].expression, LabelIn):
                orders.setdefault(alias, []).append(position)
    return orders


def live_after_filters(plan: LogicalPlan, statistics: PlanStatistics) -> dict:
    """Expected live documents per alias once its filters have run.

    A filter on a label of a table classified after the joins thins
    nothing before them.
    """
    after_joins = classified_above_joins(plan.root)
    live = {a: float(st.n_docs) for a, st in statistics.stats.items()}
    for alias, predicates in plan.operators().filters.items():
        survival = 1.0
        for predicate in predicates:
            if alias in after_joins and isinstance(predicate.expression,
                                                   LabelIn):
                continue
            survival *= effective_selectivity(predicate.selectivity)
        live[alias] *= survival
    return live


def filter_alias_work(preds, stats, order, pre: int,
                      canvas: int = 0, window: int = 0) -> Work:
    """Expected Work of one filter chain: a scan, then asks over KV."""
    total = Work()
    mean = stats.mean_doc_tokens
    n = float(stats.n_docs)
    for si, predicate_index in enumerate(order):
        p = preds[predicate_index]
        q = question_tokens(p.prompt, canvas)
        op = scan if si == 0 else ask
        total = total + op(pre + mean, q, window=window) * n
        n *= effective_selectivity(p.selectivity)
    return total


def filter_works(plan: LogicalPlan, statistics: PlanStatistics,
                 model: ModelSpec) -> dict:
    """Expected Work of every table's filter chain, in its execution order."""
    orders = filter_orders(plan)
    return {
        alias: filter_alias_work(
            predicates, statistics.stats[alias], orders.get(alias, []),
            statistics.pre, model.canvas_tokens, model.sliding_window)
        for alias, predicates in plan.operators().filters.items()
    }


def join_sequence(plan: LogicalPlan) -> list:
    """Return [(written position, anchor)] in execution order.

    Reads the ``exec_idx`` and ``exec_anchor`` the join_order rule
    recorded on each SemanticJoin. When any join has none, the joins
    run as written, each on its written anchor or, for a free full
    join, on its first table.
    """
    joins = plan.operators().joins
    if any(join.exec_idx is None for join in joins):
        return [(position, join.anchor or _join_aliases(join)[0])
                for position, join in enumerate(joins)]
    ordered = sorted(enumerate(joins), key=lambda item: item[1].exec_idx)
    return [(position, join.exec_anchor) for position, join in ordered]


def sequence_specs(plan: LogicalPlan, statistics: PlanStatistics) -> list:
    """Return [(spec, anchor)] in execution order, for walk and schedule."""
    return [(statistics.specs[position], anchor)
            for position, anchor in join_sequence(plan)]

"""Build executable query graphs with operator order and execution settings."""

from dataclasses import dataclass, field, replace
from itertools import groupby

from quail.logical import (
    Alias,
    LogicalPlan,
    classified_above_joins,
    column_name,
    effective_selectivity,
    extracted_above_joins,
    has_score,
    is_score,
    oriented_join_conditions,
)
from quail.physical import (
    AiScore,
    Barrier,
    Filter,
    Foreign,
    HashJoin,
    PortRef,
    Recombine,
)
from quail.physical import (
    Scan as PhysicalScan,
)
from quail.physical.base import input_ports
from quail.planner.graph import retention
from quail.planner.graph.pricing import PlanPricer
from quail.planner.graph.results import result_nodes
from quail.planner.graph.shards import contiguous_shards
from quail.planner.operators.classify import (
    emit_classify,
    emit_joined_classify,
)
from quail.planner.operators.extract import emit_extract
from quail.planner.operators.filter import emit_filter
from quail.planner.operators.frontier import Frontier
from quail.planner.operators.join import emit_join
from quail.planner.operators.score import (
    ScoreLowering,
    emit_projected_scores,
    emit_score_joins,
    emit_score_pair,
    emit_scores,
)
from quail.planner.optimizer.context import PlanningContext
from quail.planner.plan import PhysicalPlan, Refusal
from quail.planner.search.operator_order import choose_operator_order
from quail.planner.search.statistics import filter_stop_keys
from quail.planner.search.table_steps import (
    APPLY,
    ASK,
    CLASSIFY,
    EXTRACT,
    TEST,
    order_table_steps,
    ready_table_steps,
    resident_after,
)
from quail.planner.statistics.tables import summarize_alias, thin
from quail.planner.validation import (
    ClassifyRefusedError,
    ScoreRefusedError,
    score_projections,
    validate_query,
)
from quail.specs import DeviceSpec, ModelSpec


def hash_join_nodes(joins, pair_fractions, scan_ports) -> list:
    """One HashJoin per join with equality conditions, over the scans.

    Args:
        joins: The logical joins in written order.
        pair_fractions: written position -> pairs kept over the cross
            product, as the session measured them.
        scan_ports: The scans' id ports, one per alias, in any order.

    Returns:
        HashJoin nodes with ids ``hash_join:<left>-<right>``.
    """
    pair_fractions = pair_fractions or {}
    by_alias = {port.port.split(":", 1)[1]: port for port in scan_ports}
    nodes = []
    for position, join in enumerate(joins):
        oriented = oriented_join_conditions(join)
        if oriented is None:
            continue
        left, right, conditions = oriented
        nodes.append(HashJoin(
            node_id=f"hash_join:{left}-{right}",
            inputs=input_ports((by_alias[left], by_alias[right])),
            left=left, right=right, on=conditions, written_pos=position,
            pair_fraction=pair_fractions.get(position, 1.0)))
    return nodes


def build_physical_plan(plan: LogicalPlan, *, model: ModelSpec,
                        device: DeviceSpec, doc_tokens: dict, gpus: int = 1,
                        order: str | None = None, pair_fractions=None,
                        context: PlanningContext | None = None,
                        scan_fractions=None):
    """Compile a LogicalPlan into a PhysicalPlan or Refusal.

    The builder chooses each table's filter order and the joins' stage
    order and anchors by cost (quail.planner.search.operator_order), unless order
    is "as_written". A one-table classification above a SemanticJoin
    runs after the joins, over the documents the joins matched. One on
    its table is ordered with the table's AI.IF filters and extractions.

    The initial estimate includes filter and join work and any inference
    costs already known. The planner then runs the graph refinement rules to select
    classification implementations, KV retention, and prefix sharing before
    comparing this candidate with others.

    Args:
        plan: The logical plan to compile.
        model: Model spec.
        device: Device spec.
        doc_tokens: alias -> list of per-document token counts.
        gpus: GPU count; one model copy runs per GPU.
        order: Stage order rule, 'by_cost' or 'as_written'; None picks
            the default rule. Reported in the plan's settings.
        pair_fractions: join written position -> the fraction of the
            cross product its equality conditions keep.
        context: The planning context, needed when the plan classifies
            documents.
        scan_fractions: alias -> the fraction of its documents the
            regular predicates are expected to keep; the context's when None.

    Returns:
        A PhysicalPlan, or a Refusal explaining why the query cannot run.
    """
    operators = plan.operators()
    refusal = validate_query(plan, operators, context, model, device, gpus)
    if refusal is not None:
        return refusal
    if scan_fractions is None and context is not None:
        scan_fractions = context.scan_fractions
    operator_order = choose_operator_order(
        plan, operators, model, device, doc_tokens, order,
        pair_fractions, context, scan_fractions)
    if isinstance(operator_order, Refusal):
        return operator_order
    return GraphBuilder(plan, model, device, doc_tokens, gpus, context,
                        pair_fractions, operator_order).build()


class GraphBuilder:
    """Construct native physical nodes in the selected operator order."""

    def __init__(self, plan, model, device, doc_tokens, workers, context,
                 pair_fractions, operator_order):
        self.plan = plan
        self.model = model
        self.device = device
        self.doc_tokens = doc_tokens
        self.workers = workers
        self.context = context
        self.pair_fractions = pair_fractions
        self.operator_order = operator_order
        self.operators = operator_order.operators
        self.classified = {alias for _, alias in self.operators.labels.calls}
        self.after_joins = classified_above_joins(plan.root)
        self.extract_after = extracted_above_joins(plan.root)
        self.projected_scores = score_projections(plan)
        self.alias_applies = {}
        self.join_applies = {}
        for apply in self.operators.applies:
            if apply.ids == "pairs":
                self.join_applies.setdefault(apply.written_pos, []).append(apply)
            else:
                self.alias_applies.setdefault(apply.aliases[0], []).append(apply)
        self.ask_filters = {
            alias: [self.operators.filters[alias][position] for position in positions]
            for alias, positions in operator_order.statistics.asks.items() if positions}
        self.stop_keys = filter_stop_keys(plan)

    def build(self):
        """Wire scans, inference, joins, and result operators into a physical plan.

        Each step takes where every table's documents stand and returns
        where they stand after it, so an operator's input is always the
        output of the operator before it.
        """
        self._prepare()
        frontiers = self._emit_scans()
        try:
            frontiers, label_ports = self._emit_documents(frontiers)
            frontiers, joins = self._emit_joins(frontiers)
            label_ports += self._emit_join_answers(frontiers, joins)
            score_pairs = self._emit_pair_scores(frontiers, joins)
        except ClassifyRefusedError as refused:
            return refused.refusal()
        except ScoreRefusedError as refused:
            return refused.refusal
        self._emit_result(frontiers, label_ports, joins, score_pairs)
        return self._finish()

    def _prepare(self):
        """Prepare graph identities and KV compatibility."""
        statistics = self.operator_order.statistics
        # a second group on the same anchor gets :2, a third :3
        self.sequence_groups = retention.group_sequence(
            self.operator_order.join_sequence)
        self.group_ids = []
        self.seen_ids = {}
        for group in self.sequence_groups:
            anchor = group[0][1]
            self.seen_ids[anchor] = self.seen_ids.get(anchor, 0) + 1
            self.group_ids.append(f"ai_join:{anchor}" if self.seen_ids[anchor] == 1
                             else f"ai_join:{anchor}:{self.seen_ids[anchor]}")

        # a sequence streams only into its alias's first use, and never through
        # a barrier apply
        barrier_aliases = {alias for alias, group in self.alias_applies.items()
                           if any(apply.kind == "barrier" for apply in group)}
        for group in self.sequence_groups:
            anchor = group[0][1]
            for spec, _ in group:
                if any(apply.kind == "barrier"
                       for apply in self.join_applies.get(spec.written_pos, ())):
                    barrier_aliases.add(anchor)
        self.streamed = retention.streamed_aliases(
            self.sequence_groups, set(self.ask_filters), barrier_aliases)

        self.nodes = []
        self.value_ports = {}
        self.pending_steps = {
            alias: list(steps)
            for alias, steps in self.operator_order.step_orders.items()
            if any(step.joins for step in steps)}
        self.scheduled_aliases = set(self.pending_steps)
        self.live_steps = dict(self.operator_order.live)
        for alias in self.scheduled_aliases:
            self.live_steps[alias] = float(statistics.stats[alias].n_docs)
        self.scoring = (ScoreLowering(self.context, self.nodes,
                                     self.projected_scores, statistics.chunk)
                        if has_score(self.plan) else None)

    def _emit_scans(self) -> dict:
        """Create the scans and equality joins; return where each table starts."""
        stats = self.operator_order.statistics.stats
        frontiers = {}
        for s in self.operators.scans:
            summary = summarize_alias(
                self.doc_tokens[s.alias], self.model.sliding_window,
                memo=self.context.memo if self.context is not None else None)
            shard_ranges, loads = contiguous_shards(
                self.doc_tokens[s.alias], self.workers, total=summary.total
            )
            sid = f"scan:{s.alias}"
            self.nodes.append(PhysicalScan(
                node_id=sid,
                alias=s.alias, input_id=s.alias,
                n_docs=len(self.doc_tokens[s.alias]),
                total_tokens=summary.total,
                shard_ranges=shard_ranges,
                shard_token_loads=tuple(loads),
                predicates=s.predicates,
                expected_docs=(float(stats[s.alias].n_docs)
                               if s.predicates else None)))
            frontiers[s.alias] = Frontier(PortRef(sid, f"ids:{s.alias}"))
        # the hash join reads the scans; survivors thin its pairs at the AI join
        self.pairs_src = {}
        for node in hash_join_nodes(self.operators.joins, self.pair_fractions,
                                    [frontier.ids for frontier in frontiers.values()]):
            self.nodes.append(node)
            self.pairs_src[node.written_pos] = node

        # aliases whose filter sequence runs in one pipeline with the
        # classification or join after it
        between = self.classified - self.after_joins
        self.pipelined = self.streamed | {
            alias for alias in self.ask_filters
            if alias in between}
        return frontiers

    def _emit_documents(self, frontiers: dict) -> tuple:
        """Create the inference nodes over each table before its joins."""
        stats = self.operator_order.statistics.stats
        label_ports = []
        frontiers = dict(frontiers)
        for s in self.operators.scans:
            if s.alias in self.scheduled_aliases:
                continue
            # a classification or extraction the planner lifted
            # above the joins is emitted after them
            kinds = {ASK, APPLY, TEST}
            if s.alias not in self.after_joins:
                kinds.add(CLASSIFY)
            if s.alias not in self.extract_after:
                kinds.add(EXTRACT)
            frontiers[s.alias], ports = self._emit_steps(
                s.alias, float(stats[s.alias].n_docs), kinds, frontiers[s.alias])
            label_ports += ports
        label_ports += self._emit_ready_steps(frontiers, Joins())
        return frontiers, label_ports

    def _emit_steps(self, alias, live, kinds, frontier, steps=None) -> tuple:
        """Emit the given kinds of a table's steps in run order.

        Consecutive steps of one kind are emitted together, each run
        over the documents the steps before it keep.

        Args:
            alias: The table.
            live: The documents the first step reads.
            kinds: The step kinds to emit: ASK, CLASSIFY, EXTRACT.
            frontier: Where the documents stand before the steps.
            steps: A scheduled subset, or None for every step of the given kinds.

        Returns:
            Where the documents stand after the steps, and the rows
            ports of the labels and answers SELECT returns.
        """
        if steps is None:
            steps = [step for step in self.operator_order.step_orders.get(alias, ())
                     if step.kind in kinds]
        label_ports = []
        previous = None
        for (kind, _), run in groupby(steps, key=lambda step: (
                step.kind, step.key if step.tests is not None
                or step.kind in (APPLY, TEST) else None)):
            run = list(run)
            positions = [step.key for step in run]
            resident = previous is not None and resident_after(previous, run[0])
            if kind == APPLY:
                frontier = self.emit_apply(positions[0], frontier)
            elif kind == TEST:
                frontier, _ = self.emit_tests(alias, positions, frontier, live,
                                               detached=True)
            elif kind == ASK and self.scoring is not None:
                frontier = emit_scores(self, alias, positions, live, frontier)
            elif kind == ASK:
                frontier = emit_filter(self, alias, positions, live, frontier,
                                       followed=run[-1] is not steps[-1])
            elif kind == CLASSIFY:
                frontier, ports = emit_classify(
                    self, alias, live, positions, frontier, tests=run[0].tests)
                label_ports += ports
            else:
                frontier, ports = emit_extract(self, alias, live, positions, frontier,
                                               resident=resident, tests=run[0].tests)
                label_ports += ports
            for step in run:
                live *= step.selectivity
            previous = run[-1]
        if ASK in kinds and self.scoring is not None:
            frontier = emit_projected_scores(self, alias, live, frontier)
        return frontier, label_ports

    def _join_survivors(self, frontiers, joins, aliases):
        """Bind aliases to the documents in the currently surviving joined rows."""
        if not aliases or not joins.pairs_edges:
            return
        bid = self.unique_id("barrier", "answers")
        self.nodes.append(Barrier(
            node_id=bid, aliases=tuple(frontiers),
            inputs=input_ports(tuple(joins.pairs_edges)
                               + tuple(value.ids for value in frontiers.values()))))
        for alias in aliases:
            frontiers[alias] = Frontier(PortRef(bid, f"ids:{alias}"))

    def _emit_ready_steps(self, frontiers, joins):
        """Emit ordered table work whose preceding joins have finished."""
        ports = []
        for alias, steps in ready_table_steps(self.pending_steps, joins.ids.keys()):
            if any(step.joins for step in steps):
                self._join_survivors(frontiers, joins, (alias,))
            steps = order_table_steps(
                steps, self.operator_order.order_rule, self.live_steps[alias])
            frontiers[alias], added = self._emit_steps(
                alias, self.live_steps[alias], {step.kind for step in steps},
                frontiers[alias], steps)
            ports += added
            for step in steps:
                self.live_steps[alias] *= step.selectivity
        return ports

    def _emit_joins(self, frontiers: dict) -> tuple:
        """Create the joins and the exchanges between anchor groups.

        Returns:
            Where each table's documents stand after the joins, and the
            Joins record of what the joins left.
        """
        # group consecutive full stages on the same anchor; gates run
        # alone; anchor switches become barriers
        steps = iter(self.operator_order.join_steps)
        groups = [[(spec, next(steps)) for spec, _ in group]
                  for group in self.sequence_groups]
        frontiers = dict(frontiers)
        joins = Joins()
        exec_idx = 0
        for g, group in enumerate(groups):
            anchor = self.sequence_groups[g][0][1]
            if group[0][0].cost is not None:
                joins.score_pairs = emit_score_joins(
                    self, group, frontiers, joins.score_pairs)
                continue
            if g > 0:
                ahead = [scan.alias for scan in self.operators.scans]
                bid = self.unique_id("barrier", anchor)
                self.nodes.append(Barrier(
                    node_id=bid,
                    inputs=input_ports(tuple(joins.pairs_edges)
                                       + tuple(frontiers[a].ids for a in ahead)),
                    next_anchor=anchor,
                    aliases=tuple(ahead)))
                for a in ahead:
                    frontiers[a] = Frontier(PortRef(bid, f"ids:{a}"))
            gid = self.group_ids[g]
            for rank, (spec, step) in enumerate(group):
                jid = gid if rank == 0 else f"{gid}:predicate:{spec.written_pos}"
                joins.ids[spec.written_pos] = jid
                output = emit_join(self, spec, step, anchor, jid, exec_idx, frontiers)
                frontiers[anchor] = output.anchor
                if output.pairs is not None:
                    joins.pairs_edges.append(output.pairs)
                    for a in spec.aliases:
                        if a not in joins.out_aliases:
                            joins.out_aliases.append(a)
                thin(self.live_steps, spec)
                joins.label_ports += self._emit_ready_steps(frontiers, joins)
                exec_idx += 1
        return frontiers, joins

    def _emit_join_answers(self, frontiers: dict, joins) -> list:
        """Create the classifications and extractions that follow the joins.

        Returns:
            The rows ports of the labels and answers SELECT returns.
        """
        label_ports = [*joins.label_ports, *emit_joined_classify(self, joins.ids)]
        deferred = (self.after_joins | self.extract_after) - self.scheduled_aliases
        pruned = deferred
        if len(self.operator_order.join_sequence) == 1:
            pruned = deferred - {self.operator_order.join_sequence[0][1]}
        self._join_survivors(frontiers, joins, pruned)
        for alias in sorted(deferred):
            kinds = set()
            if alias in self.after_joins:
                kinds.add(CLASSIFY)
            if alias in self.extract_after:
                kinds.add(EXTRACT)
            frontiers[alias], ports = self._emit_steps(
                alias, self.live_steps[alias], kinds,
                frontiers[alias])
            label_ports += ports
        return label_ports

    def _emit_pair_scores(self, frontiers: dict, joins) -> PortRef | None:
        """Create the pair scores SELECT returns; return their rows port."""
        score_pairs = joins.score_pairs
        for expression in self.projected_scores:
            call = expression.expression
            if len(call.aliases()) == 2 and call.prompt not in self.scoring.names:
                score_pairs = emit_score_pair(self, call.prompt, 0, 1.0, (), frontiers)
        return score_pairs

    def _emit_result(self, frontiers, label_ports, joins, score_pairs):
        """Connect result operators to surviving rows and answers."""
        if score_pairs is not None:
            sink_inputs = (score_pairs,)
        elif (len(joins.pairs_edges) == 1
              and len(self.operator_order.join_sequence) == 1
              and not self.after_joins and not self.scheduled_aliases):
            # one full join and nothing after it: its true pairs are the
            # result rows, so no recombination is needed
            sink_inputs = (joins.pairs_edges[0],)
        elif joins.pairs_edges:
            self.nodes.append(Recombine(
                node_id="recombine",
                inputs=input_ports(
                    tuple(joins.pairs_edges)
                    + tuple(frontiers[a].ids for a in joins.out_aliases)
                ),
                alias_order=tuple(joins.out_aliases)))
            sink_inputs = (PortRef("recombine", "tuples"),)
        else:
            alias = self.operators.scans[0].alias
            sink_inputs = (frontiers[alias].source,)
        columns = []
        for c in self.plan.projection.columns:
            columns.extend(c.expression.output_columns(c.name) if isinstance(c, Alias)
                           else (column_name(c),))
        score_ports = tuple(PortRef(node.node_id, "rows") for node in self.nodes
                            if isinstance(node, AiScore)
                            and any(node.spec.name == item.name
                                    for item in self.projected_scores))
        result_inputs = (*sink_inputs,
                         *(port for port in score_ports if port not in sink_inputs),
                         *label_ports)
        self.nodes.extend(result_nodes(self.plan.result, result_inputs, tuple(columns),
                                       self.plan.projection.packing()))

    def _finish(self):
        """Build the plan and bind its pricer."""
        statistics = self.operator_order.statistics
        # a score join's work is in its AiScore's estimate
        stage_works = {step.written_pos: step.worker_work(self.workers)
                       for step in self.operator_order.join_steps
                       if not is_score(
                           self.operators.joins[step.written_pos].predicate)}
        pricer = PlanPricer(
            context=self.context, statistics=statistics, stage_works=stage_works,
            live=self.operator_order.live)
        plan = PhysicalPlan(
            model=self.model.name, device=self.device.name, workers=self.workers,
            backend="quail", nodes=tuple(self.nodes),
            settings={
                "chunk_tokens": statistics.chunk,
                "arena_pages": list(statistics.arena_pages),
                "admission_tokens": statistics.admission,
                "order_rule": self.operator_order.order_rule,
                "order_source": self.operator_order.order_source,
                **({
                    "retained_kv_tokens": self.scoring.capacity,
                    "prefix_reuse": "fixed prompt and first document within each score",
                    "survivor_assumption": "uniform independent selection",
                    "estimated_fresh_tokens": self.scoring.work.tokens,
                    "estimated_attention_pairs": self.scoring.work.pairs,
                    "batching": "token_based_admission",
                    "data_parallel_copies": self.workers,
                    "score_normalization": {
                        "reranker": "yes_no_softmax",
                        "decision": "decision_head_softmax",
                    }.get(self.model.role, "true_false_softmax"),
                } if self.scoring is not None else {}),
                **({"classify_placement": ("after joins" if self.after_joins
                                           else "before joins")}
                   if self.operators.labels.calls else {}),
                **({"canvas_draws": self.context.canvas_draws}
                   if self.operators.labels.calls and self.model.answer_canvas else {}),
            },
            estimator=pricer)
        return replace(plan, estimated_seconds=pricer.seconds(plan.graph))

    def unique_id(self, prefix, name):
        key = f"{prefix}:{name}"
        self.seen_ids[key] = self.seen_ids.get(key, 0) + 1
        return key if self.seen_ids[key] == 1 else f"{key}:{self.seen_ids[key]}"

    def emit_apply(self, apply, frontier) -> Frontier:
        """Append one callback at its position in the table's sequence."""
        (alias,) = apply.aliases
        aid = self.unique_id("apply", apply.function)
        self.nodes.append(Foreign(
            node_id=aid, inputs=input_ports((frontier.ids,)),
            function=apply.function, kind=apply.kind, ids=apply.ids,
            columns=apply.columns, aliases=(alias,)))
        return Frontier(PortRef(aid, f"ids:{alias}"))

    def emit_tests(self, alias, tests, frontier, live, *, detached=False):
        """Test computed values on the documents still present at this point."""
        for test in tests:
            source = (self.value_ports[(alias, test.condition.column.column)]
                      if detached else frontier.rows)
            inputs = (source, frontier.ids) if detached else (source,)
            lid = f"filter:{alias}:{test.position}"
            self.nodes.append(Filter(
                node_id=lid, inputs=input_ports(inputs), predicate=test.condition,
                aliases=(alias,), selectivity=test.selectivity,
                written_pos=test.position))
            frontier = Frontier(PortRef(lid, f"ids:{alias}"), PortRef(lid, "rows"))
            live *= effective_selectivity(test.selectivity)
        return frontier, live


@dataclass
class Joins:
    """What the AI joins leave for the operators after them.

    Attributes:
        ids: Join written position -> its AiJoin node id.
        pairs_edges: Every full stage's passing-pairs port.
        out_aliases: The recombination's output order.
        score_pairs: The rows of the pairs the score joins keep, or None.
        label_ports: Values computed between join stages.
    """

    ids: dict = field(default_factory=dict)
    pairs_edges: list = field(default_factory=list)
    out_aliases: list = field(default_factory=list)
    score_pairs: PortRef | None = None
    label_ports: list = field(default_factory=list)

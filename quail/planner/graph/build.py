"""Build the physical plan of a decided logical plan."""

from functools import partial

from quail.cost.sol import speed_of_light
from quail.cost.work import Work
from quail.logical import (
    PROBABILITIES_SUFFIX,
    SPAN_SUFFIX,
    Alias,
    LogicalPlan,
    classified_above_joins,
    effective_selectivity,
    extracted_above_joins,
    has_score,
    is_score,
    oriented_join_conditions,
)
from quail.physical import (
    AiScore,
    Barrier,
    Exchange,
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
from quail.planner.graph.node_estimates import node_estimates
from quail.planner.graph.results import result_nodes
from quail.planner.graph.shards import contiguous_shards
from quail.planner.operators.classify import (
    emit_classify,
    emit_joined_classify,
)
from quail.planner.operators.extract import emit_extract
from quail.planner.operators.filter import emit_filter
from quail.planner.operators.join import emit_join
from quail.planner.operators.score import (
    ScoreLowering,
    emit_score_joins,
    emit_score_pair,
    emit_scores,
)
from quail.planner.optimizer.physical import PlanningContext
from quail.planner.plan import PhysicalPlan, Refusal, inference_seconds
from quail.planner.search.decisions import choose_plan, validate_query
from quail.planner.search.statistics import filter_stop_keys
from quail.planner.statistics.tables import summarize_alias
from quail.planner.validation import (
    ClassifyRefusedError,
    ScoreRefusedError,
    score_projections,
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
        on = tuple(
            (left_ref.column, right_ref.column)
            for left_ref, right_ref in conditions
        )
        nodes.append(HashJoin(
            node_id=f"hash_join:{left}-{right}",
            inputs=input_ports((by_alias[left], by_alias[right])),
            left=left, right=right, on=on, written_pos=position,
            pair_fraction=pair_fractions.get(position, 1.0)))
    return nodes


def regular_predicates(scan) -> tuple:
    """Return a logical scan's regular predicates as (column, comparison, value)."""
    return tuple((predicate.column.column, predicate.comparison,
                  predicate.value) for predicate in scan.predicates)


def build_physical_plan(plan: LogicalPlan, *, model: ModelSpec,
                        device: DeviceSpec, doc_tokens: dict, gpus: int = 1,
                        order: str | None = None, pair_fractions=None,
                        context: PlanningContext | None = None,
                        scan_fractions=None):
    """Compile a LogicalPlan into a PhysicalPlan or Refusal.

    The builder chooses each table's filter order and the joins' stage
    order and anchors by cost (quail.planner.search.ordering), unless order
    is "as_written". A one-table classification above a SemanticJoin
    runs after the joins, over the documents the joins matched. One on
    its table runs after the table's AI.IF filters.

    The choose_classify_implementation physical rule picks each classification's scoring
    rule, and the kv_retention rule picks which KV stays resident. The
    estimate and ``search_seconds`` count only the filters, the joins,
    and the classifications of joined rows.

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
    decisions = choose_plan(plan, operators, model, device, doc_tokens, order,
                            pair_fractions, context, scan_fractions)
    if isinstance(decisions, Refusal):
        return decisions
    return GraphBuilder(plan, model, device, doc_tokens, gpus, context,
                        pair_fractions, decisions).build()


class GraphBuilder:
    """Construct native physical nodes from validated planning decisions."""

    def __init__(self, plan, model, device, doc_tokens, workers, context,
                 pair_fractions, decisions):
        self.plan = plan
        self.model = model
        self.device = device
        self.doc_tokens = doc_tokens
        self.workers = workers
        self.context = context
        self.pair_fractions = pair_fractions
        self.decisions = decisions
        self.operators = decisions.operators
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
            for alias, positions in decisions.statistics.asks.items() if positions}
        self.stop_keys = filter_stop_keys(plan)

    def build(self):
        """Wire scans, inference, joins, and result operators into a physical plan."""
        self._prepare()
        self._emit_scans()
        for emit in (self._emit_documents, self._emit_joins, self._emit_join_answers):
            refusal = emit()
            if refusal is not None:
                return refusal
        self._emit_result()
        return self._finish()

    def _prepare(self):
        """Prepare graph identities and KV compatibility."""
        statistics = self.decisions.statistics
        # a second group on the same anchor gets :2, a third :3
        self.sequence_groups = retention.group_sequence(self.decisions.join_sequence)
        self.group_ids = []
        self.join_ids = {}
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
            self.sequence_groups, set(self.ask_filters), barrier_aliases, self.workers)

        self.nodes = []
        self.ids_src = {}
        self.rows_src = {}
        self.scoring = (ScoreLowering(self.context, self.nodes,
                                     self.projected_scores, statistics.chunk)
                        if has_score(self.plan) else None)

    def _emit_scans(self):
        """Create scan and equality-join inputs."""
        stats = self.decisions.statistics.stats
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
                predicates=regular_predicates(s),
                expected_docs=(float(stats[s.alias].n_docs)
                               if s.predicates else None)))
            self.ids_src[s.alias] = PortRef(sid, f"ids:{s.alias}")
        # the hash join reads the scans; survivors thin its pairs at the AI join
        self.pairs_src = {}
        for node in hash_join_nodes(self.operators.joins, self.pair_fractions,
                                    self.ids_src.values()):
            self.nodes.append(node)
            self.pairs_src[node.written_pos] = node

        # aliases whose filter sequence runs in one pipeline with the
        # classification or join after it
        between = self.classified - self.after_joins
        self.pipelined = self.streamed | {
            alias for alias in self.ask_filters
            if self.workers == 1 and alias in between}

    def _emit_documents(self):
        """Create inference nodes over individual tables."""
        stats = self.decisions.statistics.stats
        # documents expected after a table's model predicates, before its
        # filters on labels
        self.live_asked = {}
        for alias, predicates in self.operators.filters.items():
            survival = 1.0
            for predicate in predicates:
                survival *= effective_selectivity(predicate.selectivity)
            self.live_asked[alias] = float(stats[alias].n_docs) * survival
        self.label_ports = []

        try:
            for s in self.operators.scans:
                if self.scoring is not None:
                    emit_scores(self, s.alias)
                elif s.alias in self.ask_filters:
                    emit_filter(self, s.alias)
                else:
                    self.emit_applies(s.alias)
                emit_classify(self, s.alias)
                emit_extract(self, s.alias)
        except ClassifyRefusedError as refused:
            return refused.refusal()
        except ScoreRefusedError as refused:
            return refused.refusal

    def _emit_joins(self):
        """Create joins and the exchanges between anchor groups."""
        # group consecutive full stages on the same anchor; gates run
        # alone; anchor switches become barriers
        steps = iter(self.decisions.join_steps)
        groups = [[(spec, next(steps)) for spec, _ in group]
                  for group in self.sequence_groups]
        exec_idx = 0
        self.pairs_edges = []     # every full stage's passing-pairs edge
        self.out_aliases = []     # recombination's output order
        self.score_pairs = None
        for g, group in enumerate(groups):
            anchor = self.sequence_groups[g][0][1]
            if group[0][0].cost is not None:
                try:
                    emit_score_joins(self, group)
                except ScoreRefusedError as refused:
                    return refused.refusal
                continue
            if g > 0:
                ahead = [scan.alias for scan in self.operators.scans]
                bid = self.unique_id("barrier", anchor)
                exchange_inputs = tuple(self.pairs_edges) + tuple(
                    self.ids_src[a] for a in ahead
                )
                self.nodes.append(Barrier(
                    node_id=bid,
                    inputs=input_ports(exchange_inputs),
                    next_anchor=anchor,
                    aliases=tuple(ahead)))
                for a in ahead:
                    self.ids_src[a] = PortRef(bid, f"ids:{a}")
            if self.workers > 1:
                # anchors go to the GPU holding their KV, else balanced
                xid = self.unique_id("exchange", anchor)
                self.nodes.append(Exchange(
                    node_id=xid,
                    inputs=input_ports((self.ids_src[anchor],)),
                    anchor=anchor))
                self.ids_src[anchor] = PortRef(xid, f"ids:{anchor}")
            gid = self.group_ids[g]
            for rank, (spec, step) in enumerate(group):
                jid = gid if rank == 0 else f"{gid}:predicate:{spec.written_pos}"
                self.join_ids[spec.written_pos] = jid
                emit_join(self, spec, step, anchor, jid, exec_idx)
                exec_idx += 1

    def _emit_join_answers(self):
        """Create classification and extraction nodes after joins."""
        try:
            emit_joined_classify(self)
        except ClassifyRefusedError as refused:
            return refused.refusal()

        # classifications after the joins run over the documents the joins
        # matched: a document with at least one true pair, expected as the
        # join selectivity times the partners it met, at most every document
        try:
            for alias in sorted(self.after_joins | self.extract_after):
                met = sum(
                    effective_selectivity(join.selectivity)
                    * sum(self.decisions.live[argument.alias]
                          for argument in join.prompt.args
                          if argument.alias != alias)
                    for join in self.operators.joins
                    if alias in {argument.alias for argument in join.prompt.args})
                if alias in self.after_joins:
                    emit_classify(
                        self, alias, live=self.decisions.live[alias] * min(1.0, met))
                if alias in self.extract_after:
                    emit_extract(
                        self, alias, live=self.decisions.live[alias] * min(1.0, met))
        except ClassifyRefusedError as refused:
            return refused.refusal()

        try:
            for expression in self.projected_scores:
                call = expression.expression
                if len(call.aliases()) == 2 and call.prompt not in self.scoring.names:
                    self.score_pairs = emit_score_pair(self, call.prompt, 0, 1.0, ())
        except ScoreRefusedError as refused:
            return refused.refusal

    def _emit_result(self):
        """Connect result operators to surviving rows and answers."""
        if self.score_pairs is not None:
            sink_inputs = (self.score_pairs,)
        elif (len(self.pairs_edges) == 1 and len(self.decisions.join_sequence) == 1
              and not self.after_joins):
            # one full join and nothing after it: its true pairs are the
            # result rows, so no recombination is needed
            sink_inputs = (self.pairs_edges[0],)
        elif self.pairs_edges:
            self.nodes.append(Recombine(
                node_id="recombine",
                inputs=input_ports(
                    tuple(self.pairs_edges)
                    + tuple(self.ids_src[a] for a in self.out_aliases)
                ),
                alias_order=tuple(self.out_aliases)))
            sink_inputs = (PortRef("recombine", "tuples"),)
        else:
            alias = self.operators.scans[0].alias
            sink_inputs = (self.rows_src.get(alias, self.ids_src[alias]),)
        columns = []
        for c in self.plan.projection.columns:
            columns.append(c.name if isinstance(c, Alias)
                           else f"{c.alias}.{c.column}")
            if isinstance(c, Alias) and getattr(c.expression, "probabilities", False):
                columns.append(c.name + PROBABILITIES_SUFFIX)
            if isinstance(c, Alias) and c.expression.kind == "extract":
                columns.append(c.name + SPAN_SUFFIX)
        score_ports = tuple(PortRef(node.node_id, "rows") for node in self.nodes
                            if isinstance(node, AiScore)
                            and any(node.spec.name == item.name
                                    for item in self.projected_scores))
        result_inputs = (*sink_inputs,
                         *(port for port in score_ports if port not in sink_inputs),
                         *self.label_ports)
        self.nodes.extend(result_nodes(self.plan.result, result_inputs, tuple(columns),
                                       self.plan.projection.packing()))

    def _finish(self):
        """Build the plan and bind its work estimator."""
        stats = self.decisions.statistics.stats
        statistics = self.decisions.statistics
        stage_work = sum((step.work for step in self.decisions.join_steps
                          if not is_score(
                              self.operators.joins[step.written_pos].predicate)),
                         Work())
        estimate = (speed_of_light(
            sum(self.decisions.filter_work.values(), Work()) + stage_work,
            self.model, self.device,
            statistics.chunk).seconds + inference_seconds(self.nodes))
        stage_works = {step.written_pos: step.work
                       for step in self.decisions.join_steps}

        estimator = partial(
            node_estimates, filter_works=self.decisions.filter_work,
            stage_works=stage_works,
            live=self.decisions.live, stats=stats, pre=statistics.pre,
            cap_pages=statistics.cap_pages,
            model=self.model, device=self.device, chunk=statistics.chunk)
        return PhysicalPlan(
            model=self.model.name, device=self.device.name, workers=self.workers,
            backend="quail", estimated_seconds=estimate,
            nodes=tuple(self.nodes),
            settings={
                "chunk_tokens": statistics.chunk,
                "arena_pages": list(statistics.arena_pages),
                "admission_tokens": statistics.admission,
                "order_rule": self.decisions.order_rule,
                "order_source": self.decisions.order_source,
                "search_seconds": estimate,
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
            estimator=estimator)

    def unique_id(self, prefix, name):
        key = f"{prefix}:{name}"
        self.seen_ids[key] = self.seen_ids.get(key, 0) + 1
        return key if self.seen_ids[key] == 1 else f"{key}:{self.seen_ids[key]}"

    def emit_applies(self, alias):
        for apply in self.alias_applies.get(alias, ()):
            aid = f"apply:{apply.function}"
            self.nodes.append(Foreign(
                node_id=aid,
                inputs=input_ports((self.ids_src[alias],)),
                function=apply.function, kind=apply.kind, ids=apply.ids,
                columns=tuple((ref.alias, ref.column)
                              for ref in apply.columns),
                aliases=(alias,)))
            self.ids_src[alias] = PortRef(aid, f"ids:{alias}")

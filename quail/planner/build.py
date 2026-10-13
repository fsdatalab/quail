"""Build the physical plan of a decided logical plan."""

from functools import partial

from quail.cost.sol import speed_of_light
from quail.cost.work import Work
from quail.logical import (
    PROBABILITIES_SUFFIX,
    SPAN_SUFFIX,
    Alias,
    LabelWork,
    LogicalPlan,
    classified_above_joins,
    effective_selectivity,
    extracted_above_joins,
    has_score,
    is_score,
    oriented_join_conditions,
)
from quail.physical import (
    AiClassify,
    AiExtract,
    AiFilter,
    AiJoin,
    AiScore,
    Barrier,
    Exchange,
    Filter,
    Foreign,
    HashJoin,
    InList,
    PortRef,
    Recombine,
)
from quail.physical import (
    Scan as PhysicalScan,
)
from quail.physical.base import input_ports
from quail.planner import retention
from quail.planner.decisions import choose_plan, validate_query
from quail.planner.estimates import node_estimates
from quail.planner.model_calls import (
    ScoreLowering,
    classify_builder,
    extract_spec,
    inference_node,
)
from quail.planner.physical_optimizer import PlanningContext
from quail.planner.plan import PhysicalPlan, Refusal, inference_seconds
from quail.planner.results import result_nodes
from quail.planner.shards import contiguous_shards
from quail.planner.statistics import (
    filter_stop_keys,
    pair_counts,
    question_tokens,
    summarize_alias,
)
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


def joined_calls(labels: LabelWork) -> list:
    """Return classifications of document pairs and their table aliases."""
    return [(call, call.aliases()) for call, _ in labels.calls
            if len(call.aliases()) == 2]


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
    order and anchors by cost (quail.planner.ordering), unless order
    is "as_written". A one-table classification above a SemanticJoin
    runs after the joins, over the documents the joins matched. One on
    its table runs after the table's AI.IF filters.

    The label_scoring physical rule picks each classification's scoring
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
                    self.emit_scores(s.alias)
                elif s.alias in self.ask_filters:
                    self.emit_filter(s.alias)
                else:
                    self.emit_applies(s.alias)
                self.emit_classify(s.alias)
                self.emit_extract(s.alias)
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
                    for spec, _ in group:
                        position = spec.written_pos
                        join = self.operators.joins[position]
                        pair_inputs = ()
                        if position in self.pairs_src:
                            producer = self.pairs_src[position]
                            pair_inputs = (PortRef(producer.node_id,
                                                  f"pairs:{position}"),)
                        if self.score_pairs is None:
                            self.score_pairs = self.emit_score_pair(
                                join.prompt, position, spec.pair_fraction,
                                pair_inputs)
                        self.score_pairs = self.scoring.comparison(
                            join, self.score_pairs, spec.aliases, position)
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
                pair_inputs = []
                partners = [a for a in spec.aliases if a != anchor]
                pairs_from = ""
                if spec.written_pos in self.pairs_src:
                    producer = self.pairs_src[spec.written_pos]
                    pair_inputs.append(
                        PortRef(producer.node_id, f"pairs:{spec.written_pos}"))
                    pairs_from = producer.node_id
                for apply in self.join_applies.get(spec.written_pos, ()):
                    # the function's pairs reach the join on their own port
                    aid = f"apply:{apply.function}"
                    self.nodes.append(Foreign(
                        node_id=aid,
                        inputs=input_ports(tuple(
                            self.ids_src[a] for a in apply.aliases)),
                        function=apply.function, kind=apply.kind, ids="pairs",
                        columns=tuple((ref.alias, ref.column)
                                      for ref in apply.columns),
                        aliases=tuple(apply.aliases),
                        written_pos=spec.written_pos))
                    pair_inputs.append(PortRef(aid, f"pairs:{spec.written_pos}"))
                    pairs_from = aid
                self.nodes.append(AiJoin(
                    node_id=jid,
                    inputs=input_ports(tuple(
                        self.ids_src[a] for a in [anchor, *partners])
                                       + tuple(pair_inputs)),
                    written_pos=spec.written_pos, exec_idx=exec_idx,
                    anchor=anchor, partners=tuple(partners),
                    semantics=spec.semantics,
                    selectivity=spec.selectivity,
                    expected_tuples=round(step.tuples, 1),
                    anchor_frame_tokens=spec.frame_tokens[anchor],
                    pair_tail_tokens=spec.tail_tokens,
                    anchor_resident=step.resident,
                    tuple_tokens=round(step.tokens, 1),
                    pairs_from=pairs_from))
                exec_idx += 1
                self.ids_src[anchor] = PortRef(jid, f"ids:{anchor}")
                if spec.semantics == "full":
                    self.pairs_edges.append(PortRef(
                        jid, f"join_answers:{spec.written_pos}"
                    ))
                    for a in [anchor] + partners:
                        if a not in self.out_aliases:
                            self.out_aliases.append(a)

    def _emit_join_answers(self):
        """Create classification and extraction nodes after joins."""
        stats = self.decisions.statistics.stats
        # a classification of joined rows follows the join that keeps them,
        # in the join's pipeline: each kept row's partner block, question,
        # and label paths run over the anchor's resident KV
        try:
            for call, aliases in joined_calls(self.operators.labels):
                for g, group in enumerate(self.sequence_groups):
                    anchor = group[0][1]
                    stage = next((spec for spec, _ in group
                                  if set(spec.aliases) == set(aliases)), None)
                    if stage is not None:
                        break
                else:
                    raise AssertionError(
                        "a classification of joined rows without its join")
                partner = next(alias for alias in aliases if alias != anchor)
                table = classify_builder(self.context, anchor, "quail")
                join = self.operators.joins[stage.written_pos]
                pairs = (effective_selectivity(join.selectivity)
                         * self.decisions.live[anchor] * self.decisions.live[partner])
                spec, _ = table.classify_joined(
                    call, self.operators.labels.names[call], partner, pairs,
                    stats[partner].mean_doc_tokens)
                node = table.node(
                    spec, PortRef(self.join_ids[stage.written_pos],
                                  f"join_answers:{stage.written_pos}"),
                    sum(isinstance(n, AiClassify) for n in self.nodes))
                self.nodes.append(node)
                if call in self.operators.labels.projected:
                    self.label_ports.append(PortRef(node.node_id, "rows"))
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
                    self.emit_classify(
                        alias, live=self.decisions.live[alias] * min(1.0, met))
                if alias in self.extract_after:
                    self.emit_extract(
                        alias, live=self.decisions.live[alias] * min(1.0, met))
        except ClassifyRefusedError as refused:
            return refused.refusal()

        try:
            for expression in self.projected_scores:
                call = expression.expression
                if len(call.aliases()) == 2 and call.prompt not in self.scoring.names:
                    self.score_pairs = self.emit_score_pair(call.prompt, 0, 1.0, ())
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
        self.nodes.extend(result_nodes(self.plan.result, result_inputs, tuple(columns)))

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

    def emit_filter(self, alias):
        stats = self.decisions.statistics.stats
        orders = self.decisions.filter_orders
        n = stats[alias].n_docs
        surv = 1.0
        for rank, i in enumerate(orders[alias]):
            p = self.operators.filters[alias][i]
            fid = f"ai_filter:{alias}" if rank == 0 else f"ai_filter:{alias}:{i}"
            self.nodes.append(AiFilter(
                node_id=fid, inputs=input_ports((self.ids_src[alias],)), alias=alias,
                written_pos=i,
                question_tokens=question_tokens(p.prompt, self.model.canvas_tokens),
                preamble_tokens=p.prompt.preamble_tokens,
                selectivity=p.selectivity, expected_docs=round(n * surv, 1),
                arena_writes=len(orders[alias]) > 1 or alias in self.pipelined,
                stop_key=self.stop_keys.get(alias, ())))
            surv *= effective_selectivity(p.selectivity)
            self.ids_src[alias] = PortRef(fid, f"ids:{alias}")
        self.emit_applies(alias)

    def emit_scores(self, alias):
        stats = self.decisions.statistics.stats
        orders = self.decisions.filter_orders
        live = float(stats[alias].n_docs)
        for position in orders.get(alias, ()):
            predicate = self.operators.filters[alias][position]
            source = self.rows_src.get(alias, self.ids_src[alias])
            if predicate.prompt not in self.scoring.names:
                source = self.scoring.score(
                    predicate.prompt, f"__score_{alias}_{position}", (source,),
                    live, stats[alias].mean_doc_tokens)
            source = self.scoring.comparison(predicate, source, (alias,), position)
            self.rows_src[alias] = source
            self.ids_src[alias] = PortRef(source.node_id, f"ids:{alias}")
            live *= effective_selectivity(predicate.selectivity)
        for expression in self.projected_scores:
            call = expression.expression
            if call.aliases() != (alias,) or call.prompt in self.scoring.names:
                continue
            self.rows_src[alias] = self.scoring.score(
                call.prompt, expression.name,
                (self.rows_src.get(alias, self.ids_src[alias]),), live,
                stats[alias].mean_doc_tokens)

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

    def emit_classify(self, alias, live=None):
        """Classify the alias's documents, then keep the accepted labels."""
        stats = self.decisions.statistics.stats
        calls = [call for call, owner in self.operators.labels.calls
                 if owner == alias and len(call.aliases()) == 1]
        if not calls or (alias in self.after_joins and live is None):
            return
        table = classify_builder(self.context, alias, "quail")
        if live is None:
            live = self.live_asked.get(alias, float(stats[alias].n_docs))
        steps = [(test.call, test)
                 for test in self.operators.label_filters.get(alias, ())]
        steps.extend((call, None) for call in calls
                     if call not in self.operators.labels.tests)
        classified_calls = set()
        scores = None
        for call, test in steps:
            if call not in classified_calls:
                spec, work = table.prepare(
                    call, self.operators.labels.names[call], live)
                if self.scoring is not None:
                    self.scoring.work += work
                node = table.node(
                    spec, scores or self.rows_src.get(alias, self.ids_src[alias]),
                                  sum(isinstance(n, AiClassify) for n in self.nodes))
                self.nodes.append(node)
                scores = PortRef(node.node_id, "rows")
                self.ids_src[alias] = PortRef(node.node_id, f"ids:{alias}")
                if call in self.operators.labels.projected:
                    self.label_ports.append(scores)
                classified_calls.add(call)
            if test is not None:
                lid = f"filter:{alias}:{test.position}"
                self.nodes.append(Filter(
                    node_id=lid, inputs=input_ports((scores,)),
                    predicate=InList(self.operators.labels.names[call], test.values),
                    aliases=(alias,), selectivity=test.selectivity,
                    written_pos=test.position))
                scores = PortRef(lid, "rows")
                self.ids_src[alias] = PortRef(lid, f"ids:{alias}")
                live *= effective_selectivity(test.selectivity)
        if scores is not None:
            self.rows_src[alias] = scores

    def emit_extract(self, alias, live=None):
        """Copy the answers of the alias's documents, after its labels."""
        stats = self.decisions.statistics.stats
        calls = [node for node in self.operators.extracts if node.alias == alias]
        if not calls or (alias in self.extract_after and live is None):
            return
        if live is None:
            live = self.live_asked.get(alias, float(stats[alias].n_docs))
            for test in self.operators.label_filters.get(alias, ()):
                live *= effective_selectivity(test.selectivity)
        for index, logical_node in enumerate(calls):
            # the extractions of a table run in one pipeline
            # (quail.physical.pipelines): a document is read once
            # for all of them
            shared = index > 0
            spec = extract_spec(self.context, logical_node.call,
                                logical_node.name, live, shared=shared)
            node = inference_node(
                AiExtract, spec, (self.rows_src.get(alias, self.ids_src[alias]),),
                index=sum(isinstance(n, AiExtract) for n in self.nodes),
                backend="quail", model=self.model.name)
            self.nodes.append(node)
            scores = PortRef(node.node_id, "rows")
            self.ids_src[alias] = PortRef(node.node_id, f"ids:{alias}")
            self.rows_src[alias] = scores
            self.label_ports.append(scores)

    def emit_score_pair(self, prompt, position, fraction, pair_inputs):
        stats = self.decisions.statistics.stats
        aliases = tuple(dict.fromkeys(ref.alias for ref in prompt.args))
        expected, prefixes = pair_counts(self.decisions.live, aliases, fraction)
        return self.scoring.score(
            prompt, f"__score_join_{position}",
            tuple(self.rows_src.get(a, self.ids_src[a]) for a in aliases) + pair_inputs,
            expected, sum(stats[a].mean_doc_tokens for a in aliases),
            pair_fraction=fraction, prefix_groups=prefixes)

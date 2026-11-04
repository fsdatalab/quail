"""Build executable query graphs with operator order and execution settings."""

from dataclasses import dataclass, field, replace
from itertools import groupby

from quail.common.expressions import aliased_call
from quail.logical import (
    PROBABILITIES_SUFFIX,
    SPAN_SUFFIX,
    LogicalPlan,
    column_name,
    columns_above_joins,
    condition_columns,
    effective_selectivity,
    has_score,
    is_score,
    oriented_join_conditions,
)
from quail.physical import (
    AiScore,
    Barrier,
    Exchange,
    Filter,
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
from quail.planner.operators.classify import emit_joined_classify
from quail.planner.operators.columns import column_planner
from quail.planner.operators.filter import emit_filter
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
    ASK,
    CLASSIFY,
    COLUMN_STEPS,
    resident_after,
)
from quail.planner.statistics.tables import summarize_alias
from quail.planner.validation import (
    ClassifyRefusedError,
    RefusedError,
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
            left=left, right=right, on=conditions, written_position=position,
            pair_fraction=pair_fractions.get(position, 1.0)))
    return nodes


def build_physical_plan(plan: LogicalPlan, *, model: ModelSpec,
                        device: DeviceSpec, document_tokens: dict, gpus: int = 1,
                        order: str | None = None, pair_fractions=None,
                        context: PlanningContext | None = None,
                        scan_fractions=None):
    """Compile a LogicalPlan into a PhysicalPlan or Refusal.

    The builder chooses each table's filter order and the joins' stage
    order and anchors by cost (quail.planner.search.operator_order), unless order
    is "as_written". A one-table classification above a SemanticJoin
    runs after the joins, over the documents the joins matched. One on
    its table runs after the table's AI.IF filters.

    The initial estimate includes filter and join work and any inference
    costs already known. The planner then runs the graph refinement rules to select
    classification implementations, KV retention, and prefix sharing before
    comparing this candidate with others.

    Args:
        plan: The logical plan to compile.
        model: Model spec.
        device: Device spec.
        document_tokens: alias -> list of per-document token counts.
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
        plan, operators, model, device, document_tokens, order,
        pair_fractions, context, scan_fractions)
    if isinstance(operator_order, Refusal):
        return operator_order
    return GraphBuilder(plan, model, device, document_tokens, gpus, context,
                        pair_fractions, operator_order).build()


class GraphBuilder:
    """Construct native physical nodes in the selected operator order."""

    def __init__(self, plan, model, device, document_tokens, gpus, context,
                 pair_fractions, operator_order):
        self.plan = plan
        self.model = model
        self.device = device
        self.document_tokens = document_tokens
        self.gpus = gpus
        self.context = context
        self.pair_fractions = pair_fractions
        self.operator_order = operator_order
        self.operators = operator_order.operators
        self.classified = {alias for _, alias in self.operators.labels.calls}
        # alias -> the step kinds the planner lifted above the joins
        self.after_joins = {}
        for alias, call_kind in columns_above_joins(plan.root):
            self.after_joins.setdefault(alias, set()).add(COLUMN_STEPS[call_kind])
        self.classified_after = {alias for alias, kinds in self.after_joins.items()
                                 if CLASSIFY in kinds}
        self.projected_scores = score_projections(plan)
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
        port_by_alias = self._emit_scans()
        try:
            port_by_alias, row_ports = self._emit_documents(port_by_alias)
            port_by_alias, joins = self._emit_joins(port_by_alias)
            row_ports += self._emit_join_answers(port_by_alias, joins)
            score_pairs = self._emit_pair_scores(port_by_alias, joins)
        except ClassifyRefusedError as refused:
            return refused.refusal()
        except RefusedError as refused:
            return refused.refusal
        self._emit_result(port_by_alias, row_ports, joins, score_pairs)
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

        # a sequence streams only into its alias's first use
        self.streamed = retention.streamed_aliases(
            self.sequence_groups, set(self.ask_filters), self.gpus)

        self.nodes = []
        # (alias, label or answer column) -> the last port that carries
        # it, for the Filters that read it beside another operator's
        self.computed_ports = {}
        # alias -> the label and answer columns its operators compute
        self.computed_columns = {}
        # (alias, name) -> each one-table column operator
        self.column_nodes = {}
        for node in (*self.operators.classifies, *self.operators.extracts,
                     *self.operators.sae_features, *self.operators.maps):
            if len(node.call.aliases()) == 1:
                self.computed_columns.setdefault(node.alias, set()).update(
                    node.added_columns)
                self.column_nodes[(node.alias, node.name)] = node
        # the label and answer tests emitted so far
        self.emitted_tests = ()
        self.scoring = (ScoreLowering(self.context, self.nodes,
                                     self.projected_scores, statistics.chunk)
                        if has_score(self.plan) else None)

    def _emit_scans(self) -> dict:
        """Create the scans and equality joins; return where each table starts."""
        stats = self.operator_order.statistics.stats
        port_by_alias = {}
        for s in self.operators.scans:
            summary = summarize_alias(
                self.document_tokens[s.alias], self.model.sliding_window,
                memo=self.context.memo if self.context is not None else None)
            shard_ranges, loads = contiguous_shards(
                self.document_tokens[s.alias], self.gpus, total=summary.total
            )
            scan_id = f"scan:{s.alias}"
            self.nodes.append(PhysicalScan(
                node_id=scan_id,
                alias=s.alias, input_id=s.alias,
                num_documents=len(self.document_tokens[s.alias]),
                total_tokens=summary.total,
                shard_ranges=shard_ranges,
                shard_token_loads=tuple(loads),
                predicates=tuple(s.predicates),
                expected_rows=(float(stats[s.alias].num_documents)
                               if s.predicates else None)))
            port_by_alias[s.alias] = PortRef(scan_id, f"rows:{s.alias}")
        # the hash join reads the scans; survivors thin its pairs at the AI join
        self.pairs_src = {}
        for node in hash_join_nodes(self.operators.joins, self.pair_fractions,
                                    list(port_by_alias.values())):
            self.nodes.append(node)
            self.pairs_src[node.written_position] = node

        # aliases whose filter sequence runs in one pipeline with the
        # classification or join after it
        between = self.classified - self.classified_after
        self.pipelined = self.streamed | {
            alias for alias in self.ask_filters
            if self.gpus == 1 and alias in between}
        return port_by_alias

    def _emit_documents(self, port_by_alias: dict) -> tuple:
        """Create the inference nodes over each table before its joins."""
        stats = self.operator_order.statistics.stats
        row_ports = []
        port_by_alias = dict(port_by_alias)
        for s in self.operators.scans:
            # a column operator the planner lifted above the joins is
            # emitted after them
            kinds = {ASK, *COLUMN_STEPS.values()} - self.after_joins.get(s.alias, set())
            port_by_alias[s.alias], rows = self._emit_steps(
                s.alias,
                float(stats[s.alias].num_documents),
                kinds,
                port_by_alias[s.alias],
            )
            row_ports += rows
        return port_by_alias, row_ports

    def _emit_steps(self, alias, expected_rows, kinds, port) -> tuple:
        """Emit the given kinds of a table's steps in run order.

        Consecutive steps of one kind are emitted together, each run
        over the documents the steps before it keep.

        Args:
            alias: The table.
            expected_rows: The documents the first step reads.
            kinds: The step kinds to emit: ASK, CLASSIFY, EXTRACT,
                FEATURES, MAP.
            port: The port the documents stand at before the steps.

        Returns:
            Where the documents stand after the steps, and the rows
            ports of the labels and answers SELECT returns.
        """
        steps = [step for step in self.operator_order.step_orders.get(alias, ())
                 if step.kind in kinds]
        row_ports = []
        # the scores SELECT returns follow the score predicates, before
        # the classifications and extractions read the rows
        scores_pending = ASK in kinds and self.scoring is not None
        previous = None
        for kind, run in groupby(steps, key=lambda step: step.kind):
            run = list(run)
            positions = [step.key for step in run]
            resident = previous is not None and resident_after(previous, run[0])
            if kind != ASK and scores_pending:
                port = emit_projected_scores(self, alias, expected_rows, port)
                scores_pending = False
            if kind == ASK and self.scoring is not None:
                port = emit_scores(self, alias, positions, expected_rows, port)
            elif kind == ASK:
                port = emit_filter(self, alias, positions, expected_rows, port,
                                       followed=run[-1] is not steps[-1])
            else:
                port, rows = self.emit_columns(
                    alias, expected_rows, port,
                    [self.column_nodes[(alias, name)] for name in positions],
                    resident=resident)
                row_ports += rows
            for step in run:
                expected_rows *= step.selectivity
            previous = run[-1]
        if scores_pending:
            port = emit_projected_scores(self, alias, expected_rows, port)
        return port, row_ports

    def _emit_joins(self, port_by_alias: dict) -> tuple:
        """Create the joins and the exchanges between anchor groups.

        Returns:
            Where each table's documents stand after the joins, and the
            JoinPorts record of what the joins left.
        """
        # group consecutive full stages on the same anchor; gates run
        # alone; anchor switches become barriers
        steps = iter(self.operator_order.join_steps)
        groups = [[(spec, next(steps)) for spec, _ in group]
                  for group in self.sequence_groups]
        port_by_alias = dict(port_by_alias)
        joins = JoinPorts()
        exec_index = 0
        for g, group in enumerate(groups):
            anchor = self.sequence_groups[g][0][1]
            if group[0][0].cost is not None:
                joins.score_pairs = emit_score_joins(
                    self, group, port_by_alias, joins.score_pairs)
                continue
            if g > 0:
                ahead = [scan.alias for scan in self.operators.scans]
                barrier_id = self.unique_id("barrier", anchor)
                self.nodes.append(Barrier(
                    node_id=barrier_id,
                    inputs=input_ports(tuple(joins.answers)
                                       + tuple(port_by_alias[a] for a in ahead)),
                    next_anchor=anchor,
                    aliases=tuple(ahead)))
                for a in ahead:
                    port_by_alias[a] = PortRef(barrier_id, f"rows:{a}")
            if self.gpus > 1:
                # anchors go to the GPU holding their KV, else balanced
                exchange_id = self.unique_id("exchange", anchor)
                self.nodes.append(
                    Exchange(
                        node_id=exchange_id,
                        inputs=input_ports((port_by_alias[anchor],)),
                        anchor=anchor,
                    )
                )
                port_by_alias[anchor] = PortRef(exchange_id, f"rows:{anchor}")
            group_id = self.group_ids[g]
            for rank, (spec, step) in enumerate(group):
                join_id = (group_id if rank == 0
                           else f"{group_id}:predicate:{spec.written_position}")
                joins.node_ids[spec.written_position] = join_id
                port_by_alias[anchor], answers = emit_join(
                    self, spec, step, anchor, join_id, exec_index, port_by_alias)
                if answers is not None:
                    joins.answers.append(answers)
                    for a in spec.aliases:
                        if a not in joins.aliases:
                            joins.aliases.append(a)
                exec_index += 1
        return port_by_alias, joins

    def _emit_join_answers(self, port_by_alias: dict, joins) -> list:
        """Create the column operators the planner lifted above the joins.

        Returns:
            The rows ports of the labels and answers SELECT returns.
        """
        row_ports = emit_joined_classify(self, joins.node_ids)
        # a column operator after the joins runs over the documents the
        # joins matched: a document with at least one true pair, expected as the
        # join selectivity times the partners it met, at most every document
        for alias, kinds in sorted(self.after_joins.items()):
            met = sum(
                effective_selectivity(join.selectivity)
                * (self.pair_fractions or {}).get(position, 1.0)
                * sum(self.operator_order.expected_rows[argument.alias]
                      for argument in join.prompt.args
                      if argument.alias != alias)
                for position, join in enumerate(self.operators.joins)
                if alias in {argument.alias for argument in join.prompt.args})
            port_by_alias[alias], rows = self._emit_steps(
                alias, self.operator_order.expected_rows[alias] * min(1.0, met), kinds,
                port_by_alias[alias])
            row_ports += rows
        return row_ports

    def _emit_pair_scores(self, port_by_alias: dict, joins) -> PortRef | None:
        """Create the pair scores SELECT returns; return their rows port."""
        score_pairs = joins.score_pairs
        for expression in self.projected_scores:
            call = expression.expression
            if len(call.aliases()) == 2 and call.prompt not in self.scoring.names:
                score_pairs = emit_score_pair(self, call.prompt, 0, 1.0, (),
                                              port_by_alias)
        return score_pairs

    def _emit_result(self, port_by_alias, row_ports, joins, score_pairs):
        """Connect result operators to surviving rows and answers."""
        filtered_after = {alias for alias, kinds in self.after_joins.items()
                          if kinds - {CLASSIFY}} & self.operators.column_filters.keys()
        if score_pairs is not None:
            sink_inputs = (score_pairs,)
        elif (len(joins.answers) == 1
              and len(self.operator_order.join_sequence) == 1
              and not (self.classified_after or filtered_after)):
            # A later classification or column filter can remove joined rows.
            sink_inputs = (joins.answers[0],)
        elif joins.answers:
            self.nodes.append(Recombine(
                node_id="recombine",
                inputs=input_ports(
                    tuple(joins.answers)
                    + tuple(port_by_alias[a] for a in joins.aliases)
                ),
                alias_order=tuple(joins.aliases)))
            sink_inputs = (PortRef("recombine", "tuples"),)
        else:
            alias = self.operators.scans[0].alias
            sink_inputs = (port_by_alias[alias],)
        columns = []
        for c in self.plan.projection.columns:
            columns.append(column_name(c))
            if getattr(aliased_call(c), "probabilities", False):
                columns.append(c.name + PROBABILITIES_SUFFIX)
            if aliased_call(c, "extract") is not None:
                columns.append(c.name + SPAN_SUFFIX)
        score_ports = tuple(PortRef(node.node_id, node.rows_port) for node in self.nodes
                            if isinstance(node, AiScore)
                            and any(node.spec.name == item.name
                                    for item in self.projected_scores))
        result_inputs = (*sink_inputs,
                         *(port for port in score_ports if port not in sink_inputs),
                         *row_ports)
        self.nodes.extend(result_nodes(self.plan.result, result_inputs, tuple(columns),
                                       self.plan.projection.packing()))

    def _finish(self):
        """Build the plan and bind its pricer."""
        statistics = self.operator_order.statistics
        # a score join's work is in its AiScore's estimate
        stage_works = {step.written_position: step.copy_work(self.gpus)
                       for step in self.operator_order.join_steps
                       if not is_score(
                           self.operators.joins[step.written_position].predicate)}
        pricer = PlanPricer(
            context=self.context, statistics=statistics, stage_works=stage_works,
            expected_rows=self.operator_order.expected_rows)
        plan = PhysicalPlan(
            model=self.model.name, device=self.device.name, gpus=self.gpus,
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
                    "data_parallel_copies": self.gpus,
                    "score_normalization": {
                        "reranker": "yes_no_softmax",
                        "decision": "decision_head_softmax",
                    }.get(self.model.role, "true_false_softmax"),
                } if self.scoring is not None else {}),
                **({"classify_placement": ("after joins" if self.classified_after
                                           else "before joins")}
                   if self.operators.labels.calls else {}),
                **({"canvas_draws": self.context.canvas_draws}
                   if self.operators.labels.calls and self.model.answer_canvas else {}),
            },
            estimator=pricer)
        seconds, estimates = pricer(plan.graph)
        return replace(plan, estimated_seconds=seconds, estimates=estimates)

    def emit_columns(self, alias, expected_rows, port, nodes, *,
                     resident=False) -> tuple:
        """Emit one table's column operators in run order, each followed by its tests.

        Each node comes from its kind's ColumnPlanner. The operators run
        in one pipeline (quail.physical.pipelines): a document is read
        once for all of them. A label or answer test runs right after
        the last operator whose columns it reads, in the planner's
        order, so it drops a document before the next operator reads it.

        Args:
            alias: The table.
            expected_rows: The documents the first operator reads.
            port: The port the documents stand at before the operators.
            nodes: The logical column operators, in run order.
            resident: Whether the step before the run left the
                documents' KV for the first operator to read.

        Returns:
            Where the documents stand after the operators and their
            tests, and the rows ports of the columns SELECT returns.
        """
        computed = self.computed_columns.get(alias, set())
        row_ports = []
        for index, logical_node in enumerate(nodes):
            planner = column_planner(logical_node)
            node = planner.node(self, logical_node, port, expected_rows,
                                resident or index > 0)
            self.nodes.append(node)
            port = PortRef(node.node_id, node.rows_port)
            if planner.projected(self.operators, logical_node):
                row_ports.append(port)
            span = getattr(node.spec, "span_name", None)
            names = (node.spec.name, *(() if span is None else (span,)))
            for name in names:
                self.computed_ports[(alias, name)] = port
            done = {name for owner, name in self.computed_ports if owner == alias}
            for test in self.operators.column_filters.get(alias, ()):
                read = set(condition_columns(test.condition)) & computed
                if test in self.emitted_tests or not read <= done:
                    continue
                self.emitted_tests += (test,)
                filter_id = f"filter:{alias}:{test.position}"
                self.nodes.append(Filter(
                    node_id=filter_id,
                    inputs=self.filter_inputs(alias, test.condition, port),
                    predicate=test.condition, aliases=(alias,),
                    selectivity=test.selectivity, written_position=test.position))
                port = PortRef(filter_id, f"rows:{alias}")
                expected_rows *= effective_selectivity(test.selectivity)
            # the port after the tests still carries the operator's columns
            for name in names:
                self.computed_ports[(alias, name)] = port
        return port, row_ports

    def filter_inputs(self, alias, condition, port) -> tuple:
        """Return a document Filter's input ports: its rows, then each other column's.

        A condition may read the labels or answers of operators before
        the one it follows; each comes from the last port that carries
        it, so the Filter stays in the pipeline.
        """
        others = []
        for name in condition_columns(condition):
            found = self.computed_ports.get((alias, name))
            if found is not None and found != port and found not in others:
                others.append(found)
        return input_ports((port, *others))

    def unique_id(self, prefix, name):
        key = f"{prefix}:{name}"
        self.seen_ids[key] = self.seen_ids.get(key, 0) + 1
        return key if self.seen_ids[key] == 1 else f"{key}:{self.seen_ids[key]}"


@dataclass
class JoinPorts:
    """What the AI joins leave for the operators after them.

    Attributes:
        node_ids: Join written position -> its AiJoin node id.
        answers: Every full stage's port of passing pairs.
        aliases: The recombination's output order.
        score_pairs: The rows of the pairs the score joins keep, or None.
    """

    node_ids: dict = field(default_factory=dict)
    answers: list = field(default_factory=list)
    aliases: list = field(default_factory=list)
    score_pairs: PortRef | None = None

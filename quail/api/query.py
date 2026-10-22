"""A compiled query: its plan, its explain text, and its run."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pyarrow as pa

from quail.api.execute import execute_query
from quail.catalog import MemoryTableProvider, inputs
from quail.catalog.pairs import columns_key, estimate_pair_fraction
from quail.catalog.selection import selected_rows
from quail.catalog.tokens import TokenLengths
from quail.execution.metrics import NodeMetrics, scalar_node_metrics
from quail.execution.projection import projection_for
from quail.execution.reporting import collect_answers, execution_report
from quail.execution.result import QueryResult
from quail.execution.runner import ExecutionContext, GenericRunner
from quail.execution.schema import cte_table, result_schema
from quail.execution.types import PhysicalRequest, document_input
from quail.explain import estimated_rows, explain
from quail.logical import (
    ColumnRef,
    LogicalPlan,
    WithPlan,
    column_name,
    cte_column_names,
    join_conditions,
    oriented_join_conditions,
)
from quail.logical.expressions import (
    column_references,
    evaluate_expression,
    source_columns,
)
from quail.physical import Scan, encode_graph
from quail.planner import plan_query, refine_plan
from quail.planner.optimizer.context import PlanningContext
from quail.planner.optimizer.tree_rewrites import logical_alternatives
from quail.planner.plan import PhysicalPlan, Refusal, RefusalError
from quail.planner.search.statistics import filter_stop_keys

if TYPE_CHECKING:
    from quail.api.session import Session


@runtime_checkable
class QueryLike(Protocol):
    """What ``Session.sql`` returns, whether the query runs here or remotely.

    ``Query`` runs in this process; ``quail.remote.client.RemoteQuery``
    sends the SQL to Quail Server. Both offer these methods, so code
    written against one works with the other.
    """

    def run(self, plan=None) -> QueryResult: ...

    def submit(self, **options): ...

    def execute_stream(self, batch_rows: int = 65_536,
                       limit: int | None = None) -> pa.RecordBatchReader: ...

    def collect(self, limit: int | None = None,
                batch_rows: int = 65_536) -> pa.Table: ...

    def explain(self, **options) -> str: ...

    def plan(self): ...


class Query:
    """A compiled query that plans and executes in the current process."""

    def __init__(self, session: "Session", logical: LogicalPlan,
                 order: str | None = None):
        self.session = session
        self.logical = logical
        self._logical_input = logical
        self.order = order
        self._plan = None
        self._searched_plan = None
        self._document_tokens = None
        self._token_inputs = None
        self._token_futures = {}
        self._token_finished_at = {}
        self._estimated = ()
        self._planning_started_at = None
        self._planning_finished_at = None
        self._replanning_s = 0.0
        self._scan_selectivities = {}
        self._pair_selectivities = {}
        self.token_wait_s = 0.0

    def token_inputs(self) -> dict:
        """Return the token store of every scanned alias.

        Plans first, then waits for any background tokenization, so
        planning and the speed of light estimate share these stores.
        """
        self.plan()
        self.wait_for_tokens()
        return self._token_inputs

    def plan(self):
        """Plan the query once and return its physical plan or refusal.

        Document lengths are exact for a tokenized column. For any other
        column they are estimated from a sample while a background thread
        tokenizes it. The tree rewrites and the physical planner both read
        these lengths.

        Returns:
            The physical plan, or a Refusal when no backend can run it.
        """
        if self._plan is None:
            self._planning_started_at = time.perf_counter()
            config = self.session.config
            operators = self.logical.operators()
            self._document_tokens = {}
            self._token_inputs = {}
            estimated = []
            for s in operators.scans:
                exact = self.session.inputs.token_lengths(s.provider, s.column)
                if exact is not None:
                    self._document_tokens[s.alias] = exact
                    continue
                self._document_tokens[s.alias] = self.session.inputs.estimate_lengths(
                    s.provider, s.column)
                estimated.append(s.alias)
            self._estimated = tuple(estimated)
            # the planner estimates what the regular predicates and the join
            # equalities keep from samples; the executor applies them
            scan_fractions = self._scan_fractions(operators.scans)
            pair_fractions = self._pair_fractions(
                operators.scans, operators.joins)
            self._scan_selectivities = scan_fractions
            self._pair_selectivities = pair_fractions
            context = PlanningContext(
                catalog=self.session.catalog, engine_config=config,
                model=self.session.model, device=self.session.device,
                gpus=config.gpus, document_tokens=self._document_tokens,
                backend=config.backend, order=self.order,
                canvas_draws=config.canvas_draws,
                attention=config.attention,
                tokenizer=self.session.tokenizer,
                pair_fractions=pair_fractions,
                scan_fractions=scan_fractions)
            candidates = logical_alternatives(
                self._logical_input,
                tuple(self.session.registry.logical_rules.values()),
                context)
            self.logical = candidates[0].plan
            scans = {}
            offset_aliases = set()
            for candidate in candidates:
                operators = candidate.plan.operators()
                offset_aliases.update(alias for extract in operators.extracts
                                      for alias in extract.call.aliases())
                for scan in operators.scans:
                    previous = scans.get(scan.alias, scan)
                    scans[scan.alias] = replace(scan, columns=tuple(dict.fromkeys(
                        (*previous.columns, *scan.columns))))
            backend = self.session.registry.backend(config.backend)
            # an extension backend may leave the capability flags unset
            reads_text = getattr(backend, "reads_document_text", False)
            locates_answers = getattr(backend, "locates_answers_in_text", False)
            for s in scans.values():
                columns = s.columns
                with_offsets = locates_answers and s.alias in offset_aliases
                if reads_text:
                    columns = tuple(dict.fromkeys(
                        (*columns, *source_columns(s.column))))
                if s.alias not in self._estimated:
                    self._token_inputs[s.alias] = self.session.inputs.tokenize(
                        s.provider, s.column, columns, with_offsets=with_offsets)
                    continue
                future = self.session.inputs.tokenize_async(
                    s.provider, s.column, columns, with_offsets=with_offsets)

                def record_finished(_future, alias=s.alias):
                    self._token_finished_at[alias] = time.perf_counter()

                future.add_done_callback(record_finished)
                self._token_futures[s.alias] = future
            self._plan = plan_query(
                self._logical_input, model=self.session.model,
                device=self.session.device,
                document_tokens=self._document_tokens,
                gpus=config.gpus,
                order=self.order,
                canvas_draws=config.canvas_draws,
                attention=config.attention,
                backend=config.backend,
                registry=self.session.registry,
                tokenizer=self.session.tokenizer,
                pair_fractions=pair_fractions,
                scan_fractions=scan_fractions,
                memo=context.memo, catalog=self.session.catalog,
                engine_config=config, logical_candidates=candidates)
            if not isinstance(self._plan, Refusal):
                self.logical = self._plan.logical_plan
            extra = {}
            if config.gpu_timing:
                extra["gpu_timing"] = True
            if extra and not isinstance(self._plan, Refusal):
                self._plan = replace(self._plan, settings={
                    **self._plan.settings, **extra})
            self._planning_finished_at = time.perf_counter()
            self._searched_plan = self._plan
        return self._plan

    @property
    def planning_s(self) -> float:
        """Return wall time spent planning this query."""
        if self._planning_started_at is None:
            return 0.0
        finished = self._planning_finished_at or time.perf_counter()
        return finished - self._planning_started_at + self._replanning_s

    @property
    def input_ready_s(self) -> float:
        """Return planning and tokenization time, excluding backend startup."""
        if self._planning_started_at is None:
            return 0.0
        finished = [
            self._planning_finished_at or time.perf_counter(),
            *self._token_finished_at.values(),
        ]
        return max(finished) - self._planning_started_at + self._replanning_s

    def _pair_fractions(self, scans, joins) -> dict:
        """Estimated pairs over the cross product, per join with conditions."""
        providers = {scan.alias: scan.provider for scan in scans}
        samples = {}

        def keys(expressions):
            out = []
            for expression in expressions:
                refs = column_references(expression)
                for ref in refs:
                    if (ref.alias, ref.column) not in samples:
                        samples[(ref.alias, ref.column)] = (
                            self.session.inputs.sample_columns(
                                providers[ref.alias], [ref.column])[ref.column])
                out.append(evaluate_expression(expression, pa.table({
                    ref.column: samples[(ref.alias, ref.column)] for ref in refs})))
            return out

        fractions = {}
        for position, join in enumerate(joins):
            oriented = oriented_join_conditions(join)
            if oriented is None:
                continue
            _, _, conditions = oriented
            fractions[position] = estimate_pair_fraction(
                keys([left for left, _ in conditions]),
                keys([right for _, right in conditions]))
        return fractions

    def _scan_fractions(self, scans) -> dict:
        """Estimated fraction of each tested scan's documents its tests keep."""
        fractions = {}
        for scan in scans:
            if not scan.predicates:
                continue
            columns = pa.table(self.session.inputs.sample_columns(
                scan.provider,
                [predicate.column.column for predicate in scan.predicates]))
            sampled = columns.num_rows
            kept = len(selected_rows(columns, scan.predicates))
            # a sample that keeps nothing, or everything, says little
            # about the rest, so the estimate stays off 0 and 1
            fractions[scan.alias] = (kept + 1) / (sampled + 2)
        return fractions

    def explain(self, *, verbose: bool = False,
                analyze: bool = False) -> str:
        """Return the optimized plan, optionally measured by running it.

        Args:
            verbose: Include runtime settings and internal node fields.
            analyze: Run the query first, like EXPLAIN ANALYZE, and show
                each node's measured rows, time, and tokens beside the
                planner's estimates, then the measured totals.
        """
        result = self.run() if analyze else None
        return self._explain(result, verbose)

    def _explain(self, result, verbose: bool, assumed=()) -> str:
        """Return the plan text, with the measurements of result when given.

        Args:
            result: The QueryResult of running the plan, or None.
            verbose: Include runtime settings and internal node fields.
            assumed: Aliases planned on assumed lengths, left out of the
                note on sampled token counts.
        """
        # plan() first: it replaces self.logical with the optimized tree
        physical = self.plan()
        if result is not None:
            physical = self._plan
            # the root relation is lazy, so its row count is taken here
            rows = result.count()
            root = physical.graph.root.node_id
            result.node_metrics[root] = replace(
                result.node_metrics.get(root, NodeMetrics()),
                output_rows=rows)
            result.report.setdefault("node_metrics", {}).setdefault(
                root, {})["output_rows"] = rows
        registry = self.session.registry
        text = explain(self.logical, physical, verbose=verbose,
                       result=result,
                       rules=tuple(registry.rules.values()),
                       usd_per_hour=(
                           self.session.device.usd_per_hour or None))
        sampled = [alias for alias in self._estimated if alias not in assumed]
        if sampled:
            text += ("\n\n  note: token counts for "
                     + ", ".join(repr(a) for a in sampled)
                     + f" are estimated from a {inputs.ESTIMATE_SAMPLE} document "
                     "sample")
        return text

    def wait_for_tokens(self) -> None:
        """Block until every background tokenization has finished.

        A plan made on estimated lengths is searched again with exact inputs,
        so prefix sharing can change which logical alternative wins.
        """
        started = time.perf_counter()
        refine = bool(self._token_futures)
        for alias, future in list(self._token_futures.items()):
            self._token_inputs[alias] = future.result()
            self._document_tokens[alias] = self._token_inputs[alias].lengths
            self._token_finished_at.setdefault(alias, time.perf_counter())
            del self._token_futures[alias]
        self.token_wait_s += time.perf_counter() - started
        if refine and self._plan is not None:
            started = time.perf_counter()
            options = dict(
                model=self.session.model,
                device=self.session.device, document_tokens=self._document_tokens,
                gpus=self.session.config.gpus, order=self.order,
                canvas_draws=self.session.config.canvas_draws,
                attention=self.session.config.attention,
                backend=self.session.config.backend,
                registry=self.session.registry,
                tokenizer=self.session.tokenizer,
                pair_fractions=self._pair_selectivities)
            if self._plan is self._searched_plan:
                self._plan = plan_query(
                    self._logical_input, scan_fractions=self._scan_selectivities,
                    catalog=self.session.catalog, engine_config=self.session.config,
                    **options)
                if not isinstance(self._plan, Refusal):
                    self.logical = self._plan.logical_plan
            else:
                # Explicit physical edits must survive exact tokenization.
                self._plan = refine_plan(self._plan, **options)
            if self.session.config.gpu_timing and not isinstance(self._plan, Refusal):
                self._plan = replace(self._plan, settings={
                    **self._plan.settings, "gpu_timing": True})
            self._replanning_s += time.perf_counter() - started
            self._estimated = ()

    def run(self, plan=None) -> QueryResult:
        """Execute the query in the current process.

        Args:
            plan: An edited PhysicalPlan from plan().insert(),
                remove(), or move(); the planner's own plan when
                omitted.
        """
        return execute_query(self, plan=plan)

    def submit(self, **_options):
        """Submitting needs Quail Server; local queries use run()."""
        raise RuntimeError(
            "submit() needs a Session with an endpoint; this session runs "
            "queries in the current process, so call run() or collect()")

    def execute_stream(self, batch_rows: int = 65_536,
                       limit: int | None = None) -> pa.RecordBatchReader:
        """Execute the query and stream Arrow record batches."""
        return self.run().execute_stream(
            batch_rows=batch_rows, limit=limit)

    def collect(self, limit: int | None = None,
                batch_rows: int = 65_536) -> pa.Table:
        """Execute the query and explicitly collect one Arrow table."""
        return self.run().collect(
            limit=limit, batch_rows=batch_rows)

    def _prepare_physical(self):
        """Bind the query's tokenized inputs to its physical plan."""
        self.plan()
        self.wait_for_tokens()
        plan = self.plan()
        if isinstance(plan, Refusal):
            raise RefusalError(plan)
        plan.graph.validate(runtime_keys=set(self.session.registry.runtimes))
        plan.graph.validate_backend(plan.backend)
        inputs = {}
        logical_scans = {
            scan.alias: scan for scan in self.logical.operators().scans
        }
        for node in plan.nodes:
            if not isinstance(node, Scan):
                continue
            store = self._token_inputs[node.alias]
            logical_scan = logical_scans[node.alias]
            inputs[node.input_id] = document_input(
                store.tokens,
                (
                    evaluate_expression(logical_scan.column, store)
                    if set(source_columns(logical_scan.column))
                    <= set(store.projected_columns)
                    else None
                ),
            )
        envelope = plan.to_envelope(self.session.registry.codecs)
        return PhysicalRequest(envelope, inputs, self._column_tables())

    def _column_tables(self) -> dict:
        """One value table per alias whose columns the executor reads.

        A scan's regular predicates, a HashJoin, an apply(), and a stop key
        read values; the tables carry the columns they name.
        """
        needed = {}
        operators = self.logical.operators()
        for join in operators.joins:
            for condition in join_conditions(join):
                for ref in column_references(condition.left) + column_references(
                        condition.right):
                    needed.setdefault(ref.alias, {})[ref.column] = None
        for apply in operators.applies:
            for expression in apply.columns:
                for ref in column_references(expression):
                    needed.setdefault(ref.alias, {})[ref.column] = None
        for scan in operators.scans:
            for predicate in scan.predicates:
                needed.setdefault(scan.alias, {})[predicate.column.column] = None
        for alias, key in filter_stop_keys(self.logical).items():
            for column in key:
                needed.setdefault(alias, {})[column] = None
        tables = {}
        for alias, columns in needed.items():
            store = self._token_inputs[alias]
            tables[columns_key(alias)] = pa.table(
                {name: store.column(name) for name in columns})
        return tables

    def finish(self, response, coordinator_wall: float = 0.0) -> QueryResult:
        """Finish the physical graph and attach execution details."""
        plan = self.plan()
        out = response.metrics
        report = execution_report(
            plan, out, self.session.inputs.notes, coordinator_wall)
        sources = {
            node.input_id: range(node.num_documents)
            for node in plan.nodes
            if isinstance(node, Scan)
        }
        observers = self.session.registry.new_observers()
        run = GenericRunner().run(
            plan.graph,
            ExecutionContext(
                runtimes=self.session.registry.runtimes,
                sources=sources,
                project=projection_for(
                    self.logical, self._token_inputs, self._document_tokens),
                observers=observers,
            ),
            initial_outputs=response.outputs,
            initial_metrics={
                node_id: NodeMetrics.from_scalars(metrics)
                for node_id, metrics in out.get("node_metrics", {}).items()
            },
        )
        if not isinstance(run.value, QueryResult):
            raise TypeError("physical graph root must return QueryResult")
        result = run.value
        result.plan = plan.graph
        result.node_metrics = {
            node_id: node_result.metrics
            for node_id, node_result in run.nodes.items()
        }
        # Saved reports need the graph and metrics without Python objects.
        report["executed_plan"] = encode_graph(
            plan.graph, self.session.registry.codecs)
        report["node_metrics"] = scalar_node_metrics(run.nodes)
        observer_reports = {
            observer.name: dict(observer.report())
            for observer in observers
        }
        if observer_reports:
            report["observers"] = observer_reports
        result.report = report

        (result.answer_tables, result.survivor_indices,
         result.true_join_tables, report["stages"]) = collect_answers(
             self.logical, plan, response.outputs, self._document_tokens)
        return result


# tokens assumed per row for a CTE column the model computes, such as an
# extraction answer or a label, when the CTE has not run yet
ASSUMED_ANSWER_TOKENS = 32

# report fields added up over the CTEs and the final SELECT
SUMMED_REPORT_FIELDS = (
    "wall_s", "model_wall_s", "coordinator_wall_s", "fresh_tokens",
    "cached_tokens", "estimated_seconds", "planning_s", "input_ready_s",
    "physical_prepare_s", "finish_s", "token_wait_s", "worker_total_s")


@dataclass(frozen=True)
class WithPhysicalPlan:
    """The physical plan of each CTE the query needs, then of its SELECT.

    Attributes:
        ctes: (name, PhysicalPlan) pairs in execution order.
        query: The final SELECT's plan.
    """

    ctes: tuple
    query: PhysicalPlan

    @property
    def plans(self) -> tuple:
        return (*(plan for _, plan in self.ctes), self.query)

    @property
    def estimated_seconds(self) -> float:
        return sum(plan.estimated_seconds for plan in self.plans)

    @property
    def backend(self) -> str:
        return self.query.backend

    @property
    def gpus(self) -> int:
        return max(plan.gpus for plan in self.plans)

    def to_envelope(self, codecs) -> dict:
        """Return every stage's envelope under the CTE names and ``query``."""
        return {"ctes": {name: plan.to_envelope(codecs) for name, plan in self.ctes},
                "query": self.query.to_envelope(codecs)}


class WithQuery:
    """A query with CTEs: each CTE runs before the queries that read it.

    A CTE's result is kept as an Arrow table in the session's catalog
    view, its text columns are tokenized like a registered table's,
    and the queries reading it are planned on it. Before a CTE has
    run, the queries reading it are planned on its estimated row count
    and an assumed token count per row, so ``explain()`` never runs
    the model. ``run()`` executes the CTEs the final SELECT needs, in
    written order, then the SELECT.
    """

    def __init__(self, session: "Session", logical: WithPlan,
                 order: str | None = None):
        self.session = session
        self.logical = logical
        self.order = order
        self.ctes = {cte.name: cte for cte in logical.needed()}
        self._view = session.with_ctes(self.ctes)
        self._schemas = {}
        for cte in logical.ctes:
            schema = result_schema(cte.plan, self._view.catalog)
            self._view.catalog.register(cte.name, MemoryTableProvider(
                schema.empty_table(), id_col="", identity=f"cte:{cte.name}"))
            self._schemas[cte.name] = schema
        self._queries = {}     # stage name -> its Query; "" is the SELECT
        self._results = {}     # stage name -> its QueryResult, once run
        for name in self.ctes:
            self._view.inputs.assumed_lengths[name] = (
                lambda column, name=name: self._assumed_lengths(name, column))

    def _stages(self) -> tuple:
        """Return (name, LogicalPlan) of every stage in execution order."""
        return (*((name, cte.plan) for name, cte in self.ctes.items()),
                ("", self.logical.query))

    def _stage(self, name: str, fresh: bool = False) -> Query:
        """Return the Query running one stage over the tables it has now."""
        if fresh or name not in self._queries:
            plan = self.logical.query if name == "" else self.ctes[name].plan
            self._queries[name] = Query(self._view, plan, order=self.order)
        return self._queries[name]

    def _assumption(self, name: str, column) -> tuple[int, int]:
        """Return the rows and tokens per row assumed for a CTE's column.

        The rows are the planner's estimate for the CTE, at least one.
        A column the CTE passes through from a registered table keeps
        that table's mean estimated token count; a column the model
        computes is assumed to hold ASSUMED_ANSWER_TOKENS.
        """
        physical = self._stage(name).plan()
        rows = None if isinstance(physical, Refusal) else estimated_rows(physical)
        if rows is None:
            rows = sum(len(lengths) for lengths in self._stage(name)._document_tokens
                       .values())
        cte = self.ctes[name]
        names = dict(zip(cte_column_names(cte.plan), cte.plan.root.result_columns()))
        projected = {column_name(c): c for c in cte.plan.projection.columns}
        source = projected.get(names.get(column_references(column)[0].column))
        tokens = ASSUMED_ANSWER_TOKENS
        if isinstance(source, ColumnRef):
            lengths = self._view.inputs.estimate_lengths(source.provider, source)
            tokens = round(sum(lengths) / len(lengths)) if len(lengths) else 0
        return max(1, round(rows)), tokens

    def _assumed_lengths(self, name: str, column) -> TokenLengths:
        rows, tokens = self._assumption(name, column)
        return TokenLengths([tokens] * rows)

    def plan(self):
        """Plan every stage and return the plans, or the first Refusal.

        A stage reading a CTE that has not run is planned on the
        assumptions ``explain()`` names.
        """
        plans = []
        for name, _ in self._stages():
            physical = self._stage(name).plan()
            if isinstance(physical, Refusal):
                return physical
            plans.append((name, physical))
        return WithPhysicalPlan(tuple(plans[:-1]), plans[-1][1])

    def explain(self, *, verbose: bool = False, analyze: bool = False) -> str:
        """Return each stage's plan, the CTEs first, then the final SELECT.

        Args:
            verbose: Include runtime settings and internal node fields.
            analyze: Run the query first and show each stage's
                measurements beside its estimates.
        """
        results = {}
        if analyze:
            self.run()
            results = self._results
        parts = []
        for name, plan in self._stages():
            reads = self.logical.reads if name == "" else self.ctes[name].reads
            title = "query" if name == "" else f"with {name}"
            if reads:
                title += " (reads " + ", ".join(reads) + ")"
            pending = {scan.alias: scan for scan in plan.operators().scans
                       if scan.provider in self.ctes
                       and scan.provider not in self._results}
            query = self._stage(name)
            lines = [title + ":",
                     query._explain(results.get(name), verbose, assumed=pending)]
            for alias, scan in pending.items():
                rows, tokens = self._assumption(scan.provider, scan.column)
                lines.append(
                    f"\n  note: {alias!r} reads the CTE {scan.provider!r}, "
                    f"which has not run; planned on {rows:,} rows of "
                    f"{tokens:,} tokens each")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def execute(self, physical_executor=None, on_backend_ready=None,
                on_execution_ready=None) -> QueryResult:
        """Run the CTEs the SELECT needs in written order, then the SELECT.

        Each CTE's result replaces its placeholder in the catalog view,
        so the next stage is planned on the rows it returned. The final
        result's report adds every stage's times and tokens up and keeps
        each CTE's own report under ``ctes``.
        """
        reports = {}
        first = True
        for name, _ in self._stages():
            query = self._stage(name, fresh=True)
            result = execute_query(
                query, physical_executor=physical_executor,
                on_backend_ready=on_backend_ready if first else None,
                on_execution_ready=on_execution_ready if first else None)
            first = False
            self._results[name] = result
            if name == "":
                break
            reports[name] = result.report
            table = cte_table(result.collect(), self._schemas[name])
            self._view.catalog.providers[name] = MemoryTableProvider(
                table, id_col="", identity=f"cte:{name}")
            self._view.inputs.assumed_lengths.pop(name, None)
        result.report = _summed_report(result.report, reports)
        return result

    def run(self, plan=None) -> QueryResult:
        """Execute the query in the current process.

        Args:
            plan: Must be None; a WITH query runs the planner's plans.
        """
        if plan is not None:
            raise RuntimeError(
                "a WITH query runs the planner's plan for each stage; "
                "edited plans are not supported")
        return execute_query(self)

    def submit(self, **_options):
        """Submitting needs Quail Server; local queries use run()."""
        raise RuntimeError(
            "submit() needs a Session with an endpoint; this session runs "
            "queries in the current process, so call run() or collect()")

    def execute_stream(self, batch_rows: int = 65_536,
                       limit: int | None = None) -> pa.RecordBatchReader:
        """Execute the query and stream Arrow record batches."""
        return self.run().execute_stream(batch_rows=batch_rows, limit=limit)

    def collect(self, limit: int | None = None,
                batch_rows: int = 65_536) -> pa.Table:
        """Execute the query and collect one Arrow table."""
        return self.run().collect(limit=limit, batch_rows=batch_rows)


def _summed_report(report: dict, cte_reports: dict) -> dict:
    """Return the SELECT's report with every stage's totals added up."""
    summed = dict(report)
    for key in SUMMED_REPORT_FIELDS:
        values = [stage.get(key) for stage in (*cte_reports.values(), report)]
        values = [value for value in values if value is not None]
        if values:
            summed[key] = (round(sum(values), 4) if isinstance(values[0], float)
                           else sum(values))
    summed["ctes"] = cte_reports
    return summed

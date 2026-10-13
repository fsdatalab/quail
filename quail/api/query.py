"""A compiled query: its plan, its explain text, and its run."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pyarrow as pa

from quail.api.execute import execute_query
from quail.catalog import inputs
from quail.catalog.pairs import columns_key, estimate_pair_fraction
from quail.execution.projection import projection_for
from quail.execution.reporting import (
    collect_answers,
    execution_report,
    restore_node_metrics,
)
from quail.execution.result import QueryResult
from quail.execution.runner import (
    ExecutionContext,
    GenericRunner,
    NodeMetrics,
    scalar_node_metrics,
)
from quail.execution.selection import selected_rows
from quail.execution.types import PhysicalRequest, document_input
from quail.explain import explain
from quail.logical import (
    LogicalPlan,
    join_conditions,
    oriented_join_conditions,
)
from quail.physical import Scan, encode_graph
from quail.planner import plan_query, refine_plan
from quail.planner.logical_optimizer import LogicalPlanningContext, apply_logical_rules
from quail.planner.plan import Refusal, RefusalError
from quail.planner.statistics import filter_stop_keys

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
        self.order = order
        self._plan = None
        self._doc_tokens = None
        self._token_inputs = None
        self._token_futures = {}
        self._token_finished_at = {}
        self._estimated = ()
        self._planning_started_at = None
        self._planning_finished_at = None
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
        tokenizes it. The logical rules and the physical planner both read
        these lengths.

        Returns:
            The physical plan, or a Refusal when no backend can run it.
        """
        if self._plan is None:
            self._planning_started_at = time.perf_counter()
            config = self.session.config
            operators = self.logical.operators()
            self._doc_tokens = {}
            self._token_inputs = {}
            estimated = []
            for s in operators.scans:
                exact = self.session.inputs.token_lengths(s.provider, s.column)
                if exact is not None:
                    self._doc_tokens[s.alias] = exact
                    continue
                self._doc_tokens[s.alias] = self.session.inputs.estimate_lengths(
                    s.provider, s.column)
                estimated.append(s.alias)
            self._estimated = tuple(estimated)
            # the planner estimates what the regular predicates and the join
            # equalities keep from samples; the executor applies them
            scan_fractions = self._scan_fractions(operators.scans)
            pair_fractions = self._pair_fractions(
                operators.scans, operators.joins)
            context = LogicalPlanningContext(
                self.session.catalog, config,
                model=self.session.model, device=self.session.device,
                gpu_count=config.gpus, document_tokens=self._doc_tokens,
                backend=config.backend, order=self.order,
                canvas_draws=config.canvas_draws,
                attention=config.attention,
                tokenizer=self.session.tokenizer,
                pair_fractions=pair_fractions,
                scan_fractions=scan_fractions)
            self.logical, changed = apply_logical_rules(
                self.logical,
                tuple(self.session.registry.logical_rules.values()),
                context)
            rewritten = self.logical.operators()
            backend = self.session.registry.backend(config.backend)
            # an extension backend may leave the capability flags unset
            reads_text = getattr(backend, "reads_document_text", False)
            locates_answers = getattr(backend, "locates_answers_in_text", False)
            offset_aliases = {alias for extract in rewritten.extracts
                              for alias in extract.call.aliases()}
            for s in rewritten.scans:
                columns = s.columns
                with_offsets = locates_answers and s.alias in offset_aliases
                if reads_text:
                    columns = tuple(dict.fromkeys((*columns, s.column)))
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
                self.logical, model=self.session.model,
                device=self.session.device,
                doc_tokens=self._doc_tokens,
                gpus=config.gpus,
                order=self.order,
                canvas_draws=config.canvas_draws,
                attention=config.attention,
                backend=config.backend,
                registry=self.session.registry,
                tokenizer=self.session.tokenizer,
                pair_fractions=pair_fractions,
                scan_fractions=scan_fractions,
                memo=context.memo)
            extra = {}
            if config.gpu_timing:
                extra["gpu_timing"] = True
            if extra and not isinstance(self._plan, Refusal):
                self._plan = replace(self._plan, settings={
                    **self._plan.settings, **extra})
            self._planning_finished_at = time.perf_counter()
        return self._plan

    @property
    def planning_s(self) -> float:
        """Return wall time spent planning this query."""
        if self._planning_started_at is None:
            return 0.0
        finished = self._planning_finished_at or time.perf_counter()
        return finished - self._planning_started_at

    @property
    def input_ready_s(self) -> float:
        """Return time until planning and document tokenization finished."""
        if self._planning_started_at is None:
            return 0.0
        finished = [
            self._planning_finished_at or time.perf_counter(),
            *self._token_finished_at.values(),
        ]
        return max(finished) - self._planning_started_at

    def _pair_fractions(self, scans, joins) -> dict:
        """Estimated pairs over the cross product, per join with conditions."""
        providers = {scan.alias: scan.provider for scan in scans}
        samples = {}

        def keys(refs):
            out = []
            for ref in refs:
                key = (ref.alias, ref.column)
                if key not in samples:
                    samples[key] = self.session.inputs.sample_columns(
                        providers[ref.alias], [ref.column])[ref.column]
                out.append(samples[key])
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
            columns = self.session.inputs.sample_columns(
                scan.provider,
                [predicate.column.column for predicate in scan.predicates])
            sampled = len(next(iter(columns.values())))
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
        # plan() first: it replaces self.logical with the optimized tree
        physical = self.plan()
        result = None
        if analyze:
            result = self.run()
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
                       rules=(tuple(registry.logical_rules.values()),
                              tuple(registry.physical_rules.values())),
                       usd_per_hour=(
                           self.session.device.usd_per_hour or None))
        if self._estimated:
            text += ("\n\n  note: token counts for "
                     + ", ".join(repr(a) for a in self._estimated)
                     + f" are estimated from a {inputs.ESTIMATE_SAMPLE} document "
                     "sample")
        return text

    def wait_for_tokens(self) -> None:
        """Block until every background tokenization has finished.

        A plan made on estimated lengths then gets the physical rules
        that read the token stores applied with the exact inputs.
        """
        started = time.perf_counter()
        refine = bool(self._token_futures)
        for alias, future in list(self._token_futures.items()):
            self._token_inputs[alias] = future.result()
            self._doc_tokens[alias] = self._token_inputs[alias].lengths
            self._token_finished_at.setdefault(alias, time.perf_counter())
            del self._token_futures[alias]
        self.token_wait_s += time.perf_counter() - started
        if refine and self._plan is not None:
            self._plan = refine_plan(
                self._plan, model=self.session.model,
                device=self.session.device, doc_tokens=self._doc_tokens,
                gpus=self.session.config.gpus, order=self.order,
                canvas_draws=self.session.config.canvas_draws,
                attention=self.session.config.attention,
                backend=self.session.config.backend,
                registry=self.session.registry,
                tokenizer=self.session.tokenizer)

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
        plan = self.plan()
        if isinstance(plan, Refusal):
            raise RefusalError(plan)
        self.wait_for_tokens()
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
                    store.column(logical_scan.column)
                    if logical_scan.column in store.projected_columns
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
                for ref in (condition.left, condition.right):
                    needed.setdefault(ref.alias, {})[ref.column] = None
        for apply in operators.applies:
            for ref in apply.columns:
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
            arrays = {alias: pa.array(range(len(store.lengths)), pa.int32())}
            for name in columns:
                arrays[name] = store.column(name)
            tables[columns_key(alias)] = pa.table(arrays)
        return tables

    def finish(self, response, coordinator_wall: float = 0.0) -> QueryResult:
        """Finish the physical graph and attach execution details."""
        plan = self.plan()
        out = response.metrics
        report = execution_report(
            plan, out, self.session.inputs.notes, coordinator_wall)
        sources = {
            node.input_id: range(node.n_docs)
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
                    self.logical, self._token_inputs, self._doc_tokens),
                observers=observers,
            ),
            initial_outputs=response.outputs,
            initial_metrics={
                node_id: restore_node_metrics(metrics)
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
             self.logical, plan, response.outputs, self._doc_tokens)
        return result

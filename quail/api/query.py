"""A compiled query: its plan, its explain text, and its run."""

from __future__ import annotations

import json
import time
from contextlib import closing
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pyarrow as pa
from pyarrow import compute as pc

from quail.api.execute import execute_query
from quail.api.region_inputs import input_scans
from quail.catalog.providers import ScanRequest
from quail.common.expressions import (
    CompileError,
    column_references,
    evaluate_expression,
    source_columns,
)
from quail.common.pairs import FIRST_DOCUMENT, columns_key
from quail.common.tokens import ColumnStore, ScanInput, TokenLengths, TokenStore
from quail.execution.metrics import NodeMetrics, scalar_node_metrics
from quail.execution.projection import projection_for
from quail.execution.reporting import collect_answers, execution_report
from quail.execution.result import QueryResult
from quail.execution.runner import ExecutionContext, GenericRunner
from quail.execution.types import PhysicalRequest, document_input
from quail.explain import explain
from quail.logical import (
    LogicalPlan,
    WithCTE,
    filter_stop_keys,
    join_conditions,
)
from quail.physical import Scan, encode_graph
from quail.physical.codec import decode_value, encode_value
from quail.physical.plan import Refusal, RefusalError, refusal_from_fields
from quail.planner import native, plan_query
from quail.planner.candidates import LogicalAlternative
from quail.planner.context import PlanningContext

# the start of Rust's message for a document that is an expression

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
                 order: str | None = None, *, catalog=None, inputs=None,
                 text: str | None = None, dialect: str = "snowflake",
                 whole: bool = True):
        self.session = session
        # the SQL the plan was compiled from, which DataFusion runs; the
        # plan is its one model region, the whole query or not
        self.text = text
        self.dialect = dialect
        self.whole = whole
        self.catalog = session.catalog if catalog is None else catalog
        self.inputs = session.inputs if inputs is None else inputs
        self.logical = logical
        self._logical_input = logical
        self.order = order
        self._plan = None
        self._searched_plan = None
        self._document_tokens = None
        self._token_inputs = None
        self._planning_started_at = None
        self._planning_finished_at = None
        # each filtered input's rows' positions in its registered table
        self.source_rows = {}
        # each input that holds a registered table whole: its identity
        self.input_identities = {}

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
        if self._plan is not None:
            return self._plan
        if isinstance(self.logical.root, WithCTE):
            from quail.api.stages import plan_stages

            self._plan = plan_stages(self)
            return self._plan
        if not self.logical.operators().prompts:
            (scan,) = self.logical.operators().scans
            num_documents = self.catalog.get(scan.provider).statistics().num_rows
            if self.inputs.pending(scan.provider):
                num_documents = len(self.inputs.estimate_lengths(
                    scan.provider, scan.column))
            self._plan = self._relational_plan(num_documents or 0)
            return self._plan
        self._planning_started_at = time.perf_counter()
        self._plan = self._plan_in_rust()
        self._planning_finished_at = time.perf_counter()
        self._searched_plan = self._plan
        return self._plan

    def _plan_in_rust(self):
        """Tokenize, measure, and plan the region in Rust.

        Rust writes each scan's token store and value columns; the query
        reads them as its inputs.
        """
        tables = {}
        for scan in self._logical_input.operators().scans:
            if scan.provider in tables:
                continue
            provider = self.catalog.get(scan.provider)
            with closing(provider.scan(ScanRequest(
                    columns=tuple(provider.columns)))) as reader:
                table = reader.read_all()
            # a registered table's rows are the same for every query
            identity = (provider.content_identity()
                        if scan.provider in self.session.catalog
                        else self.input_identities.get(scan.provider))
            tables[scan.provider] = (identity, table)
        config = self.session.config
        settings = {
            "model": native.untagged(encode_value(self.session.model)),
            "device": native.untagged(encode_value(self.session.device)),
            "gpus": config.gpus, "canvas_draws": config.canvas_draws,
            "attention": config.attention, "order": self.order,
            "backend": config.backend,
            "physical_rules": [rule.name for rule in
                               self.session.registry.physical_rules.values()],
        }
        planner = self.session.planner
        call = planner.plan if config.backend == "quail" else planner.prepare
        found, message = call(json.dumps(encode_value(self._logical_input)), tables,
                              json.dumps(settings))
        if message is not None:
            raise CompileError(message)
        if config.backend == "quail":
            return self.adopt_plan(json.loads(found))
        return self._lower_prepared(json.loads(found))

    def _lower_prepared(self, found: dict):
        """Lower a region Rust prepared with the session's backend.

        Args:
            found: What ``NativePlanner.prepare`` returns, read from JSON.

        Returns:
            The physical plan, or a Refusal.
        """
        self._adopt_scans(found)
        config = self.session.config
        alternatives = tuple(
            LogicalAlternative(decode_value(LogicalPlan, json.loads(text)),
                               tuple(rules))
            for text, rules in found["alternatives"])
        pair_fractions = {int(position): fraction for position, fraction
                          in found["fields"]["pair_fractions"].items()}
        plan = plan_query(
            self._logical_input, model=self.session.model,
            device=self.session.device, document_tokens=self._document_tokens,
            gpus=config.gpus, order=self.order, canvas_draws=config.canvas_draws,
            attention=config.attention, backend=config.backend,
            registry=self.session.registry, tokenizer=self.session.tokenizer,
            pair_fractions=pair_fractions, catalog=self.catalog,
            engine_config=config, logical_candidates=alternatives)
        if not isinstance(plan, Refusal):
            self.logical = plan.logical_plan
            if config.gpu_timing:
                plan = replace(plan, settings={**plan.settings, "gpu_timing": True})
        self._plan = self._searched_plan = plan
        return plan

    def _adopt_scans(self, found: dict) -> None:
        """Read the token store and column files Rust wrote as the inputs."""
        self._token_inputs, self._document_tokens = {}, {}
        for alias, files in found["scans"].items():
            store = TokenStore(files["tokens"])
            self._token_inputs[alias] = ScanInput(store, {
                name: ColumnStore(path) for name, path in files["columns"]})
            self._document_tokens[alias] = store.lengths

    def adopt_plan(self, found: dict):
        """Take the plan Rust made, and read its scans' files as the inputs.

        Args:
            found: A planned or refused region, as ``NativePlanner.plan``
                returns it and read from JSON.

        Returns:
            The physical plan, or the Refusal.
        """
        self._adopt_scans(found)
        if "refusal" in found:
            refusal = refusal_from_fields(found["refusal"])
            self._plan = self._searched_plan = refusal
            return refusal
        plan, self.logical = native.planned_region(found, self.session.tokenizer)
        if self.session.config.gpu_timing:
            plan = replace(plan, settings={**plan.settings, "gpu_timing": True})
        if self._planning_started_at is None:
            self._planning_started_at = self._planning_finished_at = time.perf_counter()
        self._plan = self._searched_plan = plan
        return plan

    @property
    def planning_s(self) -> float:
        """Return wall time spent planning this query."""
        if self._planning_started_at is None:
            return 0.0
        finished = self._planning_finished_at or time.perf_counter()
        return finished - self._planning_started_at

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

    def _explain(self, result, verbose: bool) -> str:
        """Return the plan text, with the measurements of result when given.

        Args:
            result: The QueryResult of running the plan, or None.
            verbose: Include runtime settings and internal node fields.
        """
        # plan() first: it replaces self.logical with the optimized tree
        physical = self.plan()
        if result is not None:
            physical = self._plan
            # the root relation is lazy, so its row count is taken here
            rows = (result.region_rows if result.region_rows is not None
                    else result.count())
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
        has_inputs = input_scans(self._logical_input, self.session.catalog)
        if self.text is not None and (not self.whole or has_inputs):
            from quail.api.context import SessionContext

            around = (result.datafusion_plan if result is not None
                      and result.datafusion_plan is not None
                      else SessionContext(self.session).explain(
                          self.text, self.dialect))
            text = f"DataFusion:\n{around}\n\nmodel region:\n{text}"
        return text

    def wait_for_tokens(self) -> None:
        """Read a relational stage's value columns.

        A model region's token stores are written when it is planned.
        """
        if not self.logical.operators().prompts:
            self._prepare_relational()

    def _prepare_relational(self):
        """Read a relational stage's value columns without tokenizing them."""
        if self._token_inputs is not None:
            return
        (scan,) = self.logical.operators().scans
        provider = self.catalog.get(scan.provider)
        expressions = (*self.logical.projection.columns,
                       *(key.expression for key in self.logical.result.order),
                       *(test.expression for predicate in scan.predicates
                         for test in predicate.tests()))
        columns = tuple(dict.fromkeys(
            column for expression in expressions
            for column in source_columns(expression)))
        with closing(provider.scan(ScanRequest(columns=columns))) as reader:
            table = reader.read_all()
        self._token_inputs = {scan.alias: table}
        self._document_tokens = {scan.alias: TokenLengths([0] * table.num_rows)}
        self._plan = self._relational_plan(table.num_rows)

    def _relational_plan(self, num_documents: int):
        """Plan a stage that asks no model."""
        config = self.session.config
        context = PlanningContext(
            model=self.session.model, device=self.session.device, gpus=config.gpus,
            document_tokens={}, backend=config.backend)
        return native.relational_plan(self.logical, context, num_documents)

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
        source_rows = {scan.alias: self.source_rows[scan.provider]
                       for scan in self.logical.operators().scans
                       if scan.provider in self.source_rows}
        return PhysicalRequest(envelope, inputs, self._column_tables(),
                               source_rows)

    def _column_tables(self) -> dict:
        """One value table per alias whose columns the executor reads.

        A scan's regular predicates, a HashJoin, an apply(), and a stop key
        read values; the tables carry the columns they name. A table the
        plan deduplicates also carries FIRST_DOCUMENT, computed from the
        tokens and the row columns the deduplicate setting lists for it.
        """
        needed = {}
        operators = self.logical.operators()
        for join in operators.joins:
            for condition in join_conditions(join):
                for ref in column_references(condition.left) + column_references(
                        condition.right):
                    needed.setdefault(ref.alias, {})[ref.column] = None
        for scan in operators.scans:
            for predicate in scan.predicates:
                for test in predicate.tests():
                    for reference in column_references(test.expression):
                        needed.setdefault(scan.alias, {})[reference.column] = None
        # a Filter on labels or answers reads the source columns it tests
        # by document ID
        providers = {scan.alias: scan.provider for scan in operators.scans}
        for alias, filters in operators.column_filters.items():
            columns = self.catalog.get(providers[alias]).columns
            for reference in (reference for each in filters
                              for found in each.condition.tests()
                              for reference in column_references(found.expression)):
                if reference.alias == alias and reference.column in columns:
                    needed.setdefault(alias, {})[reference.column] = None
        for alias, key in filter_stop_keys(self.logical).items():
            for column in key:
                needed.setdefault(alias, {})[column] = None
        deduplicated = self.plan().settings.get("deduplicate", {})
        for alias in deduplicated:
            needed.setdefault(alias, {})
        tables = {}
        for alias, columns in needed.items():
            store = self._token_inputs[alias]
            values = {name: store.column(name) for name in columns}
            if alias in deduplicated:
                values[FIRST_DOCUMENT] = store.tokens.first_documents_with(
                    [store.column(column) for column in deduplicated[alias]])
            tables[columns_key(alias)] = pa.table(values)
        return tables

    def finish(self, response, coordinator_wall: float = 0.0) -> QueryResult:
        """Finish the physical graph and attach execution details."""
        plan = self.plan()
        out = response.metrics
        report = execution_report(
            plan, out, self.inputs.notes, coordinator_wall)
        sources = {
            node.input_id: range(node.num_documents)
            for node in plan.nodes
            if isinstance(node, Scan)
        }
        sources.update(self._column_tables())
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
        self._to_source_rows(result)
        return result

    def _to_source_rows(self, result) -> None:
        """Number each document by its row in the registered table.

        A filtered input holds some of a table's rows; answer tables and
        survivors name its documents by their row in the table, as they
        do for a table read whole.
        """
        rows = {scan.alias: self.source_rows[scan.provider]
                for scan in self.logical.operators().scans
                if scan.provider in self.source_rows}
        if not rows:
            return

        def renumbered(table):
            for alias, mapping in rows.items():
                if alias in table.column_names:
                    index = table.column_names.index(alias)
                    table = table.set_column(index, alias, pc.take(
                        mapping, table.column(alias)))
            return table

        result.answer_tables = {
            kind: {name: renumbered(table) for name, table in tables.items()}
            for kind, tables in result.answer_tables.items()}
        result.survivor_indices = {
            alias: (pc.take(rows[alias], survivors) if alias in rows else survivors)
            for alias, survivors in result.survivor_indices.items()}
        result.true_join_tables = {
            name: renumbered(table) for name, table in result.true_join_tables.items()}

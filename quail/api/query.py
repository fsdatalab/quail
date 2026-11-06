"""A compiled query: its plan, its explain text, and its run."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import _quail_native
import pyarrow as pa

from quail.api.compile import READ_AS_SQL
from quail.api.execute import execute_query
from quail.api.region_inputs import input_scans
from quail.common.expressions import (
    CompileError,
    evaluate_expression,
    source_columns,
)
from quail.common.pairs import FIRST_DOCUMENT, columns_key
from quail.common.tokens import ColumnStore, ScanInput, TokenStore
from quail.execution.metrics import NodeMetrics, scalar_node_metrics
from quail.execution.projection import projection_for
from quail.execution.result import QueryResult
from quail.execution.rows import as_table
from quail.execution.runner import ExecutionContext, GenericRunner
from quail.execution.types import PhysicalRequest, document_input
from quail.explain import explain
from quail.physical import Scan, encode_graph
from quail.physical.codec import decode_value, encode_value, untagged
from quail.physical.plan import Refusal, RefusalError, plan_from_json

# the start of Rust's message for a document that is an expression

if TYPE_CHECKING:
    from quail.api.session import Session


def backend_refusal(session) -> Refusal | None:
    """Return why the session's backend cannot run its configuration, or None."""
    config = session.config
    if config.canvas_draws < 1:
        return Refusal(
            reasons=(f"canvas_draws must be at least 1, got {config.canvas_draws}",),
            constraint="canvas_draws", needed=1, available=config.canvas_draws,
            unit="draws")
    try:
        backend = session.registry.backend(config.backend)
    except ValueError as error:
        return Refusal(reasons=(str(error),), constraint="unknown_backend",
                       needed=1, available=0, unit="backends")
    support = backend.supports(session.model, session.device, config.gpus)
    if not support.supported:
        return Refusal(
            reasons=(support.reason or "unsupported backend configuration",),
            constraint="unsupported_backend_configuration", needed=1,
            available=0, unit="configurations")
    return None


def planning_settings(session, order: str | None) -> dict:
    """The configuration Rust plans a region for.

    Args:
        session: The session whose model, device, and backend plan it.
        order: The stage order rule, 'by_cost' or 'as_written', or None.
    """
    config = session.config
    backend = session.registry.backend(config.backend)
    return {
        "model": untagged(encode_value(session.model)),
        "device": untagged(encode_value(session.device)),
        "gpus": config.gpus, "canvas_draws": config.canvas_draws,
        "attention": config.attention, "order": order,
        "backend": config.backend,
        "physical_rules": [rule.name for rule in
                           session.registry.physical_rules.values()],
        "request": getattr(backend, "request_settings", None),
    }


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
    """A SQL query that DataFusion plans and runs, with Quail running its model regions.

    Rust's session parses and binds the SQL, plans each model region in
    Rust, and runs every operator outside the regions; the Python
    executor runs each planned region. A query of one model region also
    offers that region's ``RegionQuery`` attributes, such as ``logical``.

    Args:
        session: The session whose tables and model the query uses.
        text: The SQL statement.
        dialect: "snowflake" or "bq".
        order: The stage order rule each region is planned with.
    """

    def __init__(self, session: "Session", text: str, dialect: str = "snowflake",
                 order: str | None = None):
        self.session = session
        self.text = text
        self.dialect = dialect
        self.order = order
        self._regions = None
        self._whole = None
        self._bound = None

    def _native(self, run_query=None, physical_executor=None):
        from quail.api.context import SessionContext

        return SessionContext(self.session, physical_executor, order=self.order,
                              run_query=run_query)._native

    def _dialect(self) -> str:
        from quail.api.compile import DATAFUSION_DIALECTS

        return DATAFUSION_DIALECTS[self.dialect]

    def bound_whole(self) -> bool:
        """Bind the SQL and return whether one model region is the whole query.

        Raises:
            CompileError: The SQL does not bind.
        """
        if self._whole is None:
            native = self._native()
            model = self.session.model
            try:
                bound, self._whole = native.logical_plans(
                    self.text, tuple(model.turn), model.prompt_layout, self._dialect())
            except RuntimeError as error:
                from quail.api.compile import compile_message

                raise CompileError(compile_message(error, self.text)) from None
            for _, message in bound:
                if message is not None and READ_AS_SQL not in message:
                    raise CompileError(message)
            self._bound = bound
        return self._whole

    @property
    def whole(self) -> bool:
        """Whether the query is one model region, with no SQL around it."""
        return self.bound_whole()

    @property
    def num_regions(self) -> int:
        """The model regions DataFusion hands Quail, before any is planned."""
        self.bound_whole()
        return len(self._bound)

    @property
    def logical(self) -> str:
        """The one model region's logical plan as bound, before planning."""
        self.bound_whole()
        if len(self._bound) != 1:
            raise AttributeError(
                f"the query has {len(self._bound)} model regions; use regions()")
        ((plan, message),) = self._bound
        if message is not None:
            return self._region().logical
        return plan

    def regions(self) -> list["RegionQuery"]:
        """Plan each model region once and return them, in plan order.

        Raises:
            CompileError: The SQL does not bind.
        """
        if self._regions is None:
            from quail.api.region_inputs import region_query

            self.bound_whole()
            try:
                found = self._native().plans(self.text, self._dialect())
            except RuntimeError as error:
                from quail.api.compile import compile_message

                raise CompileError(compile_message(error, self.text)) from None
            regions = []
            for kind, payload, planned in found:
                if kind == "sql":
                    inner = self.session.sql(payload, order=self.order)
                    regions.extend(inner.regions())
                    continue
                region = region_query(self.session, payload, order=self.order)
                if planned is not None:
                    region.adopt_plan(json.loads(planned))
                regions.append(region)
            self._regions = regions
        return self._regions

    def _region(self) -> "RegionQuery":
        regions = self.regions()
        if len(regions) != 1:
            raise AttributeError(
                f"the query has {len(regions)} model regions; use regions()")
        return regions[0]

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._region(), name)

    def plan(self):
        """Return the physical plan, or Refusal, of the query's model region.

        A query with several model regions returns each one's, in a tuple.
        """
        plans = tuple(region.plan() for region in self.regions())
        return plans[0] if len(plans) == 1 else plans

    def explain(self, *, verbose: bool = False, analyze: bool = False) -> str:
        """Return DataFusion's plan around the regions, then each region's plan.

        Args:
            verbose: Include runtime settings and internal node fields.
            analyze: Run the query first, like EXPLAIN ANALYZE, and show
                each node's measured rows, time, and tokens beside the
                planner's estimates.
        """
        regions = self.regions()
        result = self.run() if analyze else None
        if result is not None and len(regions) == 1:
            texts = [regions[0]._explain(result, verbose)]
        else:
            texts = [region._explain(None, verbose) for region in regions]
        if len(regions) == 1 and self._whole and not regions[0].has_inputs():
            return texts[0]
        around = (result.datafusion_plan if result is not None
                  and result.datafusion_plan is not None
                  else self._native().explain(self.text, self._dialect()))
        if len(texts) == 1:
            return f"DataFusion:\n{around}\n\nmodel region:\n{texts[0]}"
        return "\n\n".join([around, *(f"model region {index + 1}:\n{text}"
                                       for index, text in enumerate(texts))])

    def run(self, plan=None, *, physical_executor=None) -> QueryResult:
        """Execute the query in the current process.

        Args:
            plan: An edited PhysicalPlan of the query's one model region,
                from plan().insert(), remove(), or move(); the planner's
                own plan when omitted.
            physical_executor: Optional callable(PhysicalRequest) that
                returns the PhysicalResponse, for tests without a GPU.
        """
        if physical_executor is None:
            return execute_query(self, plan=plan)
        return execute_query(self, physical_executor=physical_executor, plan=plan)

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


class RegionQuery:
    """One model region of a query: its plan, and its run in this process."""

    def __init__(self, session: "Session", logical: str,
                 order: str | None = None, *, catalog=None, assumed_rows=None):
        self.session = session
        self.catalog = session.catalog if catalog is None else catalog
        # rows of each input table not yet computed, as a CTE's before it
        # runs, by table name
        self.assumed_rows = {} if assumed_rows is None else dict(assumed_rows)
        # the region's logical plan, as Rust's JSON: as bound, then as the
        # planner chose it
        self.logical = logical
        self._logical_input = logical
        self._facts = {}
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

    def facts(self, logical: str | None = None) -> dict:
        """Return what Rust reads from the logical plan, once per plan.

        Args:
            logical: A logical plan as JSON; the query's chosen plan when
                omitted.

        Returns:
            The plan's scans, value columns, stop keys, join equalities,
            and whether it asks the model; see ``region_facts`` in Rust.
        """
        logical = self.logical if logical is None else logical
        if logical not in self._facts:
            self._facts[logical] = json.loads(_quail_native.region_facts(logical))
        return self._facts[logical]

    def token_inputs(self) -> dict:
        """Return the token store of every scanned alias.

        Plans first, then waits for any background tokenization, so
        planning and the speed of light estimate share these stores.
        """
        self.plan()
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
        # Rust plans every region a backend accepts as the query runs
        refusal = backend_refusal(self.session)
        if refusal is None:
            raise RuntimeError("a model region is planned by Rust's session")
        self._plan = self._searched_plan = refusal
        return refusal

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
        plan = plan_from_json(found, self.session.registry.codecs)
        if isinstance(plan, Refusal):
            self._plan = self._searched_plan = plan
            return plan
        self.logical = plan.logical_plan
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
        return text

    def has_inputs(self) -> bool:
        """Whether the region reads a table the session did not register."""
        return bool(input_scans(self.facts(self._logical_input), self.session.catalog))

    def run(self, plan=None) -> QueryResult:
        """Execute the query in the current process.

        Args:
            plan: An edited PhysicalPlan from plan().insert(),
                remove(), or move(); the planner's own plan when
                omitted.
        """
        return execute_query(self, plan=plan)

    def _prepare_physical(self):
        """Bind the query's tokenized inputs to its physical plan."""
        self.plan()
        plan = self.plan()
        if isinstance(plan, Refusal):
            raise RefusalError(plan)
        plan.graph.validate(runtime_keys=set(self.session.registry.runtimes))
        plan.graph.validate_backend(plan.backend)
        inputs = {}
        documents = {scan["alias"]: decode_value(Any, scan["column"])
                     for scan in self.facts()["scans"]}
        for node in plan.nodes:
            if not isinstance(node, Scan):
                continue
            store = self._token_inputs[node.alias]
            column = documents[node.alias]
            inputs[node.input_id] = document_input(
                store.tokens,
                (
                    evaluate_expression(column, store)
                    if set(source_columns(column))
                    <= set(store.projected_columns)
                    else None
                ),
            )
        envelope = plan.to_envelope(self.session.registry.codecs)
        source_rows = self._alias_source_rows()
        return PhysicalRequest(envelope, inputs, self._column_tables(),
                               source_rows)

    def _alias_source_rows(self) -> dict:
        """Each filtered input's rows in its registered table, by scan alias."""
        return {scan["alias"]: self.source_rows[scan["provider"]]
                for scan in self.facts()["scans"]
                if scan["provider"] in self.source_rows}

    def _column_tables(self) -> dict:
        """One value table per alias whose columns the executor reads.

        A scan's regular predicates, a HashJoin, an apply(), and a stop key
        read values; the tables carry the columns they name. A table the
        plan deduplicates also carries FIRST_DOCUMENT, computed from the
        tokens and the row columns the deduplicate setting lists for it.
        """
        needed = {alias: dict.fromkeys(columns)
                  for alias, columns in self.facts()["value_columns"].items()}
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
        report = json.loads(_quail_native.execution_report(
            plan.backend, plan.estimated_seconds, plan.settings.get("order_rule"),
            json.dumps(out, default=str), coordinator_wall))
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
                project=projection_for(self._token_inputs, self._document_tokens),
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

        self._collect_answers(result, plan, response.outputs)
        return result

    def _collect_answers(self, result, plan, outputs) -> None:
        """Attach the answers, survivors, and stages Rust reads from outputs.

        Each model operator's answers, the surviving documents, and each
        predicate's measured selectivity; documents are numbered by their
        row in the registered table.
        """
        ports = [
            {"node_type": node.type_name, "node": node.node_id,
             "port": output.name, "value_type": output.value_type.value,
             "answer": None if output.answer is None else {
                 "kind": output.answer.kind, "name": output.answer.name,
                 "columns": list(output.answer.columns)}}
            for node in plan.graph.topological_nodes() for output in node.outputs]
        source_rows = {alias: rows.to_pylist()
                       for alias, rows in self._alias_source_rows().items()}
        request = {
            "logical": self.logical, "ports": ports,
            "num_documents": {alias: len(tokens)
                              for alias, tokens in self._document_tokens.items()},
            "source_rows": source_rows}
        tables = {(ref.node_id, ref.port): as_table(value)
                  for ref, value in outputs.items()
                  if isinstance(as_table(value), pa.Table)}
        answers, survivors, true_joins, stages = _quail_native.collect_answers(
            json.dumps(request), tables)
        result.answer_tables = {kind: {} for kind in (
            "filters", "joins", "classifies", "scores", "extracts", "features",
            "maps")}
        for kind, key, table in answers:
            key = tuple(key) if isinstance(key, list) else key
            result.answer_tables[kind][key] = table
        result.survivor_indices = {alias: table.column(0).combine_chunks()
                                   for alias, table in survivors}
        result.true_join_tables = dict(true_joins)
        result.report["stages"] = json.loads(stages)

"""A compiled query: its plan, its explain text, and its run."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import _quail_native
import pyarrow as pa

from quail.api.execute import execute_query
from quail.api.region_inputs import input_scans
from quail.common.expressions import CompileError, SimilarCallWarning
from quail.common.pairs import FIRST_DOCUMENT, columns_key
from quail.common.tokens import ScanInput, TokenStore
from quail.execution.finished import finished_table
from quail.execution.metrics import NodeMetrics, NodeResult, scalar_node_metrics
from quail.execution.result import QueryResult
from quail.execution.rows import as_table
from quail.execution.runner import compute_subgraph
from quail.execution.types import PhysicalRequest, document_input
from quail.explain import explain
from quail.physical import Limit, Scan, decode_plan_value
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
        "model": asdict(session.model),
        "device": asdict(session.device),
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
    offers that region's ``RegionQuery`` attributes, such as ``facts()``.

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
                bound, self._whole = native.region_plans(
                    self.text, tuple(model.turn), model.prompt_layout, self._dialect())
            except RuntimeError as error:
                from quail.api.compile import compile_message

                raise CompileError(compile_message(error, self.text)) from None
            for region, message, unsupported in bound:
                # DataFusion runs the operators of a shape Quail does not read
                if message is not None and not unsupported:
                    raise CompileError(message)
                for warning in json.loads(region).get("warnings", []) if region else []:
                    warnings.warn(warning, SimilarCallWarning, stacklevel=3)
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

    def _bound_region(self) -> dict:
        """The one model region as derived, before planning."""
        self.bound_whole()
        if len(self._bound) != 1:
            raise AttributeError(
                f"the query has {len(self._bound)} model regions; use regions()")
        ((found, message, _),) = self._bound
        if message is not None:
            region = self._region()
            return {"operators": region.operators, "logical": region.logical}
        return json.loads(found)

    @property
    def logical(self) -> str:
        """DataFusion's plan of the one model region, before planning."""
        return self._bound_region()["logical"]

    @property
    def operators(self) -> str:
        """The one model region's operators as JSON, before planning."""
        return self._bound_region()["operators"]

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
            for payload, planned in found:
                found = json.loads(payload)
                region = region_query(self.session, found["operators"],
                                      logical=found["logical"], order=self.order,
                                      region_id=found["region_id"])
                if planned is not None:
                    text, scans = planned
                    region.adopt_plan(json.loads(text), scans)
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
                planner's estimates. Regions SQL did not read are labeled
                not executed and show estimates only.
        """
        regions = self.regions()
        result = self.run() if analyze else None
        if result is not None:
            unexpected = (set(result.region_results)
                          - {region.region_id for region in regions})
            if unexpected:
                raise RuntimeError(
                    f"executed model regions do not match the query: {unexpected}")
        texts = []
        for region in regions:
            executed = (result.region_results.get(region.region_id)
                        if result is not None else None)
            text = region._explain(executed, verbose)
            if result is not None and executed is None:
                # LIMIT and other SQL operators need not read every model region.
                text = "not executed (estimated plan):\n" + text
            texts.append(text)
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

    def __init__(self, session: "Session", operators: str,
                 order: str | None = None, *, logical: str = "", catalog=None,
                 region_id: int | None = None):
        self.session = session
        self.region_id = region_id
        self.catalog = session.catalog if catalog is None else catalog
        # the region's operators, as Rust's JSON: as derived, then as the
        # planner chose them
        self.operators = operators
        self._operators_input = operators
        # DataFusion's plan of the region, one operator per line
        self.logical = logical
        self._facts = {}
        self.order = order
        self._plan = None
        self._document_tokens = None
        self._token_inputs = None
        # seconds Rust took to tokenize the inputs and plan the region
        self._planning_s = 0.0
        # each filtered input's rows' positions in its registered table
        self.source_rows = {}

    def facts(self, operators: str | None = None) -> dict:
        """Return what Rust reads from the region's operators, once per value.

        Args:
            operators: Operators as JSON; the query's chosen operators
                when omitted.

        Returns:
            The operators' scans, value columns, stop keys, join
            equalities, and whether they ask the model; see
            ``region_facts`` in Rust.
        """
        operators = self.operators if operators is None else operators
        if operators not in self._facts:
            self._facts[operators] = json.loads(_quail_native.region_facts(operators))
        return self._facts[operators]

    def token_inputs(self) -> dict:
        """Return the token store of every scanned alias, as Rust planned with it."""
        self.plan()
        return self._token_inputs

    def plan(self):
        """Return the physical plan Rust made for the region, or its refusal.

        Returns:
            The physical plan, or a Refusal when no backend can run it.
        """
        if self._plan is not None:
            return self._plan
        # Rust plans every region a backend accepts as the query runs
        refusal = backend_refusal(self.session)
        if refusal is None:
            raise RuntimeError("a model region is planned by Rust's session")
        self._plan = refusal
        return refusal

    def adopt_plan(self, found: dict, scans: dict):
        """Take the plan Rust made, and its scans' tokens and columns as the inputs.

        Args:
            found: A planned or refused region, read from the JSON Rust
                planned it as.
            scans: By alias, (the token store, the value columns) of each
                scan, as pyarrow Tables over Rust's buffers.

        Returns:
            The physical plan, or the Refusal.
        """
        self._planning_s = found.pop("planning_s", 0.0)
        self._token_inputs, self._document_tokens = {}, {}
        for alias, (tokens, columns) in scans.items():
            store = TokenStore(tokens)
            self._token_inputs[alias] = ScanInput(store, columns)
            self._document_tokens[alias] = store.lengths
        plan = plan_from_json(found)
        if isinstance(plan, Refusal):
            self._plan = plan
            return plan
        self.operators = plan.operators
        if self.session.config.gpu_timing:
            plan = replace(plan, settings={**plan.settings, "gpu_timing": True})
        self._plan = plan
        return plan

    @property
    def planning_s(self) -> float:
        """Return the seconds Rust took to tokenize the inputs and plan the region."""
        return self._planning_s

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
        # A downstream region may have been inspected with empty placeholder
        # inputs. ANALYZE must use the plan chosen for its actual input rows.
        physical = (result.physical_plan if result is not None
                    and result.physical_plan is not None else self.plan())
        if result is not None:
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
        operators = physical.operators if not isinstance(physical, Refusal) else None
        text = explain(self.logical, operators or self.operators, physical,
                       verbose=verbose, result=result,
                       rules=tuple(registry.rules.values()),
                       usd_per_hour=(
                           self.session.device.usd_per_hour or None))
        return text

    def has_inputs(self) -> bool:
        """Whether the region reads a table the session did not register."""
        return bool(input_scans(self.facts(self._operators_input),
                                self.session.catalog))

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
        documents = {scan["alias"]: decode_plan_value(json.dumps(scan["column"]))
                     for scan in self.facts()["scans"]}
        for node in plan.nodes:
            if not isinstance(node, Scan):
                continue
            store = self._token_inputs[node.alias]
            column = documents[node.alias]
            inputs[node.input_id] = document_input(
                store.tokens,
                (
                    column.evaluate(store)
                    if {ref.column for ref in column.references}
                    <= set(store.projected_columns)
                    else None
                ),
            )
        envelope = plan.to_envelope()
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

    def finished_rows(self, on_rows=None):
        """Return the FinishedRows that builds the region's rows from its run.

        Args:
            on_rows: Optional callable(pyarrow Table) handed each batch
                of rows as it is built, each row once.
        """
        request = self._answer_request(self.plan(), returned=True)
        return _quail_native.FinishedRows(json.dumps(request), self._source_values,
                                          on_rows)

    def finish(self, response, coordinator_wall: float = 0.0,
               finished=None) -> QueryResult:
        """Build the region's rows from the run's outputs, with its report.

        Args:
            response: The run's PhysicalResponse.
            coordinator_wall: Seconds the run took, as the caller saw it.
            finished: The FinishedRows the run reported to, or None for
                a new one. Every returned output is reported to it, so it
                builds each row the run's operators did not report.
        """
        plan = self.plan()
        out = response.metrics
        report = json.loads(_quail_native.execution_report(
            plan.backend, plan.estimated_seconds, plan.settings.get("order_rule"),
            json.dumps(out, default=str), coordinator_wall))
        tables = _output_tables(response.outputs)
        if finished is None:
            finished = self.finished_rows()
        for node in plan.graph.topological_nodes():
            for output in node.outputs:
                table = tables.get((node.node_id, output.name))
                if table is not None and finished.wants(node.node_id, output.name):
                    finished.report(finished_table(node.node_id, output.name, table))
        finished.finish()
        rows = finished.rows()
        result = QueryResult.from_table(rows, report=report)
        result.plan = plan.graph
        result.physical_plan = plan
        result.region_id = self.region_id
        report["region_id"] = self.region_id
        measured = dict(out.get("node_metrics", {}))
        for node in plan.nodes:
            if isinstance(node, Limit):
                # Rust keeps the region's first rows as it builds them
                measured[node.node_id] = scalar_node_metrics({node.node_id: NodeResult(
                    {}, NodeMetrics(output_rows=rows.num_rows))})[node.node_id]
        result.node_metrics = {
            node_id: NodeMetrics.from_scalars(metrics)
            for node_id, metrics in measured.items()}
        # Saved reports need the graph and metrics without Python objects.
        report["executed_plan"] = json.loads(plan.graph.to_json())
        report["node_metrics"] = {
            node_id: dict(metrics) for node_id, metrics in measured.items()}
        observers = self.session.registry.new_observers()
        _observe(plan.graph, response.outputs, result.node_metrics, observers)
        observer_reports = {
            observer.name: dict(observer.report())
            for observer in observers
        }
        if observer_reports:
            report["observers"] = observer_reports
        self._collect_answers(result, plan, tables)
        return result

    def _source_values(self, name: str, expression: str) -> pa.Table:
        """Return a source column's values for each document of its table.

        Args:
            name: The column's name in the region's rows.
            expression: The column's source expression, as Rust's JSON.
        """
        source = decode_plan_value(expression)
        alias = source.references[0].alias
        store = self._token_inputs[alias]
        loaded = (store.column_names if isinstance(store, pa.Table)
                  else store.projected_columns)
        for ref in source.references:
            if ref.column not in loaded:
                raise CompileError(
                    f"the column {name!r} was not loaded by the scan of {alias!r}")
        return pa.table({name: source.evaluate(store)})

    def _answer_request(self, plan, returned: bool = False) -> dict:
        """Return what Rust reads besides the output relations.

        The operators, every output port, and each scan's documents.

        Args:
            plan: The physical plan.
            returned: List only the ports of the nodes the run returns:
                the model nodes and the inputs they read.
        """
        nodes = plan.graph.topological_nodes()
        if returned:
            try:
                kept = {node.node_id for node in compute_subgraph(plan.graph).nodes}
            except ValueError:
                kept = set()
            nodes = [node for node in nodes if node.node_id in kept]
        ports = [
            {"node_type": node.type_name, "node": node.node_id,
             "port": output.name, "value_type": output.value_type.value,
             "answer": None if output.answer is None else {
                 "kind": output.answer.kind, "name": output.answer.name,
                 "columns": list(output.answer.columns)}}
            for node in nodes for output in node.outputs]
        source_rows = {alias: rows.to_pylist()
                       for alias, rows in self._alias_source_rows().items()}
        return {
            "operators": self.operators, "ports": ports,
            "num_documents": {alias: len(tokens)
                              for alias, tokens in self._document_tokens.items()},
            "source_rows": source_rows}

    def _collect_answers(self, result, plan, tables) -> None:
        """Attach the answers, survivors, and stages Rust reads from outputs.

        Each model operator's answers, the surviving documents, and each
        predicate's measured selectivity; documents are numbered by their
        row in the registered table.

        Args:
            result: The QueryResult to attach them to.
            plan: The executed physical plan.
            tables: Each output relation, by (node id, port).
        """
        answers, survivors, true_joins, stages = _quail_native.collect_answers(
            json.dumps(self._answer_request(plan)), tables)
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


def _output_tables(outputs) -> dict:
    """Return each output relation that is an Arrow table, by (node id, port)."""
    tables = {}
    for ref, value in outputs.items():
        table = as_table(value)
        if isinstance(table, pa.Table):
            tables[(ref.node_id, ref.port)] = table
    return tables


def _observe(graph, outputs, metrics, observers) -> None:
    """Show each observer every node whose outputs the run returned."""
    if not observers:
        return
    values = {(ref.node_id, ref.port): value for ref, value in outputs.items()}
    for node in graph.topological_nodes():
        found = {output.name: values[(node.node_id, output.name)]
                 for output in node.outputs
                 if (node.node_id, output.name) in values}
        if not found or len(found) != len(node.outputs):
            continue
        result = NodeResult(found, metrics.get(node.node_id, NodeMetrics()))
        for observer in observers:
            observer.after_node(node, result)

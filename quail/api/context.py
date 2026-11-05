"""Run SQL through DataFusion, with Quail planning each model region.

DataFusion parses and binds the query. Its rule ``ClaimModelRegions``
replaces each part of the plan that calls an AI function, together
with the operators around it that Quail plans, with one model region.
Rust binds each region into Quail's logical plan, as ``Session.sql``
compiles SQL, and Quail plans and runs it; DataFusion runs every
operator outside the regions. A region the Rust binder does not read,
such as one with a subquery in FROM, reaches Quail as SQL text.

For example, imagine a ``reviews`` table with review text in ``body``:

    context = quail.SessionContext(session)
    context.register("reviews", quail.DocumentProvider.from_table(
        reviews, id_col="id"))
    table = context.sql(
        "SELECT r.id FROM reviews r "
        "WHERE AI_FILTER(PROMPT('Does the review discuss the ending? {0}', "
        "r.body))")

The DataFusion side lives in the ``quail-datafusion`` crate. Its Python
module, ``_quail_native``, is part of the quail-engine wheel, built by
maturin from ``crates/quail-native``.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pyarrow as pa

from quail.api.compile import DATAFUSION_DIALECTS
from quail.api.execute import execute_query
from quail.api.query import Query
from quail.catalog import ScanRequest, TableProvider
from quail.logical import LogicalPlan
from quail.physical.codec import decode_value


def ipc_bytes(table: pa.Table) -> bytes:
    """Return a table as Arrow IPC stream bytes."""
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


class SessionContext:
    """A DataFusion session context with Quail registered.

    Args:
        session: The Quail session that plans and runs model regions;
            its tables are the context's tables.
        physical_executor: Optional callable(PhysicalRequest) that
            returns the PhysicalResponse, for tests without a GPU.

    Raises:
        ImportError: The ``_quail_native`` module is not built.
    """

    def __init__(self, session, physical_executor: Callable | None = None, *,
                 order: str | None = None, run_region: Callable | None = None):
        from _quail_native import NativeContext

        self.session = session
        self._physical_executor = physical_executor
        self._order = order
        model = session.model
        rules = [rule.name for rule in session.registry.logical_rules.values()]
        self._native = NativeContext(run_region or self._run_region, tuple(model.turn),
                                     model.prompt_layout, rules)
        for name, provider in session.catalog.providers.items():
            self._register_table(name, provider)

    def register(self, name: str, provider: TableProvider) -> None:
        """Register a table with the session and with DataFusion.

        DataFusion keeps the table's schema only. It scans the provider
        when an operator outside every model region reads the table.
        """
        self.session.register(name, provider)
        self._register_table(name, provider)

    def _register_table(self, name: str, provider: TableProvider) -> None:
        def scan(columns: list[str]) -> pa.Table:
            reader = provider.scan(ScanRequest(columns=tuple(columns)))
            return reader.read_all()

        self._native.register(name, ipc_bytes(provider.schema().empty_table()), scan,
                              getattr(provider, "id_col", None) or None)

    def sql(self, text: str, dialect: str = "snowflake") -> pa.Table:
        """Run a query and return its rows."""
        # DataFusion's batches, shared through the Arrow C stream interface
        return self._native.sql(text, DATAFUSION_DIALECTS[dialect])

    def region_sql(self, text: str) -> list[str]:
        """Return the SQL each model region of a query sends to Quail."""
        return self._native.region_sql(text)

    def logical_plans(self, text: str) -> list[LogicalPlan]:
        """Return the logical plan Rust binds for each model region of a query.

        Raises:
            CompileError: The Rust binder refuses a region.
        """
        from quail.common.expressions import CompileError

        model = self.session.model
        bound, _ = self._native.logical_plans(text, tuple(model.turn),
                                              model.prompt_layout)
        plans = []
        for plan, message in bound:
            if message is not None:
                raise CompileError(message)
            plans.append(decode_value(LogicalPlan, json.loads(plan)))
        return plans

    def explain(self, text: str, dialect: str = "snowflake") -> str:
        """Return DataFusion's optimized logical plan for a query."""
        return self._native.explain(text, DATAFUSION_DIALECTS[dialect])

    def _run_region(self, kind: str, payload: str, tables: dict | None = None,
                    fetch: int | None = None) -> pa.Table:
        from quail.api.region_inputs import region_query

        if kind == "plan":
            logical = decode_value(LogicalPlan, json.loads(payload))
            if fetch is not None:
                logical = with_limit(logical, fetch)
            query = region_query(self.session, logical, order=self._order,
                                 tables=tables)
        else:
            query = self.session.sql(payload, order=self._order)
        result = execute_query(query, physical_executor=self._physical_executor)
        return result.collect()


def with_limit(logical: LogicalPlan, fetch: int) -> LogicalPlan:
    """Return a region's plan that returns at most ``fetch`` rows.

    DataFusion passes the limit when a LIMIT above the region reaches
    it; Quail can then stop the model once enough rows pass.
    """
    from dataclasses import replace

    from quail.logical import Result, WithCTE

    root = logical.root
    if isinstance(root, WithCTE):
        return logical
    if isinstance(root, Result):
        limit = fetch if root.limit is None else min(root.limit, fetch)
        return replace(logical, root=replace(root, limit=limit))
    return replace(logical, root=Result(input=root, limit=fetch))


class DataFusionQuery:
    """A query DataFusion runs, with Quail planning and running each model region.

    ``Session.sql`` returns one when the query needs SQL around its
    model regions, such as a UNION of two AI filters or a window over
    an AI join's rows. It offers ``Query``'s methods.

    Args:
        session: The session whose tables and model the query uses.
        text: The SQL statement.
        dialect: "snowflake" or "bq".
        order: The stage order rule each region is planned with.
    """

    def __init__(self, session, text: str, dialect: str = "snowflake",
                 order: str | None = None):
        self.session = session
        self.text = text
        self.dialect = dialect
        self.order = order

    def _context(self, physical_executor=None) -> SessionContext:
        return SessionContext(self.session, physical_executor, order=self.order)

    def _regions(self) -> list[Query]:
        model = self.session.model
        bound, _ = self._context()._native.logical_plans(
            self.text, tuple(model.turn), model.prompt_layout,
            DATAFUSION_DIALECTS[self.dialect])
        from quail.api.region_inputs import region_query

        return [region_query(self.session, decode_value(LogicalPlan, json.loads(plan)),
                             order=self.order)
                for plan, message in bound if message is None]

    def plan(self) -> tuple:
        """Return the physical plan, or Refusal, of each model region Rust binds."""
        return tuple(query.plan() for query in self._regions())

    def explain(self, **options) -> str:
        """Return DataFusion's plan, then each model region's plan."""
        parts = [self._context().explain(self.text, self.dialect)]
        parts.extend(f"model region {index + 1}:\n{query.explain(**options)}"
                     for index, query in enumerate(self._regions()))
        return "\n\n".join(parts)

    def run(self, plan=None, *, physical_executor: Callable | None = None):
        """Run the query in the current process.

        Args:
            plan: Must be None: an edited plan is per model region.
            physical_executor: Optional callable(PhysicalRequest) that
                returns the PhysicalResponse, for tests without a GPU.

        Raises:
            TypeError: A plan was given.
        """
        from quail.execution.result import QueryResult

        if plan is not None:
            raise TypeError("a query DataFusion runs takes no edited plan")
        table = self._context(physical_executor).sql(self.text, self.dialect)
        return QueryResult.from_table(table)

    def submit(self, **_options):
        """Submitting needs Quail Server; local queries use run()."""
        raise RuntimeError(
            "submit() needs a Session with an endpoint; this session runs "
            "queries in the current process, so call run() or collect()")

    def collect(self, limit: int | None = None, batch_rows: int = 65_536) -> pa.Table:
        """Run the query and collect its rows."""
        table = self.run().collect()
        return table if limit is None else table.slice(0, limit)

    def execute_stream(self, batch_rows: int = 65_536,
                       limit: int | None = None) -> pa.RecordBatchReader:
        """Run the query and stream its rows."""
        table = self.collect(limit=limit)
        return pa.RecordBatchReader.from_batches(
            table.schema, table.to_batches(max_chunksize=batch_rows))

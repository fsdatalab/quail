"""Run SQL through DataFusion, with Quail planning each model region.

DataFusion parses, binds, and optimizes the query, each AI call as an
``AiNode``. Quail's query planner then replaces each part of the
optimized plan that calls an AI function, together with the operators
around it that Quail plans, with one model region.
Rust derives each region's operators from DataFusion's plan of it, as
``Session.sql`` compiles SQL, and Quail plans and runs them; DataFusion
runs every operator outside the regions. A region whose shape Quail's
operators do not read reaches Quail as SQL text.

For example, imagine a ``reviews`` table with review text in ``body``:

    context = quail.SessionContext(session)
    context.register("reviews", quail.DocumentProvider.from_table(
        reviews, id_col="id"))
    table = context.sql(
        "SELECT r.id FROM reviews r "
        "WHERE AI_FILTER(PROMPT('Does the review discuss the ending? {0}', "
        "r.body))")

The session is the ``quail`` crate's ``Session``, the one a Rust
program runs queries with; ``_quail_native`` binds it for Python. Its
model regions call ``quail.api.engine.Engine``.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pyarrow as pa

from quail.api.compile import DATAFUSION_DIALECTS, compile_message
from quail.api.engine import Engine
from quail.catalog import ScanRequest, TableProvider


def ipc_bytes(table: pa.Table) -> bytes:
    """Return a table as Arrow IPC stream bytes."""
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


class SessionContext:
    """Quail's Rust ``Session``, over a Python session's tables and model.

    Args:
        session: The Quail session that plans and runs model regions;
            its tables are the context's tables.
        physical_executor: Optional callable(PhysicalRequest) that
            returns the PhysicalResponse, for tests without a GPU.

    Raises:
        ImportError: The ``_quail_native`` module is not built.
    """

    def __init__(self, session, physical_executor: Callable | None = None, *,
                 order: str | None = None, run_query: Callable | None = None):
        from _quail_native import NativeSession

        self.session = session
        engine = Engine(session, physical_executor, order=order, run_query=run_query)
        self._engine = engine
        self._native = NativeSession(engine, session.planner)
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
                              getattr(provider, "id_col", None) or None,
                              provider.content_identity())

    def sql(self, text: str, dialect: str = "snowflake") -> pa.Table:
        """Run a query and return its rows."""
        # DataFusion's batches, shared through the Arrow C stream interface
        try:
            return self._native.sql(text, DATAFUSION_DIALECTS[dialect])
        except RuntimeError as error:
            raise RuntimeError(compile_message(error, text)) from None

    def region_plans(self, text: str) -> list[dict]:
        """Return each model region Rust derives, as a dict.

        Each dict holds ``operators``, the region's operators as JSON, and
        ``logical``, DataFusion's plan of the region.

        Raises:
            CompileError: Quail refuses a region.
        """
        from quail.common.expressions import CompileError

        model = self.session.model
        try:
            derived, _ = self._native.region_plans(text, tuple(model.turn),
                                                 model.prompt_layout)
        except RuntimeError as error:
            raise CompileError(compile_message(error, text)) from None
        regions = []
        for found, message, _ in derived:
            if message is not None:
                raise CompileError(message)
            regions.append(json.loads(found))
        return regions

    def explain(self, text: str, dialect: str = "snowflake") -> str:
        """Return DataFusion's optimized logical plan for a query."""
        try:
            return self._native.explain(text, DATAFUSION_DIALECTS[dialect])
        except RuntimeError as error:
            raise RuntimeError(compile_message(error, text)) from None

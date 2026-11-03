"""Run SQL through DataFusion, with Quail planning each model region.

DataFusion parses and binds the query. Its rule ``ClaimModelRegions``
replaces each part of the plan that calls an AI function, together
with the operators around it that Quail plans, with one model region.
Quail plans and runs each region as ``Session.sql`` does; DataFusion
runs every operator outside the regions.

For example, imagine a ``reviews`` table with review text in ``body``:

    context = quail.SessionContext(session)
    context.register("reviews", quail.DocumentProvider.from_table(
        reviews, id_col="id"))
    table = context.sql(
        "SELECT r.id FROM reviews r "
        "WHERE AI_FILTER(PROMPT('Does the review discuss the ending? {0}', "
        "r.body))")

The DataFusion side lives in the ``quail-datafusion`` crate. Its Python
module, ``_quail_datafusion``, is a wheel that maturin builds from
``crates/quail-datafusion-python``; install the wheel into Quail's
environment.
"""

from __future__ import annotations

from collections.abc import Callable

import pyarrow as pa

from quail.api.execute import execute_query
from quail.catalog import ScanRequest, TableProvider
from quail.logical import SPAN_SUFFIX


def ipc_bytes(table: pa.Table) -> bytes:
    """Return a table as Arrow IPC stream bytes."""
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def ipc_table(data: bytes) -> pa.Table:
    """Return the table held in Arrow IPC stream bytes."""
    with pa.ipc.open_stream(data) as reader:
        return reader.read_all()


class SessionContext:
    """A DataFusion session context with Quail registered.

    Args:
        session: The Quail session that plans and runs model regions;
            its tables are the context's tables.
        physical_executor: Optional callable(PhysicalRequest) that
            returns the PhysicalResponse, for tests without a GPU.

    Raises:
        ImportError: The ``_quail_datafusion`` module is not built.
    """

    def __init__(self, session, physical_executor: Callable | None = None):
        from _quail_datafusion import NativeContext

        self.session = session
        self._physical_executor = physical_executor
        self._native = NativeContext(self._run_region)

    def register(self, name: str, provider: TableProvider) -> None:
        """Register a table with the session and with DataFusion."""
        self.session.register(name, provider)
        table = provider.scan(ScanRequest(columns=provider.columns)).read_all()
        self._native.register(name, ipc_bytes(table))

    def sql(self, text: str) -> pa.Table:
        """Run a query and return its rows."""
        return ipc_table(self._native.sql(text))

    def region_sql(self, text: str) -> list[str]:
        """Return the SQL each model region of a query sends to Quail."""
        return self._native.region_sql(text)

    def explain(self, text: str) -> str:
        """Return DataFusion's optimized logical plan for a query."""
        return self._native.explain(text)

    def _run_region(self, text: str) -> bytes:
        query = self.session.sql(text)
        result = execute_query(query, physical_executor=self._physical_executor)
        return ipc_bytes(without_spans(result.collect()))


def without_spans(table: pa.Table) -> pa.Table:
    """Drop the answer position column Quail adds beside each extraction.

    An ``AI.EXTRACT(...) AS lease`` returns ``lease`` and ``lease_span``;
    DataFusion's plan has only ``lease``.
    """
    names = set(table.column_names)
    return table.drop_columns([
        name for name in table.column_names
        if name.endswith(SPAN_SUFFIX) and name[:-len(SPAN_SUFFIX)] in names])

"""Plan and run the model regions of Quail's Rust ``Session``.

Rust's ``quail::Session`` runs SQL in DataFusion. Each model region
DataFusion reaches calls ``Engine.run_region`` with the region's plan
and the rows of the tables it reads from DataFusion. A Rust program
opens an engine with ``Engine.open``; Python's ``quail.Session`` makes
one over itself.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import _quail_native
import pyarrow as pa

from quail.api.execute import execute_query


class Engine:
    """Plans and runs model regions on a Quail session's model.

    Args:
        session: The Quail session whose model, tables, and executor
            the regions use.
        physical_executor: Optional callable(PhysicalRequest) that
            returns the PhysicalResponse, for tests without a GPU.
        order: The stage order rule each region is planned with.
        run_query: Optional callable(query, on_rows, hint) -> pyarrow
            Table that runs a region's query in place of the engine,
            handing each batch of finished rows to ``on_rows`` when it is
            not None, and skipping rows by the ``InputHint`` when given.
        owns_session: Whether ``close`` closes the session.
    """

    # the rows of a region input Rust runs at once, when the region asks
    # each row on its own; a larger input runs in windows of this size
    input_window_rows = 65_536

    def __init__(self, session, physical_executor: Callable | None = None, *,
                 order: str | None = None, run_query: Callable | None = None,
                 owns_session: bool = False):
        self.session = session
        self._physical_executor = physical_executor
        self._order = order
        self._run_query = run_query
        self._owns_session = owns_session

    @classmethod
    def open(cls, model: str, device: str, gpus: int = 1,
             backend: str = "quail") -> Engine:
        """Open an engine on a new session of ``model`` on ``device``."""
        from quail.api.config import EngineConfig
        from quail.api.session import Session

        config = EngineConfig(model=model, device=device, gpus=gpus, backend=backend)
        return cls(Session(config), owns_session=True)

    @property
    def turn(self) -> tuple[str, str]:
        """The model's chat-turn text, which Rust binds prompts in."""
        return tuple(self.session.model.turn)

    @property
    def layout(self) -> str:
        """The model's prompt layout, which Rust binds prompts in."""
        return self.session.model.prompt_layout

    @property
    def rules(self) -> list[str]:
        """The logical optimizer rules DataFusion runs, in order."""
        return [rule.name for rule in self.session.registry.logical_rules.values()]

    @property
    def tokenizer_file(self) -> str:
        """The model's ``tokenizer.json``, which Rust plans regions with."""
        return self.session.tokenizer_file

    @property
    def settings(self) -> str | None:
        """The configuration Rust plans regions for, as JSON.

        None when the session's backend refuses its configuration, so the
        region's query reports the refusal.
        """
        from quail.api.query import backend_refusal, planning_settings

        if backend_refusal(self.session) is not None:
            return None
        return json.dumps(planning_settings(self.session, self._order))

    def execute(self, planned: str, tables: dict | None = None,
                on_rows: Callable | None = None, hint=None) -> pa.Table | None:
        """Run a region Rust planned and return its rows.

        Args:
            planned: The planned region as JSON: its physical plan, the
                operators it lowers, DataFusion's plan of the region, and
                each scan's files; or its refusal.
            tables: The rows of each table the region reads, by name.
            on_rows: Optional callable(pyarrow Table) handed the region's
                rows as they finish, each row once; the call then
                returns None.
            hint: Optional ``InputHint`` over the region's one input: the
                tests above the region, by which it may skip rows before
                the model asks about them.

        Raises:
            RefusalError: No configuration can run the region.
        """
        from quail.api.region_inputs import region_query
        from quail.physical.plan import RefusalError, refusal_from_fields

        found = json.loads(planned)
        if "refusal" in found:
            raise RefusalError(refusal_from_fields(found["refusal"]))
        query = region_query(self.session, found["operators"],
                             logical=found.get("logical", ""), order=self._order,
                             tables=tables, keep_order=True)
        query.adopt_plan(found)
        return self._run(query, on_rows, hint)

    def _run(self, query, on_rows: Callable | None = None,
             hint=None) -> pa.Table | None:
        if self._run_query is not None:
            table = self._run_query(query, on_rows, hint)
        else:
            table = execute_query(query, physical_executor=self._physical_executor,
                                  on_rows=on_rows, hint=hint).collect()
        return None if on_rows is not None else table

    def run_region(self, payload: str, tables: dict | None = None,
                   fetch: int | None = None) -> pa.Table:
        """Plan and run one model region and return its rows.

        Args:
            payload: The region as JSON: its operators and DataFusion's
                plan of it.
            tables: The rows of each table the region reads from
                DataFusion, by table name.
            fetch: The most rows DataFusion reads, or None.
        """
        from quail.api.region_inputs import region_query

        found = json.loads(payload)
        operators = found["operators"]
        if fetch is not None:
            operators = _quail_native.limited_operators(operators, fetch)
        query = region_query(self.session, operators, logical=found["logical"],
                             order=self._order, tables=tables)
        return self._run(query)

    def close(self) -> None:
        """Close the session if this engine opened it."""
        if self._owns_session:
            self.session.close()

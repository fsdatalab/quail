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

import pyarrow as pa

from quail.api.execute import execute_query
from quail.logical import LogicalPlan
from quail.physical.codec import decode_value


class Engine:
    """Plans and runs model regions on a Quail session's model.

    Args:
        session: The Quail session whose model, tables, and executor
            the regions use.
        physical_executor: Optional callable(PhysicalRequest) that
            returns the PhysicalResponse, for tests without a GPU.
        order: The stage order rule each region is planned with.
        run_query: Optional callable(Query) -> pyarrow Table that runs a
            region's query in place of the engine.
        owns_session: Whether ``close`` closes the session.
    """

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
        """The optimizer rules each model region runs, in order."""
        return [rule.name for rule in self.session.registry.logical_rules.values()]

    @property
    def settings(self) -> str | None:
        """The configuration Rust plans regions for, as JSON.

        None for a backend whose regions Python plans.
        """
        config = self.session.config
        if config.backend != "quail":
            return None
        from quail.physical.codec import encode_value
        from quail.planner import native

        return json.dumps({
            "model": native.untagged(encode_value(self.session.model)),
            "device": native.untagged(encode_value(self.session.device)),
            "gpus": config.gpus, "canvas_draws": config.canvas_draws,
            "attention": config.attention, "order": self._order,
            "backend": config.backend,
            "physical_rules": [rule.name for rule in
                               self.session.registry.physical_rules.values()],
        })

    def execute(self, planned: str, tables: dict | None = None) -> pa.Table:
        """Run a region Rust planned and return its rows.

        Args:
            planned: The planned region as JSON: its physical plan, the
                logical plan it lowers, and each scan's files; or its
                refusal.
            tables: The rows of each table the region reads, by name.

        Raises:
            RefusalError: No configuration can run the region.
        """
        from quail.api.region_inputs import region_query
        from quail.physical.plan import Refusal, RefusalError

        found = json.loads(planned)
        if "refusal" in found:
            fields = found["refusal"]
            raise RefusalError(Refusal(**{**fields, "reasons": tuple(fields["reasons"])}))
        logical = decode_value(LogicalPlan, json.loads(found["logical"]))
        query = region_query(self.session, logical, order=self._order,
                             tables=tables, keep_order=True)
        query.adopt_plan(found)
        return self._run(query)

    def _run(self, query) -> pa.Table:
        if self._run_query is not None:
            return self._run_query(query)
        result = execute_query(query, physical_executor=self._physical_executor)
        return result.collect()

    def run_region(self, kind: str, payload: str, tables: dict | None = None,
                   fetch: int | None = None) -> pa.Table:
        """Plan and run one model region and return its rows.

        Args:
            kind: "plan" for a region's logical plan as JSON, or "sql" for
                the SQL of a region the Rust binder does not read.
            payload: The plan or SQL.
            tables: The rows of each table the region reads from
                DataFusion, by table name.
            fetch: The most rows DataFusion reads, or None.
        """
        from quail.api.region_inputs import region_query

        if kind == "plan":
            logical = decode_value(LogicalPlan, json.loads(payload))
            if fetch is not None:
                logical = with_limit(logical, fetch)
            query = region_query(self.session, logical, order=self._order,
                                 tables=tables)
        else:
            query = self.session.sql(payload, order=self._order)
        return self._run(query)

    def close(self) -> None:
        """Close the session if this engine opened it."""
        if self._owns_session:
            self.session.close()


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

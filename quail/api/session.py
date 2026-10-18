"""Register tables and run queries, here or on Quail Server."""

from __future__ import annotations

from quail.api.builder import BoundBuilder
from quail.api.query import Query, QueryLike
from quail.builtins import built_in_registry
from quail.catalog import Catalog, InputStore, TableProvider
from quail.catalog.tokens import GigatokenEncoder
from quail.extensions import ExtensionRegistry
from quail.frontend.builder import Query as BuilderQuery
from quail.frontend.prompts import PromptBinder
from quail.frontend.sql import SQLDialect, compile_sql
from quail.planner.plan import EngineConfig, Refusal, RefusalError, resolve_model
from quail.specs.files import resolve_model_path


class Session:
    """Register tables and run queries, here or on Quail Server.

    Without ``endpoint`` every query runs in this process. With one,
    ``register`` describes each table for the server, ``sql`` returns
    a query that ``submit()`` sends there, and ``get_run`` reattaches
    to an accepted query by id. The query API is the same either way.
    """

    def __init__(self, config: EngineConfig, *,
                 tokenizer=None,
                 registry: ExtensionRegistry | None = None,
                 endpoint: str | None = None):
        self.registry = registry or built_in_registry()
        model = resolve_model(config.model, self.registry.models)
        if isinstance(model, Refusal):
            raise RefusalError(model)
        self.config = config
        self.model = model
        self.device = self.registry.device(config.device)
        try:
            backend = self.registry.backend(config.backend)
        except ValueError as error:
            raise RefusalError(Refusal(
                reasons=(str(error),),
                constraint="unknown_backend",
                needed=1,
                available=0,
                unit="backends",
            )) from error
        support = backend.supports(model, self.device, config.gpus)
        if not support.supported:
            raise RefusalError(Refusal(
                reasons=(support.reason or
                         "unsupported backend configuration",),
                constraint="unsupported_backend_configuration",
                needed=1,
                available=0,
                unit="configurations",
            ))
        self.catalog = Catalog()
        self._tok = tokenizer      # injectable for tests
        self.inputs = InputStore(
            self.catalog, tokenizer=lambda: self.tokenizer,
            tokenizer_name="configured tokenizer" if tokenizer else "Gigatoken")
        self._remote = None
        if endpoint is not None:
            from quail.remote.client import RemoteConnection

            self._remote = RemoteConnection(endpoint, self.registry.codecs)
        elif tokenizer is None:
            _ = self.tokenizer
            self.inputs.warm_tokenizer()

    @property
    def endpoint(self) -> str | None:
        """The Quail Server this session submits to, or None."""
        return None if self._remote is None else self._remote.client.endpoint

    @property
    def session_id(self) -> str | None:
        """The id sent with every submission of a remote session, or None.

        The server saves it on each record, so ``GET /v1/queries``
        with ``session_id`` lists what this session submitted.
        """
        return None if self._remote is None else self._remote.session_id

    def close(self):
        """Wait for background tokenization and remove temporary token files.

        Closing a remote session removes its staged uploads only; the
        server keeps every accepted query and the inputs it needs.
        """
        if self._remote is not None:
            self._remote.close()
        self.inputs.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def register(self, name: str, provider: TableProvider) -> None:
        """Register a table under ``name``.

        On a remote session the provider must be one the server can
        read: an in-memory table, a Parquet, IPC, or Arrow dataset
        (uploaded as a snapshot), or a Hugging Face dataset (pinned to a
        revision). Any other provider raises TypeError here rather than
        running locally.
        """
        self.catalog.register(name, provider)
        if self._remote is not None:
            try:
                self._remote.register(name, provider)
            except Exception:
                del self.catalog.providers[name]
                raise

    def sql(self, text: str, order: str | None = None,
            dialect: SQLDialect | str = SQLDialect.SNOWFLAKE) -> QueryLike:
        """Compile SQL locally or prepare it for remote submission.

        Returns:
            A local ``Query`` when the session has no endpoint. A remote
            session returns ``RemoteQuery`` and the server compiles the SQL.
        """
        if self._remote is not None:
            from quail.remote.client import RemoteQuery

            return RemoteQuery(self, text, order=order,
                               dialect=SQLDialect(dialect).value)
        logical = compile_sql(
            text, self.catalog, dialect=dialect,
            prompts=PromptBinder(model=self.model),
        )
        return Query(self, logical, order=order)

    def get_run(self, query_id: str):
        """Reattach to a query the server already accepted.

        Returns a QueryRun; it never resubmits or starts another
        execution.
        """
        if self._remote is None:
            raise RuntimeError(
                "get_run() needs a Session with an endpoint; a local "
                "session has no saved query records")
        return self._remote.run(query_id)

    def docs(self, name: str) -> "BoundBuilder":
        if self._remote is not None:
            raise RuntimeError(
                "the builder API is not available on a remote Session; "
                "write the query with sql()")
        return BoundBuilder(self,
                            BuilderQuery(self.catalog, name,
                                         prompts=PromptBinder(model=self.model)))

    @property
    def tokenizer(self):
        """Return Gigatoken's encoder, loading it when first used.

        The encoder's ``decode`` attribute returns a token id list's text
        as bytes; the planner uses it to find the word each label token
        starts.
        """
        if self._tok is None:
            self._tok = GigatokenEncoder(resolve_model_path(
                self.model.hf_name, self.model.revision, include_weights=False))
        return self._tok

    def register_functions(self, functions: dict) -> None:
        """Register a query's apply() functions once each, by name."""
        for name, function in functions.items():
            known = self.registry.functions.get(name)
            if known is function:
                continue
            if known is not None:
                raise ValueError(
                    f"apply name {name!r} is registered for another "
                    f"function on this session")
            self.registry.register_function(function, name=name)

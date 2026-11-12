"""Register tables and run queries, here or on Quail Server."""

from __future__ import annotations

import os

import _quail_native

from quail.api.compile import SQLDialect
from quail.api.config import EngineConfig, resolve_model
from quail.api.query import QueryLike
from quail.builtins import built_in_registry
from quail.catalog import Catalog, TableProvider
from quail.common.tokens import Tokenizer
from quail.extensions import ExtensionRegistry
from quail.physical.plan import Refusal, RefusalError
from quail.specs.files import resolve_model_path


class Session:
    """Register tables and run queries, here or on Quail Server.

    Without ``endpoint`` every query runs in this process. With one,
    such as ``grpc://127.0.0.1:8642``, ``register`` uploads each table
    to the server over Arrow Flight, ``sql`` returns a query that
    ``submit()`` sends there, and ``get_run`` reattaches to a submitted
    query by id. The query API is the same either way.
    """

    def __init__(self, config: EngineConfig, *,
                 tokenizer: str | os.PathLike | None = None,
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
        self._tokenizer_directory = tokenizer
        self._tokenizer = None
        self._remote = None
        if endpoint is not None:
            from quail.remote.client import RemoteConnection

            self._remote = RemoteConnection(endpoint)
        else:
            _ = self.tokenizer

    @property
    def endpoint(self) -> str | None:
        """The Quail Server this session submits to, or None."""
        return None if self._remote is None else self._remote.client.endpoint

    def close(self):
        """Wait for background tokenization and remove temporary token files.

        Closing a remote session closes its connection only; the
        server keeps its tables and every submitted query.
        """
        if self._remote is not None:
            self._remote.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def register(self, name: str, provider: TableProvider) -> None:
        """Register a table under ``name``.

        A remote session uploads the provider's rows to the server.
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
            A ``Query`` that this process plans and runs, or a
            ``RemoteQuery`` the server compiles when the session has an
            endpoint.

        Raises:
            ValueError: ``order`` is given on a remote session; the
                server chooses each query's order.
        """
        if self._remote is not None:
            from quail.remote.client import RemoteQuery

            if order is not None:
                raise ValueError("a remote session does not take order")
            return RemoteQuery(self, text, dialect=SQLDialect(dialect).value)
        from quail.api.query import Query

        query = Query(self, text, SQLDialect(dialect).value, order=order)
        query.bound_whole()     # a statement that does not bind fails here
        return query

    def get_run(self, query_id: str):
        """Return the run of a query the server already has, by its id."""
        if self._remote is None:
            raise RuntimeError(
                "get_run() needs a Session with an endpoint; a local "
                "session has no saved query records")
        return self._remote.run(query_id)

    @property
    def tokenizer_file(self) -> str:
        """Return the path of the model's ``tokenizer.json``."""
        from pathlib import Path

        directory = self._tokenizer_directory or resolve_model_path(
            self.model.hf_name, self.model.revision, include_weights=False)
        return str(Path(directory) / "tokenizer.json")

    @property
    def planner(self):
        """Return the Rust planner of this session's regions, made when first used.

        It tokenizes with the session's ``tokenizer.json`` and keeps the
        token stores of the tables its queries read whole.
        """
        if getattr(self, "_planner", None) is None:
            self._planner = _quail_native.NativePlanner(self.tokenizer_file)
        return self._planner

    @property
    def tokenizer(self):
        """Return the model's tokenizer, loading it when first used.

        It reads the ``tokenizer.json`` in the directory the session was
        given, or else the model's. The encoder's ``decode`` attribute
        returns a token id list's text as bytes; the planner uses it to
        find the word each label token starts.
        """
        if self._tokenizer is None:
            directory = self._tokenizer_directory or resolve_model_path(
                self.model.hf_name, self.model.revision, include_weights=False)
            self._tokenizer = Tokenizer(directory)
        return self._tokenizer

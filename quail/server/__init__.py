"""Quail Server: the optional query server and its Python client.

A ``Session`` with an ``endpoint`` sends queries here instead of
running them in its own process. The server saves each query's record,
status, and result, and keeps running after the client disconnects.
Importing this package pulls in nothing beyond the standard library and
pyarrow. The HTTP application in ``quail.server.app`` needs the
``server`` installation extra.
"""

from quail.server.records import (
    ACTIVE_STATES,
    STATES,
    TERMINAL_STATES,
    InvalidRequestError,
    QueryFailedError,
    QueryIdConflictError,
    QueryStatus,
    ServerError,
    UnknownQueryError,
)

__all__ = [
    "ACTIVE_STATES",
    "STATES",
    "TERMINAL_STATES",
    "InvalidRequestError",
    "QueryFailedError",
    "QueryStatus",
    "QueryIdConflictError",
    "ServerError",
    "UnknownQueryError",
]

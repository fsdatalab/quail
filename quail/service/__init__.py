"""Optional query service: saved query records, status, and results.

Importing this package pulls in nothing beyond the standard library and
pyarrow. The HTTP application in ``quail.service.app`` needs the
``service`` installation extra.
"""

from quail.service.records import (
    ACTIVE_STATES,
    STATES,
    TERMINAL_STATES,
    InvalidRequestError,
    QueryFailedError,
    QueryStatus,
    RequestKeyConflictError,
    ServiceError,
    UnknownQueryError,
)

__all__ = [
    "ACTIVE_STATES",
    "STATES",
    "TERMINAL_STATES",
    "InvalidRequestError",
    "QueryFailedError",
    "QueryStatus",
    "RequestKeyConflictError",
    "ServiceError",
    "UnknownQueryError",
]

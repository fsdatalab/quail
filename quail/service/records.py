"""The saved query record, its states, and the service's error types."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

STATES = (
    "queued", "planning", "running",
    "succeeded", "failed", "interrupted", "cancelled",
)
ACTIVE_STATES = frozenset({"planning", "running"})
TERMINAL_STATES = frozenset({"succeeded", "failed", "interrupted", "cancelled"})


class ServiceError(RuntimeError):
    """An error the service reports to a client with an HTTP status."""

    status = 500


class InvalidRequestError(ServiceError):
    """The submission or request is malformed or refers to unknown inputs."""

    status = 400


class UnknownQueryError(ServiceError):
    """No saved record has this id."""

    status = 404


class RequestKeyConflictError(ServiceError):
    """A request key was reused for a different specification."""

    status = 409


class NotReadyError(ServiceError):
    """The result was requested before the record reached succeeded."""

    status = 409


class QueryFailedError(RuntimeError):
    """Raised by result() when the saved record ended without a result."""

    def __init__(self, status: QueryStatus):
        self.status = status
        error = status.error or {}
        detail = error.get("message") or status.state
        super().__init__(f"query {status.id} {status.state}: {detail}")


def canonical_json(value) -> str:
    """Serialize a value so equal values give equal text."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      default=str)


def spec_hash(spec: dict, config: dict, inputs: dict, timeout_s: float) -> str:
    """Hash everything a submission fixes, for request-key comparison."""
    text = canonical_json([spec, config, inputs, timeout_s])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QueryStatus:
    """One complete saved status snapshot of a query record.

    progress, plan, error, and result are plain dictionaries so the
    snapshot serializes to JSON without any Quail object.
    """

    id: str
    state: str
    revision: int
    created_at: float
    updated_at: float
    timeout_s: float
    spec: dict
    config: dict
    inputs: dict
    started_at: float | None = None
    cancel_requested: bool = False
    progress: dict | None = None
    plan: dict | None = None
    error: dict | None = None
    result: dict | None = None
    request_key: str | None = field(default=None, compare=False)

    @property
    def done(self) -> bool:
        return self.state in TERMINAL_STATES

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> QueryStatus:
        known = {name for name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items()
                      if key in known})

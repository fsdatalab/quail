"""The runtime: Session, the runnable Query, and the Modal worker.
The coordinator is the local process and is deliberately thin:
compile, plan, ship the plan, gate and assemble from the returned
answer rows, apply the projection."""

from .session import Query, RefusalError, Result, Session

__all__ = ["Query", "RefusalError", "Result", "Session"]

"""Tables for querying agent traces.

A trace is one conversation between a user, an agent, and the agent's
tools, stored as a list of messages in the chat format that OpenAI,
vLLM, and most agent frameworks write: each message has a ``role``
(``system``, ``user``, ``assistant``, or ``tool``) and a ``content``
string; an assistant message may carry ``tool_calls`` and a tool
message a ``tool_call_id``.

``trace_tables`` turns a table with one row per trace into two tables
a Session can register: ``traces``, one row per trace with its whole
transcript, and ``messages``, one row per message with the ids of the
messages before it, so a query reads only the messages it needs and
joins a message to the one it answers.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

import pyarrow as pa

from quail.logical import CompileError

TRANSCRIPT_ROLES = ("user", "assistant", "tool")

TRACE_COLUMNS = ("id", "request", "transcript", "message_count")
MESSAGE_COLUMNS = ("id", "trace_id", "turn_index", "role", "content",
                   "tool_call_id", "prev_id", "prev_user_id",
                   "prev_assistant_id")


def _json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def message_text(message: dict) -> str:
    """Render one message as the text an AI function reads.

    The content comes first. Each tool call follows on its own line as
    ``[tool call] name(arguments)``, so a message that only calls tools
    is not empty.
    """
    content = message.get("content")
    if content is None:
        content = ""
    elif not isinstance(content, str):
        content = _json_text(content)
    lines = [content] if content else []
    for call in message.get("tool_calls") or ():
        function = call.get("function", call)
        arguments = function.get("arguments", "")
        if not isinstance(arguments, str):
            arguments = _json_text(arguments)
        lines.append(f"[tool call] {function.get('name', '')}({arguments})")
    return "\n".join(lines)


def transcript(messages: Sequence[dict],
               roles: Iterable[str] = TRANSCRIPT_ROLES,
               max_message_chars: int | None = None) -> str:
    """Render a trace as one text, each message under a ``[ROLE]`` heading.

    Args:
        messages: The trace's messages, oldest first.
        roles: The roles to keep. The default leaves out the system
            prompt.
        max_message_chars: Cut each message's text to this many
            characters and mark the cut, or None to keep every message
            whole.
    """
    kept = set(roles)
    pieces = []
    for message in messages:
        role = str(message.get("role", ""))
        if role not in kept:
            continue
        text = message_text(message)
        if max_message_chars is not None and len(text) > max_message_chars:
            text = (text[:max_message_chars]
                    + f"\n[cut {len(text) - max_message_chars} characters]")
        pieces.append(f"[{role.upper()}]\n{text}")
    return "\n\n".join(pieces)


def _messages_of(value) -> list[dict]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise CompileError(
            "each trace's messages must be a list of messages or its JSON, "
            f"got {type(value).__name__}")
    messages = []
    for message in value:
        if not isinstance(message, dict) or "role" not in message:
            raise CompileError("each message must be a dict with a 'role'")
        messages.append({key: item for key, item in message.items()
                         if item is not None})
    return messages


def _rows_of(traces, id_col: str, messages_col: str, keep: Sequence[str]):
    if isinstance(traces, pa.Table):
        names = traces.schema.names
        missing = [name for name in (id_col, messages_col, *keep)
                   if name not in names]
        if missing:
            raise CompileError(
                f"columns {missing} not in trace table schema {tuple(names)}")
        return traces.select([id_col, messages_col, *keep]).to_pylist()
    rows = []
    for row in traces:
        missing = [name for name in (id_col, messages_col, *keep)
                   if name not in row]
        if missing:
            raise CompileError(f"trace record lacks {missing}")
        rows.append(row)
    return rows


def trace_tables(
    traces: pa.Table | Iterable[dict],
    *,
    id_col: str = "id",
    messages_col: str = "messages",
    keep: Sequence[str] = (),
    transcript_roles: Iterable[str] = TRANSCRIPT_ROLES,
    max_message_chars: int | None = None,
) -> dict[str, pa.Table]:
    """Split one row per trace into ``traces`` and ``messages`` tables.

    Args:
        traces: An Arrow table or records with one row per trace. The
            messages column holds a list of messages or its JSON text.
        id_col: The column that identifies each trace.
        messages_col: The column that holds the messages.
        keep: Further columns copied onto each ``traces`` row, such as
            a task id, a run number, or an outcome.
        transcript_roles: The roles the transcript keeps.
        max_message_chars: The per-message cut applied to the
            transcript, or None for none.

    Returns:
        ``traces``: ``id``, ``request`` (the first user message),
        ``transcript``, ``message_count``, then the kept columns.
        ``messages``: ``id`` (``<trace id>/<turn index>``), ``trace_id``,
        ``turn_index``, ``role``, ``content``, ``tool_call_id``,
        ``prev_id``, ``prev_user_id``, and ``prev_assistant_id``, the
        last three null for the first message of their kind.

    Raises:
        CompileError: A column is missing, a trace id repeats, or a
            message has no role.
    """
    keep = tuple(keep)
    for name in keep:
        if name in TRACE_COLUMNS:
            raise CompileError(
                f"kept column {name!r} clashes with a trace column; "
                f"trace columns are {TRACE_COLUMNS}")
    trace_rows = {name: [] for name in TRACE_COLUMNS + keep}
    message_rows = {name: [] for name in MESSAGE_COLUMNS}
    seen = set()
    for row in _rows_of(traces, id_col, messages_col, keep):
        trace_id = str(row[id_col])
        if trace_id in seen:
            raise CompileError(f"trace id {trace_id!r} repeats")
        seen.add(trace_id)
        messages = _messages_of(row[messages_col])
        request = ""
        prev_id = prev_user = prev_assistant = None
        for turn_index, message in enumerate(messages):
            role = str(message["role"])
            text = message_text(message)
            if role == "user" and not request:
                request = text
            message_id = f"{trace_id}/{turn_index}"
            message_rows["id"].append(message_id)
            message_rows["trace_id"].append(trace_id)
            message_rows["turn_index"].append(turn_index)
            message_rows["role"].append(role)
            message_rows["content"].append(text)
            tool_call_id = message.get("tool_call_id")
            message_rows["tool_call_id"].append(
                None if tool_call_id is None else str(tool_call_id))
            message_rows["prev_id"].append(prev_id)
            message_rows["prev_user_id"].append(prev_user)
            message_rows["prev_assistant_id"].append(prev_assistant)
            prev_id = message_id
            if role == "user":
                prev_user = message_id
            elif role == "assistant":
                prev_assistant = message_id
        trace_rows["id"].append(trace_id)
        trace_rows["request"].append(request)
        trace_rows["transcript"].append(
            transcript(messages, transcript_roles, max_message_chars))
        trace_rows["message_count"].append(len(messages))
        for name in keep:
            trace_rows[name].append(row[name])
    kept_fields = []
    if isinstance(traces, pa.Table):
        kept_fields = [traces.schema.field(name) for name in keep]
    else:
        kept_fields = [pa.field(name, pa.array(trace_rows[name]).type)
                       for name in keep]
    return {
        "traces": pa.table(trace_rows, schema=pa.schema([
            ("id", pa.string()), ("request", pa.string()),
            ("transcript", pa.string()), ("message_count", pa.int32()),
            *kept_fields])),
        "messages": pa.table(message_rows, schema=pa.schema([
            ("id", pa.string()), ("trace_id", pa.string()),
            ("turn_index", pa.int32()), ("role", pa.string()),
            ("content", pa.string()), ("tool_call_id", pa.string()),
            ("prev_id", pa.string()), ("prev_user_id", pa.string()),
            ("prev_assistant_id", pa.string())])),
    }


def register_traces(session, traces, *, prefix: str = "",
                    **options) -> dict[str, pa.Table]:
    """Register a trace set as ``<prefix>traces`` and ``<prefix>messages``.

    Args:
        session: The Session to register on.
        traces: One row per trace, as ``trace_tables`` takes.
        prefix: Put before both table names, such as ``support_``.
        **options: Passed to ``trace_tables``.

    Returns:
        The two tables, by their unprefixed names.
    """
    from quail.catalog import DocumentProvider

    tables = trace_tables(traces, **options)
    for name, table in tables.items():
        session.register(f"{prefix}{name}",
                         DocumentProvider.from_table(table, id_col="id"))
    return tables

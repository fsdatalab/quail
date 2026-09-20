"""Python adaptation of fast-jev-compaction for OpenHands trajectories.

The state fitting, retention questions, and reconstruction rules follow
tamaratran/fast-jev-compaction at e3f262a7f4d42bd8dd32ced30d26176f7cb545b0.
"""

# MIT License
#
# Copyright (c) 2025
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace

REFERENCE_REVISION = "e3f262a7f4d42bd8dd32ced30d26176f7cb545b0"
PRESERVE_RECENT = 6
MAX_STATE_TOKENS = 25_000
RESULT_HEAD_CHARS = 300

CONTEXT_TEXT = (
    "A coding assistant conversation is being compacted to free context. "
    "`history` is the whole conversation so far, oldest first; tool outputs "
    "are replaced by a short `result` note and long texts may be abridged. "
    "Each question asks whether one tool call, or the full output of that "
    "call, still needs to stay in the history verbatim. Whatever is not kept "
    "is deleted permanently, but the assistant can always re-run a tool "
    "or re-read a file."
)


def json_text(value) -> str:
    """Serialize state fields with the reference library's compact spacing."""
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    # A UTF-16 truncation can leave a lone surrogate; JSON.stringify escapes it.
    return text.encode("utf-8", errors="backslashreplace").decode("utf-8")


def text_length(text: str) -> int:
    """Count UTF-16 units, as JavaScript does in the original library."""
    return len(text.encode("utf-16-le", errors="surrogatepass")) // 2


def text_slice(text: str, start: int, end: int | None = None) -> str:
    """Slice using the reference library's UTF-16 offsets."""
    raw = text.encode("utf-16-le", errors="surrogatepass")
    return raw[2 * start:None if end is None else 2 * end].decode(
        "utf-16-le", errors="surrogatepass")


def truncate(text: str, limit: int) -> str:
    return text if text_length(text) <= limit else text_slice(text, 0, limit - 1) + "…"


def estimate_tokens(text: str) -> int:
    """Apply the original heuristic; this is not a model tokenizer."""
    total = 0.0
    for piece in re.findall(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]", text):
        if piece[0].isascii() and piece[0].isdigit():
            total += len(piece) / 2
        elif piece[0].isascii() and piece[0].isalpha():
            total += 1 + (len(piece) - 1) // 6
        else:
            total += 0.9 * text_length(piece)
    return math.ceil(total)


def is_pinned(index: int, count: int) -> bool:
    return index == 0 or index >= count - PRESERVE_RECENT


@dataclass(frozen=True)
class ToolCall:
    """One paired call and result in a recorded trajectory."""

    id: str
    source_id: str
    tool: str
    arguments: dict
    call_index: int
    result_index: int
    result: str
    is_error: bool
    pinned: bool


def collect_tool_calls(messages: list[dict]) -> list[ToolCall]:
    """Pair OpenHands tool calls with results without guessing parallel matches."""
    pending, seen, paired = {}, set(), []
    for index, message in enumerate(messages):
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise ValueError("message content must be text")
        if message["role"] == "tool":
            source_id = message.get("tool_call_id")
            # The dataset omits result IDs; a single pending call is unambiguous.
            if source_id is None and len(pending) == 1:
                source_id = next(iter(pending))
            if source_id not in pending:
                raise ValueError("ambiguous or unmatched tool result")
            call_index, tool, arguments = pending.pop(source_id)
            paired.append(ToolCall(
                "", source_id, tool, arguments, call_index, index, content,
                bool(message.get("is_error", False)),
                is_pinned(call_index, len(messages)) or is_pinned(index, len(messages)),
            ))
        for call in message.get("tool_calls") or []:
            source_id, function = call["id"], call["function"]
            if source_id in seen:
                raise ValueError(f"duplicate tool call ID: {source_id}")
            seen.add(source_id)
            arguments = json.loads(function["arguments"])
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be a JSON object")
            pending[source_id] = index, function["name"], arguments
    # The original IDs follow call order, even if results arrive out of order.
    order = {call["id"]: i for i, call in enumerate(
        call for message in messages for call in message.get("tool_calls") or [])}
    return [replace(call, id=f"t{i + 1}") for i, call in enumerate(
        sorted(paired, key=lambda call: order[call.source_id]))]


def retention_questions(call: ToolCall) -> list[dict]:
    """Fill the original templates with a call's ID, tool name, and result size."""
    call_prompt = (
        f"Tool call {call.id} ({call.tool}) should stay in the history: knowing "
        "this call was made, with its input, still matters for what the "
        "assistant does next"
    )
    result_prompt = (
        f"The full output of tool call {call.id} ({call.tool}, "
        f"{text_length(call.result)} chars) should stay in the history verbatim: "
        "the assistant still needs its contents and re-running the tool would not do"
    )
    return [{"key": f"{kind}_{call.id}", "tool_call_id": call.source_id,
             "kind": kind, "statement": statement}
            for kind, statement in (("call", call_prompt), ("result", result_prompt))]


def build_state(messages: list[dict], calls: list[ToolCall]) -> tuple[dict, int, str]:
    """Fit the decision context using the reference library's reduction stages."""
    goal = "\n".join(truncate(message["content"], 500) for message in [
        m for m in messages if m["role"] == "user" and (m.get("content") or "").strip()
    ][-3:])
    by_message = {}
    for call in calls:
        by_message.setdefault(call.call_index, []).append(call)
    state = {"context": CONTEXT_TEXT, "goal": goal, "history": []}
    base_tokens = estimate_tokens(json_text(state))

    def result(stage):
        return state, base_tokens + sum(sizes), stage

    def fits():
        return base_tokens + sum(sizes) <= MAX_STATE_TOKENS

    def resize(index):
        sizes[index] = estimate_tokens(json_text(history[index])) + 1

    for cap in (1000, 200, 60):
        history = []
        for i, message in enumerate(messages):
            if message["role"] == "tool":
                continue
            text = message.get("content") or ""
            own = by_message.get(i, [])
            if not text.strip() and not own:
                continue
            entry = {"i": i, "role": message["role"], "text": text}
            if own:
                entry["tool_calls"] = [{
                    "id": call.id, "tool": call.tool,
                    "input": truncate(json_text(call.arguments), cap),
                    "result": f"{'error' if call.is_error else 'ok'}, "
                              f"{text_length(call.result)} chars (omitted)",
                } for call in own]
            history.append(entry)
        state["history"] = history
        sizes = [estimate_tokens(json_text(entry)) + 1 for entry in history]
        if fits():
            return result("full" if cap == 1000 else f"inputs<={cap}")

    def pinned(entry):
        return is_pinned(entry["i"], len(messages))

    order = sorted(range(len(history)), key=lambda i: pinned(history[i]))
    for i in order:
        text = history[i]["text"]
        length = text_length(text)
        if length > 590:
            history[i]["text"] = (text_slice(text, 0, 400)
                                  + f"\n[… {length - 550} chars omitted …]\n"
                                  + text_slice(text, -150))
            resize(i)
            if fits():
                return result("texts abridged")
    for i in order:
        entry = history[i]
        if not pinned(entry) and entry["text"]:
            length = text_length(messages[entry["i"]].get("content") or "")
            entry["text"] = f"[… {length} chars omitted …]"
            resize(i)
            if fits():
                return result("old messages collapsed")
    for i in order:
        entry = history[i]
        own = by_message.get(entry["i"])
        if not pinned(entry) and own:
            compact = []
            for call in own:
                values = []
                for key, value in call.arguments.items():
                    text = value if isinstance(value, str) else truncate(
                        json_text({key: value}), 200)
                    values.append(f"{key}={re.sub(r'\s+', ' ', text)}")
                compact.append(
                    f"{call.id} {call.tool} {truncate(' '.join(values), 60)} → "
                    f"{'error' if call.is_error else 'ok'} "
                    f"{text_length(call.result)}ch")
            entry["tool_calls"] = compact
            resize(i)
            if fits():
                return result("old calls compacted")
    removed = set()
    for i in order:
        if not pinned(history[i]) and "tool_calls" not in history[i]:
            removed.add(i)
            sizes[i] = 0
            if fits():
                state["history"] = [entry for j, entry in enumerate(history)
                                    if j not in removed]
                return result("old messages left out")
    merged = []
    for i, entry in enumerate(history):
        if i in removed:
            continue
        foldable = (not pinned(entry) and not entry["text"]
                    and isinstance(entry.get("tool_calls", [None])[0], str))
        previous = merged[-1] if merged else None
        if (foldable and previous and not pinned(previous) and not previous["text"]
                and isinstance(previous.get("tool_calls", [None])[0], str)
                and previous["role"] == entry["role"]):
            previous["tool_calls"] += entry["tool_calls"]
        else:
            merged.append(dict(entry))
    state["history"] = merged
    sizes = [estimate_tokens(json_text(entry)) + 1 for entry in merged]
    if fits():
        return result("old calls merged")
    raise ValueError("history exceeds the state budget after all reduction stages")


def reconstruct_trace(messages: list[dict], calls: list[ToolCall],
                      answers: dict[str, bool]) -> tuple[list[dict], list[dict]]:
    """Apply retention decisions while preserving the OpenHands message format."""
    dropped, truncated, decisions = set(), {}, []
    for call in calls:
        if call.pinned:
            action = "pinned"
        else:
            keys = f"call_{call.id}", f"result_{call.id}"
            if any(type(answers.get(key)) is not bool for key in keys):
                raise ValueError(f"missing Boolean retention answers for {call.id}")
            action = ("keep" if answers[keys[1]] else
                      "truncate" if answers[keys[0]] else "drop")
        decisions.append({"tool_call_id": call.source_id, "action": action})
        if action == "drop":
            dropped.add(call.source_id)
        elif action == "truncate":
            text, length = call.result, text_length(call.result)
            if length > RESULT_HEAD_CHARS + 120:
                text = text_slice(text, 0, RESULT_HEAD_CHARS) + "\n"
                text += (f"[fast-jev-compaction truncated {length - RESULT_HEAD_CHARS} "
                         "chars of this tool result"
                         f"{' (error)' if call.is_error else ''}; "
                         "re-run the tool if needed]")
            truncated[call.result_index] = text
    dropped_results = {call.result_index for call in calls if call.source_id in dropped}
    kept = []
    for index, message in enumerate(messages):
        if index in dropped_results:
            continue
        if index in truncated:
            message = {**message, "content": truncated[index]}
        tools = message.get("tool_calls") or []
        remaining = [tool for tool in tools if tool["id"] not in dropped]
        if len(remaining) != len(tools):
            if not remaining and not (message.get("content") or "").strip():
                continue
            message = {**message, "tool_calls": remaining}
        kept.append(message)
    return kept, decisions


def message_chars(messages: list[dict]) -> int:
    """Count message text and tool arguments using the reference convention."""
    return sum(text_length(message.get("content") or "") + sum(
        text_length(json_text(json.loads(call["function"]["arguments"])))
        for call in message.get("tool_calls") or []) for message in messages)

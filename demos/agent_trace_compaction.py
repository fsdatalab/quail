"""Compact recorded OpenHands traces with a Quail semantic join.

Run locally on a machine with one or more H100 GPUs:

    uv run python demos/agent_trace_compaction.py \
      --output-dir /tmp/compaction --limit 100 --seed 42

Or run on Modal with the companion script:

    uv run modal run --detach demos/agent_trace_compaction_modal.py \
      --limit 100 --seed 42 \
      2>&1 | tee /tmp/quail-agent-compaction.log

Sample one trajectory per issue with a fixed seed. --limit counts issues.
Each selected complete trajectory is the conversation at compaction time.
The Python adaptation of fast-jev-compaction prepares the state and questions
and reconstructs messages. Quail replaces Jev's probabilities with Boolean
retention decisions. These are not calibrated probabilities.
The original tool outputs are omitted from the model's decision context.

Inputs, decisions, compacted messages, and timing are saved under the
output directory. The source ID index, sampled trajectories, and model
weights are cached by huggingface_hub. No Jev key is needed.
This measures compaction, not whether an agent can finish after compaction.

Reference: https://github.com/tamaratran/fast-jev-compaction (MIT).
Dataset: https://huggingface.co/datasets/nvidia/SWE-Zero-openhands-trajectories
(CC BY 4.0; retain the source repository and license in saved records).
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

import itertools
import json
import math
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DATASET = "nvidia/SWE-Zero-openhands-trajectories"
DATASET_REVISION = "7b3cd106d00f60918e722d33a1d74bc67072a7ea"
MODEL = "diffusion-gemma-26b-a4b-fp8"
DEVICE = "h100-sxm"

SQL = """
SELECT c.id, q.id, q.tool_call_id, q.kind
FROM conversations c
JOIN tool_questions q
  ON c.id = q.conversation_id
 AND AI.IF(
     PROMPT(
         'Using the compaction state in DOCUMENT {0}, evaluate whether
          the retention statement in DOCUMENT {1} is true.',
         c.state, q.statement
     ),
     {'anchor': 'c'}
 )
"""

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


def sample_trajectories(index: pa.Table, limit: int, seed: int) -> pa.Table:
    """Choose one attempt per issue, then sample distinct issues reproducibly."""
    if limit < 1:
        raise ValueError("limit must be positive")
    ordered = index.sort_by([("instance_id", "ascending"),
                             ("trajectory_id", "ascending")])
    groups = pc.run_end_encode(ordered["instance_id"].combine_chunks())
    ends = groups.run_ends.to_numpy()
    starts = np.concatenate(([0], ends[:-1]))
    if limit > len(ends):
        raise ValueError(f"requested {limit} issues, but only {len(ends)} exist")
    rng = np.random.default_rng(seed)
    attempts = starts + rng.integers(ends - starts)
    issues = rng.choice(len(ends), size=limit, replace=False)
    return ordered.take(attempts[issues])


def cache_table(table: pa.Table, path: Path) -> None:
    """Publish a complete cached Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    pq.write_table(table, temporary)
    temporary.replace(path)


def source_index(filesystem, cache: Path) -> pa.Table:
    """Index all attempts using only ID columns, without reading trace bodies."""
    from huggingface_hub import HfApi

    path = cache / "index.parquet"
    if path.exists():
        return pq.read_table(path)
    files = sorted(name for name in HfApi().list_repo_files(
        DATASET, repo_type="dataset", revision=DATASET_REVISION)
        if name.startswith("data/") and name.endswith(".parquet"))
    parts = []
    for name in files:
        remote = f"datasets/{DATASET}@{DATASET_REVISION}/{name}"
        with filesystem.open(remote, block_size=1 << 20) as source:
            parquet = pq.ParquetFile(source)
            for group in range(parquet.num_row_groups):
                ids = parquet.read_row_group(
                    group, columns=["instance_id", "trajectory_id"])
                parts.append(ids
                             .append_column("shard", pa.array([name] * len(ids)))
                             .append_column("row_group", pa.array([group] * len(ids)))
                             .append_column("row_index", pa.array(range(len(ids)))))
        print(f"indexed {name}", flush=True)
    index = pa.concat_tables(parts)
    cache_table(index, path)
    return index


def read_source_group(filesystem, cache: Path, entries: pa.Table) -> pa.Table:
    """Read one record group and cache only the sampled trajectories."""
    shard = entries["shard"][0].as_py()
    group = entries["row_group"][0].as_py()
    path = cache / "selected" / Path(shard).stem / f"{group}.parquet"
    cached = pq.read_table(path) if path.exists() else None
    positions = (pc.index_in(entries["trajectory_id"],
                             value_set=cached["trajectory_id"])
                 if cached is not None else None)
    if positions is not None and positions.null_count == 0:
        selected = cached.take(positions)
    else:
        full_group = cache / Path(shard).stem / f"{group}.parquet"
        if full_group.exists():
            table = pq.read_table(full_group)
        else:
            remote = f"datasets/{DATASET}@{DATASET_REVISION}/{shard}"
            with filesystem.open(remote, block_size=1 << 20) as source:
                table = pq.ParquetFile(source).read_row_group(group, columns=[
                    "instance_id", "trajectory_id", "repo", "license", "trajectory"])
        selected = table.take(entries["row_index"])
        if cached is not None:
            cached = cached.filter(pc.invert(pc.is_in(
                cached["trajectory_id"], value_set=selected["trajectory_id"])))
        cache_table(pa.concat_tables([cached, selected]) if cached is not None
                    else selected, path)
    for column in ("instance_id", "trajectory_id"):
        if not selected[column].equals(entries[column]):
            raise ValueError("cached trajectory does not match the source index")
    print(f"loaded {len(selected)} sampled trajectories from {shard}, "
          f"row group {group}", flush=True)
    return selected


def source_rows(limit: int, seed: int):
    """Sample with Arrow and fetch required record groups four at a time."""
    from huggingface_hub import HfFileSystem
    from huggingface_hub.constants import HF_HUB_CACHE

    cache = Path(HF_HUB_CACHE) / "quail-agent-compaction" / DATASET_REVISION
    filesystem = HfFileSystem()
    started = time.perf_counter()
    index = source_index(filesystem, cache)
    print(f"source index: {len(index)} attempts, "
          f"{time.perf_counter() - started:.2f} s", flush=True)
    started = time.perf_counter()
    selected = sample_trajectories(index, limit, seed).sort_by([
        ("shard", "ascending"), ("row_group", "ascending"),
        ("row_index", "ascending")])
    print(f"sampled {len(selected)} issues in "
          f"{time.perf_counter() - started:.3f} s", flush=True)
    same_group = pc.and_(
        pc.equal(selected["shard"].slice(1), selected["shard"].slice(0, limit - 1)),
        pc.equal(selected["row_group"].slice(1),
                 selected["row_group"].slice(0, limit - 1)))
    boundaries = np.concatenate(([0], np.flatnonzero(~same_group.to_numpy()) + 1,
                                 [limit]))
    groups = [selected.slice(int(start), int(end - start))
              for start, end in itertools.pairwise(boundaries)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for batch in itertools.batched(groups, 4):
            pending = [pool.submit(read_source_group, filesystem, cache, entries)
                       for entries in batch]
            for future in pending:
                yield from future.result().to_pylist()


def prepare_tables(rows, directory: Path) -> dict:
    """Write bounded Parquet batches of conversations, questions, and messages."""
    for name in ("conversations", "tool_questions", "messages"):
        (directory / name).mkdir(parents=True)
    counts = {"conversations": 0, "questions": 0, "pinned_calls": 0}
    seen, seen_issues = set(), set()
    for batch_id, batch in enumerate(itertools.batched(rows, 100)):
        conversations, questions, originals = [], [], []
        for row in batch:
            conv_id = row["trajectory_id"]
            if conv_id in seen:
                raise ValueError(f"duplicate trajectory ID: {conv_id}")
            if row["instance_id"] in seen_issues:
                raise ValueError(f"duplicate issue: {row['instance_id']}")
            seen.add(conv_id)
            seen_issues.add(row["instance_id"])
            messages = row["trajectory"]
            try:
                calls = collect_tool_calls(messages)
                candidates = [call for call in calls if not call.pinned]
                state, tokens, stage = (build_state(messages, calls) if candidates
                                        else (None, 0, "no candidates"))
            except ValueError as error:
                raise ValueError(f"cannot prepare {conv_id}: {error}") from error
            conversations.append({"id": conv_id, "instance_id": row["instance_id"],
                                  "state": json_text(state)})
            originals.append({
                "id": conv_id, "instance_id": row["instance_id"],
                "repo": row["repo"], "license": row["license"],
                "messages": json.dumps(messages),
                "state_tokens_estimated": tokens,
                "state_fitting_stage": stage,
            })
            for question in itertools.chain.from_iterable(
                    retention_questions(call) for call in candidates):
                questions.append({"id": f"{conv_id}:{question['key']}",
                                  "conversation_id": conv_id, **question})
            counts["conversations"] += 1
            counts["questions"] += 2 * len(candidates)
            counts["pinned_calls"] += len(calls) - len(candidates)
        filename = f"{batch_id:06d}.parquet"
        pq.write_table(pa.Table.from_pylist(conversations),
                       directory / "conversations" / filename)
        pq.write_table(pa.Table.from_pylist(originals),
                       directory / "messages" / filename)
        schema = pa.schema([(name, pa.string()) for name in (
            "id", "conversation_id", "key", "tool_call_id", "kind", "statement")])
        pq.write_table(pa.Table.from_pylist(questions, schema=schema),
                       directory / "tool_questions" / filename)
        print(f"prepared {counts['conversations']} conversations, "
              f"{counts['questions']} questions", flush=True)
    return counts


def prepare(directory: Path, limit: int, seed: int) -> dict:
    """Sample trajectories and write Parquet tables to *directory*."""
    if limit < 1:
        raise ValueError("limit must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    counts = prepare_tables(source_rows(limit, seed), directory)
    metadata = {**counts, "dataset": DATASET, "dataset_revision": DATASET_REVISION,
                "reference_revision": REFERENCE_REVISION,
                "seed": seed, "issue_limit": limit,
                "sampling": "one random trajectory per sampled instance_id",
                "preserve_recent_messages": PRESERVE_RECENT,
                "max_state_tokens_estimated": MAX_STATE_TOKENS,
                "truncate_result_chars": RESULT_HEAD_CHARS,
                "sql": SQL, "prepare_s": time.perf_counter() - started}
    (directory / "inputs.json").write_text(json.dumps(metadata, indent=2))
    print(f"output directory: {directory}", flush=True)
    return metadata


def complete_answers(result, conversations, questions):
    """Validate all pairs and return answers in question-table order."""
    answers = pa.concat_tables(list(result.answer_tables["joins"].values()))
    ordered = answers.sort_by([("q", "ascending")])
    if not np.array_equal(ordered["q"].to_numpy(), np.arange(len(questions))):
        raise ValueError("query did not answer every question exactly once")
    actual = conversations["id"].take(ordered["c"])
    if actual.to_pylist() != questions["conversation_id"].to_pylist():
        raise ValueError("query evaluated a question against the wrong conversation")
    return ordered


def run_remote(query, directory: Path, endpoint: str):
    """Submit the join to a query service, watch it, and return its result.

    The query id goes to the log and to ``query_id.txt`` beside the
    inputs, so a later run or a browser can reattach to the same query:
    the status page is ``<endpoint>/queries/<id>``.
    """
    run = query.submit(request_key=f"agent-compaction:{directory.name}")
    (directory / "query_id.txt").write_text(run.id)
    print(f"query id: {run.id}", flush=True)
    print(f"status page: {endpoint.rstrip('/')}/queries/{run.id}", flush=True)
    for status in run.watch():
        progress = status.progress or {}
        print(f"{status.state} (revision {status.revision})"
              + (f": {progress.get('label')} {progress.get('done')}"
                 f"/{progress.get('total')} {progress.get('unit')}"
                 if progress else ""), flush=True)
    status = run.status()
    if status.plan is not None:
        print(status.plan["text"], flush=True)
        (directory / "plan.txt").write_text(status.plan["text"])
    return run.result()


def token_lengths(session, name: str, column: str) -> np.ndarray:
    """Token counts of one registered column, from the session or the tokenizer.

    A local session has tokenized the column already. A remote session
    has not, so the column is tokenized here with the model tokenizer.
    """
    exact = session.token_lengths(name, column)
    if exact is None:
        tokenizer = session.tokenizer
        exact = [len(tokenizer(text))
                 for text in session.column_values(name, column).to_pylist()]
    return np.asarray(exact)


def evaluate(directory: Path, gpus: int, endpoint: str | None = None) -> dict:
    """Run the join and save every Boolean decision and its execution report.

    Args:
        directory: The prepared inputs; outputs are written beside them.
        gpus: GPUs the query asks for.
        endpoint: A query service to submit to; None runs on this host.
    """
    import quail
    from quail.frontend.sql import compile_sql
    from quail.specs import MODAL_GPU_USD_PER_HOUR

    conversations = pq.read_table(directory / "conversations", columns=["id"])
    questions = pq.read_table(directory / "tool_questions", columns=[
        "id", "conversation_id", "key", "tool_call_id", "kind"])
    report = {"wall_s": 0.0, "boot_s": 0.0, "fresh_tokens": 0}
    input_tokens = 0
    if len(questions):
        with quail.Session(config=quail.EngineConfig(
                model=MODEL, device=DEVICE, gpus=gpus),
                endpoint=endpoint) as session:
            session.register("conversations", quail.DocumentProvider.from_parquet(
                str(directory / "conversations"), id_col="id"))
            session.register("tool_questions", quail.DocumentProvider.from_parquet(
                str(directory / "tool_questions"), id_col="id"))
            query = session.sql(SQL, dialect="bq")
            if endpoint is None:
                explanation = query.explain()
                print(explanation, flush=True)
                (directory / "plan.txt").write_text(explanation)
                result = query.run()
                logical = query.logical
            else:
                result = run_remote(query, directory, endpoint)
                # the service compiled the query; compile it here too for
                # the prompt token pieces the throughput number needs
                logical = compile_sql(SQL, session.catalog, session.tokenizer,
                                      dialect="bq", turn=session.model.turn)
            answers = complete_answers(result, conversations, questions)
            pq.write_table(result.collect(), directory / "retained.parquet")
            prompt = logical.operators().joins[0].prompt
            pieces = {alias: (label, frame)
                      for alias, label, frame in prompt.label_token_ids}
            overhead = (len(prompt.preamble_token_ids) + len(pieces["c"][1])
                        + len(pieces["q"][0]) + len(prompt.tail_token_ids))
            c_lengths = token_lengths(session, "conversations", "state")
            q_lengths = token_lengths(session, "tool_questions", "statement")
            input_tokens = int(c_lengths[answers["c"].to_numpy()].sum()
                               + q_lengths.sum() + overhead * len(questions))
            decisions = questions.append_column("answer", answers["answer"])
            report = dict(result.report)
    else:
        decisions = questions.append_column("answer", pa.array([], type=pa.bool_()))
    pq.write_table(decisions, directory / "decisions.parquet")
    seconds = report["wall_s"]
    report.update(
        model=MODEL, device=DEVICE, gpus=gpus, input_tokens=input_tokens,
        endpoint=endpoint,
        input_tokens_per_second=input_tokens / seconds if seconds else None,
        evaluated_pairs=len(questions),
        gpu_cost_usd=seconds * gpus * MODAL_GPU_USD_PER_HOUR[DEVICE] / 3600,
        gpu_startup_cost_usd=report.get("boot_s", 0) * gpus
        * MODAL_GPU_USD_PER_HOUR[DEVICE] / 3600,
    )
    (directory / "execution.json").write_text(json.dumps(report, indent=2))
    return report


def reconstruct(directory: Path) -> dict:
    """Apply the original retention rules and save the compacted conversations."""
    started = time.perf_counter()
    decisions = {}
    for row in pq.read_table(directory / "decisions.parquet").to_pylist():
        decisions.setdefault(row["conversation_id"], {})[row["key"]] = row["answer"]
    totals = {"characters_before": 0, "characters_after": 0,
              "keep": 0, "truncate": 0, "drop": 0, "pinned": 0}
    output = directory / "compacted"
    output.mkdir()
    for filename in sorted((directory / "messages").glob("*.parquet")):
        rows = []
        for original in pq.read_table(filename).to_pylist():
            messages = json.loads(original["messages"])
            calls = collect_tool_calls(messages)
            compacted, actions = reconstruct_trace(
                messages, calls, decisions.get(original["id"], {}))
            totals["characters_before"] += message_chars(messages)
            totals["characters_after"] += message_chars(compacted)
            for action in actions:
                totals[action["action"]] += 1
            rows.append({"id": original["id"], "instance_id": original["instance_id"],
                         "repo": original["repo"],
                         "license": original["license"],
                         "messages": json.dumps(compacted),
                         "decisions": json.dumps(actions)})
        pq.write_table(pa.Table.from_pylist(rows), output / filename.name)
    summary = {**json.loads((directory / "inputs.json").read_text()),
               **json.loads((directory / "execution.json").read_text()), **totals,
               "reconstruct_s": time.perf_counter() - started,
               "output_directory": str(directory)}
    before = totals["characters_before"]
    summary["character_reduction_fraction"] = (
        1 - totals["characters_after"] / before if before else 0)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compact OpenHands traces")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=1, choices=[1, 2, 4, 8])
    parser.add_argument("--endpoint", help="submit the join to a query service")
    args = parser.parse_args()

    directory = args.output_dir / uuid.uuid4().hex
    prepare(directory, args.limit, args.seed)
    evaluate(directory, args.gpus, args.endpoint or None)
    reconstruct(directory)

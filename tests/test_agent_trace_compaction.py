"""Sampling, retained messages, and query answer coverage for the demo."""

import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from demos._trace_compaction import (
    build_state,
    collect_tool_calls,
    reconstruct_trace,
    retention_questions,
)
from demos.agent_trace_compaction import (
    SQL,
    complete_answers,
    prepare_tables,
    source_rows,
)
from quail.catalog import Catalog, DocumentProvider
from quail.frontend.sql import compile_sql
from quail.logical import join_conditions


def test_seeded_sampling_and_cached_trajectories(tmp_path, monkeypatch):
    import huggingface_hub
    import huggingface_hub.constants

    files = ["data/first.parquet", "data/second.parquet"]
    originals = []
    for attempt, filename in enumerate(files):
        rows = [{"instance_id": f"issue-{i}", "trajectory_id": f"trace-{i}-{attempt}",
                 "repo": "example/repo", "license": "MIT", "trajectory": [
                     {"role": "user", "content": f"Fix issue {i}, attempt {attempt}"}]}
                for i in range(5)]
        originals.extend(rows)
        pq.write_table(pa.Table.from_pylist(rows), tmp_path / filename.split("/")[-1],
                       row_group_size=2)
    monkeypatch.setattr(huggingface_hub, "HfFileSystem", lambda: SimpleNamespace(
        open=lambda path, **_: (tmp_path / path.rsplit("/", 1)[-1]).open("rb")))
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(
        list_repo_files=lambda *_args, **_kwargs: files))
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE",
                        str(tmp_path / "cache"))
    selected = list(source_rows(limit=5, seed=42))
    assert len(selected) == len({row["instance_id"] for row in selected}) == 5
    assert all(row in originals for row in selected)
    cached = list((tmp_path / "cache").glob("*/*/selected/*/*.parquet"))
    assert sum(pq.read_metadata(path).num_rows for path in cached) == 5
    another_seed = list(source_rows(limit=5, seed=43))
    assert another_seed != selected

    def offline(*_args, **_kwargs):
        raise AssertionError("the selected trajectories should be cached")

    monkeypatch.setattr(huggingface_hub, "HfFileSystem", lambda: SimpleNamespace(
        open=offline))
    monkeypatch.setattr(huggingface_hub, "HfApi", offline)
    assert list(source_rows(limit=5, seed=42)) == selected
    assert list(source_rows(limit=5, seed=43)) == another_seed


def trace():
    def call(call_id):
        return {"id": call_id, "type": "function", "function": {
            "name": "execute_bash", "arguments": '{"command":"ls"}'}}

    return [
        {"role": "user", "content": "Fix the failing test."},
        {"role": "assistant", "content": "old observation " * 30_000,
         "tool_calls": [call("old-call")]},
        {"role": "tool", "content": "🐦" * 300},
        *[{"role": "assistant", "content": f"Recent message {i}"} for i in range(4)],
        {"role": "assistant", "content": "Check the files.",
         "tool_calls": [call("recent-call")]},
        {"role": "tool", "content": "a.py\nb.py\n"},
    ]


def test_compaction_preserves_messages_and_applies_both_retention_decisions():
    source = trace()
    calls = collect_tool_calls(source)
    state, tokens, _ = build_state(source, calls)
    assert tokens <= 25_000
    assert "🐦" not in str(state)
    assert "600 chars" in retention_questions(calls[0])[1]["statement"]
    for keep_call, keep_result, action in (
            (False, False, "drop"), (True, False, "truncate"),
            (False, True, "keep"), (True, True, "keep")):
        compacted, decisions = reconstruct_trace(source, calls, {
            "call_t1": keep_call, "result_t1": keep_result,
        })
        assert decisions == [{"tool_call_id": "old-call", "action": action},
                             {"tool_call_id": "recent-call", "action": "pinned"}]
        assert compacted[-6:] == source[-6:]
        assert compacted[1]["content"] == source[1]["content"]
        if action == "drop":
            assert not compacted[1]["tool_calls"]
            assert len(compacted) == len(source) - 1
        elif action == "truncate":
            assert compacted[2]["content"].startswith("🐦" * 150 + "\n")
        else:
            assert compacted == source
    with pytest.raises(ValueError, match="missing Boolean"):
        reconstruct_trace(source, calls, {"call_t1": False})
    assert source[2]["content"] == "🐦" * 300


def test_demo_join_and_complete_answer_coverage(tmp_path):
    counts = prepare_tables([
        {"instance_id": f"issue-{i}", "trajectory_id": f"trace-{i}",
         "repo": "example/repo", "license": "MIT", "trajectory": trace()}
        for i in range(2)
    ], tmp_path)
    assert counts == {"conversations": 2, "questions": 4, "pinned_calls": 2}
    conversations = pq.read_table(tmp_path / "conversations")
    questions = pq.read_table(tmp_path / "tool_questions")
    saved = pq.read_table(tmp_path / "messages")
    assert json.loads(saved["messages"][0].as_py()) == trace()
    catalog = Catalog()
    for name, table in (("conversations", conversations),
                        ("tool_questions", questions)):
        catalog.register(name, DocumentProvider.from_table(table, id_col="id"))
    plan = compile_sql(SQL, catalog, lambda text: list(text.encode()), dialect="bq")
    assert [str(condition) for condition in join_conditions(plan.root.input)] == [
        "c.id = q.conversation_id"]

    def result(c, q, answers):
        return SimpleNamespace(answer_tables={"joins": {0: pa.table({
            "c": c, "q": q, "answer": answers,
        })}})

    ordered = complete_answers(
        result([1, 0, 1, 0], [3, 0, 2, 1], [False, True, True, False]),
        conversations, questions)
    assert ordered["answer"].to_pylist() == [True, False, True, False]
    for c, q in (([0], [0]), ([0, 0], [0, 0]), ([1, 1, 0, 0], [0, 1, 2, 3])):
        with pytest.raises(ValueError, match="every question|wrong conversation"):
            complete_answers(result(c, q, [False] * len(q)), conversations, questions)

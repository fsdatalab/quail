"""The compaction demo's remote path against an in-process Quail Server."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import server_fakes
from test_session import fake_tok, make_executor

import quail
from quail.server.executor import Hooks

pytest.importorskip("modal")
pytest.importorskip("starlette")
pytest.importorskip("uvicorn")

from demos import agent_trace_compaction as demo  # noqa: E402
from quail.server.app import ServerSettings, create_app  # noqa: E402

hooks = Hooks(
    physical_executor=make_executor({}, {("c", "q"): lambda a, q: q % 2 == 0}),
    tokenizer=fake_tok)


def write_inputs(directory):
    (directory / "conversations").mkdir(parents=True)
    (directory / "tool_questions").mkdir(parents=True)
    # one conversation: the fake executor evaluates every pair the plan
    # lists, so with one conversation the pairs are exactly the four
    # questions the equality join keeps
    pq.write_table(pa.table({
        "id": ["c0"],
        "state": ["state zero " + "word " * 30],
    }), directory / "conversations" / "part.parquet")
    pq.write_table(pa.table({
        "id": ["q0", "q1", "q2", "q3"],
        "conversation_id": ["c0", "c0", "c0", "c0"],
        "key": ["k0", "k1", "k2", "k3"],
        "tool_call_id": ["t0", "t1", "t2", "t3"],
        "kind": ["result", "result", "call", "result"],
        "statement": ["keep the file list", "keep the error",
                      "keep the plan", "keep the diff"],
    }), directory / "tool_questions" / "part.parquet")


def test_evaluate_submits_to_a_server_and_records_the_query_id(
        tmp_path, monkeypatch):
    settings = ServerSettings(
        data_dir=tmp_path / "data", models=(demo.MODEL,), device=demo.DEVICE,
        in_process=True, hooks=hooks)
    url, stop = server_fakes.start_server(create_app(settings))
    # the client tokenizes the columns for the throughput number; use the
    # fake tokenizer instead of downloading the model's
    monkeypatch.setattr(quail.Session, "tokenizer", property(lambda s: fake_tok))
    directory = tmp_path / "run"
    write_inputs(directory)
    try:
        report = demo.evaluate(directory, 1, url)
    finally:
        stop()
    query_id = (directory / "query_id.txt").read_text()
    assert query_id == f"agent-compaction-{directory.name}"
    assert report["endpoint"] == url
    assert report["backend"] == "quail"
    assert report["input_tokens"] > 0
    assert "Scan" in (directory / "plan.txt").read_text()
    decisions = pq.read_table(directory / "decisions.parquet")
    assert decisions.column("id").to_pylist() == ["q0", "q1", "q2", "q3"]
    assert len(decisions.column("answer").to_pylist()) == 4
    retained = pq.read_table(directory / "retained.parquet")
    assert retained.num_rows == decisions.column("answer").to_pylist().count(True)

"""The compaction demo exposes only local execution."""

import inspect
import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quail
from demos import agent_trace_compaction as demo
from quail.specs import MODAL_GPU_USD_PER_HOUR


def test_evaluate_has_no_remote_endpoint():
    assert list(inspect.signature(demo.evaluate).parameters) == [
        "directory",
        "gpus",
    ]


def test_evaluate_uses_public_session_api(monkeypatch, tmp_path):
    conversations = tmp_path / "conversations"
    questions = tmp_path / "tool_questions"
    conversations.mkdir()
    questions.mkdir()
    pq.write_table(
        pa.table({"id": ["conversation"], "state": ["state"]}),
        conversations / "part.parquet",
    )
    pq.write_table(
        pa.table({
            "id": ["question"],
            "conversation_id": ["conversation"],
            "key": ["call_t1"],
            "tool_call_id": ["source"],
            "kind": ["call"],
            "statement": ["keep it"],
        }),
        questions / "part.parquet",
    )

    prompt = SimpleNamespace(
        preamble_token_ids=(1, 2),
        label_token_ids=(
            ("c", (3,), (4, 5)),
            ("q", (6,), (7, 8, 9)),
        ),
        tail_token_ids=(10,),
    )
    logical = SimpleNamespace(
        operators=lambda: SimpleNamespace(
            joins=(SimpleNamespace(prompt=prompt),),
        ),
    )
    answer_table = pa.table({"q": [0], "c": [0], "answer": [True]})
    result = SimpleNamespace(
        answer_tables={"joins": {0: answer_table}},
        collect=lambda: pa.table({"id": ["conversation"]}),
        report={"wall_s": 2.0, "boot_s": 1.0, "fresh_tokens": 30},
    )

    class FakeQuery:
        def __init__(self):
            self.logical = logical

        def explain(self):
            return "plan"

        def run(self):
            return result

    class FakeSession:
        instance = None

        def __init__(self, config):
            self.config = config
            self.registered = []
            self.length_calls = []
            FakeSession.instance = self

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def register(self, name, provider):
            self.registered.append((name, provider))

        def sql(self, sql, dialect):
            assert sql == demo.SQL
            assert dialect == "bq"
            return FakeQuery()

        def token_lengths(self, provider_name, column):
            self.length_calls.append((provider_name, column))
            return {
                ("conversations", "state"): [10],
                ("tool_questions", "statement"): [20],
            }[(provider_name, column)]

    def from_parquet(path, id_col):
        return path, id_col

    monkeypatch.setattr(quail, "Session", FakeSession)
    monkeypatch.setattr(
        quail.DocumentProvider,
        "from_parquet",
        staticmethod(from_parquet),
    )

    report = demo.evaluate(tmp_path, gpus=2)

    session = FakeSession.instance
    assert session.config == quail.EngineConfig(
        model=demo.MODEL,
        device=demo.DEVICE,
        gpus=2,
    )
    assert session.registered == [
        ("conversations", (str(conversations), "id")),
        ("tool_questions", (str(questions), "id")),
    ]
    assert session.length_calls == [
        ("conversations", "state"),
        ("tool_questions", "statement"),
    ]
    assert report["input_tokens"] == 36
    assert report["input_tokens_per_second"] == 18
    assert report["gpu_cost_usd"] == pytest.approx(
        4 * MODAL_GPU_USD_PER_HOUR[demo.DEVICE] / 3600
    )
    assert json.loads((tmp_path / "execution.json").read_text()) == report
    assert pq.read_table(tmp_path / "decisions.parquet")["answer"].to_pylist() == [
        True
    ]

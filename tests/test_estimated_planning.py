"""Planning on length estimates while the token file is written."""

import threading

import pyarrow as pa
from test_session import make_executor

import quail
from quail.execution import execute as execution

DOCS = pa.table({
    "id": [f"d{i}" for i in range(6)],
    "body": ["pad " * (2 * i + 2) for i in range(6)],
})
SQL = ("SELECT d.id FROM docs d WHERE "
       "AI_FILTER(PROMPT('question {0}', d.body), {'selectivity': 0.5})")
TRUTH = {"d": {"question": [1, 0, 1, 0, 1, 0]}}


def _session(tokenizer=str.split):
    session = quail.Session(
        quail.EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
        tokenizer=tokenizer)
    session.register("docs", quail.DocumentProvider.from_table(DOCS, id_col="id"))
    return session


def test_estimated_planning_token_reuse_and_concurrent_boot(monkeypatch):
    lengths = [len(body.split()) for body in DOCS["body"].to_pylist()]
    session = _session()
    estimates = session.estimate_lengths("docs", "body")
    # tokens per byte is constant here, so the estimate is exact
    assert estimates == lengths
    assert session.estimate_lengths("docs", "body") is estimates
    session.close()

    monkeypatch.setattr(execution, "gpu_problem", lambda: None)
    monkeypatch.setattr(execution, "_prepare_backend", lambda *args: None)
    executor = make_executor(TRUTH)
    monkeypatch.setattr(execution, "_execute_physical",
                        lambda request, registry: executor(request))
    session = _session()
    query = session.sql(SQL)
    assert "estimated from a" in query.explain()
    assert query._estimated == ("d",)
    assert sorted(query.run().to_rows()) == [("d0",), ("d2",), ("d4",)]
    assert query.token_wait_s >= 0.0
    assert not query._token_futures
    assert list(session.token_lengths("docs", "body")) == lengths
    again = session.sql(SQL)
    assert "estimated from a" not in again.explain()
    assert again._estimated == ()
    session.close()

    booted = threading.Event()
    order = []

    def waiting_tokenizer(text):
        # the token file cannot finish until the backend has booted; the
        # sample tokenized for the estimate runs on the main thread
        in_background = threading.current_thread().name.startswith(
            "quail-tokenize")
        if in_background and not booted.wait(timeout=10):
            raise AssertionError("boot did not start before tokenizing")
        return text.split()

    def prepare(plan, registry):
        order.append("prepare")
        booted.set()

    def execute(request, registry):
        order.append("execute")
        return executor(request)

    monkeypatch.setattr(execution, "_prepare_backend", prepare)
    monkeypatch.setattr(execution, "_execute_physical", execute)
    session = _session(tokenizer=waiting_tokenizer)
    query = session.sql(SQL)
    assert sorted(query.run().to_rows()) == [("d0",), ("d2",), ("d4",)]
    assert order == ["prepare", "execute"]
    assert query.token_wait_s > 0.0
    session.close()

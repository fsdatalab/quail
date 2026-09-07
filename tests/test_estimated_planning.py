"""Planning on length estimates while the token file is written."""

import threading

import pyarrow as pa
from test_session import make_executor

import quail
from quail.runtime import compute, local

DOCS = pa.table({
    "id": [f"d{i}" for i in range(6)],
    "body": ["pad " * (2 * i + 2) for i in range(6)],
})
SQL = ("SELECT d.id FROM docs d WHERE "
       "AI_FILTER(PROMPT('question {0}', d.body), {'selectivity': 0.5})")
TRUTH = {"d": {"question": [1, 0, 1, 0, 1, 0]}}


def _session(tokenizer=str.split, **kwargs):
    session = quail.Session(tokenizer=tokenizer, **kwargs)
    session.register("docs", quail.DocumentProvider.from_table(DOCS, id_col="id"))
    return session


def test_estimates_scale_byte_lengths_by_a_sample_ratio():
    session = _session()
    estimates = session.estimate_lengths("docs", "body")
    # every document is n copies of "pad ", so tokens per byte is
    # constant and the estimate lands on the exact count
    assert estimates == [len(body.split()) for body in DOCS["body"].to_pylist()]
    assert session.estimate_lengths("docs", "body") is estimates
    session.close()


def test_plan_uses_estimates_and_run_uses_exact_tokens():
    session = _session(
        compute_provider=quail.InProcessComputeProvider(make_executor(TRUTH))
    )
    query = session.sql(SQL)
    text = query.explain()
    assert "estimated from a" in text
    assert query._estimated == ("d",)

    assert sorted(query.run().to_rows()) == [("d0",), ("d2",), ("d4",)]
    assert query.token_wait_s >= 0.0
    assert not query._token_futures
    assert list(session.token_lengths("docs", "body")) == [
        len(body.split()) for body in DOCS["body"].to_pylist()
    ]

    # a second query on the same session plans on the exact counts
    again = session.sql(SQL)
    assert "estimated" not in again.explain()
    assert again._estimated == ()
    session.close()


def test_backend_boots_before_tokens_are_ready(monkeypatch):
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

    executor = make_executor(TRUTH)

    def execute(request, registry):
        order.append("execute")
        return executor(request)

    monkeypatch.setattr(compute, "local_gpu_problem", lambda: None)
    monkeypatch.setattr(local, "_prepare_backend", prepare)
    monkeypatch.setattr(local, "_execute_physical", execute)

    session = _session(tokenizer=waiting_tokenizer)
    query = session.sql(SQL)
    assert sorted(query.run().to_rows()) == [("d0",), ("d2",), ("d4",)]
    assert order == ["prepare", "execute"]
    assert query.token_wait_s > 0.0
    session.close()

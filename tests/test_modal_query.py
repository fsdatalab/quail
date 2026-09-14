"""Whole-query Modal function execution with a fake model."""

import json
from pathlib import Path

import pyarrow as pa
import pytest
from test_session import make_executor

from demos import quickstart, quickstart_modal
from quail.runtime import execute as execution
from quail.runtime.session import Session


def test_quickstarts_return_collected_rows_after_session_closes(monkeypatch, tmp_path):
    reviews = pa.table({
        "id": ["rv0", "rv1"],
        "body": ["The acting was excellent.", "The acting was poor."],
    })
    monkeypatch.setattr(quickstart, "load_reviews", lambda: reviews)
    commits = []
    monkeypatch.setattr(quickstart_modal, "RESULTS_DIR", tmp_path / "chosen-path")
    monkeypatch.setattr(quickstart_modal.volume, "commit",
                        lambda: commits.append("volume"))
    monkeypatch.setattr(Session, "tokenizer", property(lambda self: str.split))
    monkeypatch.setattr(Session, "_fast_tokenizer", lambda self: None)
    monkeypatch.setattr(execution, "gpu_problem", lambda: None)
    monkeypatch.setattr(execution, "_prepare_backend", lambda *args: None)
    executor = make_executor({"r": {"Instruction": [1, 0]}})
    monkeypatch.setattr(execution, "_execute_physical",
                        lambda request, registry: executor(request))
    closed = []
    close = Session.close

    def record_close(session):
        close(session)
        closed.append(session)

    monkeypatch.setattr(Session, "close", record_close)

    for on_modal in [False, True]:
        closed.clear()
        run = (quickstart_modal.run_query.get_raw_f() if on_modal
               else quickstart.run_query)
        rows, report = run()

        assert len(closed) == 1
        assert closed[0]._token_directory is None
        assert rows.to_pylist() == [{"r.id": "rv0"}]
        assert report["fresh_tokens"] == 1234
        assert report["worker_total_s"] >= 0
        if on_modal:
            path = Path(report["result_volume_path"])
            assert path.parent == tmp_path / "chosen-path"
            assert json.loads(path.read_text()) == report
            assert commits == ["volume"]
        else:
            assert "result_volume_path" not in report
            assert commits == []

    def fail():
        raise RuntimeError("query failed")

    commits.clear()
    monkeypatch.setattr(quickstart, "run_query", fail)
    with pytest.raises(RuntimeError, match="query failed"):
        quickstart_modal.run_query.get_raw_f()()
    assert commits == ["volume"]

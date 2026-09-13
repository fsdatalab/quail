"""Whole-query Modal function execution with a fake model."""

import ast
import json
import tomllib
from pathlib import Path

import pyarrow as pa
import pytest
from test_session import make_executor

from demos import quickstart, quickstart_modal
from quail.runtime import execute as execution
from quail.runtime.session import Session


def test_release_dependencies_are_exact_and_match_modal_demo():
    project = tomllib.loads(Path("pyproject.toml").read_text())
    assert project["project"]["name"] == "quail-engine"
    runtime = tuple(
        requirement.split(";", 1)[0].strip()
        for requirement in project["project"]["dependencies"]
        if "sys_platform == 'emscripten' or sys_platform == 'win32'"
        not in requirement
    )
    assert runtime == quickstart_modal.IMAGE_REQUIREMENTS
    assert all(
        "==" in requirement
        for requirement in project["project"]["dependencies"]
    )

    dev = project["dependency-groups"]["dev"]
    pinned_dev = (requirement for requirement in dev if requirement != "quail-b")
    assert all("==" in requirement for requirement in pinned_dev)
    quail_b_source = project["tool"]["uv"]["sources"]["quail-b"]
    assert len(quail_b_source["rev"]) == 40
    assert project["build-system"]["requires"] == ["hatchling==1.32.0"]

    docs_example = ast.parse(
        Path("docs/examples/imdb_queries_modal.py").read_text()
    )
    docs_requirements = next(
        ast.literal_eval(node.value)
        for node in docs_example.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "IMAGE_REQUIREMENTS"
            for target in node.targets
        )
    )
    assert docs_requirements == runtime

    compute_guide = Path(
        "docs/content/docs/user-guide/compute.mdx"
    ).read_text()
    assert all(f'"{requirement}"' in compute_guide for requirement in runtime)


def test_quickstarts_return_collected_rows_after_session_closes(monkeypatch, tmp_path):
    reviews = pa.table({
        "id": ["rv0", "rv1"],
        "body": ["The acting was excellent.", "The acting was poor."],
    })
    loads = []

    def load_table(name, *, limit):
        loads.append((name, limit))
        return reviews

    monkeypatch.setattr(quickstart.benchmark, "load_table", load_table)
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

    assert loads == [("reviews", 100), ("reviews", 100)]

    def fail():
        raise RuntimeError("query failed")

    commits.clear()
    monkeypatch.setattr(quickstart, "run_query", fail)
    with pytest.raises(RuntimeError, match="query failed"):
        quickstart_modal.run_query.get_raw_f()()
    assert commits == ["volume"]

"""Saved and remote results preserve every model answer table."""

import io
import json
from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal

import pyarrow as pa
import pytest
from helpers import server_fakes

from quail.execution.result import QueryResult, extract_answers, filter_answer_table
from quail.remote.artifacts import load_result, verify_result, write_result
from quail.remote.client import QueryRun, ServerClient


@pytest.fixture()
def result():
    result = QueryResult.from_table(pa.table({"id": [1]}))
    result.answer_tables = {
        "filters": {("r", 0): filter_answer_table("r", [0, 1], [0, 0], [True, False])},
        "joins": {3: pa.table({"r": [0], "p": [1], "answer": [True]})},
        "classifies": {
            name: pa.table({"r": [0, 1],
                            name: pa.array(["a", "b"]).dictionary_encode()})
            for name in ("topic/a", "topic_a", "Topic", "topic", "x" * 300)},
        "scores": {"score": pa.table({"r": [0, 1], "score": [0.8, 0.2]})},
        "extracts": {"lease.ends": extract_answers(
            "r", "lease.ends", "lease.ends.span", pa.array([0, 1], pa.int32()),
            ["March 3", None], [(10, 17), None]).table()},
    }
    return result


def assert_same_result(actual, expected):
    assert actual.collect().equals(expected.collect(), check_metadata=True)
    assert actual.answer_tables.keys() == expected.answer_tables.keys()
    for kind, tables in expected.answer_tables.items():
        assert actual.answer_tables[kind].keys() == tables.keys()
        for key, table in tables.items():
            assert actual.answer_tables[kind][key].equals(table, check_metadata=True)


def test_saved_result_preserves_all_answer_tables(result, tmp_path):
    manifest = json.loads(json.dumps(write_result(result, tmp_path)))
    verify_result(tmp_path, manifest)
    assert_same_result(load_result(tmp_path, manifest), result)
    files = [entry["file"] for entries in manifest["files"]["answers"].values()
             for entry in entries]
    assert len({name.casefold() for name in files}) == len(files)


@pytest.fixture()
def saved_result(tmp_path):
    pytest.importorskip("starlette")
    from starlette.testclient import TestClient

    from quail.server.app import ServerSettings, create_app

    @contextmanager
    def serve(result):
        app = create_app(ServerSettings(
            data_dir=tmp_path, models=("qwen3-4b-fp8",), device="h100-sxm",
            in_process=True, hooks=server_fakes.hooks))
        server = app.state.server
        status = server.store.create(
            spec={"sql": server_fakes.FILTER_SQL}, config=server_fakes.CONFIG,
            inputs={}, timeout_s=60)
        epoch = server.store.begin(status.id)
        directory = server.scheduler.result_directory(status.id)
        manifest = write_result(result, directory)
        server.store.finish(status.id, epoch, "succeeded", result={
            **manifest, "directory": str(directory.relative_to(tmp_path))})
        with TestClient(app) as http:
            yield http, status.id

    return serve


def test_remote_result_preserves_all_answer_tables(result, saved_result):
    with saved_result(result) as (http, query_id):
        def open_response(method, path, **_kwargs):
            response = http.request(method, path)
            response.raise_for_status()
            return io.BytesIO(response.content)

        client = ServerClient("http://testserver")
        client._open = open_response
        assert_same_result(QueryRun(client, query_id).result(), result)


def test_preview_formats_arrow_values_without_losing_precision(saved_result):
    table = pa.table({
        "id": [2**53 + 1, None],
        "date": [date(2026, 10, 17), None],
        "time": [datetime(2026, 10, 17, tzinfo=timezone.utc), None],
        "amount": [Decimal("123456789.0123456789"), None],
        "number": [float("nan"), float("inf")],
        "bytes": [b"\xff", None],
        "span": [{"start": 2, "end": 5}, None],
    })
    with saved_result(QueryResult.from_table(table)) as (http, query_id):
        response = http.get(f"/v1/queries/{query_id}/rows?limit=1")
        assert response.status_code == 200
        assert response.json() == {
            "columns": table.column_names, "total_rows": 2,
            "rows": [["9007199254740993", "2026-10-17",
                      "2026-10-17 00:00:00+00:00", "123456789.0123456789",
                      "nan", "b'\\xff'", "{'start': 2, 'end': 5}"]],
        }
        rows = http.get(f"/v1/queries/{query_id}/rows?limit=2").json()["rows"]
        assert rows[1] == [None, None, None, None, "inf", None, None]
        payload = http.get(f"/v1/queries/{query_id}/result").content
        with pa.ipc.open_stream(payload) as reader:
            actual = reader.read_all()
        assert actual.schema == table.schema
        assert actual.drop(["number"]).equals(table.drop(["number"]))


def test_saved_result_preview_preserves_rows_with_no_columns(saved_result):
    table = pa.table({"id": [1, 2, 3]}).select([])
    with saved_result(QueryResult.from_table(table)) as (http, query_id):
        response = http.get(f"/v1/queries/{query_id}/rows?limit=2")
        assert response.status_code == 200
        assert response.json() == {"columns": [], "rows": [[], []], "total_rows": 3}


@pytest.mark.parametrize("kind", ["classifies", "scores", "extracts"])
def test_verification_reads_all_answer_files(result, tmp_path, kind):
    manifest = write_result(result, tmp_path)
    entry = manifest["files"]["answers"][kind][0]
    (tmp_path / entry["file"]).unlink()
    with pytest.raises(FileNotFoundError):
        verify_result(tmp_path, manifest)

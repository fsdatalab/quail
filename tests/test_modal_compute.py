"""Modal Function compute provider tests."""

from dataclasses import replace

import pyarrow as pa
import pytest

import quail
from quail.logical import ColumnRef, LogicalPlan, Project, Scan
from quail.planner.plan import EngineConfig
from quail.runtime.compute import QueryRequest, _modal_request
from quail.runtime.result import QueryResult


def _logical_plan() -> LogicalPlan:
    scan = Scan("docs", "d", "body")
    return LogicalPlan(Project(
        scan,
        (ColumnRef("d", "docs", "id"),),
    ))


def _request() -> QueryRequest:
    provider = quail.DocumentProvider.from_table(
        pa.table({
            "id": ["a", "b", "c"],
            "body": ["one", "two", "three"],
            "unused": [1, 2, 3],
        }),
        id_col="id",
    )
    return QueryRequest(
        logical_plan=_logical_plan(),
        providers={"docs": provider},
        config=EngineConfig(),
    )


def test_modal_rejects_rtx_before_allocating_an_h100(monkeypatch):
    provider = quail.ModalComputeProvider()

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("allocated an H100 for an RTX plan")

    monkeypatch.setattr(provider, "_function", unexpected_allocation)
    request = _request()
    request = replace(request, config=replace(
        request.config, device="rtx-pro-6000-blackwell-server"))
    with pytest.raises(ValueError, match="provisions H100s only"):
        provider.execute(request)


def test_modal_request_leaves_remote_source_on_worker():
    class RemoteProvider:
        id_col = "id"
        columns = ("id", "body")

        def remote_source(self):
            return {
                "type": "parquet",
                "paths": ["s3://bucket/docs.parquet"],
                "id_col": "id",
            }

        def scan(self, request):
            raise AssertionError("the client scanned a remote source")

    request = QueryRequest(
        logical_plan=_logical_plan(),
        providers={"docs": RemoteProvider()},
        config=EngineConfig(),
    )

    prepared = _modal_request(request)

    assert prepared["sources"] == {"docs": {"remote": {
        "type": "parquet",
        "paths": ["s3://bucket/docs.parquet"],
        "id_col": "id",
    }}}


def test_modal_provider_calls_function_and_releases_minimum_container(
    monkeypatch, capsys
):
    from quail.runtime import worker

    autoscaler = []
    closed = []
    submitted = []

    class AppContext:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            closed.append("app")

    class App:
        def run(self, detach=False):
            return AppContext()

    class Call:
        object_id = "fc-test"

        def get(self):
            return QueryResult.from_table(
                pa.table({"ok": [True]}),
                {"wall_s": 1.0, "fresh_tokens": 2},
            )

    class Function:
        def update_autoscaler(self, **settings):
            autoscaler.append(settings)

        def spawn(self, request, initialize_worker):
            assert initialize_worker is None
            submitted.append(request)
            return Call()

    function = Function()

    class Worker:
        app = App()

        def function(self, gpu_count):
            assert gpu_count == 1
            return function

    monkeypatch.setattr(worker, "modal_worker", lambda **kwargs: Worker())

    provider = quail.ModalComputeProvider()
    result = provider.execute(_request())
    provider.close()

    assert result.collect().to_pydict() == {"ok": [True]}
    assert isinstance(result, QueryResult)
    assert submitted[0]["sources"]["docs"]["table"].column_names == [
        "id", "body"
    ]
    assert autoscaler == [
        {"min_containers": 1},
        {"min_containers": 0},
    ]
    assert closed == ["app"]
    assert "function call id: fc-test" in capsys.readouterr().out

"""Default compute provider tests."""

from dataclasses import replace

import pyarrow as pa
import pytest
from test_session import make_executor

import quail
from quail.runtime import compute


def _tokens(text):
    return text.split()


def _session(**kwargs):
    session = quail.Session(tokenizer=_tokens, **kwargs)
    session.register("docs", quail.DocumentProvider.from_table(
        pa.table({"id": ["a", "b"], "body": ["one two", "three four"]}),
        id_col="id",
    ))
    return session


FILTER_SQL = (
    "SELECT d.id FROM docs d WHERE AI_FILTER(PROMPT('question {0}', d.body))"
)


def test_session_defaults_to_in_process_compute():
    session = _session()
    assert isinstance(session.compute_provider, quail.InProcessComputeProvider)
    session.close()


def test_in_process_provider_names_modal_when_no_gpu(monkeypatch):
    monkeypatch.setattr(
        compute, "local_gpu_problem", lambda: "no CUDA GPU is visible"
    )
    session = _session()
    with pytest.raises(RuntimeError, match="ModalComputeProvider") as error:
        session.sql(FILTER_SQL).run()
    assert "no CUDA GPU is visible" in str(error.value)
    session.close()


def test_fake_physical_executor_skips_gpu_check(monkeypatch):
    def fail():
        raise AssertionError("the GPU check must not run with a fake executor")

    monkeypatch.setattr(compute, "local_gpu_problem", fail)
    executor = make_executor({"d": {"question": [1, 0]}})
    session = _session(
        compute_provider=quail.InProcessComputeProvider(executor)
    )
    result = session.sql(FILTER_SQL).run()
    assert result.to_rows() == [("a",)]
    session.close()


def test_in_process_run_reuses_the_planned_query(monkeypatch):
    from quail.runtime import local

    session = _session(
        compute_provider=quail.InProcessComputeProvider(
            make_executor({"d": {"question": [1, 0]}})
        ),
    )
    query = session.sql(FILTER_SQL)
    query.explain()
    query.wait_for_tokens()
    stores_after_plan = dict(session._token_stores)

    def no_second_session(*args, **kwargs):
        raise AssertionError("run() must not build a worker session")

    # a worker session would tokenize the documents a second time
    monkeypatch.setattr(local, "Session", no_second_session)
    assert query.run().to_rows() == [("a",)]
    assert session._token_stores == stores_after_plan
    session.close()


def test_device_config_reaches_planning_and_query_request(monkeypatch):
    from quail.planning import SupportResult
    from quail.specs import H100_SXM

    registry = quail.ExtensionRegistry.with_built_ins()
    device = replace(H100_SXM, name="test-h100")
    registry.register_device(device)
    selected = []

    def supports(model, hardware, gpu_count):
        selected.append((hardware, gpu_count))
        return SupportResult.accept()

    monkeypatch.setattr(registry.backend("quail"), "supports", supports)
    config = quail.EngineConfig(gpus=1, device=device.name)
    with _session(config=config, registry=registry) as session:
        query = session.sql(FILTER_SQL)
        assert session.device is device
        assert query.plan().device == device.name
        assert query._request().config == config
        assert selected and all(item == (device, 1) for item in selected)


def test_default_device_is_h100():
    with _session() as session:
        assert session.config.device == "h100-sxm"
        assert session.device.name == session.config.device


def test_unknown_device_in_config_is_rejected():
    with pytest.raises(ValueError, match="missing-device"):
        _session(config=quail.EngineConfig(device="missing-device"))

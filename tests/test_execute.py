"""In-process query execution tests."""

import subprocess
import sys
from dataclasses import replace

import pyarrow as pa
import pytest
from test_session import make_executor

import quail
from quail.runtime import execute as execution


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

IMPORT_TEXT = """
import sys
import quail
from quail.bench import quailb
from quail.runtime import execute
from demos import quickstart

assert "modal" not in sys.modules
"""


def test_engine_import_and_gpu_requirement(monkeypatch):
    subprocess.run([sys.executable, "-c", IMPORT_TEXT], check=True)

    with monkeypatch.context() as patch:
        patch.setattr(
            execution, "gpu_problem", lambda: "no CUDA GPU is visible"
        )
        session = _session()
        with pytest.raises(RuntimeError, match="process with a CUDA GPU") as error:
            session.sql(FILTER_SQL).run()
        assert "no CUDA GPU is visible" in str(error.value)
        session.close()


def test_query_reuses_plan_and_device(monkeypatch):
    with monkeypatch.context() as patch:
        from quail.runtime import session as session_module

        patch.setattr(execution, "gpu_problem", lambda: None)
        patch.setattr(execution, "_prepare_backend", lambda *args: None)
        executor = make_executor({"d": {"question": [1, 0]}})
        patch.setattr(execution, "_execute_physical",
                            lambda request, registry: executor(request))
        session = _session()
        query = session.sql(FILTER_SQL)
        query.explain()
        query.wait_for_tokens()
        stores_after_plan = dict(session._token_stores)

        def no_second_session(*args, **kwargs):
            raise AssertionError("run() must reuse its existing session")

        patch.setattr(session_module, "Session", no_second_session)
        assert query.run().to_rows() == [("a",)]
        assert session._token_stores == stores_after_plan
        session.close()

    with monkeypatch.context() as patch:
        from quail.planning import SupportResult
        from quail.specs import H100_SXM

        registry = quail.ExtensionRegistry.with_built_ins()
        device = replace(H100_SXM, name="test-h100")
        registry.register_device(device)
        selected = []

        def supports(model, hardware, gpu_count):
            selected.append((hardware, gpu_count))
            return SupportResult.accept()

        patch.setattr(registry.backend("quail"), "supports", supports)
        config = quail.EngineConfig(gpus=1, device=device.name)
        with _session(config=config, registry=registry) as session:
            query = session.sql(FILTER_SQL)
            assert session.device is device
            assert query.plan().device == device.name
            assert selected and all(item == (device, 1) for item in selected)

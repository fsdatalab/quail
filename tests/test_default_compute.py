"""Default compute provider tests."""

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

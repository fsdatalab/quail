"""Local query execution and engine switching on one H100 through Modal."""

import gc
import os
import subprocess
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pyarrow as pa
import pytest

import quail
from quail.remote.artifacts import write_result

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a CUDA device")

BODIES = [
    "The item is blue and round.",
    "The item is blue and square.",
    "The item is red and round.",
    "The item is red and square.",
]


def _session(backend, model="qwen3-4b-fp8", gpus=1):
    session = quail.Session(quail.EngineConfig(
        model=model, device="h100-sxm", backend=backend, gpus=gpus))
    session.register("items", quail.DocumentProvider.from_table(pa.table({
        "id": [f"d{index}" for index in range(len(BODIES))],
        "body": BODIES,
    }), id_col="id"))
    session.register("colors", quail.DocumentProvider.from_table(pa.table({
        "id": ["blue", "red"], "body": ["The color is blue.", "The color is red."],
    }), id_col="id"))
    return session


def _filters(session):
    return (session.docs("items").alias("d")
            .ai_filter(quail.prompt("Does {0} say the item is blue?",
                                    quail.col("d.body")), selectivity=0.5)
            .ai_filter(quail.prompt("Does {0} say the item is round?",
                                    quail.col("d.body")), selectivity=0.5))


def _save(result, name):
    destination = Path(os.environ["QUAIL_TEST_RESULTS"]) / name
    write_result(result, destination)
    print(f"{name}: {result.report['boot_kind']}, "
          f"query {result.report['wall_s']} s, "
          f"startup {result.report['boot_s']} s, "
          f"rows {result.count()}, saved {destination}", flush=True)


def test_local_execution_plugins_in_fresh_process():
    plugin_test_text = """
from importlib.metadata import entry_points
from vllm.plugins import load_general_plugins
from vllm import ModelRegistry
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers import TokenizerRegistry

plugin, = entry_points(group="vllm.general_plugins", name="quail")
load_general_plugins()
plugin.load()()
assert TokenizerRegistry.tokenizers["gigatoken"] == (
    "quail.baselines.vllm", "GigatokenVLLMTokenizer")
assert RENDERER_REGISTRY.renderers["gigatoken"] == (
    "vllm.renderers.hf", "HfRenderer")
assert {"Qwen3DecisionModel", "Decision2ForCausalLM"} <= set(
    ModelRegistry.get_supported_archs())
"""
    subprocess.run([sys.executable, "-c", plugin_test_text], check=True, timeout=120)


def test_local_execution_diffusion_to_qwen():
    from vllm import envs
    from vllm.v1.engine.core_client import InprocClient

    from quail.api.execute import _BACKEND_STATE

    name = "VLLM_ENABLE_V1_MULTIPROCESSING"
    previous = os.environ.get(name)
    previous_mode = envs.VLLM_ENABLE_V1_MULTIPROCESSING
    models = []
    try:
        for model, in_process in (("diffusion-gemma-26b-a4b-fp8", True),
                                  ("qwen3-4b-fp8", not previous_mode)):
            with _session("stock_vllm", model=model) as session:
                result = _filters(session).select("d.id").run()
                _save(result, f"switch-model-{model}")
                assert result.report["boot_kind"] == "cold"
                assert envs.VLLM_ENABLE_V1_MULTIPROCESSING is not in_process
                assert isinstance(
                    _BACKEND_STATE.state["engine"].settings[
                        "client"].llm.llm_engine.engine_core,
                    InprocClient) is in_process
                assert all(reference() is None for reference in models)
                if in_process:
                    models.append(weakref.ref(
                        _BACKEND_STATE.state["engine"].settings[
                            "client"].llm.llm_engine.model_executor.driver_worker
                        .worker.model_runner.model))
                warm = _filters(session).select("d.id").run()
                _save(warm, f"switch-model-{model}-warm")
                assert warm.report["boot_kind"] == "warm"
                assert sorted(warm.to_rows()) == sorted(result.to_rows())
    finally:
        _BACKEND_STATE.close()
    assert os.environ.get(name) == previous
    assert envs.VLLM_ENABLE_V1_MULTIPROCESSING == previous_mode
    assert all(reference() is None for reference in models)


def test_local_execution_reuses_and_switches_engines():
    from quail.api.execute import _BACKEND_STATE

    try:
        with _session("quail") as session:
            first = _filters(session).select("d.id").run()
            _save(first, "quail-cold")
            assert first.report["boot_kind"] == "cold"
            expected = sorted(first.to_rows())
            assert expected
            second = _filters(session).select("d.id").run()
            _save(second, "quail-warm")
            assert second.report["boot_kind"] == "warm"
            assert sorted(second.to_rows()) == expected

            classified = _filters(session).ai_classify(
                quail.prompt("What shape does {0} describe?", quail.col("d.body")),
                ["round", "square"], name="shape", probabilities=True)
            joined = classified.ai_join(
                session.docs("colors").alias("c"),
                quail.prompt("Do {0} and {1} describe the same color?",
                             quail.col("d.body"), quail.col("c.body")),
                selectivity=0.5).select("d.id", "c.id", "shape").run()
            _save(joined, "quail-classify-join")
            answers = {row["d"]: row for row in
                       joined.answer_tables["classifies"]["shape"].to_pylist()}
            rows = joined.collect().to_pylist()
            assert rows
            for row in rows:
                answer = answers[int(row["d.id"][1:])]
                assert row["shape"] == answer["shape"]
                assert row["shape_probabilities"] == answer["shape_probabilities"]

        for backend, boot in (("stock_vllm", "cold"),
                              ("pipelined_vllm", "warm"), ("quail", "cold")):
            with _session(backend) as session:
                result = _filters(session).select("d.id").run()
                _save(result, f"switch-{backend}")
                assert result.report["boot_kind"] == boot
                assert sorted(result.to_rows()) == expected
    finally:
        _BACKEND_STATE.close()


def _keep_ids(rows):
    return rows["d"]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_local_execution_across_gpu_counts():
    from quail.api.execute import _BACKEND_STATE
    from quail.backends.quail import worker

    expected = {}
    previous = []
    killed = None
    try:
        for gpus in (1, 2, 1, 2):
            with _session("quail", gpus=gpus) as session:
                queries = {
                    "filters": _filters(session).select("d.id"),
                    "classify": session.docs("items").alias("d").ai_classify(
                        quail.prompt("What shape does {0} describe?",
                                     quail.col("d.body")),
                        ["round", "square"], name="shape", probabilities=True
                    ).select("d.id", "shape"),
                    "extract": session.sql(
                        "SELECT d.id, AI.EXTRACT(d.body, "
                        "{'color': 'What color is the item?', "
                        "'shape': 'What shape is the item?'}) AS item "
                        "FROM items d WHERE item.color IS NOT NULL",
                        order="as_written"),
                    "join": _filters(session).ai_join(
                        session.docs("colors").alias("c"),
                        quail.prompt("Do {0} and {1} describe the same color?",
                                     quail.col("d.body"), quail.col("c.body")),
                        selectivity=0.5).select("d.id", "c.id"),
                    "paired_classify": session.sql(
                        "SELECT d.id, c.id, AI.CLASSIFY(PROMPT("
                        "'Is the color in {0} the color in {1}?', d.body, c.body), "
                        "ARRAY['same', 'different']) AS match FROM items d "
                        "JOIN colors c ON AI_FILTER(PROMPT("
                        "'Do {0} and {1} describe the same color?', d.body, c.body))"),
                    "scores": session.docs("items").alias("d").ai_score(
                        quail.prompt("Does {0} say the item is blue?",
                                     quail.col("d.body")), name="score"
                    ).select("d.id", "score"),
                    "distinct": _filters(session).distinct().select("d.id"),
                    "limit": _filters(session).limit(1).select("d.id"),
                    "apply": _filters(session).apply(
                        _keep_ids, columns=[quail.col("d.id")], ids="preserve"
                    ).select("d.id"),
                    "empty": session.docs("items").alias("d").where(
                        quail.col("d.id") == "absent").ai_classify(
                            quail.prompt("What shape does {0} describe?",
                                         quail.col("d.body")),
                            ["round", "square"], name="shape"
                        ).select("d.id", "shape"),
                }
                for index, (name, query) in enumerate(queries.items()):
                    result = query.run()
                    _save(result, f"gpus-{gpus}-{len(previous)}-{name}")
                    assert result.report["boot_kind"] == (
                        "cold" if index == 0 else "warm")
                    rows = sorted(result.to_rows(), key=lambda row: repr(row[:2]))
                    if name == "classify":
                        rows = [(row[0], row[1]) for row in rows]
                        for probabilities in result.collect()[
                                "shape_probabilities"].to_pylist():
                            assert sum(dict(probabilities).values()) == pytest.approx(1)
                    if name == "scores":
                        assert dict(rows) == pytest.approx(
                            dict(expected.setdefault(name, rows)), abs=0.01)
                    else:
                        assert rows == expected.setdefault(name, rows)
                    assert not any(process.is_alive() for process in previous)
                if gpus > 1:
                    children = [process for process, _ in worker._WORKERS.children]
                    assert len(children) == gpus and all(
                        process.is_alive() for process in children)
                    previous.extend(children)
                    if len(previous) == 2:
                        killed = children[0]
                        children[0].kill()
                        children[0].join()
                        recovered = queries["filters"].run()
                        _save(recovered, "gpus-2-after-worker-exit")
                        assert recovered.report["boot_kind"] == "cold"
                        assert sorted(recovered.to_rows()) == expected["filters"]
                        assert not any(process.is_alive() for process in previous)
                        previous.extend(
                            process for process, _ in worker._WORKERS.children)
    finally:
        _BACKEND_STATE.close()
    assert not any(process.is_alive() for process in previous)
    assert all(process.exitcode == 0 for process in previous if process is not killed)


def test_local_execution_recovers_after_failed_startup(monkeypatch):
    from quail.api.execute import _BACKEND_STATE
    from quail.baselines.vllm import VLLMClient

    shutdowns = []
    close = VLLMClient.close

    def fail_warmup(self, *args, **kwargs):
        raise RuntimeError("injected warmup failure")

    def shutdown(self):
        close(self)
        shutdowns.append(True)

    try:
        with _session("stock_vllm") as session:
            with monkeypatch.context() as startup:
                startup.setattr(VLLMClient, "generate", fail_warmup)
                startup.setattr(VLLMClient, "close", shutdown)
                with pytest.raises(RuntimeError, match="injected warmup failure"):
                    _filters(session).select("d.id").run()
            assert shutdowns == [True]
            assert not _BACKEND_STATE.state
            print("failed startup shut down its engine", flush=True)

            result = _filters(session).select("d.id").run()
            _save(result, "retry-after-failed-startup")
            assert result.report["boot_kind"] == "cold"
            expected = sorted(result.to_rows())
            assert expected
            warm = _filters(session).select("d.id").run()
            _save(warm, "warm-after-retry")
            assert warm.report["boot_kind"] == "warm"
            assert sorted(warm.to_rows()) == expected
        with _session("quail") as session:
            result = _filters(session).select("d.id").run()
            _save(result, "quail-after-retry")
            assert result.report["boot_kind"] == "cold"
            assert sorted(result.to_rows()) == expected
    finally:
        _BACKEND_STATE.close()


def test_local_execution_releases_failed_diffusion_with_traceback(monkeypatch):
    from quail.api.execute import _BACKEND_STATE
    from quail.baselines.vllm import VLLMClient

    models = []

    def fail_warmup(self, *args, **kwargs):
        models.append(weakref.ref(
            self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model))
        raise RuntimeError("injected warmup failure")

    try:
        with _session("stock_vllm", "diffusion-gemma-26b-a4b-fp8") as session:
            with monkeypatch.context() as startup:
                startup.setattr(VLLMClient, "generate", fail_warmup)
                with pytest.raises(
                        RuntimeError, match="injected warmup failure") as saved:
                    _filters(session).select("d.id").run()
            gc.collect()
            assert not _BACKEND_STATE.state
            assert models and all(reference() is None for reference in models)
        with _session("stock_vllm") as session:
            result = _filters(session).select("d.id").run()
            _save(result, "qwen-after-failed-diffusion")
            assert result.report["boot_kind"] == "cold"
            assert sorted(result.to_rows()) == [("d0",)]
        assert saved.value.__traceback__ is not None
    finally:
        _BACKEND_STATE.close()


@pytest.mark.parametrize("phase", ["arena", "warm"])
def test_local_execution_releases_failed_quail_with_traceback(monkeypatch, phase):
    from quail.api.execute import _BACKEND_STATE
    from quail.backends.quail import worker

    models = []
    load_model = worker.load_model

    def load(*args, **kwargs):
        model = load_model(*args, **kwargs)
        models.append(weakref.ref(model))
        return model

    def fail(*args, **kwargs):
        raise RuntimeError("injected startup failure")

    try:
        with _session("quail") as session:
            with monkeypatch.context() as startup:
                startup.setattr(worker, "load_model", load)
                if phase == "arena":
                    startup.setattr(worker, "KVArena", fail)
                else:
                    startup.setattr(worker.LoadedGpu, "warm", fail)
                with pytest.raises(
                        RuntimeError, match="injected startup failure") as saved:
                    _filters(session).select("d.id").run()
            assert not _BACKEND_STATE.state
            assert models and all(reference() is None for reference in models)
            retry = _filters(session).select("d.id").run()
            _save(retry, f"quail-after-failed-{phase}")
            assert retry.report["boot_kind"] == "cold"
            assert sorted(retry.to_rows()) == [("d0",)]
            warm = _filters(session).select("d.id").run()
            _save(warm, f"quail-warm-after-failed-{phase}")
            assert warm.report["boot_kind"] == "warm"
            assert sorted(warm.to_rows()) == [("d0",)]
        with _session("stock_vllm") as session:
            result = _filters(session).select("d.id").run()
            _save(result, f"vllm-after-failed-quail-{phase}")
            assert result.report["boot_kind"] == "cold"
            assert sorted(result.to_rows()) == [("d0",)]
        assert models and all(reference() is None for reference in models)
        assert saved.value.__traceback__ is not None
    finally:
        _BACKEND_STATE.close()


def test_local_execution_reuses_engine_after_failed_submission(monkeypatch):
    from quail.api.execute import _BACKEND_STATE

    try:
        with _session("pipelined_vllm") as session:
            first = _filters(session).select("d.id").run()
            _save(first, "before-failed-submission")
            engine = _BACKEND_STATE.state["engine"].settings["client"].llm.llm_engine
            add_request = engine.add_request
            submitted = []

            def fail_submission(request_id, *args, **kwargs):
                result = add_request(request_id, *args, **kwargs)
                submitted.append(request_id)
                if len(submitted) == 2:
                    raise RuntimeError("injected submission failure")
                return result

            with monkeypatch.context() as submission:
                submission.setattr(engine, "add_request", fail_submission)
                with pytest.raises(RuntimeError, match="injected submission failure"):
                    _filters(session).select("d.id").run()
            assert len(submitted) == 2
            assert engine.get_num_unfinished_requests() == 0
            assert engine.reset_prefix_cache()
            retry = _filters(session).select("d.id").run()
            _save(retry, "after-failed-submission")
            assert retry.report["boot_kind"] == "warm"
            assert sorted(retry.to_rows()) == sorted(first.to_rows()) == [("d0",)]
    finally:
        _BACKEND_STATE.close()


@pytest.mark.parametrize("backend", ["quail", "stock_vllm"])
def test_local_execution_serializes_concurrent_queries(backend):
    from quail.api.execute import _BACKEND_STATE

    start = Barrier(2)

    def run(session):
        start.wait(timeout=30)
        return _filters(session).select("d.id").run()

    try:
        with _session(backend) as first, _session(backend) as second:
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(run, (first, second)))
            for index, result in enumerate(results):
                _save(result, f"concurrent-{backend}-{index}")
            assert sorted(result.report["boot_kind"] for result in results) == [
                "cold", "warm"]
            expected = sorted(results[0].to_rows())
            assert expected
            assert sorted(results[1].to_rows()) == expected
    finally:
        _BACKEND_STATE.close()

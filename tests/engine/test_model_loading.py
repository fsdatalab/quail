"""CPU tests for model files, retained answer weights, and readouts."""

import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from quail.backends.quail.executor import model_loading
from quail.backends.quail.executor.model_loading import (
    answer_weights,
    retain_answer_head,
)
from quail.backends.quail.executor.readout import AnswerRows, AsyncAnswers, AsyncScores
from quail.builtins import built_in_registry


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def _model(torch, tied=False):
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.embed_tokens = torch.nn.Embedding(8, 4, dtype=torch.bfloat16)
    model.lm_head = (model.model.embed_tokens if tied else
                     torch.nn.Linear(4, 8, bias=False, dtype=torch.bfloat16))
    return model


def test_retained_answer_weights_ownership_and_answer_rows(torch):
    for tied in [False, True]:
        model = _model(torch, tied)
        embedding = model.model.embed_tokens.weight
        reference = weakref.ref(model.lm_head.weight)
        expected = model.lm_head.weight.detach()[[1, 3, 5]].clone()
        retain_answer_head(torch, model, [5, 1, 3, 1])
        gc.collect()
        assert model.lm_head is None
        assert (reference() is not None) == tied
        assert model.model.embed_tokens.weight is embedding
        assert model.quail_answer_token_ids == (1, 3, 5)
        assert torch.equal(model.quail_answer_weights, expected)
        assert (model.quail_answer_weights.untyped_storage().nbytes()
                == expected.numel() * 2)
        assert not model.quail_answer_weights.requires_grad

    model = _model(torch)
    retain_answer_head(torch, model, [1, 3, 5])
    weights = answer_weights(model, [1, 3, 5])
    retain_answer_head(torch, model, [5, 3, 1])
    assert answer_weights(model, [1, 3, 5]) is weights
    with pytest.raises(ValueError, match="differ"):
        answer_weights(model, [1, 3, 6])
    assert answer_weights(model, [1, 3, 5]) is weights

    model = _model(torch)
    original = model.lm_head
    with pytest.raises(ValueError, match="empty"):
        retain_answer_head(torch, model, [])
    assert model.lm_head is original

    # answer rows preserve the scores and share the retained weights
    model = _model(torch)
    hidden = torch.tensor([[1, 2, -1, 0], [0, 1, 3, -2]], dtype=torch.bfloat16)
    full_scores = torch.nn.functional.linear(hidden, model.lm_head.weight)
    expected = (full_scores[:, [1, 3]].amax(1) > full_scores[:, 5]).int().tolist()
    retain_answer_head(torch, model, [1, 3, 5])
    cpu_torch = SimpleNamespace(tensor=lambda values, **kwargs: torch.tensor(values))

    first = AnswerRows(cpu_torch, torch.nn.functional, model, [1, 3], [5])
    second = AnswerRows(cpu_torch, torch.nn.functional, model, [3, 1], [5])
    assert first.weights is second.weights is model.quail_answer_weights
    assert first(hidden) == second(hidden) == expected
    assert AsyncAnswers.dtype is None
    assert AsyncScores.dtype is np.float32


@pytest.fixture
def clear_model_paths():
    model_loading.resolve_model_path.cache_clear()
    yield
    model_loading.resolve_model_path.cache_clear()


def test_model_files_are_resolved_once_and_cached_before_children_start(
        monkeypatch, tmp_path, clear_model_paths):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / kwargs["revision"])

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    first = model_loading.resolve_model_path("Qwen/model", "revision-a")
    assert model_loading.resolve_model_path("Qwen/model", "revision-a") == first
    assert model_loading.resolve_model_path("Qwen/model", "revision-b") != first
    assert [call["revision"] for call in calls] == ["revision-a", "revision-b"]
    assert all(call["repo_id"] == "Qwen/model" for call in calls)
    assert "*.safetensors" in calls[0]["allow_patterns"]
    # a tokenizer-only download is its own cache entry without the weights
    assert model_loading.resolve_model_path(
        "Qwen/model", "revision-a", include_weights=False) == first
    assert "*.safetensors" not in calls[2]["allow_patterns"]

    # the files are cached before the GPU children start
    from quail.backends.quail import worker

    events = []
    registry = built_in_registry()

    def resolve(name, revision):
        events.append("resolve")
        assert revision == registry.model("qwen3-4b-fp8").revision
        return str(tmp_path)

    def round_fn(kind, subs):
        events.append("boot")
        assert kind == "boot"
        assert len(subs) == 8
        assert all(sub["registry"] is registry
                   and sub["model_path"] == str(tmp_path) for sub in subs)
        return [{"boot_s": 3.0, "kind": "cold"}] * 8

    def ensure(count, configuration):
        events.append("children")
        assert count == 8
        assert configuration == ("qwen3-4b-fp8", "h100-sxm", "quail", 8)
        return SimpleNamespace(round=round_fn)

    def run_graph(*args):
        assert events == ["resolve", "children", "boot"]
        events.append("query")
        return {"_outputs": {}, "boot_s": 0.0, "wall_s": 7.0}

    clock = iter([10.0, 12.0, 15.0])
    monkeypatch.setattr(worker.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(worker, "resolve_model_path", resolve)
    monkeypatch.setattr(worker, "_ensure_children", ensure)
    monkeypatch.setattr(worker, "execute_staged_graph", run_graph)
    result = worker.execute_quail_multi({
        "documents": {}, "gpus": 8, "model": "qwen3-4b-fp8",
        "physical_plan": {"device": "h100-sxm", "backend": "quail"},
        "chunk_tokens": 100, "true_ids": [1], "false_ids": [2],
    }, registry, object())
    assert events == ["resolve", "children", "boot", "query"]
    assert result.metrics["boot_s"] == 5.0
    assert result.metrics["boot"]["model_files_s"] == 2.0
    assert result.metrics["wall_s"] == 7.0


def test_a_base_model_without_a_turn_close_ends_answers_at_its_config_eos(
        tmp_path):
    import json

    from helpers.sessions import TOKENIZER

    from quail.common.tokens import Tokenizer

    # a base checkpoint keeps its end of text as an added token
    tokenizer = json.loads((TOKENIZER / "tokenizer.json").read_text())
    eos = len(tokenizer["model"]["vocab"])
    tokenizer["added_tokens"] = [{
        "id": eos, "content": "<|endoftext|>", "single_word": False,
        "lstrip": False, "rstrip": False, "normalized": False, "special": True}]
    (tmp_path / "tokenizer.json").write_text(json.dumps(tokenizer))
    (tmp_path / "tokenizer_config.json").write_text(
        '{"eos_token": "<|endoftext|>"}')
    assert list(Tokenizer(tmp_path).stop_ids) == [eos]

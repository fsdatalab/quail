"""Tokenizers and engines resolve the same pinned model files."""

import sys
from contextlib import ExitStack
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from quail.baselines.sglang import SGLangEngine
from quail.baselines.vllm import (
    DefaultVLLMEngine,
    GigatokenVLLMTokenizer,
    VLLMEngine,
)
from quail.builtins import built_in_registry
from quail.specs.files import resolve_model_path


@pytest.fixture
def snapshots(tmp_path, monkeypatch):
    requests = []
    alphabet = ["a", "b"] + sorted(set(pre_tokenizers.ByteLevel.alphabet())
                                  - {"a", "b"})
    first_vocab = {token: index for index, token in enumerate(alphabet)}
    for revision, vocab in (("first", first_vocab),
                            ("second", first_vocab | {"a": 1, "b": 0})):
        directory = tmp_path / revision
        directory.mkdir()
        tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.save(str(directory / "tokenizer.json"))

    def download(*, repo_id, revision, allow_patterns):
        requests.append((repo_id, revision, allow_patterns))
        return str(tmp_path / revision)

    resolve_model_path.cache_clear()
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    yield tmp_path, requests
    resolve_model_path.cache_clear()


def test_tokenizer_revision_changes_the_loaded_vocabulary(snapshots):
    _, requests = snapshots
    first = GigatokenVLLMTokenizer.from_pretrained("test/model", revision="first")
    second = GigatokenVLLMTokenizer.from_pretrained("test/model", revision="second")
    assert first.encode("a") == [0]
    assert second.encode("a") == [1]
    assert [request[1] for request in requests] == ["first", "second"]
    assert all("*.safetensors" not in request[2] for request in requests)


def test_tokenizer_download_does_not_replace_weight_download(snapshots):
    directory, requests = snapshots
    tokenizer = resolve_model_path("test/model", "first", include_weights=False)
    model = resolve_model_path("test/model", "first")
    assert tokenizer == model == str(directory / "first")
    assert len(requests) == 2
    assert "*.safetensors" not in requests[0][2]
    assert "*.safetensors" in requests[1][2]
    assert resolve_model_path("test/model", "first") == model
    assert len(requests) == 2


@pytest.mark.parametrize("adapter", [VLLMEngine, DefaultVLLMEngine, SGLangEngine])
def test_baseline_loads_the_spec_revision(monkeypatch, snapshots, adapter):
    directory, requests = snapshots
    spec = built_in_registry().model("qwen3-4b-fp8")
    received = []

    class EngineReachedError(Exception):
        pass

    def construct(**kwargs):
        received.append(kwargs)
        raise EngineReachedError

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        LLM=construct, SamplingParams=object))
    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(Engine=construct))
    with ExitStack() as resources, pytest.raises(EngineReachedError):
        adapter().boot(spec, [1, 2], resources=resources)
    keyword = "model_path" if adapter is SGLangEngine else "model"
    assert received[0][keyword] == str(directory / spec.revision)
    assert len(requests) == 1
    assert requests[0][:2] == (spec.hf_name, spec.revision)
    assert "*.safetensors" in requests[0][2]

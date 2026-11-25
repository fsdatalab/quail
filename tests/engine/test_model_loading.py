"""CPU tests for model files, retained answer weights, and readouts."""

import gc
import json
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from quail.backends.quail.executor import model_loading
from quail.backends.quail.executor.model_loading import (
    E2M1_VALUES,
    answer_weights,
    dequantize_nvfp4,
    nvfp4_output_head,
    retain_answer_head,
)
from quail.backends.quail.executor.readout import AnswerRows, AsyncAnswers, AsyncScores


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


def _nvfp4(torch, codes, scales, global_scale):
    """Pack E2M1 codes two per byte, the even column low, with their scales."""
    codes = torch.tensor(codes, dtype=torch.uint8)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return (packed, torch.tensor(scales).to(torch.float8_e4m3fn),
            torch.tensor(global_scale, dtype=torch.float32))


def test_nvfp4_weights_dequantize_to_their_codes_times_both_scales(torch):
    codes = [list(range(16)) + list(range(15, -1, -1)),
             [7] * 16 + [9] * 16]
    scales = [[2.0, 0.5], [1.0, 4.0]]
    packed, scale, global_scale = _nvfp4(torch, codes, scales, 0.25)

    weights = dequantize_nvfp4(torch, packed, scale, global_scale, rows_per_step=1)

    expected = [[E2M1_VALUES[code] * scales[row][column // 16] * 0.25
                 for column, code in enumerate(codes[row])] for row in range(2)]
    assert weights.dtype == torch.bfloat16
    assert weights.float().tolist() == expected


def test_an_nvfp4_head_is_read_from_the_checkpoint_and_kept_in_bf16(
        torch, tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    packed, scale, global_scale = _nvfp4(torch, [[1, 2] * 8, [3, 4] * 8],
                                         [[1.0], [2.0]], 1.0)
    safetensors.save_file({"lm_head.weight": packed,
                           "lm_head.weight_scale": scale,
                           "lm_head.weight_scale_2": global_scale},
                          tmp_path / "model.safetensors")
    layers = {"lm_head": {"quant_algo": "NVFP4", "group_size": 16}}
    (tmp_path / "config.json").write_text(json.dumps(
        {"quantization_config": {"quant_algo": "MIXED_PRECISION",
                                 "quantized_layers": layers}}))
    model = _model(torch)
    model.lm_head = torch.nn.Linear(8, 2, bias=False)

    head = nvfp4_output_head(torch, tmp_path, device="cpu")
    retain_answer_head(torch, model, [0, 1], head=head)

    assert model.quail_full_head.float().tolist() == [[0.5, 1.0] * 8, [3.0, 4.0] * 8]
    assert model.quail_answer_weights.dtype == torch.bfloat16
    (tmp_path / "config.json").write_text(json.dumps(
        {"quantization_config": {"quant_method": "fp8"}}))
    assert nvfp4_output_head(torch, tmp_path, device="cpu") is None


@pytest.fixture
def clear_model_paths():
    model_loading.resolve_model_path.cache_clear()
    yield
    model_loading.resolve_model_path.cache_clear()


def test_model_files_are_resolved_once_and_cached(
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


def test_a_base_model_without_a_turn_close_ends_answers_at_its_config_eos(
        tmp_path):

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

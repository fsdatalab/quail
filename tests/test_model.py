"""CPU tests for model initialization and retained answer weights."""

import gc
import weakref
from types import SimpleNamespace

import pytest

from quail.backends.quail.executor.model import answer_weights, retain_answer_head


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


def test_retained_answer_weights_and_embedding_ownership(torch):
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
        assert model.quail_answer_weights.data_ptr() != embedding.data_ptr()
        assert model.quail_answer_weights.device == embedding.device
        assert not model.quail_answer_weights.requires_grad
        assert model.model.embed_tokens(torch.tensor([0, 7])).shape == (2, 4)

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

    model = _model(torch)
    model.lm_head.weight = model.model.embed_tokens.weight
    embedding = model.model.embed_tokens.weight
    retain_answer_head(torch, model, [1, 3, 5])
    assert model.lm_head is None
    assert model.model.embed_tokens.weight is embedding
    assert torch.equal(model.quail_answer_weights, embedding[[1, 3, 5]])


def test_answer_rows_preserve_scores_and_share_retained_weights(torch):
    from quail.backends.quail.executor.readout import AnswerRows

    model = _model(torch)
    hidden = torch.tensor([[1, 2, -1, 0], [0, 1, 3, -2]], dtype=torch.bfloat16)
    full_scores = torch.nn.functional.linear(hidden, model.lm_head.weight)
    expected = (full_scores[:, [1, 3]].amax(1) > full_scores[:, 5]).int().tolist()
    retain_answer_head(torch, model, [1, 3, 5])
    cpu_torch = SimpleNamespace(tensor=lambda values, **kwargs: torch.tensor(values))

    def tokenizer(word, **kwargs):
        return {"input_ids": [
            5 if "false" in word.lower() else 3 if word.startswith(" ") else 1]}

    first = AnswerRows.from_tokenizer(cpu_torch, torch.nn.functional, model,
                                      tokenizer)
    second = AnswerRows(cpu_torch, torch.nn.functional, model, [3, 1], [5])
    assert first.weights is second.weights is model.quail_answer_weights
    assert first(hidden) == second(hidden) == expected

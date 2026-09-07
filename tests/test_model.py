"""CPU tests for model initialization and cached answer weights."""

from types import SimpleNamespace

import pytest

from quail.executor.model import _SingleRank, answer_weights, cache_answer_weights


class _Torch:
    class device:
        def __init__(self, name):
            self.name = name


def test_single_rank_collectives_are_identity():
    group = _SingleRank(_Torch)
    assert group.world_size == 1
    assert group.is_first_rank
    assert group.is_last_rank
    x = object()
    assert group.all_reduce(x) is x
    assert group.all_gather(x) is x
    assert group.broadcast(x) is x
    assert group.broadcast_object("ok") == "ok"


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


@pytest.mark.parametrize("tied", [False, True])
def test_caches_exact_rows_and_preserves_full_model(torch, tied):
    model = _model(torch, tied)
    output_head = model.lm_head
    embedding = model.model.embed_tokens.weight
    expected = model.lm_head.weight.detach()[[1, 3, 5]].clone()
    cache_answer_weights(torch, model, [5, 1, 3, 1])
    assert model.lm_head is output_head
    assert model.model.embed_tokens.weight is embedding
    assert model.quail_answer_token_ids == (1, 3, 5)
    assert torch.equal(model.quail_answer_weights, expected)
    assert model.quail_answer_weights.untyped_storage().nbytes() == expected.numel() * 2
    assert model.quail_answer_weights.data_ptr() != embedding.data_ptr()
    assert model.quail_answer_weights.device == embedding.device
    assert not model.quail_answer_weights.requires_grad
    assert model.model.embed_tokens(torch.tensor([0, 7])).shape == (2, 4)


def test_repeated_queries_reuse_weights_and_reject_different_ids(torch):
    model = _model(torch)
    cache_answer_weights(torch, model, [1, 3, 5])
    weights = answer_weights(model, [1, 3, 5])
    cache_answer_weights(torch, model, [5, 3, 1])
    assert answer_weights(model, [1, 3, 5]) is weights
    with pytest.raises(ValueError, match="differ"):
        answer_weights(model, [1, 3, 6])
    assert answer_weights(model, [1, 3, 5]) is weights


def test_empty_ids_do_not_change_weights(torch):
    model = _model(torch)
    original = model.lm_head
    with pytest.raises(ValueError, match="empty"):
        cache_answer_weights(torch, model, [])
    assert model.lm_head is original


def test_separate_head_module_with_shared_weight_keeps_embeddings(torch):
    model = _model(torch)
    model.lm_head.weight = model.model.embed_tokens.weight
    embedding = model.model.embed_tokens.weight
    output_head = model.lm_head
    cache_answer_weights(torch, model, [1, 3, 5])
    assert model.lm_head is output_head
    assert model.model.embed_tokens.weight is embedding
    assert torch.equal(model.quail_answer_weights, embedding[[1, 3, 5]])


def test_answerers_preserve_scores_and_share_retained_weights(torch):
    from quail.backends.quail.worker import _PayloadAnswerer
    from quail.executor.loop import Answerer

    model = _model(torch)
    hidden = torch.tensor([[1, 2, -1, 0], [0, 1, 3, -2]], dtype=torch.bfloat16)
    full_scores = torch.nn.functional.linear(hidden, model.lm_head.weight)
    expected = (full_scores[:, [1, 3]].amax(1) > full_scores[:, 5]).int().tolist()
    cache_answer_weights(torch, model, [1, 3, 5])
    cpu_torch = SimpleNamespace(tensor=lambda values, **kwargs: torch.tensor(values))

    def tokenizer(word, **kwargs):
        return {"input_ids": [5 if "false" in word.lower() else 3 if word.startswith(" ") else 1]}

    first = Answerer(cpu_torch, torch.nn.functional, model, tokenizer)
    second = _PayloadAnswerer(cpu_torch, torch.nn.functional, model, [3, 1], [5])
    assert first.weights is second.weights is model.quail_answer_weights
    assert first(hidden) == second(hidden) == expected

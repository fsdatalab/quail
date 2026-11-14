"""Rust's tokenizer gives a model's ids and bytes."""

import pytest


def test_a_model_tokenizer_in_rust_gives_gigatokens_ids_and_bytes():
    from gigatoken import Tokenizer as Gigatoken

    from quail.builtins import built_in_registry
    from quail.common.tokens import Tokenizer
    from quail.specs.files import resolve_model_path

    model = built_in_registry().model("qwen3-4b-fp8")
    try:
        directory = resolve_model_path(model.hf_name, model.revision,
                                       include_weights=False)
    except Exception as error:  # noqa: BLE001 - offline machines skip
        pytest.skip(f"the model's tokenizer is not available: {error}")
    texts = ["Lease ends in 2027.", "A moving ending — and a café.", "", "  spaced  "]
    reference = Gigatoken(directory)
    tokenizer = Tokenizer(directory)
    found = [tokenizer(text) for text in texts]
    assert found == [list(reference.encode(text)) for text in texts]
    assert dict(enumerate(tokenizer.token_bytes())) == dict(reference.vocab)
    assert [tokenizer.decode(ids).decode() for ids in found] == texts

"""Extraction head work and tensor dispatches are batched across fields."""

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from helpers.extract_fakes import make_spec
from test_map_decode import events_torch
from torch.utils._python_dispatch import TorchDispatchMode

from quail.backends.quail.executor.extract.stages import ExtractionInputs
from quail.backends.quail.executor.map_decoder import MapDecoder
from quail.backends.quail.executor.readout import AsyncTopLogprobs


class Dispatches(TorchDispatchMode):
    """Count CPU tensor operations as a proxy for GPU launch submission."""

    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[str(func)] += 1
        return func(*args, **(kwargs or {}))


def copy_batch(num_fields, num_documents=4, document_tokens=24, draft_tokens=2,
               num_map_documents=0):
    """Build independent fields over repeated document spans with a small head."""
    text = "five" + " years" * (document_tokens - 1)
    spec, vocabulary = make_spec(text)
    spec = replace(spec, top_tokens=4, draft_tokens=draft_tokens,
                   max_answer_tokens=32, first_window_chars=8, window_chars=16)
    state = SimpleNamespace(loaded_model=SimpleNamespace(arena=SimpleNamespace()))
    documents = ExtractionInputs(
        state, [spec] * num_fields, vocabulary, np.arange(num_documents), None,
        [text] * num_documents)
    slots = np.arange(num_fields * num_documents).reshape(num_fields, -1)
    size = len(vocabulary.vocab)
    num_slots = slots.size + num_map_documents
    decoder = MapDecoder(events_torch(0), torch.eye(size), num_slots,
                         [32] * (num_fields + bool(num_map_documents)),
                         vocabulary.stop_ids, 16)
    decoder.configure_copy(0, slots, documents, [spec] * num_fields)
    for q, selected in enumerate(slots):
        for slot in selected:
            decoder.start(int(slot), q, 0, [0, 1, 2])
    for slot in range(slots.size, num_slots):
        decoder.start(slot, num_fields, 0, [0, 1, 2])
    hidden = torch.full((num_slots, size), -10.0)
    hidden[:, vocabulary.encode_ids("five")[0]] = 1
    decoder.submit(hidden, np.arange(num_slots), np.arange(num_slots), None)
    decoder.harvest(wait=True)
    return decoder, vocabulary


def continuation(decoder, vocabulary):
    """Run one pass that accepts every draft token."""
    rows = decoder.rows()
    chunk = SimpleNamespace(decode_rows=rows, tokens=len(rows), meta={},
                            input_ids=torch.empty(len(rows), dtype=torch.int64))
    decoder.feed(chunk)
    hidden = torch.full((len(rows), len(vocabulary.vocab)), -10.0)
    hidden[:, vocabulary.encode_ids(" years")[0]] = 1
    decoder.submit(hidden, [], [], rows)
    decoder.harvest(wait=True)
    return len(rows)


def test_tensor_dispatch_count_does_not_scale_with_fields_or_documents(monkeypatch):
    monkeypatch.delenv("QUAIL_EXTRACT_TRACE", raising=False)
    counts = []
    for num_fields, num_documents in [(1, 1), (3, 4), (41, 4)]:
        decoder, vocabulary = copy_batch(num_fields, num_documents)
        continuation(decoder, vocabulary)
        with Dispatches() as dispatches:
            num_rows = continuation(decoder, vocabulary)
        assert num_rows == num_fields * num_documents * 3
        assert dispatches.counts["aten.mm.default"] == 1
        counts.append(sum(dispatches.counts.values()))
    assert counts[1] == counts[2] <= counts[0], counts


def test_all_draft_rows_use_bounded_head_blocks(monkeypatch):
    decoder, vocabulary = copy_batch(3, num_map_documents=2)
    monkeypatch.setattr(AsyncTopLogprobs, "BLOCK_ROWS", 7)
    with Dispatches() as dispatches:
        num_rows = continuation(decoder, vocabulary)
    assert dispatches.counts["aten.mm.default"] == -(-num_rows // 7)
    assert all((written[written > 0] == 4).all() for written in decoder.written[:3])
    assert decoder.written[3][-2:].tolist() == [2, 2]


def test_maps_keep_argmax_tie_order_in_a_shared_projection():
    decoder, vocabulary = copy_batch(3, num_map_documents=2)
    rows = decoder.rows()
    decoder.feed(SimpleNamespace(
        decode_rows=rows, tokens=len(rows), meta={},
        input_ids=torch.empty(len(rows), dtype=torch.int64)))
    tokens = [vocabulary.encode_ids(word)[0] for word in ("five", " years")]
    hidden = torch.full((len(rows), len(vocabulary.vocab)), -10.0)
    hidden[:, tokens] = 1
    with Dispatches() as dispatches:
        decoder.submit(hidden, [], [], rows)
    decoder.harvest(wait=True)
    assert dispatches.counts["aten.mm.default"] == 1
    assert decoder.tokens[3][-2:, 1].tolist() == [min(tokens)] * 2


def test_copy_rows_and_selection_need_no_preparation_after_harvest(monkeypatch):
    decoder, vocabulary = copy_batch(3, document_tokens=100)
    constraint = decoder.copy_groups[0].constraint
    prepare = constraint.prepare
    monkeypatch.setattr(constraint, "prepare", lambda *args: pytest.fail(
        "copy selection must already have its inputs"))
    rows = decoder.rows()
    decoder.feed(SimpleNamespace(
        decode_rows=rows, tokens=len(rows), meta={},
        input_ids=torch.empty(len(rows), dtype=torch.int64)))
    hidden = torch.full((len(rows), len(vocabulary.vocab)), -10.0)
    hidden[:, vocabulary.encode_ids(" years")[0]] = 1
    decoder.submit(hidden, [], [], rows)
    monkeypatch.setattr(constraint, "prepare", prepare)
    decoder.harvest(wait=True)
    assert all((written[written > 0] == 4).all() for written in decoder.written)

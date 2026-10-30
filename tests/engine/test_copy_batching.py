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


class Dispatches(TorchDispatchMode):
    """Count CPU tensor operations as a proxy for GPU launch submission."""

    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[str(func)] += 1
        return func(*args, **(kwargs or {}))


def copy_batch(num_fields, num_documents=4, document_tokens=24, draft_tokens=2,
               num_map_documents=0, prepare=True, answer_limit=32, text=None):
    """Build independent fields over repeated document spans with a small head."""
    text = text or "five" + " years" * (document_tokens - 1)
    spec, vocabulary = make_spec(text)
    spec = replace(spec, top_tokens=4, draft_tokens=draft_tokens,
                   max_answer_tokens=answer_limit, first_window_chars=8,
                   window_chars=16)
    state = SimpleNamespace(loaded_model=SimpleNamespace(arena=SimpleNamespace()))
    documents = ExtractionInputs(
        state, [spec] * num_fields, vocabulary, np.arange(num_documents), None,
        [text] * num_documents)
    slots = np.arange(num_fields * num_documents).reshape(num_fields, -1)
    size = len(vocabulary.vocab)
    num_slots = slots.size + num_map_documents
    decoder = MapDecoder(events_torch(0), torch.eye(size), num_slots,
                         [answer_limit] * (num_fields + bool(num_map_documents)),
                         vocabulary.stop_ids, 16,
                         workspace_bytes=64 * 2**20)
    decoder.configure_copy(0, slots, documents, [spec] * num_fields)
    for q, selected in enumerate(slots):
        for slot in selected:
            decoder.start(int(slot), q, 0, [0, 1, 2])
    for slot in range(slots.size, num_slots):
        decoder.start(slot, num_fields, 0, [0, 1, 2])
    hidden = torch.full((num_slots, size), -10.0)
    hidden[:, vocabulary.encode_ids("five")[0]] = 1
    decoder.submit(hidden, np.arange(num_slots), np.arange(num_slots), None)
    decoder.harvest(wait=True, prepare=prepare)
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


def sequential_constrain(self, group, projected, documents, indices):
    """Run the pre-batching verifier, including unused-row trace values."""
    torch, constraint = self.torch, group.constraint
    slots, first, inverse, counts = np.unique(
        documents, return_index=True, return_inverse=True, return_counts=True)
    local = constraint.tensor(self.copy_document[slots])
    index = constraint.tensor(group.index[slots])
    limits = constraint.tensor(self.limits[self.map[slots]])
    stream_width = int(group.widths[group.index[slots]].max())
    streams = group.streams[index, :stream_width]
    alive = group.alive[index, :stream_width]
    num_streams = int(np.where(indices[first] == 0, stream_width,
                              self.num_streams[slots]).max())
    num_streams = max(1, num_streams)
    if num_streams < stream_width:
        streams, alive = constraint.compact(streams, alive, num_streams)
    source = group.source[index]
    active = torch.ones(len(slots), dtype=torch.bool, device=self.device)
    width = 4 + self.copy_words + self.copy_trace_words
    out = torch.full((len(documents), width), -2,
                     dtype=torch.int64, device=self.device)
    for step in range(int(counts.max())):
        valid = step < counts
        row = np.minimum(first + step, len(documents) - 1)
        at = constraint.tensor(np.where(valid, indices[row], 1))
        selected = constraint.tensor(row)
        chosen, ended, keep = constraint.choose_readout(
            projected[selected], streams, alive, at, local,
            first=bool((valid & (indices[row] == 0)).any()))
        use = active & constraint.tensor(valid)
        writes = use & ~ended
        alive = torch.where(writes[:, None], keep, alive)
        following = constraint.source(streams, alive)
        hit = chosen == constraint.tokens_at(source, at)
        values = torch.stack((torch.where(ended, -1, chosen), following,
                              alive.sum(dim=1), hit.to(torch.int64)), dim=1)
        # Only valid rows may scatter; padded rows can alias a real row.
        valid_index = constraint.tensor(np.flatnonzero(valid))
        out[selected[valid_index], :4] = torch.where(
            use[valid_index, None], values[valid_index], -2)
        if self.copy_trace is not None:
            bits = constraint.trace_values.view(torch.int32).to(torch.int64)
            out[selected[valid_index], 4 + self.copy_words:] = bits[valid_index]
        active &= writes & hit & (at + 1 < limits)
        source = following
    group.alive[index, :stream_width] = False
    group.alive[index, :num_streams] = alive
    group.streams[index, :num_streams] = streams
    group.widths[group.index[slots]] = num_streams
    group.source[index] = source
    packed = constraint.packed_positions(streams, alive, local)
    out[:, 4:4 + constraint.mask_words] = packed[constraint.tensor(inverse)]
    return out


def verifier_inputs(decoder, vocabulary, mixed=False):
    """Build projected choices over long, optionally ragged draft rows."""
    rows = decoder.rows()
    documents = rows.slots
    indices = rows.positions - decoder.origin[documents] + 1
    if mixed:
        valid = indices <= 40 + documents % 9
        documents, indices = documents[valid], indices[valid]
    hidden = torch.full((len(documents), len(vocabulary.vocab)), -10.0)
    hidden[:, vocabulary.encode_ids(" years")[0]] = 1
    if mixed:
        rejects = ((documents % 4 == 0) & (indices == 20)
                   | (documents % 4 == 1) & (indices == 40))
        hidden[rejects, vocabulary.encode_ids(" days")[0]] = 2
        stops = (documents % 4 == 2) & (indices == 1 + documents % 43)
        hidden[stops, vocabulary.stop_ids[0]] = 2
    group = decoder.copy_groups[0]
    _, projected = decoder._project(hidden, documents)
    return group, projected[group.key], documents, indices


@pytest.mark.parametrize("trace,num_fields,workspace_bytes", [
    (False, 1, 64 * 2**20),
    (True, 41, 64 * 2**20),
    (True, 3, 128000),
    (False, 41, 128000),
])
def test_long_draft_choices_and_state_match_sequential_verification(
    monkeypatch, tmp_path, trace, num_fields, workspace_bytes
):
    if trace:
        monkeypatch.setenv("QUAIL_EXTRACT_TRACE", str(tmp_path / "trace.jsonl"))
    else:
        monkeypatch.delenv("QUAIL_EXTRACT_TRACE", raising=False)
    text = " ".join("five" + " years" * length + " days" for length in [64, 19, 39])
    decoder, vocabulary = copy_batch(num_fields, draft_tokens=47, answer_limit=96,
                                     text=text)
    inputs = verifier_inputs(decoder, vocabulary, mixed=True)
    group = inputs[0]
    decoder.limits[::3] = 30
    ranks = group.alive.to(torch.int64).cumsum(dim=1)
    group.alive &= ranks <= (1 + torch.arange(len(group.alive)) % 3)[:, None]
    decoder.num_streams[:] = group.alive.sum(dim=1).numpy()[group.index]
    columns = torch.arange(group.alive.shape[1])
    last = torch.where(group.alive, columns, 0).max(dim=1).values
    group.source[1::2] = group.streams.gather(1, last[:, None]).squeeze(1)[1::2]
    saved = (group.alive.clone(), group.streams.clone(), group.source.clone(),
             group.widths.copy())
    expected = sequential_constrain(decoder, *inputs)
    final = (group.alive.clone(), group.streams.clone(), group.source.clone())
    group.alive.copy_(saved[0])
    group.streams.copy_(saved[1])
    group.source.copy_(saved[2])
    group.widths[:] = saved[3]
    decoder.workspace_bytes = workspace_bytes
    actual = decoder._constrain(*inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(group.alive, final[0], rtol=0, atol=0)
    torch.testing.assert_close(group.streams[group.alive], final[1][final[0]],
                               rtol=0, atol=0)
    torch.testing.assert_close(group.source, final[2], rtol=0, atol=0)
    assert (group.widths >= group.alive.sum(dim=1).numpy()).all()
    assert (actual[:, 0] == -2).any()
    assert (actual[:, 0] >= 0).any()

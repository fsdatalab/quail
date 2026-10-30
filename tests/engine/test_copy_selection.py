"""Lazy copy preparation and diagnostic parity with the host token rule."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from helpers.extract_fakes import make_spec
from helpers.extract_reference import Decoder
from helpers.extract_reference import ReferenceInputs as ExtractionInputs
from test_map_decode import events_torch

from quail.backends.quail.executor.extract.copy import CopyConstraint
from quail.backends.quail.executor.map_decoder import MapDecoder
from quail.backends.quail.executor.readout import AsyncTopLogprobs


def inputs(spec, vocabulary, texts):
    state = SimpleNamespace(loaded_model=SimpleNamespace(arena=SimpleNamespace()))
    return ExtractionInputs(state, [spec], vocabulary, range(len(texts)), None, texts)


def test_copy_preparation_stops_extending_positions_after_their_draft_diverges(
    monkeypatch,
):
    monkeypatch.delenv("QUAIL_EXTRACT_TRACE", raising=False)
    text = "five years " + "tail " * 40 + "five days " * 100
    spec, vocabulary = make_spec(text, "five", "tail")
    spec = replace(
        spec,
        top_tokens=4,
        max_answer_tokens=32,
        draft_tokens=16,
        first_window_chars=1,
        window_chars=8,
    )
    documents = inputs(spec, vocabulary, [text])
    encoded = []
    original = vocabulary.encode_batch

    def encode(texts):
        encoded.extend(texts)
        return original(texts)

    monkeypatch.setattr(vocabulary, "encode_batch", encode)
    constraint = CopyConstraint(torch, "cpu", documents, spec)
    assert not encoded
    assert constraint.num_positions == 0
    logits = torch.full((1, len(vocabulary.vocab)), -20.0)
    token = vocabulary.encode_ids("five")[0]
    logits[0, token] = 1
    streams = constraint.stream_ids.clone()
    chosen, ended, alive = constraint.choose(
        logits, streams, streams >= 0, torch.tensor([0]), torch.tensor([0])
    )
    assert chosen.tolist() == [token] and not ended.any()
    source = constraint.source(streams, alive)
    packed = constraint.packed_positions(streams, alive, torch.tensor([0])).numpy()
    args = (
        np.array([0]),
        np.array([token]),
        packed,
        np.array([1]),
        np.array([16]),
        source.numpy(),
    )
    constraint.prepare(*args)
    positions = (streams[alive] // constraint.num_options).numpy()
    first = int(source[0]) // constraint.num_options
    others = positions[positions != first]
    assert len(others) == 100
    assert constraint.position_lengths[first] >= 16
    assert constraint.position_lengths[others].max() <= 2
    assert constraint.num_positions == len(positions)
    assert len(constraint.fillers) < constraint.num_options
    num_encoded = len(encoded)
    constraint.prepare(*args)
    assert len(encoded) == num_encoded


def test_position_masks_preserve_the_high_bit_and_word_boundaries():
    text = "five " * 150
    spec, vocabulary = make_spec(text)
    constraint = CopyConstraint(torch, "cpu", inputs(spec, vocabulary, [text]), spec)
    alive = torch.zeros_like(constraint.stream_ids, dtype=torch.bool)
    selected = [0, 63, 64, 127, 128, 149]
    alive[0, selected] = True
    packed = constraint.packed_positions(
        constraint.stream_ids, alive, torch.tensor([0])
    )
    bits = np.unpackbits(packed.numpy().view(np.uint8), axis=1, bitorder="little")
    assert np.flatnonzero(bits[0]).tolist() == selected


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("draft_tokens", [0, 4])
@pytest.mark.parametrize("num_questions", [1, 3, 41])
def test_copy_traces_match_the_host_rule_on_the_same_logits(
    monkeypatch, tmp_path, dtype, draft_tokens, num_questions
):
    text = "five days or five years. five years."
    spec, vocabulary = make_spec(text)
    spec = replace(
        spec,
        top_tokens=4,
        max_answer_tokens=12,
        draft_tokens=draft_tokens,
        first_window_chars=1,
        window_chars=8,
    )
    extra_stop = vocabulary.encode_ids("none")[0]
    vocabulary.stop_ids = (*vocabulary.stop_ids, extra_stop)
    vocabulary.is_stop[extra_stop] = True
    state = SimpleNamespace(loaded_model=SimpleNamespace(arena=SimpleNamespace()))
    documents = ExtractionInputs(
        state, [spec] * num_questions, vocabulary, [0], None, [text]
    )
    size = len(vocabulary.vocab)
    head = torch.eye(size, dtype=dtype)
    logits = torch.full((num_questions, 20, size), -10.0, dtype=dtype)
    first = vocabulary.encode_ids("five")[0]
    alternate = vocabulary.encode_ids(" five")[0]
    continuations = [vocabulary.encode_ids(word)[0] for word in (" years", " days")]
    logits[:, 0, [first, alternate]] = 2
    for q in range(num_questions):
        logits[q, 1, continuations[q % 2]] = 2 + q / 64
    logits[:, 2:, vocabulary.stop_ids[0]] = 1
    logits[:, 2:, extra_stop] = 0
    readout = AsyncTopLogprobs(
        torch, torch.nn.functional, head, 4, [vocabulary.stop_ids]
    )
    old_path, new_path = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
    monkeypatch.setenv("QUAIL_EXTRACT_TRACE", str(old_path))
    reference = Decoder(vocabulary, 4, 9, num_questions, spec)
    document = documents.documents[0]
    for q, answer in enumerate(document.answers):
        fed = []
        while not answer.done:
            at = len(answer.written)
            values = readout.values(logits[q, at : at + max(1, len(fed))]).numpy()
            reference.verify(document, q, values, fed)
            if not answer.done:
                fed = reference.draft(answer, document.rho, q)
    monkeypatch.setenv("QUAIL_EXTRACT_TRACE", str(new_path))
    decoder = MapDecoder(
        events_torch(0), head, num_questions, [12] * num_questions,
        vocabulary.stop_ids, 16,
        workspace_bytes=64 * 2**20,
    )
    slots = np.arange(num_questions)
    decoder.configure_copy(0, slots[:, None], documents, [spec] * num_questions)
    for q in slots:
        decoder.start(int(q), int(q), 0, [0, 1])
    decoder.submit(logits[:, 0], slots, slots, None)
    decoder.harvest(wait=True)
    while decoder.active:
        rows = decoder.rows()
        decoder.feed(
            SimpleNamespace(
                decode_rows=rows,
                tokens=len(rows),
                meta={},
                input_ids=torch.empty(len(rows), dtype=torch.int64),
            )
        )
        decoder.submit(logits[rows.slots, rows.positions + 1], [], [], rows)
        decoder.harvest(wait=True)
    for q, answer in enumerate(document.answers):
        assert decoder.output_answers(q, [q]) == ([answer.answer], [answer.span])
        assert decoder.answers(q, [q])[0].tolist() == answer.written
    old = [json.loads(line) for line in old_path.read_text().splitlines()]
    new = [json.loads(line) for line in new_path.read_text().splitlines()]
    assert len(new) == len(old)
    assert {row["question"] for row in new} == set(range(num_questions))
    for q in range(num_questions):
        expected = [row for row in old if row["question"] == q]
        actual = [row for row in new if row["question"] == q]
        assert [row["index"] for row in actual] == list(range(len(expected)))
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected):
            assert got == pytest.approx(want, abs=1e-6)
        assert expected[0]["margin"] == 0
        assert expected[-1]["chosen"] == -1


def test_a_replayed_copy_prepares_the_continuation_after_a_rejected_draft():
    text = "five days or five years long lease"
    spec, vocabulary = make_spec(text, "five")
    spec = replace(
        spec,
        top_tokens=4,
        max_answer_tokens=8,
        draft_tokens=4,
        first_window_chars=1,
        window_chars=1,
    )
    documents = inputs(spec, vocabulary, [text])
    size = len(vocabulary.vocab)
    decoder = MapDecoder(
        events_torch(3), torch.eye(size), 1, [8], vocabulary.stop_ids, 16,
        workspace_bytes=64 * 2**20,
    )
    decoder.configure_copy(0, np.array([[0]]), documents, [spec])

    def logits(token, count=1):
        values = torch.full((count, size), -10.0)
        values[:, token] = 1
        return values

    first, years, longer = [
        vocabulary.encode_ids(word)[0] for word in ("five", " years", " long")
    ]
    decoder.start(0, 0, 0, [0, 1])
    decoder.submit(logits(first), [0], [0], None)
    decoder.harvest(wait=True)
    rows = decoder.rows()
    decoder.feed(
        SimpleNamespace(
            decode_rows=rows,
            tokens=len(rows),
            meta={},
            input_ids=torch.empty(len(rows), dtype=torch.int64),
        )
    )
    decoder.submit(logits(years, len(rows)), [], [], rows)
    decoder.pause(0)
    decoder.harvest(wait=True)
    decoder.resume(0, [0, 1])
    decoder.submit(logits(longer), [0], [0], None)
    decoder.harvest(wait=True)
    rows = decoder.rows()
    decoder.feed(
        SimpleNamespace(
            decode_rows=rows,
            tokens=len(rows),
            meta={},
            input_ids=torch.empty(len(rows), dtype=torch.int64),
        )
    )
    decoder.submit(logits(vocabulary.stop_ids[0], len(rows)), [], [], rows)
    decoder.harvest(wait=True)
    assert decoder.answers(0, [0])[0].tolist() == [first, years, longer]
    start = text.index("five years")
    assert decoder.output_answers(0, [0]) == (
        ["five years long"],
        [(start, start + 15)],
    )

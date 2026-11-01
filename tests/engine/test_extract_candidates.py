"""The extraction's candidates: word keys, first tokens, and streams."""

import numpy as np
import pytest
from helpers.extract_fakes import RANKED, make_vocabulary, squeezed

from quail.backends.quail.executor.extract.candidates import (
    DocumentTokens,
    ensure_tokens,
    first_tokens,
    make_streams,
)


def test_first_tokens_follow_the_models_spelling():
    tables = make_vocabulary()
    text = RANKED
    ids, offsets = tables.encode(text)
    scan = DocumentTokens.from_tokens(tables, ids, offsets, 1)
    the, march = tables.encode_ids("The")[0], tables.encode_ids("March")[0]
    space_march = tables.encode_ids(" March")[0]
    lp = {the: -0.5, march: -1.0, space_march: -3.0}
    starts = first_tokens(tables, _scores(tables, lp), scan)
    by_char = {scan.starts[i - 1]: value for i, value in starts.items(tables)}
    # the model writes "March" without its space; the document's token
    # has one, and the likelier spelling starts it; "The" twice
    assert by_char[text.index(" March")] == (march, [])
    assert by_char[0] == (the, []) and by_char[26] == (the, [])
    assert set(by_char) == {0, 26, text.index(" March")}
    assert {scan.starts[i - 1] for i in starts} == set(by_char)
    # the chosen first token names the streams, each the text from
    # its start as the model would write it, with its character ends
    streams = make_streams(tables, text, starts, the)
    assert [stream.char for stream in streams] == [0, 26]
    # a stream holds its first token until the writing reaches it
    assert streams[1].stream == [the]
    streams[1].extend(tables)
    assert streams[1].stream[:3] == tables.encode_ids("The rent is")
    # a start's offset includes its token's leading space
    assert text[streams[1].char:streams[1].ends[2]] == " The rent is"
    (march_stream,) = make_streams(tables, text, starts, march)
    march_stream.extend(tables)
    assert march_stream.ends[:4] == [23, 24, 25, 26]
    assert text[march_stream.char:march_stream.ends[3]] == " March 3."


@pytest.mark.parametrize("first_window,window", [(9, 19)])
def test_stream_batches_preserve_tokens_and_offsets_across_windows(
        monkeypatch, first_window, window):
    texts = [" ".join(["word"] * 300), "word   \n word  ", "word"]
    tables = make_vocabulary(*texts)
    streams = []
    expected = []
    word = tables.encode_ids("word")[0]
    for text in texts:
        ids, offsets = tables.encode(text)
        scan = DocumentTokens.from_tokens(tables, ids, offsets)
        starts = first_tokens(tables, _scores(tables, {word: -0.5}), scan)
        streams.append(make_streams(tables, text, starts, word,
                                    first_window_chars=first_window,
                                    window_chars=window)[0])
        rest, ends = squeezed(text)
        ids, offsets = tables.encode(rest)
        expected.append((ids, [ends[b - 1] for _, b in offsets]))
    batches = []
    encode_batch = tables.encode_batch

    def recorded(texts):
        batches.append(tuple(texts))
        return encode_batch(texts)

    def scalar(text):
        raise AssertionError("stream windows must be tokenized together")

    monkeypatch.setattr(tables, "encode_batch", recorded)
    monkeypatch.setattr(tables, "encode", scalar)
    ensure_tokens(streams, 80, tables)
    assert len(batches[0]) == 2
    largest_window = max(len(text) for batch in batches for text in batch)
    assert largest_window <= max(first_window, window) + 4
    assert not streams[0].exhausted
    assert streams[1].exhausted and streams[2].exhausted
    ensure_tokens(streams, 10_000, tables)
    assert [(stream.stream, stream.ends) for stream in streams] == expected
    assert all(stream.exhausted for stream in streams)
    before = list(batches)
    ensure_tokens(streams, 10_000, tables)
    assert batches == before


def _scores(vocabulary, pairs):
    values = np.full(len(vocabulary.vocab), -np.inf, dtype=np.float32)
    for token, score in pairs.items():
        values[token] = score
    return values

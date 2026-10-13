"""The extraction's candidates: word keys, first tokens, and streams."""

import numpy as np
import pytest
from extract_fakes import LEASE, RANKED, TAIL, FakeTokenizer, make_vocabulary

from quail.backends.quail.executor.extract import candidates as candidates_mod
from quail.backends.quail.executor.extract.candidates import (
    DocumentTokens,
    SqueezedText,
    Stream,
    document_tokens,
    ensure_tokens,
    first_tokens,
    make_streams,
    squeezed,
)
from quail.backends.quail.executor.extract.vocabulary import Vocabulary, same_word
from quail.execution.tokens import GigatokenEncoder


def test_gigatoken_vocabulary_encodes_unicode_without_transformers(monkeypatch):
    import sys

    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    texts = ['  Café\n\n東京 🙂', 'none "March"', "", "a  b"]
    native = Tokenizer(models.BPE())
    native.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    native.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=256, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    expected = native.encode_batch(texts, add_special_tokens=False)
    encoder = GigatokenEncoder(native)
    monkeypatch.setitem(sys.modules, "transformers", None)
    tables = Vocabulary(encoder)
    monkeypatch.setattr(encoder, "batch_arrow", lambda _: pytest.fail(
        "continuations must not convert through Arrow token documents"))
    assert tables.quote_id in tables.stop_ids
    for (ids, offsets), original in zip(tables.encode_batch(texts), expected):
        assert list(ids) == original.ids
        assert offsets.tolist() == [list(pair) for pair in original.offsets]
    text = 'n  Café\n東京 🙂 ' * 80
    stream = Stream(0, tables.encode_ids("n"), [1], text)
    stream.next(100_000, tables)
    normalized, character_ends = squeezed(text)
    expected = native.encode(normalized, add_special_tokens=False)
    assert stream.stream == expected.ids
    assert stream.ends == [character_ends[end - 1] for _, end in expected.offsets]
    encode = encoder.batch_numpy
    monkeypatch.setattr(encoder, "batch_numpy", lambda texts: encode(["ab"]))
    with pytest.raises(ValueError, match="preserve the source text"):
        tables.encode("a b")


def test_word_keys_and_token_offsets():
    assert same_word("ĠMarch") == "march" and same_word("the") == "the"
    assert same_word("Ċ") == "" and same_word(5) == "" and same_word("ĠaĠb") == ""
    tables = make_vocabulary()
    ids, offsets = tables.encode(LEASE)
    tokens = document_tokens(offsets, 3)
    assert [LEASE[a:b] for _, (a, b) in tokens][:4] == [
        "Contract", " of", " lease", "."]
    assert [index for index, _ in tokens] == list(range(3, 3 + len(ids)))


def test_first_tokens_follow_the_models_spelling():
    tables = make_vocabulary()
    text = RANKED
    ids, offsets = tables.encode(text)
    prefix = [99] + ids
    tokens = document_tokens(offsets, 1)
    the, march = tables.encode_ids("The")[0], tables.encode_ids("March")[0]
    space_march = tables.encode_ids(" March")[0]
    lp = {the: -0.5, march: -1.0, space_march: -3.0}
    starts = first_tokens(tables, _scores(tables, lp), prefix, tokens)
    by_char = {tokens[i - 1][1][0]: value for i, value in starts.items(tables)}
    # the model writes "March" without its space; the document's token
    # has one, and the likelier spelling starts it; "The" twice
    assert by_char[text.index(" March")] == (march, [])
    assert by_char[0] == (the, []) and by_char[26] == (the, [])
    assert set(by_char) == {0, 26, text.index(" March")}
    assert {tokens[i - 1][1][0] for i in starts} == set(by_char)
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


def test_a_stream_writes_a_run_of_whitespace_as_one_space():
    text = "laws of the State of          Arkansas\nshall apply."
    tables = make_vocabulary(text, "laws of the State of Arkansas shall apply.")
    ids, offsets = tables.encode(text)
    prefix = [99] + ids
    tokens = document_tokens(offsets, 1)
    laws = tables.encode_ids("laws")[0]
    starts = first_tokens(tables, _scores(tables, {laws: -0.5}), prefix, tokens)
    (stream,) = make_streams(tables, text, starts, laws)
    stream.extend(tables)
    # the model writes " Arkansas" right after "of"; the stream does too,
    # and its end is the document's, past the spaces
    assert stream.stream == tables.encode_ids(
        "laws of the State of Arkansas shall apply.")
    assert text[stream.char:stream.ends[5]] == "laws of the State of          Arkansas"
    assert text[stream.char:stream.ends[6]].endswith("Arkansas\nshall")
    assert squeezed("a  b\n\n c") == ("a b c", [1, 3, 4, 7, 8])


def test_a_stream_tokenizes_a_window_at_a_time():

    text = " ".join(["word"] * 300)
    tables = make_vocabulary(text)
    ids, offsets = tables.encode(text)
    prefix = [99] + ids
    tokens = document_tokens(offsets, 1)
    word = tables.encode_ids("word")[0]
    starts = first_tokens(tables, _scores(tables, {word: -0.5}), prefix, tokens)
    streams = make_streams(tables, text, starts, word)
    assert len(streams) == 300
    first = streams[0]
    # one window of the text is tokenized ahead; the rest on demand
    window_tokens = len(first.stream)
    assert window_tokens < 300 and not first.exhausted
    assert first.next(window_tokens + 5, tables) == tables.encode_ids(" word")[0]
    assert len(first.stream) > window_tokens
    assert first.next(10_000, tables) is None and first.exhausted
    assert text[first.char:first.ends[-1]] == text
    assert candidates_mod.STREAM_WINDOW == 512

def test_a_first_token_that_begins_a_longer_word_carries_its_rest():
    text = "The lease is their lease."
    tables = make_vocabulary(text, "their", "ir")
    ids, offsets = tables.encode(text)
    prefix = [99] + ids
    tokens = document_tokens(offsets, 1)
    the = tables.encode_ids("The")[0]
    starts = first_tokens(tables, _scores(tables, {the: -0.5}), prefix, tokens)
    by_char = {tokens[i - 1][1][0]: value for i, value in starts.items(tables)}
    # "the" begins "their" too: that start continues with "ir"
    assert by_char[0] == (the, [])
    assert by_char[12] == (the, tables.encode_ids("ir"))
    streams = make_streams(tables, text, starts, the)
    assert [stream.char for stream in streams] == [0, 12]
    assert streams[1].stream[:2] == [the, tables.encode_ids("ir")[0]]
    assert text[streams[1].char:streams[1].ends[1]] == " their"

def test_vocabulary_pairs_each_token_with_its_quote_closing_merge():
    tokenizer = FakeTokenizer([LEASE, TAIL])
    tokenizer.tokens.extend(['."', '""', 'Ġ"'])
    tokenizer.ids = {token: index for index, token in enumerate(tokenizer.tokens)}
    tables = Vocabulary(tokenizer)
    ids = tokenizer.ids
    expected = np.full(len(tokenizer), -1, dtype=np.int64)
    expected[[ids["."], ids['"'], ids[" "]]] = [
        ids['."'], ids['""'], ids['Ġ"']]
    np.testing.assert_array_equal(tables.merged, expected)
    assert set(tables.stop_ids) == {ids['"'], ids['""']}
    assert np.flatnonzero(tables.is_stop).tolist() == sorted(tables.stop_ids)


def test_stream_batches_preserve_tokens_and_offsets_across_windows(monkeypatch):
    texts = [" ".join(["word"] * 300), "word   \n word  ", "word"]
    tables = make_vocabulary(*texts)
    streams = []
    expected = []
    word = tables.encode_ids("word")[0]
    for text in texts:
        ids, offsets = tables.encode(text)
        starts = first_tokens(tables, _scores(tables, {word: -0.5}), ids,
                              document_tokens(offsets, 0))
        streams.append(make_streams(tables, text, starts, word)[0])
        rest, ends = squeezed(text)
        ids, offsets = tables.encode(rest)
        expected.append((ids, [ends[b - 1] for _, b in offsets]))
    batches = []
    encode_batch = tables.encode_batch

    def recorded(texts):
        batches.append(len(texts))
        return encode_batch(texts)

    def scalar(text):
        raise AssertionError("stream windows must be tokenized together")

    monkeypatch.setattr(tables, "encode_batch", recorded)
    monkeypatch.setattr(tables, "encode", scalar)
    ensure_tokens(streams, 80, tables)
    assert batches == [2, 1]
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


def test_normalized_windows_preserve_partial_whitespace_runs():
    text = " a\t\tbc\r\n  d\u2003e\u00a0 f   "
    normalized = SqueezedText(text)
    for start in range(len(text) + 1):
        for end in range(start, len(text) + 1):
            want_text, want_ends = squeezed(text[start:end])
            got_text, got_ends = normalized.window(start, end)
            assert got_text == want_text
            assert got_ends.tolist() == [start + value for value in want_ends]


def test_questions_reuse_candidate_positions_and_normalized_text():
    tables = make_vocabulary()
    ids, offsets = tables.encode(RANKED)
    scan = DocumentTokens.of(tables, ids, document_tokens(offsets, 0))
    the, march = (tables.encode_ids(word)[0] for word in ("The", "March"))
    first = first_tokens(tables, _scores(tables, {the: -0.5}), ids, None, scan)
    streams = make_streams(tables, RANKED, first, the)
    index = scan.candidates
    normalized = scan.normalized
    second = first_tokens(tables, _scores(tables, {march: -0.1}), ids, None, scan)
    more = make_streams(tables, RANKED, second, march)
    assert scan.candidates is index
    assert all(stream.normalized is normalized for stream in streams + more)
    assert first.best_token == the and second.best_token == march
    assert first.best_lp == -0.5


def test_empty_document_has_no_candidate_starts():
    tables = make_vocabulary()
    starts = first_tokens(tables, _scores(tables, {}), [], [])
    assert starts.best_lp == -np.inf
    assert not starts.allows(tables.none_id)


def test_candidate_ties_prefer_longer_spelling():
    text = "their"
    tables = make_vocabulary(text, "the", "The", "ir")
    ids, offsets = tables.encode(text)
    whole, short = (tables.encode_ids(word)[0] for word in ("their", "The"))
    scan = DocumentTokens.of(tables, ids, document_tokens(offsets, 0))
    tied = first_tokens(tables, _scores(tables, {whole: -1, short: -1}),
                        ids, None, scan)
    assert tied.best_token == whole
    higher = first_tokens(tables, _scores(tables, {whole: -2, short: -1}),
                          ids, None, scan)
    assert higher.best_token == short
    assert list(higher.items(tables)) == [(0, (short, tables.encode_ids("ir")))]

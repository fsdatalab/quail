"""The extraction's candidates: word keys, first tokens, and streams."""

from extract_fakes import LEASE, RANKED, TAIL, FakeTokenizer, make_vocabulary

from quail.backends.quail.executor.extract import candidates as candidates_mod
from quail.backends.quail.executor.extract.candidates import (
    document_tokens,
    first_tokens,
    make_streams,
    squeezed,
)
from quail.backends.quail.executor.extract.vocabulary import Vocabulary, same_word


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
    starts = first_tokens(tables, lp, prefix, tokens)
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
    starts = first_tokens(tables, {laws: -0.5}, prefix, tokens)
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
    starts = first_tokens(tables, {word: -0.5}, prefix, tokens)
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
    starts = first_tokens(tables, {the: -0.5}, prefix, tokens)
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
    assert tables.merged == {ids["."]: ids['."'], ids['"']: ids['""'],
                             ids[" "]: ids['Ġ"']}
    assert set(tables.stop_ids) == {ids['"'], ids['""']}

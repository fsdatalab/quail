"""The vocabulary index an extraction reads: each token's word and its spellings.

A document token may be written without its leading space, in another
case, or as a prefix token followed by the rest of the word. The index
groups the vocabulary by word so a document's positions can be
matched against a row's likeliest tokens by word, and it names the
tokens that close an answer.
"""

from functools import cached_property

import numpy as np
import pyarrow as pa

from quail.catalog.tokens import GigatokenEncoder, TokenOffsets, TokenView
from quail.progress import say


def same_word(token) -> str:
    """The key under which tokens differing only in a leading space or case meet."""
    if not isinstance(token, str):
        return ""
    text = token[1:] if token.startswith("Ġ") else token
    return text.lower() if text and "Ġ" not in text and "Ċ" not in text else ""


class Vocabulary:
    """One model's tokenizer and the vocabulary index its extractions read.

    The tokenizer and the stop tokens are ready at once; AI.MAP needs only
    those. The word index that AI.EXTRACT reads is built on first use.

    Args:
        tokenizer: A Gigatoken encoder or a tokenizer supplied by the caller.
        turn_suffix: The model's chat-turn text after a prompt; its first
            token closes a turn. Empty for a model without chat turns.
    """

    def __init__(self, tokenizer, turn_suffix: str = ""):
        self.encoder = tokenizer if isinstance(tokenizer, GigatokenEncoder) else None
        if self.encoder is not None:
            tokenizer = self.encoder.tokenizer.as_hf()
        self.tokenizer = tokenizer
        # the answer ends where the model ends its turn: the turn close
        # or the end of text, which no document holds
        stops = {self.encode_ids(turn_suffix)[0]} if turn_suffix else set()
        if tokenizer.eos_token_id is not None:
            stops.add(int(tokenizer.eos_token_id))
        if not stops:
            raise ValueError("AI.EXTRACT needs a turn close or an end-of-text token")
        self.stop_ids = tuple(sorted(stops))
        self.is_stop = np.zeros(len(tokenizer), dtype=bool)
        self.is_stop[list(self.stop_ids)] = True

    @cached_property
    def offsets(self):
        """Return the token byte table that maps tokens to source characters."""
        if self.encoder is None:
            return None
        return TokenOffsets(self.encoder.token_bytes())

    @cached_property
    def vocab(self) -> list:
        """Return each token id's token text."""
        return self.tokenizer.convert_ids_to_tokens(list(range(len(self.tokenizer))))

    @cached_property
    def _word_index(self) -> tuple:
        """Return the spellings of each word, every key, and each token's key."""
        # a document token may be written without its leading space
        # or in another case after the opening quote: every
        # vocabulary token with the same text up to those can start it
        variants = {}
        # every token's key: its word, or the token itself without one
        keys = []
        key_ids = {}
        key_of = np.empty(len(self.vocab), dtype=np.int64)
        for index, token in enumerate(self.vocab):
            key = same_word(token)
            if key:
                variants.setdefault(key, []).append(index)
            else:
                key = index
            kid = key_ids.get(key)
            if kid is None:
                kid = len(keys)
                keys.append(key)
                key_ids[key] = kid
            key_of[index] = kid
        return variants, keys, key_of

    @property
    def variants(self) -> dict:
        """Return the tokens of each word, keyed by the word."""
        return self._word_index[0]

    @property
    def keys(self) -> list:
        """Return every key: a word, or a token id for a token without one."""
        return self._word_index[1]

    @property
    def key_of(self) -> np.ndarray:
        """Return each token id's key index."""
        return self._word_index[2]

    @cached_property
    def _options(self) -> list:
        return [None] * len(self.keys)

    @cached_property
    def _fillers(self) -> list:
        return [None] * len(self.vocab)

    def options(self, kid: int) -> tuple:
        """Return the tokens that can begin a key's word, and their lengths.

        A word's token, in any spelling, begins it whole; so does any
        token spelling a prefix of an ASCII word, followed by the rest
        of the word. A token without a word begins itself alone.
        """
        found = self._options[kid]
        if found is None:
            key = self.keys[kid]
            if isinstance(key, str) and key.isascii():
                pairs = [(t, j) for j in range(len(key), 0, -1)
                         for t in self.variants.get(key[:j], [])]
            elif isinstance(key, str):
                pairs = [(t, len(key)) for t in self.variants[key]]
            else:
                pairs = [(key, len(self.vocab[key]))]
            found = (np.array([t for t, _ in pairs], dtype=np.int64),
                     np.array([j for _, j in pairs], dtype=np.int64))
            self._options[kid] = found
        return found

    def option_table(self, kids):
        """Pack the requested word spellings into one vocabulary-indexed table."""
        kids = np.unique(np.asarray(kids, dtype=np.int64))
        options = [self.options(int(kid)) for kid in kids]
        counts = np.zeros(len(self.keys), dtype=np.int64)
        counts[kids] = [len(tokens) for tokens, _ in options]
        tokens = np.concatenate([tokens for tokens, _ in options]) if options else []
        lengths = np.concatenate([lengths for _, lengths in options]) if options else []
        return (np.r_[0, np.cumsum(counts)], np.asarray(tokens, dtype=np.int64),
                np.asarray(lengths, dtype=np.int64))

    def filler(self, token: int, length: int) -> list:
        """Return the rest of a token's word after its first length characters."""
        text = self.vocab[token]
        text = text[1:] if text.startswith("Ġ") else text
        if length >= len(text):
            return []
        cached = self._fillers[token]
        if cached is None:
            cached = self._fillers[token] = [None] * len(text)
        found = cached[length]
        if found is None:
            found = cached[length] = self.encode_ids(text[length:])
        return found

    def document_batch(self, texts, stored) -> list[TokenView]:
        """Return document token views with source character ranges.

        Stored views with offsets are reused. Other inputs are encoded
        together through the same tokenizer used for continuations.
        """
        missing = [index for index, tokens in enumerate(stored)
                   if not isinstance(tokens, TokenView)
                   or tokens.character_offsets is None]
        result = list(stored)
        for index, (ids, offsets) in zip(
                missing, self.encode_batch([texts[index] for index in missing])):
            bounds = np.asarray(offsets, dtype=np.int64).reshape(-1)
            result[index] = TokenView(
                pa.array(ids, type=pa.int32()),
                pa.FixedSizeListArray.from_arrays(pa.array(bounds), 2))
        return result

    def encode(self, text: str) -> tuple:
        """Return the token ids of a text and each token's character range."""
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list) -> list:
        """Return (token ids, character ranges) of each text, encoded together."""
        if not texts:
            return []
        if self.encoder is not None:
            ids, offsets, boundaries = self.encode_packed(texts)
            return [(ids[start:end], offsets[start:end])
                    for start, end in zip(boundaries[:-1], boundaries[1:])]
        backend = getattr(self.tokenizer, "backend_tokenizer", None)
        if backend is not None:
            return [(enc.ids, enc.offsets) for enc in backend.encode_batch(
                list(texts), add_special_tokens=False)]
        enc = self.tokenizer(list(texts), return_offsets_mapping=True,
                             add_special_tokens=False)
        return [(list(ids), list(offsets))
                for ids, offsets in zip(enc["input_ids"], enc["offset_mapping"])]

    def encode_packed(self, texts):
        """Return flat tokens, character ranges and row boundaries."""
        source = pa.array(texts, type=pa.large_string())
        if not len(source):
            return (np.empty(0, dtype=np.int64), np.empty((0, 2), dtype=np.int64),
                    np.zeros(1, dtype=np.int64))
        if self.encoder is not None:
            ids, boundaries = self.encoder.batch_numpy(source)
            tokens = pa.LargeListArray.from_arrays(boundaries, pa.array(ids))
            offsets, exact = self.offsets.numpy(tokens, source)
            if not exact.to_numpy(zero_copy_only=False).all():
                raise ValueError(
                    "AI.EXTRACT requires token bytes that preserve the source text")
            return ids, offsets, boundaries
        encoded = self.encode_batch(source.to_pylist())
        boundaries = np.r_[0, np.cumsum([len(ids) for ids, _ in encoded])]
        ids = np.concatenate([ids for ids, _ in encoded]).astype(np.int64)
        offsets = np.concatenate([
            np.asarray(offsets, dtype=np.int64).reshape(-1, 2)
            for _, offsets in encoded])
        return ids, offsets, boundaries

    def encode_ids(self, text: str) -> list:
        return list(self.tokenizer.encode(text, add_special_tokens=False))


def load_vocabulary(state) -> Vocabulary:
    """Return the loaded model's vocabulary index, loading its tokenizer once."""
    loaded = state.loaded_model
    if loaded.vocabulary is None:
        from quail.specs.files import resolve_model_path

        spec = loaded.model_spec
        path = resolve_model_path(spec.hf_name, spec.revision)
        say(f"loading the tokenizer of {spec.name} for AI.EXTRACT and AI.MAP")
        loaded.vocabulary = Vocabulary(GigatokenEncoder(path), spec.turn_suffix)
    return loaded.vocabulary

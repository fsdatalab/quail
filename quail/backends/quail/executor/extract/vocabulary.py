"""The vocabulary index an extraction reads: each token's word and its spellings.

A document token may be written without its leading space, in another
case, or as a prefix token followed by the rest of the word. The index
groups the vocabulary by word so a document's positions can be
matched against a row's likeliest tokens by word.
"""

import numpy as np

from quail.backends.quail.executor.tokenizer import ModelTokenizer


def same_word(token) -> str:
    """The key under which tokens differing only in a leading space or case meet."""
    if not isinstance(token, str):
        return ""
    text = token[1:] if token.startswith("Ġ") else token
    return text.lower() if text and "Ġ" not in text and "Ċ" not in text else ""


def word_index(vocab: list) -> tuple[dict, list, np.ndarray]:
    """Group a vocabulary's tokens by word, as same_word keys them.

    Args:
        vocab: Each token id's token text.

    Returns:
        The tokens of each word, keyed by the word; every key in order of
        its first token, a word or, for a token without one, its token
        id; and each token id's key index.
    """
    variants = {}
    key_ids = {}
    key_of = []
    for index, token in enumerate(vocab):
        key = same_word(token)
        if key:
            variants.setdefault(key, []).append(index)
        else:
            key = index
        key_of.append(key_ids.setdefault(key, len(key_ids)))
    return variants, list(key_ids), np.array(key_of, dtype=np.int64)


class Vocabulary:
    """The word index of one model's vocabulary, which its extractions read.

    Args:
        tokenizer: The model's tokenizer.
    """

    def __init__(self, tokenizer: ModelTokenizer):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.token_texts()
        # a document token may be written without its leading space
        # or in another case after the opening quote: every
        # vocabulary token with the same text up to those can start it
        self.variants, self.keys, self.key_of = word_index(self.vocab)
        self._options = [None] * len(self.keys)
        self._fillers = [None] * len(self.vocab)

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
            found = cached[length] = self.tokenizer.encode_ids(text[length:])
        return found

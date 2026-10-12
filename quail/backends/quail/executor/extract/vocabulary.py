"""The vocabulary index an extraction reads: each token's word and its spellings.

A document token may be written without its leading space, in another
case, or as a prefix token followed by the rest of the word. The index
groups the vocabulary by word so a document's positions can be
matched against a row's likeliest tokens by word, and it names the
tokens that close an answer.
"""

import numpy as np

from quail.progress import say


def same_word(token) -> str:
    """The key under which tokens differing only in a leading space or case meet."""
    if not isinstance(token, str):
        return ""
    text = token[1:] if token.startswith("Ġ") else token
    return text.lower() if text and "Ġ" not in text and "Ċ" not in text else ""


class Vocabulary:
    """One model's tokenizer and the vocabulary index its extractions read.

    Args:
        tokenizer: A Hugging Face fast tokenizer, which returns the
            character offsets of the tokens it produces.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
        # a document token may be written without its leading space
        # or in another case after the opening quote: every
        # vocabulary token with the same text up to those can start it
        self.variants = {}
        # every token's key: its word, or the token itself without one
        self.keys = []
        key_ids = {}
        key_of = np.empty(len(self.vocab), dtype=np.int64)
        for index, token in enumerate(self.vocab):
            key = same_word(token)
            if key:
                self.variants.setdefault(key, []).append(index)
            else:
                key = index
            kid = key_ids.get(key)
            if kid is None:
                kid = len(self.keys)
                self.keys.append(key)
                key_ids[key] = kid
            key_of[index] = kid
        self.key_of = key_of
        self._options = {}      # key id -> (its first tokens, their lengths)
        self._fillers = {}      # (token, length) -> the rest of its word
        self.quote_id = tokenizer.convert_tokens_to_ids('"')
        # a decoder closes the answer with any token that opens with the
        # quote, such as `".` or `",`: together they are the stop
        self.stop_ids = tuple(index for index, token in enumerate(self.vocab)
                              if token.startswith('"'))
        self._stops = set(self.stop_ids)
        # a decoder may copy a token and close the quote in one token,
        # such as `."` or `)"`: the merged token of each token that has one
        index = {token: i for i, token in enumerate(self.vocab)}
        self.merged = {index[token[:-1]]: i for token, i in index.items()
                       if len(token) > 1 and token.endswith('"')
                       and token[:-1] in index}
        self.none_id = self.encode_ids("none")[0]

    def options(self, kid: int) -> tuple:
        """Return the tokens that can begin a key's word, and their lengths.

        A word's token, in any spelling, begins it whole; so does any
        token spelling a prefix of an ASCII word, followed by the rest
        of the word. A token without a word begins itself alone.
        """
        found = self._options.get(kid)
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

    def filler(self, token: int, length: int) -> list:
        """Return the rest of a token's word after its first length characters."""
        found = self._fillers.get((token, length))
        if found is None:
            text = self.vocab[token]
            text = text[1:] if text.startswith("Ġ") else text
            found = self.encode_ids(text[length:]) if length < len(text) else []
            self._fillers[(token, length)] = found
        return found

    def encode(self, text: str) -> tuple:
        """Return the token ids of a text and each token's character range."""
        enc = self.tokenizer(text, return_offsets_mapping=True,
                             add_special_tokens=False)
        return list(enc["input_ids"]), list(enc["offset_mapping"])

    def encode_batch(self, texts: list) -> list:
        """Return (token ids, character ranges) of each text, encoded together."""
        if not texts:
            return []
        backend = getattr(self.tokenizer, "backend_tokenizer", None)
        if backend is not None:
            return [(enc.ids, enc.offsets) for enc in backend.encode_batch(
                list(texts), add_special_tokens=False)]
        enc = self.tokenizer(list(texts), return_offsets_mapping=True,
                             add_special_tokens=False)
        return [(list(ids), list(offsets))
                for ids, offsets in zip(enc["input_ids"], enc["offset_mapping"])]

    def encode_ids(self, text: str) -> list:
        return list(self.tokenizer.encode(text, add_special_tokens=False))


def load_vocabulary(state) -> Vocabulary:
    """Return the loaded model's vocabulary index, loading its tokenizer once."""
    loaded = state.loaded_model
    if loaded.vocabulary is None:
        from transformers import AutoTokenizer

        from quail.backends.quail.executor.model import resolve_model_path

        spec = loaded.model_spec
        path = resolve_model_path(spec.hf_name, spec.revision)
        say(f"loading the tokenizer of {spec.name} for AI.EXTRACT")
        loaded.vocabulary = Vocabulary(AutoTokenizer.from_pretrained(path))
    return loaded.vocabulary

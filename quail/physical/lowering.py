"""A prompt lowered to the token ids one model reads.

A logical plan's prompts are text. Planning lowers each once per query
with the session's tokenizer: the token ids before each document and
after it, the frame around a join's records, and a classification's
lettered form when the tokenizer has a one-token letter per label. The
Rust planner in ``_quail_native`` does the lowering. A tokenizer that
returns words instead of integer ids, as the tests' does, crosses to
Rust with each word interned as an integer id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from quail.common.expressions import CompileError, Prompt
from quail.physical.codec import decode_value, encode_value

# interned word ids start here, above any vocabulary
_WORDS = 1 << 40



class Vocabulary:
    """A tokenizer's ids as Rust reads them, with words interned."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids, self.words = {}, {}

    def intern(self, token):
        """The integer id of a token: itself, or an interned word's."""
        if isinstance(token, (int, np.integer)) and not isinstance(token, bool):
            return int(token)
        if token not in self.ids:
            self.ids[token] = _WORDS | len(self.ids)
            self.words[self.ids[token]] = token
        return self.ids[token]

    def word(self, token):
        """The token an id stands for."""
        return self.words.get(token, token)

    def tokens(self, text):
        """Tokenize text into integer ids."""
        return [self.intern(token) for token in self.tokenizer(text)]

    def decoder(self):
        """Decode a list of ids, or None for a tokenizer that cannot decode."""
        decode = getattr(self.tokenizer, "decode", None)
        if decode is None:
            return None

        def decoded(ids):
            text = decode([self.word(token) for token in ids])
            return (text.decode("utf-8", errors="replace")
                    if isinstance(text, bytes) else str(text))

        return decoded


@dataclass(frozen=True)
class LoweredPrompt:
    """A prompt's text with the token ids one tokenizer gives it.

    Attributes:
        prompt: The text prompt.
        preamble_token_ids: The preamble before the document.
        tail_token_ids: The text after the document, or after a join's
            last partner.
        frame_tokens: Tokens of the frame after the anchor document.
        label_token_ids: Per placeholder of a join or a classification
            of joined rows, (alias, its label's ids, its frame's ids).
        lettered: The classification prompt lowered with one letter
            per label, or None.
    """

    prompt: Prompt
    preamble_token_ids: tuple
    tail_token_ids: tuple
    frame_tokens: int
    label_token_ids: tuple = ()
    lettered: "LoweredPrompt | None" = None

    @property
    def preamble_tokens(self) -> int | None:
        """The preamble's token count; None for a score, laid out later."""
        return None if self.prompt.kind == "score" else len(self.preamble_token_ids)

    @property
    def tail_tokens(self) -> int | None:
        """The tail's token count; None for a score, laid out later."""
        return None if self.prompt.kind == "score" else len(self.tail_token_ids)

    @property
    def labels(self) -> tuple:
        """Per placeholder, (alias, label tokens, frame tokens)."""
        return tuple((alias, len(label), len(frame))
                     for alias, label, frame in self.label_token_ids)

    @property
    def template(self) -> str:
        return self.prompt.template

    @property
    def args(self) -> tuple:
        return self.prompt.args

    @property
    def preamble(self) -> str:
        return self.prompt.preamble

    @property
    def tail(self) -> str:
        return self.prompt.tail

    @property
    def layout(self) -> str:
        return self.prompt.layout

    @property
    def tail_segments(self) -> tuple:
        return self.prompt.tail_segments

    @property
    def label_prefix(self) -> str:
        return self.prompt.label_prefix

    @property
    def letters(self) -> tuple:
        return self.prompt.letters


def _decode_prompt(value):
    return decode_value(Prompt, value)


def _lowered(found: dict, vocabulary: Vocabulary) -> LoweredPrompt:
    def words(ids):
        return tuple(vocabulary.word(token) for token in ids)

    return LoweredPrompt(
        prompt=_decode_prompt(found["prompt"]),
        preamble_token_ids=words(found["preamble_token_ids"]),
        tail_token_ids=words(found["tail_token_ids"]),
        frame_tokens=found["frame_tokens"],
        label_token_ids=tuple((alias, words(label), words(frame))
                              for alias, label, frame in found["label_token_ids"]),
        lettered=(None if found.get("lettered") is None
                  else _lowered(found["lettered"], vocabulary)))


def _call(name: str, arguments: list, tokenizer):
    import _quail_native

    vocabulary = Vocabulary(tokenizer)
    found, message = _quail_native.prompt_lowering(
        name, json.dumps(arguments), vocabulary.tokens)
    if message is not None:
        raise CompileError(message)
    return json.loads(found), vocabulary


def model_fields(model) -> dict:
    """The model fields prompt lowering reads."""
    return {"role": model.role,
            "answer_spellings": [list(words) for words in model.answer_spellings],
            "turn": list(model.turn), "prompt_layout": model.prompt_layout}


def lower_prompt(prompt: Prompt, tokenizer) -> LoweredPrompt:
    """Lower a prompt's text to the token ids the tokenizer gives it.

    Raises:
        CompileError: The prompt cannot be lowered for this tokenizer.
    """
    found, vocabulary = _call("lower_prompt", [encode_value(prompt)], tokenizer)
    return _lowered(found, vocabulary)


def answer_ids(model, tokenizer) -> tuple[list, list]:
    """Return the token ids a model answers true and false with."""
    (true_ids, false_ids), vocabulary = _call(
        "answer_ids", [model_fields(model)], tokenizer)
    return ([vocabulary.word(token) for token in true_ids],
            [vocabulary.word(token) for token in false_ids])

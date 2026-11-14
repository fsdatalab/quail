"""Fakes for the extraction tests: a word-level tokenizer and small documents."""

import re

import numpy as np

from quail.backends.quail.executor.extract.vocabulary import Vocabulary
from quail.physical import ExtractSpec

NEG = float("-inf")
QUESTION = "When does the lease end?"
LEASE = ("Contract of lease.\nThe lease ends on\nMarch 3, 2027 at noon.\n"
         "Rent is $2,000.")
TERM = "Term: five years"
NOTHING = "No dates here."
HEAD = "DOCUMENT:\n"
TAIL = " question\nPhrase:"
QUOTED = 'Lease of the "Premises" for five years.'
NONE_HELD = "None of the parties may assign this Agreement."


class FakeTokenizer:
    """Words, numbers, and punctuation as tokens, spaced like a byte-level BPE.

    It has Tokenizer's methods. A token after a space is spelled with
    "Ġ" and a line break "Ċ", as a byte-level BPE spells them.
    """

    PIECES = re.compile(r"\n| ?[A-Za-z]+|[0-9]| ?[^\sA-Za-z0-9]| ")

    def __init__(self, texts):
        self.tokens = ['"', "none", " none", "-", "\n", "<|im_end|>", "Phrase", ":",
                       " question", "Lines"]
        self.tokens.extend(str(digit) for digit in range(10))
        for text in texts:
            for piece in self.PIECES.findall(text):
                if piece not in self.tokens:
                    self.tokens.append(piece)
        self.ids = {token: index for index, token in enumerate(self.tokens)}
        self.eos_token_id = self.ids["<|im_end|>"]
        self.stop_ids = [self.eos_token_id]

    def __len__(self):
        return len(self.tokens)

    def _pieces(self, text):
        at = 0
        for match in self.PIECES.finditer(text):
            if match.start() != at:
                raise ValueError(f"cannot tokenize {text[at:match.start()]!r}")
            yield match.group(0), match.span()
            at = match.end()
        if at != len(text):
            raise ValueError(f"cannot tokenize {text[at:]!r}")

    def __call__(self, text):
        return [self.ids[piece] for piece, _ in self._pieces(text)]

    def encode_packed(self, texts):
        ids, ranges, boundaries = [], [], [0]
        for text in texts:
            for piece, span in self._pieces(text):
                ids.append(self.ids[piece])
                ranges.extend(span)
            boundaries.append(len(ids))
        return tuple(np.asarray(values, dtype="<i8").tobytes()
                     for values in (ids, ranges, boundaries))

    def token_bytes(self):
        return [token.encode() for token in self.tokens]

    def token_texts(self):
        return [self.spelled(index) for index in range(len(self.tokens))]

    def decode_batch(self, tokens):
        return ["".join(self.tokens[int(t)] for t in ids if int(t) != self.eos_token_id)
                for ids in tokens]

    def spelled(self, index):
        token = self.tokens[index]
        if token == "\n":
            return "Ċ"
        return "Ġ" + token[1:] if token.startswith(" ") else token


RANKED = "The lease ends on March 3. The rent is due."


def make_vocabulary(*extra):
    # a stream writes a run of whitespace as one space, so the
    # vocabulary holds the squeezed texts' tokens too
    tokenizer = FakeTokenizer([LEASE, TERM, NOTHING, HEAD, TAIL, QUOTED, NONE_HELD,
                               "March",
                               "five", RANKED,
                               *(squeezed(text)[0] for text in (LEASE, TERM,
                                                                NOTHING)),
                               *extra])
    return Vocabulary(tokenizer)


def make_spec(*extra):
    tables = make_vocabulary(*extra)
    return ExtractSpec(
        name="ends", aliases=("c",), query_template="{0}",
        arguments=(("c", "text"),), expected_rows=3, estimated_seconds=0.0,
        prompt_token_parts=(tuple(tables.tokenizer(HEAD)),
                            tuple(tables.tokenizer(TAIL))),
        question=QUESTION, draft_tokens=16, draft_policy="fixed"), tables


def squeezed(text: str) -> tuple:
    """Return the text with each run of whitespace as one space, and a map back.

    The model writes a run of spaces or line breaks as one space, so
    the stream follows the squeezed text; the map gives, for each
    squeezed character, the document character it ends at.
    """
    out, ends = [], []
    run = False
    for at, char in enumerate(text):
        if char.isspace():
            if not run:
                out.append(" ")
                ends.append(at + 1)
            else:
                ends[-1] = at + 1
            run = True
        else:
            out.append(char)
            ends.append(at + 1)
            run = False
    return "".join(out), ends

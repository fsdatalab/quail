"""Write the tests' tokenizer, ``tests/fixtures/tokenizer/tokenizer.json``.

Each UTF-8 byte is one token, whose id is the byte's value, except that a
space and one or two letters, such as " A" or " no", is one token, and so
is each answer word in ANSWER_WORDS, with or without a space before it, so
a short answer label and a model's yes and no are one token as a model's
are. Run from the repository
root:

    uv run python tests/helpers/make_tokenizer.py
"""

import json
import string
from pathlib import Path

from tokenizers.pre_tokenizers import ByteLevel

ANSWER_WORDS = ("yes", "Yes", "YES", "no", "No", "NO", "true", "True", "TRUE",
                "false", "False", "FALSE")

PATH = Path(__file__).resolve().parents[1] / "fixtures" / "tokenizer" / "tokenizer.json"


def build() -> dict:
    """Return the tokenizer's JSON."""
    alphabet = ByteLevel.alphabet()
    # the byte-level alphabet in byte order, so a byte's token id is its value
    by_byte = {}
    for character in alphabet:
        encoded = ord(character)
        byte = encoded if encoded < 256 and chr(encoded) == character and (
            33 <= encoded <= 126 or 161 <= encoded <= 172 or 174 <= encoded <= 255
        ) else None
        by_byte[character] = byte
    printable = {byte for byte in by_byte.values() if byte is not None}
    unprintable = [byte for byte in range(256) if byte not in printable]
    others = sorted(character for character, byte in by_byte.items() if byte is None)
    for character, byte in zip(others, unprintable, strict=True):
        by_byte[character] = byte
    vocab = {character: byte for character, byte in by_byte.items()}
    space = next(character for character, byte in by_byte.items() if byte == 32)
    merges = []
    for first in string.ascii_letters:
        merges.append([space, first])
        vocab[space + first] = len(vocab)
    for first in string.ascii_letters:
        for second in string.ascii_letters:
            merges.append([space + first, second])
            vocab[space + first + second] = len(vocab)
    known = {tuple(merge) for merge in merges}
    for word in ANSWER_WORDS:
        for spelled in (word, " " + word):
            pieces = [space if character == " " else character for character in spelled]
            joined = pieces[0]
            for piece in pieces[1:]:
                if (joined, piece) not in known:
                    known.add((joined, piece))
                    merges.append([joined, piece])
                joined += piece
                vocab.setdefault(joined, len(vocab))
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False,
                          "trim_offsets": False, "use_regex": False},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": False,
                    "trim_offsets": False, "use_regex": False},
        "model": {"type": "BPE", "dropout": None, "unk_token": None,
                  "continuing_subword_prefix": None, "end_of_word_suffix": None,
                  "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": merges},
    }


if __name__ == "__main__":
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(build(), ensure_ascii=False))

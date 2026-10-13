"""Reading a copied answer from decoded text and locating it in the document."""

import re

NONE_ANSWER = "none"
PHRASE_CLOSE = '"'


def read_phrase(decoded: str, close: str = PHRASE_CLOSE) -> str | None:
    """Return the words the model wrote after the phrase cue.

    The answer runs to the closing quote, or to the end of the text
    when the model did not close it.

    Args:
        decoded: The text the model wrote after the phrase cue.
        close: The string that closes the answer.

    Returns:
        The answer, or None for an empty answer or the word "none".
    """
    phrase = decoded.split(close, 1)[0].strip()
    if not phrase or phrase.casefold().rstrip(".") == NONE_ANSWER:
        return None
    return phrase


def locate_span(document: str, phrase: str) -> tuple[int, int] | None:
    """Return the character span of the phrase's first occurrence in the document.

    An exact match wins. Otherwise case is ignored and any run of
    whitespace in the phrase matches any run of whitespace in the
    document.

    Returns:
        (start, end), or None when the document does not hold the phrase.
    """
    start = document.find(phrase)
    if start >= 0:
        return start, start + len(phrase)
    words = phrase.split()
    if not words:
        return None
    pattern = r"\s+".join(re.escape(word) for word in words)
    match = re.search(pattern, document, re.IGNORECASE)
    return None if match is None else match.span()


def line_span(document: str, span: tuple[int, int]) -> tuple[int, int]:
    """Widen a span to the whole lines of the document that hold it."""
    start = document.rfind("\n", 0, span[0]) + 1
    end = document.find("\n", max(span[1], start))
    return start, len(document) if end < 0 else end

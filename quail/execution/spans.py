"""Reading a copied answer from decoded text and locating it in the document."""

import re


def read_phrase(decoded: str) -> str | None:
    """Return the words the model wrote after the phrase cue.

    The answer is the whole text, up to where the model ended its turn.

    Args:
        decoded: The text the model wrote after the phrase cue.

    Returns:
        The answer, or None for an empty answer.
    """
    return decoded.strip() or None


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


def _word_span(document: str, phrase: str) -> tuple[int, int] | None:
    """Return the span where the document holds the phrase as whole words.

    An exact match wins; otherwise case and runs of whitespace are
    forgiven, as in locate_span.
    """
    words = phrase.split()
    if not words:
        return None
    head = r"(?<!\w)" if phrase.strip()[0].isalnum() else ""
    tail = r"(?!\w)" if phrase.strip()[-1].isalnum() else ""
    for pattern, flags in ((re.escape(phrase.strip()), 0),
                           (r"\s+".join(re.escape(word) for word in words),
                            re.IGNORECASE)):
        match = re.search(head + pattern + tail, document, flags)
        if match is not None:
            return match.span()
    return None


def held_prefix(document: str, phrase: str) -> tuple[str, tuple[int, int]] | None:
    """Return the longest prefix of the phrase the document holds, and its span.

    The prefix is whole words and marks of the phrase, and the document
    holds it as whole words, as copy decoding copies them.

    Returns:
        (prefix, (start, end)), or None when the document holds no word
        of the phrase.
    """
    ends = [match.end() for match in re.finditer(r"\w+|[^\w\s]", phrase)]
    low, high = 0, len(ends)
    while low < high:
        middle = (low + high + 1) // 2
        if _word_span(document, phrase[:ends[middle - 1]]) is None:
            high = middle - 1
        else:
            low = middle
    if low == 0:
        return None
    prefix = phrase[:ends[low - 1]].strip()
    return prefix, _word_span(document, prefix)

"""The exact prompt text a predicate asks, for filters and joins.

The reference labels answer this text. An engine that runs QUAIL-B
sends the same text, so the text is defined here and not borrowed
from any engine. Every document prefix starts with `SHARED_PRE`; the
question comes after the document so the document's prefix does not
depend on the query.
"""

from __future__ import annotations

import re

# These wrappers match Qwen3 apply_chat_template(enable_thinking=False).
# The user message stays open across the reusable document prefix.
CHAT_PREFIX = "<|im_start|>user\n"
CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PROMPT_FORMAT = "qwen3-chat-nonthinking-v1"
DOCUMENT_PRE = "DOCUMENT:\n"
SHARED_PRE = CHAT_PREFIX + DOCUMENT_PRE

# Fixed strings for join prompt layout.
JOIN_DOC_LABEL = "\n\nDOCUMENT {}:\n"      # each partner block
JOIN_ANCHOR_NOTE = "\n\n(The document above is DOCUMENT {}.)"
JOIN_QUESTION_SEP = "\n\n"                 # anchor note -> question
DATA_PROCESSING_INSTRUCTION = "You are performing a data processing task."
TASK_INSTRUCTION = (
    f"{DATA_PROCESSING_INSTRUCTION} "
    "Evaluate TRUE or FALSE for the following question: "
)
ANSWER_CUE = "\nANSWER:" + CHAT_SUFFIX


def _marker(placeholder: int) -> str:
    return "{%d}" % placeholder


def _check_placeholders(template: str, n_args: int) -> None:
    slots = [int(m) for m in re.findall(r"\{(\d+)\}", template)]
    if sorted(set(slots)) != list(range(n_args)):
        raise ValueError(
            f"template placeholders {sorted(set(slots))} do not match "
            f"{n_args} document(s): expected {{0}}..{{{n_args - 1}}} "
            "each used at least once")


def split_template(template: str) -> tuple[str, str]:
    """Split into (pre-placeholder text, placeholder-onward text)."""
    i = template.find("{")
    if i < 0:
        return template, ""
    return template[:i], template[i:]


def split_frame(template: str) -> tuple[str, str]:
    """Split into (frame, canonical_template).

    Relocates text before the first placeholder to after it, so the
    document comes first and its prefix stays query-independent.
    """
    user_pre, tail = split_template(template)
    if not tail:
        return "", template
    m = re.match(r"\{\d+\}", tail)
    if m is None:
        raise ValueError(
            "template text before the first placeholder must not contain "
            f"a brace that is not a placeholder: {tail[:40]!r}")
    frame = user_pre.strip()
    rest = tail[m.end():]
    return frame, (DOCUMENT_PRE + m.group(0)
                   + (f"\n\n{frame}" if frame else "") + rest)


def render_filter_question(tail: str) -> str:
    """Wrap a filter question with the task instruction and answer cue."""
    content = tail.lstrip("\n")
    sep = tail[:len(tail) - len(content)]
    if not sep:
        sep = "\n\n"
    return sep + TASK_INSTRUCTION + content + ANSWER_CUE


def render_filter_prompt(template: str, document: str) -> str:
    """Return the complete text of one filter question over one document."""
    _check_placeholders(template, 1)
    _frame, canonical = split_frame(template)
    preamble, tail = split_template(canonical)
    m = re.match(r"(\{\d+\})(.*)", tail, re.DOTALL)
    if m is None or not tail.startswith("{0}"):
        raise ValueError(f"unexpected filter template layout: {template!r}")
    return CHAT_PREFIX + preamble + document + render_filter_question(m.group(2))


def join_label(placeholder: int) -> str:
    return JOIN_DOC_LABEL.format(_marker(placeholder))


def render_join_frame(template: str, placeholder: int) -> str:
    """The anchor note and question written once after the anchor document."""
    return (JOIN_ANCHOR_NOTE.format(_marker(placeholder))
            + JOIN_QUESTION_SEP + TASK_INSTRUCTION + template)


def render_join_prompt(template: str, documents, anchor: int = 0) -> str:
    """Return the complete text of one join question over one tuple.

    Args:
        template: Join template with one placeholder per document.
        documents: The document texts in placeholder order.
        anchor: The placeholder whose document comes first.
    """
    documents = tuple(documents)
    _check_placeholders(template, len(documents))
    if len(documents) < 2:
        raise ValueError("a join prompt needs at least two documents")
    if anchor < 0 or anchor >= len(documents):
        raise ValueError(f"join anchor placeholder {anchor} is out of range")
    out = SHARED_PRE + documents[anchor] + render_join_frame(template, anchor)
    for i, document in enumerate(documents):
        if i != anchor:
            out += join_label(i) + document
    return out + ANSWER_CUE


def true_false_ids(tok):
    """The token ids that mean TRUE and FALSE.

    Args:
        tok: A HuggingFace tokenizer; called with add_special_tokens=False.
    """
    true, false = set(), set()
    for w in ("TRUE", " TRUE", "True", " True"):
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if ids:
            true.add(ids[0])
    for w in ("FALSE", " FALSE", "False", " False"):
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if ids:
            false.add(ids[0])
    return true, false

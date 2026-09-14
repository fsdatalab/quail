"""Prompt text, prompt token ids, and template binding."""

from quail.logical.nodes import CompileError, Prompt

# Fixed preamble before every document. Must be a formatting label,
# not an instruction; instruction text here biases short-document
# completions.
SHARED_PRE = "DOCUMENT:\n"


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

# Fixed strings for join prompt layout.
JOIN_DOC_LABEL = "\n\nDOCUMENT {}:\n"      # each partner block
JOIN_ANCHOR_NOTE = "\n\n(The document above is DOCUMENT {}.)"
JOIN_QUESTION_SEP = "\n\n"                 # anchor note -> question
TASK_INSTRUCTION = "Evaluate TRUE or FALSE for the following question: "
ANSWER_CUE = "\nANSWER:"


def _marker(placeholder: int) -> str:
    return "{%d}" % placeholder


def join_label(placeholder: int) -> str:
    return JOIN_DOC_LABEL.format(_marker(placeholder))


def join_anchor_note(placeholder: int) -> str:
    return JOIN_ANCHOR_NOTE.format(_marker(placeholder))


def render_join_question(template: str) -> str:
    """The static join question written once into each anchor's KV."""
    return JOIN_QUESTION_SEP + TASK_INSTRUCTION + template


def render_join_frame(template: str, placeholder: int) -> str:
    """The complete anchor frame, tokenized as one string."""
    return join_anchor_note(placeholder) + render_join_question(template)


def join_anchor_prefix_ids(prompt, placeholder: int, document_ids,
                           tokenizer) -> list:
    """The canonical token prefix shared by every tuple of an anchor."""
    if placeholder < 0 or placeholder >= len(prompt.args):
        raise ValueError(f"join anchor placeholder {placeholder} is out of range")
    return (list(tokenizer(prompt.preamble)) + list(document_ids)
            + list(tokenizer(render_join_frame(prompt.template,
                                               placeholder))))


def join_tuple_suffix_ids(prompt, partners, tokenizer) -> list:
    """Build token ids for partner blocks and answer cue of one tuple."""
    out = []
    for placeholder, document_ids in partners:
        if placeholder < 0 or placeholder >= len(prompt.args):
            raise ValueError(
                f"join partner placeholder {placeholder} is out of range")
        out += tokenizer(join_label(placeholder))
        out += list(document_ids)
    out += tokenizer(prompt.tail)
    return out


def render_join_prompt_ids(prompt, documents, anchor: int,
                           tokenizer) -> list:
    """The complete canonical token ids for one join tuple."""
    if len(documents) != len(prompt.args):
        raise ValueError(
            f"join has {len(prompt.args)} placeholders but received "
            f"{len(documents)} documents")
    prefix = join_anchor_prefix_ids(prompt, anchor, documents[anchor],
                                    tokenizer)
    partners = [(i, ids) for i, ids in enumerate(documents) if i != anchor]
    return prefix + join_tuple_suffix_ids(prompt, partners, tokenizer)


def render_join_prompt_text(prompt, documents, anchor: int) -> str:
    """The complete canonical text for one join tuple."""
    if len(documents) != len(prompt.args):
        raise ValueError(
            f"join has {len(prompt.args)} placeholders but received "
            f"{len(documents)} documents")
    if anchor < 0 or anchor >= len(prompt.args):
        raise ValueError(f"join anchor placeholder {anchor} is out of range")
    out = prompt.preamble + documents[anchor]
    out += render_join_frame(prompt.template, anchor)
    for i, document in enumerate(documents):
        if i != anchor:
            out += join_label(i) + document
    return out + prompt.tail


def render_filter_question(tail: str) -> str:
    """Wrap a filter tail with the task instruction and answer cue."""
    content = tail.lstrip("\n")
    sep = tail[:len(tail) - len(content)]
    if not sep:
        sep = "\n\n"
    return sep + TASK_INSTRUCTION + content + ANSWER_CUE


def render_filter_prompt_ids(prompt, document_ids, tokenizer) -> list:
    """The complete canonical token ids for one filter document."""
    if len(prompt.args) != 1 or not prompt.tail.startswith("{0}"):
        raise ValueError("a filter prompt must start its tail with {0}")
    tail = prompt.tail.replace("{0}", "", 1)
    return (list(tokenizer(prompt.preamble)) + list(document_ids)
            + list(tokenizer(tail)))


def split_template(template: str) -> tuple[str, str]:
    """Split into (pre-placeholder text, placeholder-onward text)."""
    i = template.find("{")
    if i < 0:
        return template, ""
    return template[:i], template[i:]


def split_frame(template: str) -> tuple[str, str]:
    """Split into (frame, canonical_template).

    Relocates user text before the first placeholder to after it,
    so the document's KV stays query-independent.
    """
    import re
    user_pre, tail = split_template(template)
    if not tail:
        return "", template
    m = re.match(r"\{\d+\}", tail)
    if m is None:
        raise CompileError(
            f"template text before the first placeholder must not "
            f"contain a brace that is not a placeholder: {tail[:40]!r}")
    frame = user_pre.strip()
    rest = tail[m.end():]
    return frame, (SHARED_PRE + m.group(0)
                   + (f"\n\n{frame}" if frame else "") + rest)


def canonicalize_template(template: str) -> str:
    return split_frame(template)[1]


def _check_placeholders(template: str, n_args: int) -> None:
    import re
    slots = [int(m) for m in re.findall(r"\{(\d+)\}", template)]
    if sorted(set(slots)) != list(range(n_args)):
        raise CompileError(
            f"PROMPT placeholders {sorted(set(slots))} do not match "
            f"{n_args} argument(s): expected {{0}}..{{{n_args - 1}}} "
            f"each used at least once")


def bind_prompt(template: str, args: tuple, tokenizer=None) -> Prompt:
    """Build a filter Prompt from a template and column arguments.

    Args:
        template: Prompt template with {0}, {1}, ... placeholders.
        args: Column references in placeholder order.
        tokenizer: Optional callable (text -> token list) for counting.
    """
    import re
    _check_placeholders(template, len(args))
    frame, template = split_frame(template)
    preamble, tail = split_template(template)
    # Wrap the question text (after the placeholder) with the task
    # instruction and answer cue.
    m = re.match(r"(\{\d+\})(.*)", tail, re.DOTALL)
    if m:
        ph, question = m.group(1), m.group(2)
        tail = ph + render_filter_question(question)
    pre_tok = tail_tok = frame_tok = None
    pre_ids = tail_ids = ()
    if tokenizer is not None:
        pre_ids = tuple(tokenizer(preamble))
        pre_tok = len(pre_ids)
        tail_text = re.sub(r"\{\d+\}", "", tail)
        tail_ids = tuple(tokenizer(tail_text))
        tail_tok = len(tail_ids)
        frame_tok = len(tokenizer(f"\n\n{frame}")) if frame else 0
    return Prompt(template=template, args=tuple(args), preamble=preamble,
                  tail=tail, preamble_tokens=pre_tok, tail_tokens=tail_tok,
                  frame=frame, frame_tokens=frame_tok,
                  preamble_token_ids=pre_ids, tail_token_ids=tail_ids)


def bind_join_prompt(template: str, args: tuple,
                     tokenizer=None) -> Prompt:
    """Build a join Prompt with one placeholder per table.

    Args:
        template: Prompt template with {0}, {1}, ... placeholders.
        args: Column references in placeholder order (one per table).
        tokenizer: Optional callable (text -> token list) for counting.
    """
    _check_placeholders(template, len(args))
    if len(args) < 2:
        raise CompileError("a join prompt needs at least two "
                           "placeholders, one per joined table")
    aliases = [r.alias for r in args]
    if len(set(aliases)) != len(aliases):
        raise CompileError(
            f"each join placeholder must name a distinct table (one "
            f"document block per table), got aliases {aliases}; to "
            f"mention a table's document again, use its marker in the "
            f"question text, not a second placeholder")
    question = render_join_question(template)
    pre_tok = tail_tok = frame_tok = None
    labels = tuple((a, None, None) for a in aliases)
    pre_ids = tail_ids = ()
    label_ids = ()
    if tokenizer is not None:
        pre_ids = tuple(tokenizer(SHARED_PRE))
        tail_ids = tuple(tokenizer(ANSWER_CUE))
        pre_tok = len(pre_ids)
        tail_tok = len(tail_ids)
        frame_tok = len(tokenizer(question))
        labels = tuple((a, len(tokenizer(join_label(i))),
                        len(tokenizer(render_join_frame(template, i))))
                       for i, a in enumerate(aliases))
        label_ids = tuple(
            (
                alias,
                tuple(tokenizer(join_label(index))),
                tuple(tokenizer(render_join_frame(template, index))),
            )
            for index, alias in enumerate(aliases)
        )
    return Prompt(template=template, args=tuple(args),
                  preamble=SHARED_PRE, tail=ANSWER_CUE,
                  preamble_tokens=pre_tok, tail_tokens=tail_tok,
                  frame=question, frame_tokens=frame_tok, labels=labels,
                  preamble_token_ids=pre_ids, tail_token_ids=tail_ids,
                  label_token_ids=label_ids)

"""Prompt text, prompt token ids, and template binding."""

from dataclasses import replace

from quail.logical.nodes import LABEL_PREFIX, CompileError, Prompt

# Fixed preamble before every document. Must be a formatting label,
# not an instruction; instruction text here biases short-document
# completions. A model's chat-turn text (ModelSpec.turn), when it has
# any, wraps the whole prompt: its opening piece goes before this
# preamble and its closing piece after the answer cue, so the user
# message stays open across the reusable document prefix.
SHARED_PRE = "DOCUMENT:\n"


def shared_preamble(turn_prefix: str) -> str:
    """The text before every document: the model's turn opener, then the preamble."""
    return turn_prefix + SHARED_PRE


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


def true_false_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """The sorted token ids that mean TRUE and FALSE.

    Args:
        tokenizer: Callable text -> token ids, without special tokens.
    """
    true, false = set(), set()
    for words, ids in (
            (("TRUE", " TRUE", "True", " True"), true),
            (("FALSE", " FALSE", "False", " False"), false)):
        for word in words:
            tokens = tokenizer(word)
            if tokens:
                ids.add(tokens[0])
    return sorted(true), sorted(false)

# Fixed strings for join prompt layout.
JOIN_DOC_LABEL = "\n\nDOCUMENT {}:\n"      # each partner block
JOIN_ANCHOR_NOTE = "\n\n(The document above is DOCUMENT {}.)"
JOIN_QUESTION_SEP = "\n\n"                 # anchor note -> question
DATA_PROCESSING_INSTRUCTION = "You are performing a data processing task."
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


def filter_question_text(prompt) -> str:
    """Return the text appended after a filter document."""
    if len(prompt.args) != 1 or not prompt.tail.startswith("{0}"):
        raise ValueError("a filter prompt must start its tail with {0}")
    return prompt.tail[len("{0}"):]


def render_filter_prompt_ids(prompt, document_ids, tokenizer) -> list:
    """The complete canonical token ids for one filter document."""
    return (list(tokenizer(prompt.preamble)) + list(document_ids)
            + list(tokenizer(filter_question_text(prompt))))


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


def bind_prompt(template: str, args: tuple, tokenizer=None,
                turn: tuple[str, str] = ("", "")) -> Prompt:
    """Build a filter Prompt from a template and column arguments.

    Args:
        template: Prompt template with {0}, {1}, ... placeholders.
        args: Column references in placeholder order.
        tokenizer: Optional callable (text -> token list) for counting.
        turn: The model's chat-turn text: the piece before the
            preamble and the piece after the answer cue.
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
    preamble = turn[0] + preamble
    tail = tail + turn[1]
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


CLASSIFY_INSTRUCTION = ("Answer with exactly one of the categories below "
                        "for the following question: ")
LETTERS_INSTRUCTION = ("Answer with the letter of exactly one of the "
                       "categories below for the following question: ")
CATEGORIES_HEADER = "\n\nCategories:"
# Jev's limit on one choice's options, which OpenJev keeps
MAX_LETTERS = 255


def letter_candidates() -> list[str]:
    """A to Z, then a to z, then AA to ZZ."""
    upper = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
    lower = [chr(code) for code in range(ord("a"), ord("z") + 1)]
    return upper + lower + [first + second for first in upper for second in upper]


def choice_letters(count: int, tokenizer=None,
                   prefix: str = LABEL_PREFIX) -> tuple[str, ...]:
    """The letters standing for ``count`` labels, in order.

    Each letter is one token after ``prefix`` under the tokenizer, and
    that token differs from the earlier letters'; without a tokenizer
    the candidates are taken in order. Fewer than ``count`` letters
    come back when the tokenizer has no more one-token letters.

    Args:
        count: How many labels need a letter.
        tokenizer: Optional callable (text -> token list).
        prefix: The text before a scored label.

    Raises:
        CompileError: More labels than MAX_LETTERS.
    """
    if count > MAX_LETTERS:
        raise CompileError(
            f"AI.CLASSIFY takes at most {MAX_LETTERS} labels, got {count}")
    letters, seen = [], set()
    for letter in letter_candidates():
        if tokenizer is not None:
            ids = tuple(tokenizer(prefix + letter))
            if len(ids) != 1 or ids[0] in seen:
                continue
            seen.add(ids[0])
        letters.append(letter)
        if len(letters) == count:
            break
    return tuple(letters)


def answer_prefix(turn: tuple[str, str]) -> str:
    """The text before a scored label under the model's chat turn.

    A label after the answer cue follows a space, as a word would; a
    model whose turn suffix opens its reply after the cue starts the
    reply without one.
    """
    return "" if turn[1] else LABEL_PREFIX


def render_categories(labels, descriptions=(), letters=()) -> str:
    """Return the category list, one line per label.

    A line is ``- label``, or ``- label: description``; lettered,
    ``- A: label`` or ``- A: label (description)``.
    """
    descriptions = descriptions or ("",) * len(labels)
    if letters:
        return CATEGORIES_HEADER + "".join(
            f"\n- {letter}: {label}"
            + (f" ({description})" if description else "")
            for letter, label, description in zip(letters, labels, descriptions))
    return CATEGORIES_HEADER + "".join(
        f"\n- {label}" + (f": {description}" if description else "")
        for label, description in zip(labels, descriptions))


def render_classify_question(question: str, labels, descriptions=(),
                             task_description: str = "",
                             letters=()) -> str:
    """Wrap a classification question with its instruction, categories, and cue."""
    content = question.lstrip("\n")
    sep = question[:len(question) - len(content)] or "\n\n"
    if task_description:
        content = (content + "\n" if content else "") + task_description
    instruction = LETTERS_INSTRUCTION if letters else CLASSIFY_INSTRUCTION
    return (sep + instruction + content
            + render_categories(labels, descriptions, letters) + ANSWER_CUE)


CLASSIFY_LAYOUTS = ("document_first", "labels_first")


def label_text(label: str, prefix: str = LABEL_PREFIX) -> str:
    """The text a label is scored as, appended after the answer cue."""
    return prefix + label


def bind_classify_prompt(template: str, args: tuple, labels, descriptions=(),
                         tokenizer=None,
                         turn: tuple[str, str] = ("", ""),
                         layout: str = "document_first",
                         task_description: str = "") -> Prompt:
    """Build an AI.CLASSIFY Prompt: the document, then the question and labels.

    The layout is the filter layout with the classification
    instruction, a category list, and the answer cue after the
    question. Label text is not part of the tail; each label is scored
    as ``label_text(label, prefix)`` after it. Under the
    ``labels_first`` layout the instruction and categories come before
    the document in the preamble, which every document then shares,
    and the tail is the document and the answer cue. The prompt's
    ``lettered`` is the same prompt with the categories lettered, for
    the ``letters`` scoring rule.

    Args:
        template: Prompt template with one {0} placeholder.
        args: The one column reference.
        labels: The labels in written order.
        descriptions: One description per label, empty for none.
        tokenizer: Optional callable (text -> token list) for counting.
        turn: The model's chat-turn text: the piece before the
            preamble and the piece after the answer cue.
        layout: ``document_first`` or ``labels_first``.
        task_description: Task text added after the question.
    """
    _check_placeholders(template, len(args))
    if len(args) == 2:
        return _bind_joined_classify_prompt(
            template, args, labels, descriptions, tokenizer, turn, layout,
            task_description)
    if len(args) != 1:
        raise CompileError("AI.CLASSIFY reads one document per row, or one "
                           "from each side of a join")
    if layout not in CLASSIFY_LAYOUTS:
        raise CompileError(
            f"unknown AI.CLASSIFY layout {layout!r}; the layouts are "
            f"{CLASSIFY_LAYOUTS}")
    prefix = answer_prefix(turn)
    named = _bind_document_classify_prompt(
        template, args, labels, descriptions, tokenizer, turn, layout,
        task_description, prefix)
    letters = choice_letters(len(labels), tokenizer, prefix)
    if len(letters) < len(labels):
        return named
    lettered = _bind_document_classify_prompt(
        template, args, labels, descriptions, tokenizer, turn, layout,
        task_description, prefix, letters)
    return replace(named, lettered=lettered)


def _bind_document_classify_prompt(template, args, labels, descriptions,
                                   tokenizer, turn, layout, task_description,
                                   prefix, letters=()):
    """Bind one AI.CLASSIFY prompt over one document, lettered when asked."""
    import re
    frame, template = split_frame(template)
    preamble, tail = split_template(template)
    m = re.match(r"(\{\d+\})(.*)", tail, re.DOTALL)
    if m is None:
        raise CompileError(f"unexpected AI.CLASSIFY template: {template!r}")
    question = render_classify_question(
        m.group(2), labels, descriptions, task_description, letters)
    if layout == "labels_first":
        body = question[:len(question) - len(ANSWER_CUE)].lstrip("\n")
        preamble = turn[0] + body + "\n\n" + preamble
        tail = m.group(1) + ANSWER_CUE + turn[1]
    else:
        preamble = turn[0] + preamble
        tail = m.group(1) + question + turn[1]
    pre_tok = tail_tok = frame_tok = None
    pre_ids = tail_ids = ()
    if tokenizer is not None:
        pre_ids = tuple(tokenizer(preamble))
        pre_tok = len(pre_ids)
        tail_ids = tuple(tokenizer(re.sub(r"\{\d+\}", "", tail)))
        tail_tok = len(tail_ids)
        frame_tok = len(tokenizer(f"\n\n{frame}")) if frame else 0
    return Prompt(template=template, args=tuple(args), preamble=preamble,
                  tail=tail, preamble_tokens=pre_tok, tail_tokens=tail_tok,
                  frame=frame, frame_tokens=frame_tok,
                  preamble_token_ids=pre_ids, tail_token_ids=tail_ids,
                  label_prefix=prefix, letters=tuple(letters))


def _bind_joined_classify_prompt(template, args, labels, descriptions,
                                 tokenizer, turn, layout, task_description):
    """Bind a classification of joined rows: the join layout, then the question.

    The anchor document (placeholder {0}) comes first, then the anchor
    note, the partner's label and document, and the classification
    question with its categories and cue. ``label_token_ids`` holds,
    as for a join, each alias's partner label and the anchor note.
    """
    if layout != "document_first":
        raise CompileError(
            "a classification of joined rows uses the document_first layout")
    aliases = [r.alias for r in args]
    if len(set(aliases)) != len(aliases):
        raise CompileError(
            "each placeholder of a classification of joined rows names a "
            "distinct table")
    question = render_classify_question(template, labels, descriptions,
                                        task_description)
    preamble = shared_preamble(turn[0])
    tail = question + turn[1]
    frame = join_anchor_note(0)
    pre_tok = tail_tok = frame_tok = None
    pre_ids = tail_ids = ()
    label_ids = ()
    if tokenizer is not None:
        pre_ids = tuple(tokenizer(preamble))
        tail_ids = tuple(tokenizer(tail))
        pre_tok, tail_tok = len(pre_ids), len(tail_ids)
        frame_tok = len(tokenizer(frame))
        label_ids = tuple(
            (alias, tuple(tokenizer(join_label(index))),
             tuple(tokenizer(join_anchor_note(index))))
            for index, alias in enumerate(aliases))
    return Prompt(template=template, args=tuple(args), preamble=preamble,
                  tail=tail, preamble_tokens=pre_tok, tail_tokens=tail_tok,
                  frame=frame, frame_tokens=frame_tok,
                  preamble_token_ids=pre_ids, tail_token_ids=tail_ids,
                  label_token_ids=label_ids, label_prefix=answer_prefix(turn))


def render_joined_classify_prompt_text(prompt, anchor: str,
                                       partner: str) -> str:
    """The complete text of a classification prompt for one joined row."""
    return (prompt.preamble + anchor + join_anchor_note(0) + join_label(1)
            + partner + prompt.tail)


def bind_score_prompt(template: str, args: tuple,
                      tokenizer=None,
                      turn: tuple[str, str] = ("", "")) -> Prompt:
    """Build an AI.SCORE Prompt that keeps the template as written.

    The reranker renders its own layout, with the query text and the
    document in separate fields, so the template is not rearranged.

    Args:
        template: Prompt template with {0}, and {1} for a pair.
        args: Column references in placeholder order.
        tokenizer: Unused; the planner tokenizes the rendered layout.
        turn: Unused; the reranker layout carries its own turn text.
    """
    _check_placeholders(template, len(args))
    aliases = [r.alias for r in args]
    if len(set(aliases)) != len(aliases):
        raise CompileError(
            f"each AI.SCORE placeholder must name a distinct table, got "
            f"aliases {aliases}")
    return Prompt(template=template, args=tuple(args), preamble="", tail="")


def bind_join_prompt(template: str, args: tuple,
                     tokenizer=None,
                     turn: tuple[str, str] = ("", "")) -> Prompt:
    """Build a join Prompt with one placeholder per table.

    Args:
        template: Prompt template with {0}, {1}, ... placeholders.
        args: Column references in placeholder order (one per table).
        tokenizer: Optional callable (text -> token list) for counting.
        turn: The model's chat-turn text: the piece before the
            preamble and the piece after the answer cue.
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
    preamble = shared_preamble(turn[0])
    tail = ANSWER_CUE + turn[1]
    pre_tok = tail_tok = frame_tok = None
    labels = tuple((a, None, None) for a in aliases)
    pre_ids = tail_ids = ()
    label_ids = ()
    if tokenizer is not None:
        pre_ids = tuple(tokenizer(preamble))
        tail_ids = tuple(tokenizer(tail))
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
                  preamble=preamble, tail=tail,
                  preamble_tokens=pre_tok, tail_tokens=tail_tok,
                  frame=question, frame_tokens=frame_tok, labels=labels,
                  preamble_token_ids=pre_ids, tail_token_ids=tail_ids,
                  label_token_ids=label_ids)

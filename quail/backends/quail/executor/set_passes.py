"""Shared pieces of set-attention passes: the state, and passes of questions.

Under set attention a pass is a state, a document and the text after
it, then up to the model's set size of questions. Each operator builds
its questions; these helpers split questions into passes. The state
ends each document as the plan joined it (JoinDocumentEnds).
"""


def split_passes(state: int, members, lengths, header: int, set_size: int,
                 budget: int) -> list[list]:
    """Split an anchor's members into passes, in order.

    A pass holds at most set_size questions and fits the chunk budget
    with the state; every question after a pass's first carries the
    header.

    Raises:
        ValueError: One question does not fit a chunk with the state.
    """
    passes, current, used = [], [], state
    for member in members:
        tokens = int(lengths[member])
        if state + tokens > budget:
            raise ValueError(
                f"a join question of {tokens} tokens does not fit one "
                f"{budget}-token chunk with its {state}-token state")
        if current and (len(current) == set_size
                        or used + header + tokens > budget):
            passes.append(current)
            current, used = [], state
        used += tokens + (header if current else 0)
        current.append(member)
    if current:
        passes.append(current)
    return passes


def group_passes(prefixes, questions, header, set_size: int, budget: int) -> list:
    """Split a pass group's questions, asked of every document, into passes.

    Every document is asked the same questions, so the passes are the
    same for each; the longest document decides what fits a chunk.

    Args:
        prefixes: Each document's state tokens.
        questions: Each question's tokens, in order.
        header: The tokens that open a question after another.
        set_size: The most questions in one pass.
        budget: The chunk token budget.

    Returns:
        Per pass: its question indices, its suffix tokens, and each
        question's length in the suffix.
    """
    passes = split_passes(max(map(len, prefixes), default=0), range(len(questions)),
                          [len(question) for question in questions], len(header),
                          set_size, budget)
    out = []
    for members in passes:
        suffix, shape = [], []
        for index, member in enumerate(members):
            question = (list(header) if index else []) + list(questions[member])
            suffix += question
            shape.append(len(question))
        out.append((members, suffix, tuple(shape)))
    return out


def spec_questions(specs) -> tuple[list, list, list]:
    """The prompt head, state tail, and questions of specs sharing passes.

    Args:
        specs: Specs whose prompt_token_parts are the head before the
            document and the tail after it, and whose first
            state_tail_tokens tail tokens a pass reads with the document.

    Raises:
        ValueError: The specs differ in prompt head or state tail.
    """
    first = specs[0]
    head, tail = first.prompt_token_parts
    state_tail = list(tail[:first.state_tail_tokens])
    questions = []
    for spec in specs:
        spec_head, spec_tail = spec.prompt_token_parts
        if (tuple(spec_head) != tuple(head)
                or list(spec_tail[:spec.state_tail_tokens]) != state_tail):
            raise ValueError("the questions of one pass share their prompt "
                             "head and state tail")
        questions.append(list(spec_tail[spec.state_tail_tokens:]))
    return list(head), state_tail, questions

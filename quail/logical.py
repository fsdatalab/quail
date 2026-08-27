"""Logical plan operators and prompt binding."""

from dataclasses import asdict, dataclass
from typing import Optional, Union


class CompileError(ValueError):
    """Raised when a query is malformed or outside the language."""


# Fixed preamble before every document. Must be a formatting label,
# not an instruction; instruction text here biases short-document
# completions.
SHARED_PRE = "DOCUMENT:\n"

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


@dataclass(frozen=True)
class ColumnRef:
    alias: str       # table alias in the query ("r")
    provider: str    # provider name in the catalog ("reviews")
    column: str      # column name ("review")


@dataclass(frozen=True)
class Prompt:
    """A bound PROMPT call, split into preamble, frame, and tail.

    Token counts are filled at bind time when a tokenizer is given;
    None means the planner must supply counts.
    """
    template: str
    args: tuple    # tuple[ColumnRef, ...] in placeholder order
    preamble: str
    tail: str
    preamble_tokens: Optional[int] = None
    tail_tokens: Optional[int] = None
    frame: str = ""
    frame_tokens: Optional[int] = None
    # join prompts only: (alias, label_tokens, anchor_frame_tokens)
    # per placeholder, in order. Empty for filter prompts.
    labels: tuple = ()


@dataclass(frozen=True)
class FilterPredicate:
    prompt: Prompt
    selectivity: Optional[float] = None   # fraction of documents that
    #                                       pass; ordering only


@dataclass(frozen=True)
class Scan:
    """Which column of which provider supplies the document text."""
    provider: str
    alias: str
    column: str


@dataclass(frozen=True)
class SemanticFilter:
    input: "Operator"
    predicates: tuple    # tuple[FilterPredicate, ...], written order


@dataclass(frozen=True)
class SemanticJoin:
    """One n-way join: cross product filtered by a single prompt."""
    inputs: tuple    # tuple[Operator]: accumulated tree first, then
    #                  one scan per newly joined table
    predicate: Prompt
    semantics: str = "full"            # full | exists | anti
    selectivity: Optional[float] = None    # fraction of tuples that pass
    anchor: Optional[str] = None       # table alias whose KV is kept;
    #                                    None = planner picks


@dataclass(frozen=True)
class Project:
    """Column projection. Always the root operator."""
    input: "Operator"
    columns: tuple    # tuple[ColumnRef, ...]
    limit: Optional[int] = None


Operator = Union[Scan, SemanticFilter, SemanticJoin, Project]


@dataclass(frozen=True)
class LogicalPlan:
    root: Project

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class JoinSpec:
    """The join predicate (or one EXISTS/anti term), pre-assembly."""
    aliases: tuple             # newly joined table aliases
    prompt: Prompt
    semantics: str = "full"
    selectivity: Optional[float] = None
    anchor: Optional[str] = None


@dataclass(frozen=True)
class QueryDesc:
    """Intermediate description collected by both entry points before assembly."""
    tables: tuple               # ((alias, provider), ...) appearance order
    doc_columns: dict           # alias -> the document column its prompts use
    filters: dict               # alias -> tuple[FilterPredicate], written order
    joins: tuple                # tuple[JoinSpec], written order
    columns: tuple              # tuple[ColumnRef], the projection
    limit: Optional[int] = None


def assemble_plan(desc: QueryDesc) -> LogicalPlan:
    """Build the operator tree from a QueryDesc."""
    def subtree(alias: str):
        provider = dict(desc.tables)[alias]
        node = Scan(provider=provider, alias=alias,
                    column=desc.doc_columns.get(alias, ""))
        preds = desc.filters.get(alias, ())
        if preds:
            node = SemanticFilter(input=node, predicates=tuple(preds))
        return node

    first = desc.tables[0][0]
    tree = subtree(first)
    for j in desc.joins:
        tree = SemanticJoin(
            inputs=(tree,) + tuple(subtree(a) for a in j.aliases),
            predicate=j.prompt, semantics=j.semantics,
            selectivity=j.selectivity, anchor=j.anchor)
    return LogicalPlan(root=Project(input=tree,
                                    columns=tuple(desc.columns),
                                    limit=desc.limit))


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
    if tokenizer is not None:
        pre_tok = len(tokenizer(preamble))
        tail_text = re.sub(r"\{\d+\}", "", tail)
        tail_tok = len(tokenizer(tail_text))
        frame_tok = len(tokenizer(f"\n\n{frame}")) if frame else 0
    return Prompt(template=template, args=tuple(args), preamble=preamble,
                  tail=tail, preamble_tokens=pre_tok, tail_tokens=tail_tok,
                  frame=frame, frame_tokens=frame_tok)


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
            f"document block per table), got aliases {aliases}; refer "
            f"to a table's document again in the question text with "
            f"its marker instead of adding a second placeholder")
    question = render_join_question(template)
    pre_tok = tail_tok = frame_tok = None
    labels = tuple((a, None, None) for a in aliases)
    if tokenizer is not None:
        pre_tok = len(tokenizer(SHARED_PRE))
        tail_tok = len(tokenizer(ANSWER_CUE))
        frame_tok = len(tokenizer(question))
        labels = tuple((a, len(tokenizer(join_label(i))),
                        len(tokenizer(render_join_frame(template, i))))
                       for i, a in enumerate(aliases))
    return Prompt(template=template, args=tuple(args),
                  preamble=SHARED_PRE, tail=ANSWER_CUE,
                  preamble_tokens=pre_tok, tail_tokens=tail_tok,
                  frame=question, frame_tokens=frame_tok, labels=labels)

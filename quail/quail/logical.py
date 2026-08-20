"""The logical plan: four operators, three semantic and one relational.

Both entry points (AI SQL and the builder) produce this tree; the
planner prices it and the coordinator applies the Project at the sink.
Everything here is a frozen dataclass so two compilations of the same
query compare equal.
"""

from dataclasses import asdict, dataclass
from typing import Optional, Union


class CompileError(ValueError):
    """The query is malformed or outside the language. Raised at
    sess.sql() / builder time; nothing is planned, nothing runs."""


# The engine-owned preamble, identical for every operator, every
# query, and every document. Because it is one fixed string, the KV
# of [SHARED_PRE + document] is identical wherever the document
# appears (same tokens, same positions, same attention context), so
# the KV store can hand a filter-scanned document to a join anchor
# and back. Changing this text invalidates every stored extent - the
# session folds it into the store content hash.
#
# The wording is measured, not guessed. Any instruction-like sentence
# here breaks the flag value completions on short documents: "Evaluate
# whether the following is true or false" dropped B1's observed
# selectivity from 0.398 to 0.074, and "You will read a document and
# answer a yes-or-no question about it" dropped it to 0.0014. Long
# documents (B4's ~5,850-token reports) were immune both times - the
# preamble sits too far from the answer position to prime it. So the
# preamble is a formatting label, not an instruction; the task text
# lives in the suffix, after the document.
SHARED_PRE = "DOCUMENT:\n"

# The join prompt's fixed strings. A join renders each tuple as
# labeled document blocks followed by the user's template as the
# question, appended VERBATIM - the placeholders stay in it as
# written ({0}, {1}, ...) and nothing is ever substituted into it.
# Each partner block is labeled with the same marker ("DOCUMENT
# {1}:"), so a marker in the question resolves to its block. The
# anchor's block keeps the bare SHARED_PRE label (so its KV is
# byte-identical to a filter scan of the same document and the store
# serves both); the naming line written right after it - into kept
# KV, once per anchor, never per tuple - maps the top block to its
# marker. Like SHARED_PRE these are formatting labels, not
# instructions.
JOIN_DOC_LABEL = "\n\nDOCUMENT {}:\n"      # each partner block
JOIN_ANCHOR_NOTE = "\n\n(The document above is {}.)"
JOIN_QUESTION_SEP = "\n\n"                 # blocks -> question


def _marker(placeholder: int) -> str:
    return "{%d}" % placeholder


def join_label(placeholder: int) -> str:
    return JOIN_DOC_LABEL.format(_marker(placeholder))


def join_anchor_note(placeholder: int) -> str:
    return JOIN_ANCHOR_NOTE.format(_marker(placeholder))


def render_join_question(template: str) -> str:
    """The per-tuple question text: the user's template exactly as
    written - braces kept, nothing filled in - behind the separator
    that ends the block list."""
    return JOIN_QUESTION_SEP + template


@dataclass(frozen=True)
class ColumnRef:
    alias: str       # table alias in the query ("r")
    provider: str    # provider name in the catalog ("reviews")
    column: str      # column name ("review")


@dataclass(frozen=True)
class Prompt:
    """A PROMPT('template {0} ...', cols...) call, bound and split.

    Filters (bind_prompt): the template is canonicalized so the
    document is inlined at its placeholder - `preamble` is the
    engine's fixed text before it (shared across every document,
    computed once), `tail` is everything from the placeholder on,
    and `frame` is the user's pre-document text, relocated after the
    document and carried at the head of each stage's question.

    Joins (bind_join_prompt): the template is never filled in - it is
    the per-tuple question, appended verbatim after the labeled
    document blocks; its {0}, {1}, ... markers stay in it and refer
    to the blocks. `preamble` is still SHARED_PRE (the anchor block's
    label), `tail` is the question, `labels` carries each
    placeholder's block-label and naming-line token counts, and
    `frame` is empty.

    Token counts are filled at bind time when a tokenizer is
    available; None means the planner must be given counts."""
    template: str
    args: tuple    # tuple[ColumnRef, ...] in placeholder order
    preamble: str
    tail: str
    preamble_tokens: Optional[int] = None
    tail_tokens: Optional[int] = None
    frame: str = ""
    frame_tokens: Optional[int] = None
    # join prompts only: per placeholder, in order,
    # (alias, label_tokens, note_tokens) - the partner block label
    # "\n\nDOCUMENT {i}:\n" and the anchor naming line for that
    # placeholder, counted at bind time so the planner prices any
    # anchor choice without a tokenizer. Empty for filter prompts.
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
    """One n-way join: the cross product of its tables, filtered by a
    single prompt that holds every document at once (the BigQuery /
    Snowflake AI-join shape - never a chain of pairwise stages). One
    placeholder per table; each tuple is one model call. exists/anti
    are the two-table gate form: anchor documents are kept (exists)
    or dropped (anti) on whether any partner answers YES."""
    inputs: tuple    # tuple[Operator]: the accumulated tree first,
    #                  then one (optionally filtered) scan per newly
    #                  joined table, joined order
    predicate: Prompt
    semantics: str = "full"            # full | exists | anti
    selectivity: Optional[float] = None    # fraction of tuples that pass
    anchor: Optional[str] = None       # table alias whose documents
    #                                    anchor (KV kept, partners
    #                                    stream); None = planner picks
    #                                    the cheaper side. exists/anti
    #                                    always anchor on the outer
    #                                    table - the gate applies to it


@dataclass(frozen=True)
class Project:
    """Always the root, never anywhere else. Column selection only:
    ids and pass-through text, nothing computed."""
    input: "Operator"
    columns: tuple    # tuple[ColumnRef, ...]


Operator = Union[Scan, SemanticFilter, SemanticJoin, Project]


@dataclass(frozen=True)
class LogicalPlan:
    root: Project

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class JoinSpec:
    """The join predicate (or one EXISTS/anti term), pre-assembly."""
    aliases: tuple             # newly joined table aliases, joined
    #                            order (one entry: the inner table,
    #                            for exists/anti)
    prompt: Prompt
    semantics: str = "full"
    selectivity: Optional[float] = None
    anchor: Optional[str] = None


@dataclass(frozen=True)
class QueryDesc:
    """What both entry points collect before assembly. Assembling the
    tree from this one description is what makes a SQL query and its
    builder translation compare equal."""
    tables: tuple               # ((alias, provider), ...) appearance order
    doc_columns: dict           # alias -> the document column its prompts use
    filters: dict               # alias -> tuple[FilterPredicate], written order
    joins: tuple                # tuple[JoinSpec], written order
    columns: tuple              # tuple[ColumnRef], the projection


def assemble_plan(desc: QueryDesc) -> LogicalPlan:
    """The deterministic tree: scans (with their filters attached
    directly above - filters always run before joins, section 4's
    unconditional pushdown), folded in written order by the join
    specs (each spec brings every table it joins), Project at the
    root."""
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
                                    columns=tuple(desc.columns)))


def split_template(template: str) -> tuple[str, str]:
    """(preamble, tail): the text before the first placeholder and
    everything from it on."""
    i = template.find("{")
    if i < 0:
        return template, ""
    return template[:i], template[i:]


def split_frame(template: str) -> tuple[str, str]:
    """(frame, canonical_template). The frame is the user's text
    before the first placeholder, stripped; the canonical template is
    SHARED_PRE, then the placeholder (the document whose KV the
    engine owns), then the frame, then everything else.

    User text before the first placeholder would sit inside the
    document's KV and make it query-specific, so it is relocated to
    just after the placeholder instead. A template without a
    placeholder has no document and is returned unchanged (it cannot
    execute anyway)."""
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
    """Build a filter Prompt, checking placeholders against arguments.

    Placeholders must be {0}, {1}, ... matching the argument count and
    order. The template is canonicalized to the engine layout
    (SHARED_PRE + document + frame + suffix) before splitting.
    `tokenizer` is any callable text -> token list; when given, the
    preamble, tail (question text, placeholders excluded), and frame
    are counted once here so the planner never tokenizes."""
    import re
    _check_placeholders(template, len(args))
    frame, template = split_frame(template)
    preamble, tail = split_template(template)
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
    """Build a join Prompt: one prompt over the whole tuple, one
    placeholder per table, evaluated on the cross product.

    The template is never filled in: at run time each tuple renders
    as labeled document blocks (the anchor's block first, under the
    bare SHARED_PRE label plus its naming line; each partner under
    "DOCUMENT {i}:", its own placeholder marker) followed by this
    template as the question, verbatim - the {0}, {1}, ... markers
    stay in it and refer to the blocks. So:

        preamble      SHARED_PRE - the anchor block's label, paid
                      once per anchor document (and byte-identical to
                      a filter scan's stored prefix, which is what
                      lets the store serve both)
        tail          the question (separator + raw template), paid
                      once per tuple
        labels        (alias, label_tokens, note_tokens) per
                      placeholder, in order: a partner block's label
                      is paid once per tuple, the anchor's naming
                      line once per anchor

    Each placeholder must name a distinct table (one block per
    table)."""
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
        tail_tok = len(tokenizer(question))
        frame_tok = 0
        labels = tuple((a, len(tokenizer(join_label(i))),
                        len(tokenizer(join_anchor_note(i))))
                       for i, a in enumerate(aliases))
    return Prompt(template=template, args=tuple(args),
                  preamble=SHARED_PRE, tail=question,
                  preamble_tokens=pre_tok, tail_tokens=tail_tok,
                  frame="", frame_tokens=frame_tok, labels=labels)

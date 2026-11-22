"""A document's last token joined with the text after it, as the tokenizer reads them.

A model that tokenizes a whole prompt at once can merge a document's
last token with the text that follows it: d1-2B reads "great." and a
blank line as "great" and one token for the period and the blank line.
Quail tokenizes documents and prompt
pieces apart, so the planner finds each table's distinct last tokens
and joins each with the text after it once (JoinDocumentEnds); a pass
then ends the document with that join (joined_state).
"""

from quail.common.tokens import concat_tokens


def ends_for(entries, alias, tail) -> dict | None:
    """A node's joins for one table and tail, as last token -> tokens; None for none.

    Args:
        entries: The node's document_ends: (alias, tail, joins) per table
            and tail it reads.
        alias: The table.
        tail: The token ids after its documents.
    """
    tail = tuple(tail)
    for entry_alias, entry_tail, joins in entries:
        if entry_alias == alias and tuple(entry_tail) == tail:
            return dict(joins)
    return None


def joined_state(document, tail, ends):
    """A document followed by the tail, its last token joined as ends gives.

    Args:
        document: The document's token ids, with any prompt head.
        tail: The token ids after it.
        ends: Last token -> its tokens joined with the tail, from the
            plan (JoinDocumentEnds).

    Raises:
        ValueError: The plan joined no ending with this tail, or not this
            document's.
    """
    if not len(document):
        return list(tail)
    if ends is None:
        raise ValueError("the plan joined no document ends with this tail; "
            "plan the query on its documents' exact tokens")
    last = int(document[-1])
    if last not in ends:
        raise ValueError(f"the plan joined no document ending in token {last}")
    return concat_tokens(document[:-1], list(ends[last]))

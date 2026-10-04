"""Label scoring rules and the token trie shared by planning and execution."""

# The label scoring rules the executor runs; each rule returns one of
# the classification's labels.
LABEL_SCORINGS = ("letters", "trie_tree", "trie_decode", "decision_choice")

# every label is one letter token, read at one row
LETTERS_SCORING = "letters"
# every trie node once, in one request: tree attention only
TREE_SCORING = "trie_tree"
# one token per round along the greedy path: fewest tokens, most rounds
DECODE_SCORING = "trie_decode"
# a decision model's head scores one option block per label, in one request
DECISION_SCORING = "decision_choice"


def decodable(labels) -> bool:
    """Return whether every label's token sequence ends at a trie leaf.

    A label that is a proper prefix of another would make the decode
    choose between stopping and continuing, and no rule scores an end
    token.
    """
    ids = {tuple(label) for label in labels}
    return not any(tuple(label[:depth]) in ids
                   for label in labels for depth in range(1, len(label)))


def label_trie(label_ids) -> dict[tuple[int, ...], list[int]]:
    """Return each proper label prefix and the tokens that can follow it.

    Args:
        label_ids: One token id sequence per label.

    Raises:
        ValueError: A label has no tokens.
    """
    children = {}
    for ids in label_ids:
        if not ids:
            raise ValueError("a label has no tokens")
        for depth in range(len(ids)):
            children.setdefault(tuple(ids[:depth]), set()).add(ids[depth])
    return {prefix: sorted(tokens) for prefix, tokens in children.items()}

"""Label scoring rules and the token trie shared by planning and execution."""

from dataclasses import dataclass

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


def word_starts(label_ids, decode) -> tuple[tuple[bool, ...], ...]:
    """Return which tokens of each label begin a new word.

    A token begins a word when its text starts with whitespace and the
    token before it holds a letter or digit; a label's first token never
    does. Punctuation such as "," joins the word after it.

    Args:
        label_ids: One token id sequence per label.
        decode: Callable from a token id list to its text, as str or
            bytes. None marks no token, so every trie node is read.
    """
    if decode is None:
        return tuple((False,) * len(ids) for ids in label_ids)

    def text(token) -> str:
        piece = decode([token])
        return (piece.decode("utf-8", errors="replace")
                if isinstance(piece, bytes) else str(piece))

    flags = []
    for ids in label_ids:
        pieces = [text(token) for token in ids]
        flags.append(tuple(
            depth > 0 and pieces[depth][:1].isspace()
            and any(ch.isalnum() for ch in pieces[depth - 1])
            for depth in range(len(ids))))
    return tuple(flags)


def choice_nodes(label_ids) -> set:
    """Return every trie node with two or more children.

    These are the nodes where a greedy decode chooses between labels;
    a node with one child is passed without a read.
    """
    trie = label_trie(label_ids)
    return {node for node, tokens in trie.items() if len(tokens) > 1}


def read_nodes(label_ids, starts=()) -> set:
    """Return the trie nodes whose rows the trie_tree rule reads.

    The root is read, and so is every node where the labels split: two
    or more children, or a label that ends where another continues.
    After a read node, the next node is read while its token continues
    the chosen word, so a label is scored through the end of the word
    that tells it apart. Nodes after that are fed, when a later read
    needs their KV, but never read.

    Args:
        label_ids: One token id sequence per label.
        starts: Per label, which tokens begin a new word, from
            ``word_starts``. Empty marks no token, so every node is read.
    """
    trie = label_trie(label_ids)
    ends = {tuple(ids) for ids in label_ids}
    read = {()}
    for index, ids in enumerate(label_ids):
        ids = tuple(ids)
        flags = starts[index] if starts else (False,) * len(ids)
        for depth in range(1, len(ids)):
            node = ids[:depth]
            if (len(trie[node]) > 1 or node in ends
                    or (ids[:depth - 1] in read and not flags[depth])):
                read.add(node)
    return read


@dataclass(frozen=True)
class DecodeRuns:
    """The token runs a greedy decode feeds, one request each.

    A choice node is a trie node with two or more children. Between
    choice nodes every token is forced, so a run feeds a choice node's
    chosen token and the forced tokens after it, up to the next choice
    node, whose row the request reads. A label is decided at the last
    choice node on its path; its tokens after that are never fed.

    Attributes:
        runs: Request token sequences, without the answer cue that the
            first starts with. The first holds the forced tokens to the
            first choice node; each later run starts at a choice node's
            child.
        first: The node the first request ends at: a choice node, or
            the only label's leaf.
        after: (choice node, chosen token) -> (the node the choice
            leads to, the index of the run that feeds it, or None at a
            leaf).
        rounds: The most requests any label needs.
    """

    runs: list
    first: tuple
    after: dict
    rounds: int


def decode_runs(label_ids) -> DecodeRuns:
    """Split the label trie into the runs a greedy decode feeds.

    Args:
        label_ids: One token id sequence per label.

    Raises:
        ValueError: A label has no tokens.
    """
    trie = label_trie(label_ids)
    choices = choice_nodes(label_ids)

    def follow(node):
        while node in trie and node not in choices:
            node = node + (trie[node][0],)
        return node

    first = follow(())
    # the only label is decided at the cue's row, with nothing fed
    runs = [list(first) if first in trie else []]
    after = {}
    reads = {first: 1}
    for node in sorted(choices, key=len):
        for token in trie[node]:
            target = follow(node + (token,))
            reads[target] = reads[node] + (target in trie)
            if target in trie:
                runs.append(list(target[len(node):]))
                after[node, token] = (target, len(runs) - 1)
            else:
                after[node, token] = (target, None)
    rounds = max(reads[node] for node in reads if node not in trie)
    return DecodeRuns(runs, first, after, rounds)


def decode_round_tokens(label_ids) -> list[list[int]]:
    """Return the tokens a greedy decode feeds for each label, per round.

    The first round's count includes the answer cue. A label's list
    ends at the round that decides it.
    """
    runs = decode_runs(label_ids)
    trie = label_trie(label_ids)
    counts = []
    for ids in label_ids:
        ids = tuple(ids)
        node = runs.first
        fed = [1 + len(runs.runs[0])]
        while node in trie:
            target, run = runs.after[node, ids[len(node)]]
            if run is None:
                break
            fed.append(len(runs.runs[run]))
            node = target
        counts.append(fed)
    return counts


def fed_nodes(read) -> set:
    """Return the read nodes and their ancestors, whose tokens are fed."""
    return {node[:depth] for node in read for depth in range(len(node) + 1)}

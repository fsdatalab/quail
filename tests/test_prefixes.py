"""CPU checks for the shared prefix credit and the token minimum."""

from quail.runtime.prefixes import shared_prefix_tokens


def test_shared_prefix_tokens_credits_the_trie_savings():
    class Store:
        def __init__(self, documents):
            self.documents = documents

        def __iter__(self):
            return iter(self.documents)

    # two documents share [1, 2]; the third shares nothing
    assert shared_prefix_tokens(Store([[1, 2, 3], [1, 2, 4], [9]])) == 2


def _tokens(text):
    return [byte + 1 for byte in text.encode("utf-8")]


def _lcp(left, right):
    n = 0
    while n < min(len(left), len(right)) and left[n] == right[n]:
        n += 1
    return n


def test_prefix_trie_size_counts_shared_prefixes_once():
    from quail.runtime.minimum import prefix_trie_size

    assert prefix_trie_size([[1, 2, 3], [1, 2, 4], [9]]) == 5
    assert prefix_trie_size([[1, 2], [1, 2]]) == 2
    assert prefix_trie_size([(), (7,)]) == 1
    assert prefix_trie_size([]) == 0


def test_minimum_input_tokens_counts_each_distinct_prefix_once():
    import pyarrow as pa

    import quail
    from quail.planner import collect_operators
    from quail.planner.plan import EngineConfig
    from quail.runtime.minimum import minimum_input_tokens
    from quail.runtime.result import answer_table

    with quail.Session(EngineConfig(backend="quail"), tokenizer=_tokens) as session:
        session.register("docs", quail.DocumentProvider.from_table(pa.table({
            "id": ["a", "b", "c"],
            "body": ["same start one", "same start two", "other"],
        }), id_col="id"))
        session.register("aspects", quail.DocumentProvider.from_table(pa.table({
            "id": ["x", "y"],
            "name": ["aa", "ab"],
        }), id_col="id"))
        query = (
            session.docs("docs").alias("d")
            .ai_filter(quail.prompt("useful {0}", quail.col("d.body")))
            .join(session.docs("aspects").alias("a"))
            .ai_filter(quail.prompt("{0} mentions {1}", quail.col("d.body"),
                                    quail.col("a.name")))
            .select("d.id", "a.id")
        )
        stores = query.token_inputs()
        _, filters, joins = collect_operators(query.logical)
        question = filters["d"][0].prompt.tail_token_ids
        parts = {alias: (label, frame)
                 for alias, label, frame in joins[0].predicate.label_token_ids}
        frame, label = parts["d"][1], parts["a"][0]
        tail = joins[0].predicate.tail_token_ids
        pre = len(joins[0].predicate.preamble_token_ids)

        filter_answers = {("d", 0): pa.table({
            "d": [0, 1, 2], "answer": [True, True, False]})}
        join_answers = {0: answer_table(
            {"d": [0, 0, 1, 1], "a": [0, 1, 0, 1]},
            [True, False, True, True], "join_answers",
            metadata={"anchor": "d", "partners": "a", "semantics": "full"})}
        minimum = minimum_input_tokens(
            query.logical, stores, filter_answers, join_answers)

    # the three documents share the preamble, and the first two share
    # "same start " (11 tokens) beyond it; every document gets the
    # question once, the two anchors get the frame once, and each
    # anchor's pairs share the label and the partners' first token
    documents = 3 * pre + 14 + 14 + 5 - (pre + pre + 11)
    anchored = len(question) + len(frame) - _lcp(question, frame)
    pairs = len(label) + 3 + 2 * len(tail)
    assert minimum == documents + len(question) + 2 * anchored + 2 * pairs


def test_minimum_input_tokens_counts_a_document_once_across_uses():
    import pyarrow as pa

    import quail
    from quail.planner import collect_operators
    from quail.planner.plan import EngineConfig
    from quail.runtime.minimum import minimum_input_tokens

    with quail.Session(EngineConfig(backend="quail"), tokenizer=_tokens) as session:
        session.register("docs", quail.DocumentProvider.from_table(pa.table({
            "id": ["a", "b"], "body": ["alpha", "beta"],
        }), id_col="id"))
        query = (
            session.docs("docs").alias("d1")
            .join(session.docs("docs").alias("d2")
                  .ai_filter(quail.prompt("short {0}", quail.col("d2.body")))
                  .ai_filter(quail.prompt("clear {0}", quail.col("d2.body"))))
            .ai_filter(quail.prompt("{0} before {1}", quail.col("d1.body"),
                                    quail.col("d2.body")))
            .select("d1.id", "d2.id")
        )
        stores = query.token_inputs()
        _, filters, joins = collect_operators(query.logical)
        first, second = (p.prompt.tail_token_ids for p in filters["d2"])
        parts = {alias: (label, frame)
                 for alias, label, frame in joins[0].predicate.label_token_ids}
        frame, label = parts["d1"][1], parts["d2"][0]
        tail = joins[0].predicate.tail_token_ids
        pre = len(joins[0].predicate.preamble_token_ids)

        # both rows pass the first stage, only "alpha" reaches the
        # second; the join anchors on d1, the same two documents
        filter_answers = {
            ("d2", 0): pa.table({"d2": [0, 1], "answer": [True, True]}),
            ("d2", 1): pa.table({"d2": [0], "answer": [True]}),
        }
        join_answers = {0: pa.table({
            "d1": [0, 0, 1, 1], "d2": [0, 1, 0, 1],
            "answer": [True, True, False, True]})}
        minimum = minimum_input_tokens(
            query.logical, stores, filter_answers, join_answers,
            anchors={0: "d1"})

    documents = 2 * pre + 5 + 4 - pre
    alpha = len(first) + len(second) + len(frame) - sum((
        _lcp(first, second),
        max(_lcp(frame, first), _lcp(frame, second))))
    beta = len(first) + len(frame) - _lcp(first, frame)
    pairs = len(label) + 9 + 2 * len(tail)
    assert minimum == documents + alpha + beta + 2 * pairs

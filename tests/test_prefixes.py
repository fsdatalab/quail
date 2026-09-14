"""CPU check for the shared prefix credit."""

from quail.planner.prefixes import shared_prefix_tokens


def test_shared_prefix_tokens_credits_the_trie_savings():
    class Store:
        def __init__(self, documents):
            self.documents = documents

        def __iter__(self):
            return iter(self.documents)

    # two documents share [1, 2]; the third shares nothing
    assert shared_prefix_tokens(Store([[1, 2, 3], [1, 2, 4], [9]])) == 2


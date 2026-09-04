"""Stock vLLM filter and join clients.

The scheduling loops live in quail.backends.request_scheduling, which
the request backends and these baselines share.
"""

import itertools

from quail.backends.request_scheduling import (  # noqa: F401
    run_filter_chain,
    run_join_grouped,
    suffix_major_tiled_order,
)
from quail.logical import join_anchor_prefix_ids, join_tuple_suffix_ids


def build_join_grouped_inputs(prompt, documents, anchor: int, tokenizer):
    """Build canonical join token parts for stock vLLM.

    Args:
        prompt: Bound join prompt.
        documents: One list of tokenized documents per placeholder.
        anchor: Index of the anchor placeholder.
        tokenizer: Tokenizer for encoding.

    Returns:
        Tuple of (prefixes, suffixes, members). Prefixes are in anchor
        document order, suffixes in Cartesian order of the remaining
        placeholders. members records each suffix's partner indices.
    """
    if len(documents) != len(prompt.args):
        raise ValueError(
            f"join has {len(prompt.args)} placeholders but received "
            f"{len(documents)} document tables")
    if anchor < 0 or anchor >= len(documents):
        raise ValueError(f"join anchor placeholder {anchor} is out of range")
    partner_slots = [i for i in range(len(documents)) if i != anchor]
    members = list(itertools.product(
        *(range(len(documents[i])) for i in partner_slots)))
    prefixes = [
        join_anchor_prefix_ids(prompt, anchor, ids, tokenizer)
        for ids in documents[anchor]
    ]
    suffixes = [
        join_tuple_suffix_ids(
            prompt,
            [(slot, documents[slot][member[j]])
             for j, slot in enumerate(partner_slots)],
            tokenizer)
        for member in members
    ]
    return prefixes, suffixes, members

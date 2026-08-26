"""Filter/Join operators for vLLM-opbench.

Built over Quail's own document tables and predicates so the baseline
measures the same documents and questions with only the serving strategy
different. Prompts are raw prompt_token_ids with constrained TRUE/FALSE
token ids.
"""

from pathlib import Path

import pyarrow.parquet as pq

from baselines.stock import build_join_grouped_inputs
from quail.logical import ColumnRef, bind_join_prompt


def true_false_ids(tokenizer) -> tuple[list[int], list[int]]:
    """First-token ids of the TRUE/FALSE spellings."""
    true, false = set(), set()
    for w in ("TRUE", " TRUE", "True", " True"):
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            true.add(ids[0])
    for w in ("FALSE", " FALSE", "False", " False"):
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            false.add(ids[0])
    return sorted(true), sorted(false)


def read_table(data_dir: str, sf: float, name: str, id_col: str,
              text_col: str) -> tuple[list, list]:
    """Read (ids, texts) from one quailb parquet table. build_sets() must
    have been called first."""
    path = Path(data_dir) / f"sf{sf}" / f"{name}.parquet"
    t = pq.read_table(path)
    return t.column(id_col).to_pylist(), t.column(text_col).to_pylist()


class Filter:
    """One predicate over one table's documents."""

    def __init__(self, name: str, template: str):
        self.name = name
        self.template = template

    def build_prompts(self, texts: list[str], tokenizer) -> list[list[int]]:
        return [tokenizer.encode(self.template.format(t),
                                 add_special_tokens=False) for t in texts]


class Join:
    """Full cross product of one predicate over two tables.

    Requests use anchor-major order, which lets vLLM reuse the
    canonical anchor prefix across its partner stream.
    """

    def __init__(self, name: str, template: str, anchor: int = 0):
        self.name = name
        self.template = template
        self.anchor = anchor

    def build_prompts(self, left_texts: list[str], right_texts: list[str],
                      tokenizer) -> tuple[list[list[int]], list[tuple[int, int]]]:
        prefixes, suffixes, members = self.build_grouped_inputs(
            left_texts, right_texts, tokenizer)
        partner = 1 - self.anchor
        prompts, pairs = [], []
        for anchor_idx, prefix in enumerate(prefixes):
            for suffix, member in zip(suffixes, members):
                indices = [None, None]
                indices[self.anchor] = anchor_idx
                indices[partner] = member[0]
                prompts.append(prefix + suffix)
                pairs.append((indices[0], indices[1]))
        return prompts, pairs

    def build_grouped_inputs(self, left_texts: list[str],
                             right_texts: list[str], tokenizer):
        def tok(text):
            return tokenizer.encode(text, add_special_tokens=False)

        args = (ColumnRef("left", "left", "document"),
                ColumnRef("right", "right", "document"))
        prompt = bind_join_prompt(self.template, args, tok)
        documents = (
            [tok(text) for text in left_texts],
            [tok(text) for text in right_texts],
        )
        return build_join_grouped_inputs(prompt, documents, self.anchor, tok)

"""Filter/Join operators for vLLM-opbench, built directly over quail's
own document tables (schema from quail/bench/quailb.py's
build_sets/register_sets) and predicates - so vLLM-opbench measures
the same documents and the same questions quail's own engine does,
with only the serving strategy different.

Prompts are raw prompt_token_ids (matching baselines/stock.py
and quail's own worker.py), not chat-template text - a plain
str.format() substitution into the predicate template, then a plain
tokenizer.encode() of the whole rendered string per request. This is
deliberately the naive approach stock vLLM would take: quail's own
engine relocates a predicate's instruction text to just after the
document so it's paid once per anchor's kept KV rather than once per
pair (see quailb.py's module docstring) - that relocation is quail's
own optimization, and reproducing it here would defeat the point of
this baseline.

The answer is a constrained TRUE/FALSE token id, matching quail's own
engine's convention (quail/runtime/session.py's
_true_false_ids): predicate text instructs YES/NO, but the actual
constrained decode always picks between TRUE/FALSE token ids - the
same mismatch-tolerant convention quail's engine already uses (see
c1f619f in this repo's history), kept here for a fair comparison of
serving strategy rather than decoding convention.
"""

from pathlib import Path

import pyarrow.parquet as pq


def true_false_ids(tokenizer) -> tuple[list[int], list[int]]:
    """First-token ids of the TRUE/FALSE spellings - same rule as
    quail's own _true_false_ids, reimplemented here so vLLM-opbench
    doesn't reach into quail's runtime internals."""
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
    """(ids, texts) for one quailb.py parquet table, sf-scaled. Reads
    the same on-disk parquet quail's own engine reads - quailb.build_
    sets(data_dir, sf) must have already been called to produce it."""
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
    """Full cross product of one predicate over two tables, anchor-
    major order (all of one left-side row's pairs consecutive) -
    matching baselines/stock.py's run_join_grouped: this is the
    order that lets vLLM's own prefix cache re-serve an anchor's KV
    across its whole partner stream, without any explicit padding."""

    def __init__(self, name: str, template: str):
        self.name = name
        self.template = template

    def build_prompts(self, left_texts: list[str], right_texts: list[str],
                      tokenizer) -> tuple[list[list[int]], list[tuple[int, int]]]:
        prompts, pairs = [], []
        for li, lt in enumerate(left_texts):
            for ri, rt in enumerate(right_texts):
                prompts.append(tokenizer.encode(
                    self.template.format(lt, rt), add_special_tokens=False))
                pairs.append((li, ri))
        return prompts, pairs

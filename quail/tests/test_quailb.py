"""QUAIL-B's CPU half: planting arithmetic, and every query compiling
and planning against stand-in sets (no downloads, no GPU)."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quail
from quail.bench.quailb import (SETS, concat_to_chars, flags_line,
                                plant_flags, queries, register_sets)
from quail.planner.plan import EngineConfig, Refusal


def test_plant_flags_rates_and_determinism():
    rates = (0.9, 0.2)
    a = plant_flags(5000, rates, seed_offset=1)
    b = plant_flags(5000, rates, seed_offset=1)
    assert (a == b).all()
    means = a.mean(axis=0)
    assert abs(means[0] - 0.9) < 0.03
    assert abs(means[1] - 0.2) < 0.03


def test_flags_line_format():
    assert flags_line([1, 0], "T") == "\n\n[FLAGS] T_1=TRUE T_2=FALSE"


def test_concat_to_chars_reaches_target_and_cycles():
    pool = ["abcd", "efgh"]
    text, cursor = concat_to_chars(pool, 30, start=0)
    assert len(text) >= 30
    assert cursor > 2       # cycled past the pool once


def _standin_sets(tmp_path):
    """Tiny parquet files with the real schemas."""
    def write(name, col, n, words):
        pq.write_table(pa.table({
            "id": [f"{name}{i}" for i in range(n)],
            col: [f"{words} {i} " + "pad " * 20
                  + flags_line([1, 1, 1, 0, 1, 0, 1, 0], "FLAG")
                  + flags_line([1, 0, 1], "T")
                  + flags_line([1, 0], "R")
                  for i in range(n)]}),
            tmp_path / f"{name}.parquet")
    write("reviews", "body", 12, "review")
    write("threads", "thread", 10, "thread")
    write("reports", "report", 8, "report")
    write("products", "description", 6, "product")
    write("terms", "term", 6, "term")
    write("reviews5k", "body", 8, "review")
    write("reviews2k", "body", 6, "review")
    write("threads2k", "thread", 6, "thread")
    return tmp_path


def test_all_queries_compile_and_plan(tmp_path):
    _standin_sets(tmp_path)
    sess = quail.Session(EngineConfig(gpus=1), tokenizer=str.split)
    register_sets(sess, tmp_path)
    qdefs = queries(sess)
    assert len(qdefs) == 16        # B1-B15 with B3 twice
    for qid, (desc, make) in qdefs.items():
        q = make()
        plan = q.plan()
        assert not isinstance(plan, Refusal), f"{qid} refused: {plan}"
        text = q.explain()
        assert "physical:" in text, qid


def test_set_table_matches_design():
    assert SETS["reviews"][0] == 50_000
    assert SETS["terms"][2] is False    # fixed, never scales
"""Hash operations compare numeric values independently of their bit patterns."""

import math
from types import SimpleNamespace

import pyarrow as pa
import pytest
from helpers.sessions import CONFIG, _run, fake_tok, make_executor

import quail
from quail.backends.quail.executor.operators.filter import stop_groups


@pytest.mark.parametrize("dtype", [pa.float32(), pa.float64()])
@pytest.mark.parametrize("dictionary", [False, True])
def test_grouping_and_distinct_share_numeric_equality(dtype, dictionary):
    values = pa.array([-0.0, 0.0, -float("nan"), float("nan"), 1.0, 1.0, None], dtype)
    if dictionary:
        values = values.dictionary_encode()
    with quail.Session(CONFIG, tokenizer=fake_tok) as session:
        session.register("docs", quail.DocumentProvider.from_table(pa.table({
            "id": range(len(values)), "body": ["text"] * len(values), "value": values,
        }), id_col="id"))

        def run(selection, suffix=""):
            query = session.sql(
                f"SELECT {selection} FROM docs r "
                f"WHERE AI_FILTER(PROMPT('keep: {{0}}', r.body)) {suffix}")
            return _run(query, make_executor({"r": {"keep:": [True] * len(values)}}))

        grouped = run(
            "r.value, COUNT(*) AS n, COUNT(DISTINCT r.value) AS distinct_values, "
            "COUNT(r.value) AS present", "GROUP BY r.value").collect()
        assert grouped["n"].to_pylist() == [2, 2, 2, 1]
        assert grouped["distinct_values"].to_pylist() == [1, 1, 1, 0]
        assert grouped["present"].to_pylist() == [2, 2, 2, 0]
        distinct = run("DISTINCT r.value").collect()
        for result in (grouped, distinct):
            zero, nan, one, null = result["r.value"].to_pylist()
            assert zero == 0.0 and math.copysign(1, zero) == -1
            assert math.isnan(nan) and math.copysign(1, nan) == -1
            assert one == 1.0 and null is None
        assert run("COUNT(DISTINCT r.value) AS n").to_rows() == [(3,)]
        query = (session.docs("docs").alias("r")
                 .ai_filter(quail.prompt("keep: {0}", quail.col("r.body")))
                 .distinct().select("r.value"))
        estimate = quail.speed_of_light_estimate(query, lambda *_args: True)
        assert estimate.filter_stages[0]["evaluated"] == 4
        assert estimate.post_filter_counts == {"r": 4}

    groups = stop_groups(SimpleNamespace(stop_key=("value",)),
                         pa.table({"value": values}), list(range(len(values))))
    assert len(set(groups)) == 4
    assert groups[0] == groups[1]
    assert groups[2] == groups[3]
    assert groups[4] == groups[5]

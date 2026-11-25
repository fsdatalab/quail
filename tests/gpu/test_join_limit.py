"""A LIMIT over an AI self-join of repeated rows, on qwen3-4b-fp8.

Each anchor's partners take more tokens than one forward pass holds, so
the LIMIT is reached while anchors are partway through their partners.

Run on Modal with ``experiments/run_gpu_tests.py --keyword join_limit``.
"""

import pyarrow as pa
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a CUDA device")

TONES = ("great", "awful", "moving", "dull", "fun", "boring", "superb", "weak",
         "lovely", "tedious", "brilliant", "clumsy", "charming", "bland",
         "gripping", "messy")
THINGS = ("acting", "plot", "score", "pacing", "dialogue", "ending", "cast",
          "camera work")
SETTING_TEXT = (
    " I saw it on a rainy evening in a small theater near the station, with a"
    " friend who had read the book, and we talked about it on the long walk"
    " home past the closed shops, the bakery, the old library, and the park"
    " where the lamps were already lit and the benches were wet.")
BODIES = [f"The {thing} was {tone}.{SETTING_TEXT * 3}"
          for tone in TONES for thing in THINGS]
REVIEWS = pa.table({"id": [f"r{index}" for index in range(2 * len(BODIES))],
                    "body": BODIES * 2})
PAIRS_SQL = (
    "SELECT a.id, b.id FROM reviews a JOIN reviews b "
    "ON AI_FILTER(PROMPT('Do both reviews praise the movie?\n{0}\n{1}', "
    "a.body, b.body)) WHERE a.id <> b.id")
NEXT_SQL = ("SELECT r.id FROM reviews r WHERE "
            "AI_FILTER(PROMPT('Does the review praise the movie?\n{0}', r.body)) "
            "LIMIT 5")


def _rows(session, sql):
    table = session.sql(sql).collect()
    return sorted(zip(*(column.to_pylist() for column in table.columns)))


def test_a_limit_over_a_self_join_of_repeated_rows_returns_pairs_of_the_whole_join():
    import quail

    config = quail.EngineConfig(model="qwen3-4b-fp8", device="h100-sxm")
    with quail.Session(config) as session:
        session.register("reviews", quail.DocumentProvider.from_table(
            REVIEWS, id_col="id"))
        pairs = _rows(session, f"{PAIRS_SQL} LIMIT 10")
        after = _rows(session, NEXT_SQL)
        every = _rows(session, PAIRS_SQL)
    assert len(pairs) == 10
    assert set(pairs) <= set(every)
    assert len(after) == 5

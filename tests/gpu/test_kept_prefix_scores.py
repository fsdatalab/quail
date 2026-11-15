"""A query's scores are the same when the arena reuses a kept prefix."""

import random

import pyarrow as pa
import pytest

import quail

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a CUDA device")

WORDS = ("river contract payment the of and party shall terminate notice law "
         "state agreement within days written delivery mountain license fee "
         "service data customer year term renewal breach court").split()
SCORE_SQL = ("SELECT d.id, AI.SCORE(PROMPT('Does the text concern a payment "
             "dispute? {0}', d.body)) AS score FROM docs d")


def test_a_second_run_of_a_query_returns_the_scores_of_the_first():
    generator = random.Random(0)
    bodies = [" ".join(generator.choice(WORDS)
                       for _ in range(generator.randrange(150, 4000)))
              for _ in range(64)]
    docs = pa.table({"id": [f"d{index}" for index in range(64)], "body": bodies})
    config = quail.EngineConfig(model="qwen3-4b-fp8", device="h100-sxm")
    scores, reused = [], []
    with quail.Session(config) as session:
        session.register("docs", quail.DocumentProvider.from_table(docs, id_col="id"))
        for _ in range(2):
            result = session.sql(SCORE_SQL).run()
            scores.append(result.collect().sort_by("d.id").column("score").to_pylist())
            reused.append(result.report["kv_manager"]["reused_tokens"])
    # the second run takes over the prefix the first left in the arena
    assert reused[0] == 0 and reused[1] > 0
    assert scores[0] == scores[1]

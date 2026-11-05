"""L40S and B200 plans, KV budgets, and $/query without a GPU."""

import pyarrow as pa
import pyarrow.parquet as pq
from helpers.sessions import TOKENIZER

import quail
from bench.quailb import queries, register_tables
from quail.api.config import EngineConfig
from quail.physical.plan import Refusal


def _standin_filter_and_join(tmp_path):
    reviews = pa.table({
        "id": [f"r{i}" for i in range(4)],
        "body": [f"review text {i}" for i in range(4)],
    })
    claims = pa.table({
        "id": [f"c{i}" for i in range(4)],
        "claim": [f"claim {i}" for i in range(4)],
        "label": ["SUPPORTS", "REFUTES", "SUPPORTS", "REFUTES"],
        "evidence_wiki_url": ["e0", "e1", "e0", "e1"],
    })
    evidence = pa.table({
        "id": ["e0", "e1"],
        "text": ["Wikipedia passage 0", "Wikipedia passage 1"],
    })
    pq.write_table(reviews, tmp_path / "reviews.parquet")
    pq.write_table(claims, tmp_path / "claims.parquet")
    pq.write_table(evidence, tmp_path / "evidence.parquet")




def test_filter_and_join_plan_on_l40s_and_b200(tmp_path):
    _standin_filter_and_join(tmp_path)
    for device in ("l40s", "b200"):
        with quail.Session(
            EngineConfig(model="qwen3-4b-fp8", device=device),
            tokenizer=TOKENIZER,
        ) as sess:
            register_tables(sess, tmp_path)
            listed = queries(sess)
            for query_id in ("IMDB-1", "FEV-1"):
                plan = listed[query_id][1]().plan()
                assert not isinstance(plan, Refusal), f"{device} {query_id}"





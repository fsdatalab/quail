"""Gate for LIMIT N early termination: run the same filter query with
and without a limit on the real Modal worker. Measures that the limited
run processes fewer tokens.

Uses 3000 IMDB reviews with a sentiment question. The store is disabled
so both runs are cold (access=read) and the comparison is fair.

Prediction:
- 3000 IMDB reviews, one filter: "Is this review negative?"
- IMDB is ~50/50 positive/negative; the model's TRUE rate will be in
  the 40-60% range.
- chunk_tokens budget is ~110k; each doc is ~200-400 tokens + 30 tokens
  of suffix, so maybe 300-400 docs per chunk.
- With LIMIT 10 and ~50% pass rate on a single-stage filter,
  FilterAdmission hits 10 survivors in the first chunk's answers and
  stops admitting for subsequent chunks.
- fresh_tokens with LIMIT should be well under half.

Run from the quail/ directory:

    uv run python tests/gpu/limit_gate.py 2>&1 | tee results/limit_gate.log
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import quail                                    # noqa: E402
from quail.planner.plan import EngineConfig     # noqa: E402
from tests.gpu.corpus import build_pool         # noqa: E402

N_DOCS = 3000
LIMIT = 10
SELECTIVITY = 0.5


def make_corpus(path):
    reviews = build_pool(N_DOCS)
    pq.write_table(pa.table({
        "id": [f"d{i}" for i in range(N_DOCS)],
        "body": reviews}), path)
    print(f"corpus: {N_DOCS} IMDB reviews")


def run_query(sess, limit=None):
    lim = f" LIMIT {limit}" if limit else ""
    sql = (f"SELECT d.id FROM docs d "
           f"WHERE AI_FILTER(PROMPT("
           f"'{{0}}\\n\\nIs this movie review negative?', d.body), "
           f"{{'selectivity': {SELECTIVITY}}}){lim}")
    q = sess.sql(sql)
    print(q.explain(), flush=True)
    return q.run()


def main():
    tmp = tempfile.mkdtemp()
    corpus_path = f"{tmp}/docs.parquet"
    make_corpus(corpus_path)

    sess = quail.Session(EngineConfig(gpus=1))
    sess.set_store(False)
    sess.register("docs", quail.DocumentProvider.from_parquet(
        corpus_path, id_col="id"))

    # run 1: no limit (processes every document)
    print("\n=== RUN 1: no limit ===", flush=True)
    res_all = run_query(sess, limit=None)

    # run 2: with limit (should stop early)
    print(f"\n=== RUN 2: LIMIT {LIMIT} ===", flush=True)
    res_lim = run_query(sess, limit=LIMIT)

    sess.close()

    # compare
    r1, r2 = res_all.report, res_lim.report
    summary = {
        "unlimited": {
            "rows": len(res_all.rows),
            "fresh_tokens": r1["fresh_tokens"],
            "wall_s": r1["wall_s"],
        },
        "limited": {
            "limit": LIMIT,
            "rows": len(res_lim.rows),
            "fresh_tokens": r2["fresh_tokens"],
            "wall_s": r2["wall_s"],
        },
        "reduction": {
            "token_ratio": r2["fresh_tokens"] / max(r1["fresh_tokens"], 1),
            "wall_ratio": r2["wall_s"] / max(r1["wall_s"], 0.001),
        },
    }

    print("\n=== RESULTS ===")
    print(json.dumps(summary, indent=2), flush=True)

    token_ratio = summary["reduction"]["token_ratio"]
    assert len(res_lim.rows) <= LIMIT, (
        f"expected at most {LIMIT} rows, got {len(res_lim.rows)}")
    assert token_ratio < 0.5, (
        f"expected <50% token ratio, got {token_ratio:.1%}; "
        f"early termination did not reduce work")
    print(f"\nGATE PASSED: LIMIT {LIMIT} processed "
          f"{token_ratio:.0%} of the tokens ({r2['fresh_tokens']} "
          f"vs {r1['fresh_tokens']})")

    Path("results").mkdir(exist_ok=True)
    with open("results/limit_gate.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("saved results/limit_gate.json")


if __name__ == "__main__":
    main()

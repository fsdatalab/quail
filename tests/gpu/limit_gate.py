"""LIMIT N early-termination gate: verify that a limited filter query processes fewer tokens than an unlimited one."""

import json
import sys
import tempfile
from pathlib import Path

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

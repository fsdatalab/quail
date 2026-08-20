"""Gate for LIMIT N early termination: run the same filter query with
and without a limit on the real Modal worker. Measures that the limited
run processes fewer tokens and finishes faster.

Prediction (stated before the run, per project convention):
- Corpus: 500 documents, one planted filter at ~60% selectivity.
  Without LIMIT, all 500 are processed; ~300 survive.
- With LIMIT 10 and 60% pass rate, FilterAdmission should stop
  admitting after roughly 17 documents (10 / 0.6), not all 500.
- So fresh_tokens with LIMIT should be well under half of the
  unlimited run, and wall_s should be shorter.

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

SEED = 20260820
N_DOCS = 500
SELECTIVITY = 0.6
LIMIT = 10
FILLER = ("The projector hummed while the reel changed and nobody in "
          "the back row noticed the splice. ")

FILTER_Q = ("\n\nExample: if the line said [FLAGS] FLAG_9=NO, then "
            "FLAG_9 has value NO.\nInstruction: output only the value "
            "of FLAG_1 from the [FLAGS] line above.\nFLAG_1=")


def make_corpus(path):
    rng = np.random.default_rng(SEED)
    flags = (rng.random(N_DOCS) < SELECTIVITY).astype(int)
    bodies = []
    for i in range(N_DOCS):
        line = f"FLAG_1={'YES' if flags[i] else 'NO'}"
        bodies.append(FILLER * 8 + f"\n\n[FLAGS] {line}")
    pq.write_table(pa.table({
        "id": [f"d{i}" for i in range(N_DOCS)],
        "body": bodies}), path)
    planted = sum(flags)
    print(f"corpus: {N_DOCS} docs, {planted} planted survivors "
          f"({planted/N_DOCS:.0%})")
    return flags


def run_query(sess, limit=None):
    lim = f" LIMIT {limit}" if limit else ""
    sql = (f"SELECT d.id FROM docs d "
           f"WHERE AI_FILTER(PROMPT('{{0}}{FILTER_Q}', d.body), "
           f"{{'selectivity': {SELECTIVITY}}}){lim}")
    q = sess.sql(sql)
    print(q.explain(), flush=True)
    res = q.run()
    return res


def main():
    tmp = tempfile.mkdtemp()
    corpus_path = f"{tmp}/docs.parquet"
    flags = make_corpus(corpus_path)

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
    assert len(res_lim.rows) == LIMIT, (
        f"expected {LIMIT} rows, got {len(res_lim.rows)}")
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

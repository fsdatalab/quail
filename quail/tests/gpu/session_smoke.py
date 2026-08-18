"""End-to-end smoke of the Session surface on the real Modal worker:
synthetic planted corpora, one filter query and one join query, run
through sess.sql(...).run().

What this smoke gates, measured:
- filter: 200 documents, two planted flags at rates 0.6 and 0.5. The
  value-completion question style works on the 4B (measured 76/76
  agreement with planted survivors); gating structure and the
  provided-vs-observed report are the checks.
- join: 12 reports x 36 candidates = 432 pairs. The PLUMBING is the
  gate here (pair count, projection, report), NOT accuracy: the 4B
  answers YES to essentially every constrained one-token equality
  judgment. Measured twice through a trivially-correct causal
  reference path (milestone1.py::run_debug_join, with and without a
  few-shot example): all-YES both times, 0 disagreements against the
  packed executor. Content-style predicates (the BioDEX shape) discriminate;
  symbolic equality does not. QUAIL-B's join predicates must use a
  checkpoint-verified phrasing.
- walls are compile-dominated: a single-rep smoke pays the DeepGEMM
  and Triton JIT for its shapes inside the measured wall. Each run()
  boots its own container today; a warm session-held worker is a
  later step.

Run from the quail/ directory:

    uv run python tests/gpu/session_smoke.py 2>&1 | tee results/session_smoke.log
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

SEED = 20260818
FILLER = ("The projector hummed while the reel changed and nobody in "
          "the back row noticed the splice. ")


def make_filter_parquet(path, n_docs=200, rates=(0.6, 0.5)):
    rng = np.random.default_rng(SEED)
    flags = rng.random((n_docs, len(rates))) < np.array(rates)
    bodies = []
    for i in range(n_docs):
        line = " ".join(f"FLAG_{j+1}={'YES' if flags[i, j] else 'NO'}"
                        for j in range(len(rates)))
        bodies.append(FILLER * 8 + f"\n\n[FLAGS] {line}")
    pq.write_table(pa.table({
        "id": [f"d{i}" for i in range(n_docs)],
        "body": bodies}), path)
    return flags


COLORS = ("blue", "red", "green", "yellow", "purple", "orange")


def make_join_parquets(rdir, cdir, n_reports=12, n_cands=36):
    """Single-lookup predicate, the shape the join findings proved the
    4B can answer: the report states one fact (its dominant color),
    the candidate names one color, the question compares them. The
    two-planted-key form (X=<k> on both sides) is known to fail - the
    committed nway3 run measured the model answering YES to nearly
    every such pair."""
    keys = len(COLORS)
    reports = []
    for i in range(n_reports):
        reports.append(FILLER * 30
                       + f"\n\nThe dominant color in this scene is "
                         f"{COLORS[i % keys]}.")
    cands = [f"The candidate color is {COLORS[j % keys]}."
             for j in range(n_cands)]
    pq.write_table(pa.table({
        "id": [f"r{i}" for i in range(n_reports)],
        "report": reports}), rdir)
    pq.write_table(pa.table({
        "id": [f"c{j}" for j in range(n_cands)],
        "body": cands}), cdir)
    truth = {(i, j): int(i % keys == j % keys)
             for i in range(n_reports) for j in range(n_cands)}
    return truth


FILTER_Q = ("\n\nExample: if the line said [FLAGS] FLAG_9=NO, then "
            "FLAG_9 has value NO.\nInstruction: output only the value "
            "of FLAG_{j} from the [FLAGS] line above.\nFLAG_{j}=")


def main():
    tmp = tempfile.mkdtemp()
    flags = make_filter_parquet(f"{tmp}/docs.parquet")
    truth = make_join_parquets(f"{tmp}/reports.parquet",
                               f"{tmp}/cands.parquet")

    sess = quail.Session(EngineConfig(gpus=1))
    sess.register("docs", quail.DocumentProvider.from_parquet(
        f"{tmp}/docs.parquet", id_col="id"))
    sess.register("reports", quail.DocumentProvider.from_parquet(
        f"{tmp}/reports.parquet", id_col="id"))
    sess.register("cands", quail.DocumentProvider.from_parquet(
        f"{tmp}/cands.parquet", id_col="id"))

    summary = {}

    # ---- the filter query, AI SQL entry point
    q1_text = FILTER_Q.replace("{j}", "1")
    q2_text = FILTER_Q.replace("{j}", "2")
    fq = sess.sql(f"""
        SELECT d.id FROM docs d
        WHERE AI_FILTER(PROMPT('{{0}}{q1_text}', d.body),
                        {{'selectivity': 0.6}})
          AND AI_FILTER(PROMPT('{{0}}{q2_text}', d.body),
                        {{'selectivity': 0.5}})
    """)
    print(fq.explain(), flush=True)
    res = fq.run()
    planted = sorted(f"d{i}" for i in range(len(flags))
                     if flags[i].all())
    got = sorted(r[0] for r in res.rows)
    agree = len(set(got) & set(planted))
    summary["filter"] = dict(
        report=res.report,
        rows=len(res.rows), planted_survivors=len(planted),
        agree_with_planted=agree)
    print(json.dumps(summary["filter"], indent=2), flush=True)

    # ---- the join query, builder entry point
    jq = (sess.docs("reports").alias("r")
          .ai_join(sess.docs("cands").alias("c"),
                   quail.prompt(
                       "You will be shown a scene report and one "
                       "candidate color. Decide from the report's "
                       "own words.\n\nREPORT:\n{0}\n\nCANDIDATE:\n{1}"
                       "\nInstruction: answer YES if the report says "
                       "its dominant color is the candidate color, "
                       "NO otherwise.\nANSWER=",
                       quail.col("r.report"), quail.col("c.body")),
                   selectivity=1 / 6)
          .select("r.id", "c.id"))
    print(jq.explain(), flush=True)
    jres = jq.run()
    planted_pairs = {(f"r{i}", f"c{j}") for (i, j), v in truth.items()
                     if v}
    got_pairs = set(jres.rows)
    summary["join"] = dict(
        report=jres.report,
        pairs_returned=len(got_pairs),
        planted_pairs=len(planted_pairs),
        agree_with_planted=len(got_pairs & planted_pairs))
    print(json.dumps(summary["join"], indent=2), flush=True)

    Path("results").mkdir(exist_ok=True)
    with open("results/session_smoke.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("saved results/session_smoke.json")


if __name__ == "__main__":
    main()

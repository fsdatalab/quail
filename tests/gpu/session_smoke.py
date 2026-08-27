"""Session smoke test on GPU: filter and join queries on synthetic planted corpora."""

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
        line = " ".join(
            f"FLAG_{j+1}={'TRUE' if flags[i, j] else 'FALSE'}"
                        for j in range(len(rates)))
        bodies.append(FILLER * 8 + f"\n\n[FLAGS] {line}")
    pq.write_table(pa.table({
        "id": [f"d{i}" for i in range(n_docs)],
        "body": bodies}), path)
    return flags


COLORS = ("blue", "red", "green", "yellow", "purple", "orange")


def make_join_parquets(rdir, cdir, n_reports=12, n_cands=36):
    """Build report and candidate parquets with planted color matches, returning a truth dict."""
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


FILTER_Q = ("\n\nExample: if the line said [FLAGS] FLAG_9=FALSE, "
            "then FLAG_9 has value FALSE.\nInstruction: output only the value "
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
        rows=res.count(), planted_survivors=len(planted),
        agree_with_planted=agree)
    print(json.dumps(summary["filter"], indent=2), flush=True)

    # the same query again: if the worker container stayed warm, the
    # second run skips the boot
    res2 = sess.sql(f"""
        SELECT d.id FROM docs d
        WHERE AI_FILTER(PROMPT('{{0}}{q1_text}', d.body),
                        {{'selectivity': 0.6}})
          AND AI_FILTER(PROMPT('{{0}}{q2_text}', d.body),
                        {{'selectivity': 0.5}})
    """).run()
    summary["filter_warm"] = dict(
        report=res2.report, rows=len(res2.rows),
        rows_match_first_run=sorted(res2.rows) == sorted(res.rows))
    print(json.dumps(summary["filter_warm"], indent=2), flush=True)

    # ---- the join query, builder entry point
    jq = (sess.docs("reports").alias("r")
          .ai_join(sess.docs("cands").alias("c"),
                   quail.prompt(
                       "Judge strictly from {0} whether it says its "
                       "dominant color is the color named in {1}. "
                       "Answer TRUE if it does, FALSE otherwise."
                       "\nANSWER=",
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

    sess.close()
    Path("results").mkdir(exist_ok=True)
    with open("results/session_smoke.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("saved results/session_smoke.json")


if __name__ == "__main__":
    main()

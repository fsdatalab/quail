"""Barrier-path smoke test on GPU: two joins with different anchors, verified on 1 and 2 GPUs."""

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

SEED = 20260824
FILLER = ("The projector hummed while the reel changed and nobody in "
          "the back row noticed the splice. ")
COLORS = ("blue", "red", "green", "yellow", "purple", "orange")
# reports use only the first 4 colors; candidates cover all 6, so the
# 4 candidates of the unused colors can match no report and the
# barrier's thinning always has something to remove
N_REPORT_COLORS = 4
N_REPORTS = 10
N_CANDS = 12
FLAG_RATE = 0.7

FILTER_Q = ("\n\nExample: if the line said [FLAGS] FLAG_9=FALSE, "
            "then FLAG_9 has value FALSE.\nInstruction: output only "
            "the value of FLAG_1 from the [FLAGS] line above."
            "\nFLAG_1=")


def make_parquets(tmp):
    rng = np.random.default_rng(SEED)
    flags = rng.random(N_REPORTS) < FLAG_RATE
    reports = []
    for i in range(N_REPORTS):
        reports.append(
            FILLER * 20
            + f"\n\nThe dominant color in this scene is "
              f"{COLORS[i % N_REPORT_COLORS]}."
            + f"\n\n[FLAGS] FLAG_1={'TRUE' if flags[i] else 'FALSE'}")
    cands = [f"The candidate color is {COLORS[j % len(COLORS)]}."
             for j in range(N_CANDS)]
    labels = [f"The label names the color {c}." for c in COLORS]
    pq.write_table(pa.table({
        "id": [f"r{i}" for i in range(N_REPORTS)],
        "report": reports}), f"{tmp}/reports.parquet")
    pq.write_table(pa.table({
        "id": [f"c{j}" for j in range(N_CANDS)],
        "body": cands}), f"{tmp}/cands.parquet")
    pq.write_table(pa.table({
        "id": [f"g{k}" for k in range(len(COLORS))],
        "label": labels}), f"{tmp}/labels.parquet")
    truth1 = {(i, j): int(i % N_REPORT_COLORS == j % len(COLORS))
              for i in range(N_REPORTS) for j in range(N_CANDS)}
    truth2 = {(j, k): int(j % len(COLORS) == k)
              for j in range(N_CANDS) for k in range(len(COLORS))}
    return flags, truth1, truth2


def build_query(sess):
    return (sess.docs("reports").alias("r")
            .ai_filter(quail.prompt("{0}" + FILTER_Q,
                                    quail.col("r.report")),
                       selectivity=FLAG_RATE)
            .ai_join(sess.docs("cands").alias("c"),
                     quail.prompt(
                         "Judge strictly from {0} whether it says its "
                         "dominant color is the color named in {1}. "
                         "Answer TRUE if it does, FALSE otherwise."
                         "\nANSWER=",
                         quail.col("r.report"), quail.col("c.body")),
                     selectivity=1 / 6, anchor="r")
            .ai_join(sess.docs("labels").alias("g"),
                     quail.prompt(
                         "Judge strictly whether {0} and {1} name the "
                         "same color. Answer TRUE if they do, FALSE "
                         "otherwise.\nANSWER=",
                         quail.col("c.body"), quail.col("g.label")),
                     selectivity=1 / 6, anchor="g")
            .select("r.id", "c.id", "g.id"))


def brute_force_rows(res):
    """Recombine answer rows on CPU as a brute-force reference for the engine's output."""
    frows = res.answer_rows["filters"]["r"]
    keep_r = {d for d, row in frows.items()
              if len(row) == 1 and all(row)}
    s1, s2 = res.answer_rows["joins"]
    pairs1 = set()      # (r global, c global)
    for la, row in s1["rows"].items():
        for ti, bit in enumerate(row):
            if bit:
                pairs1.add((s1["anchor_index"][la],
                            s1["partner_index"][ti][0]))
    pairs2 = set()      # (g global, c global)
    for la, row in s2["rows"].items():
        for ti, bit in enumerate(row):
            if bit:
                pairs2.add((s2["anchor_index"][la],
                            s2["partner_index"][ti][0]))
    triples = sorted(
        (f"r{r}", f"c{c}", f"g{g}")
        for r, c in pairs1 if r in keep_r
        for g, c2 in pairs2 if c2 == c)
    return triples, pairs1, pairs2


def run_one(gpus, flags, truth1, truth2, tmp):
    sess = quail.Session(EngineConfig(gpus=gpus))
    sess.register("reports", quail.DocumentProvider.from_parquet(
        f"{tmp}/reports.parquet", id_col="id"))
    sess.register("cands", quail.DocumentProvider.from_parquet(
        f"{tmp}/cands.parquet", id_col="id"))
    sess.register("labels", quail.DocumentProvider.from_parquet(
        f"{tmp}/labels.parquet", id_col="id"))
    q = build_query(sess)
    plan = q.plan()
    kinds = [n["op"] for n in plan.nodes]
    print(f"[{gpus} gpu] plan nodes: {kinds}", flush=True)
    assert kinds.count("JoinGroup") == 2, kinds
    assert kinds.count("Barrier") == 1, kinds
    print(q.explain(), flush=True)

    res = q.run()
    expected, pairs1, pairs2 = brute_force_rows(res)
    got = sorted(res.rows)

    # HARD: engine recombination == CPU brute force of the same rows
    assert got == expected, (
        f"recombination mismatch: {len(got)} rows vs "
        f"{len(expected)} brute-forced")

    # HARD: stage 2 saw only barrier-thinned candidates
    thinned_c = {c for _, c in pairs1}
    jstages = [s for s in res.report["stages"] if s["op"] == "join"]
    assert jstages[1]["tuples"] == len(COLORS) * len(thinned_c), (
        jstages[1]["tuples"], len(COLORS), len(thinned_c))
    # HARD: the count check above is meaningless if nothing thinned;
    # the corpus guarantees candidates with no possible match, so a
    # full survivor set means thinning did not run
    assert len(thinned_c) < N_CANDS, (
        "every candidate survived the barrier; the thinning gate "
        "was vacuous on this run")

    # informational: agreement with the planted truth
    keep_r = {i for i in range(N_REPORTS) if flags[i]}
    planted = sorted(
        (f"r{i}", f"c{j}", f"g{k}")
        for i in keep_r for j in range(N_CANDS)
        for k in range(len(COLORS))
        if truth1[(i, j)] and truth2[(j, k)])
    agree = len(set(got) & set(planted))
    planted_thinned = {j for j in range(N_CANDS)
                       if any(truth1[(i, j)] for i in keep_r)}
    summary = dict(
        gpus=gpus, rows=len(got),
        rows_match_brute_force=True,
        stage2_tuples=jstages[1]["tuples"],
        thinned_candidates=len(thinned_c),
        planted_thinned_candidates=len(planted_thinned),
        candidates_total=N_CANDS,
        planted_triples=len(planted),
        agree_with_planted=agree,
        stage1_observed_sel=jstages[0]["observed_selectivity"],
        stage2_observed_sel=jstages[1]["observed_selectivity"],
        report=res.report)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "report"}, indent=2), flush=True)
    print(json.dumps(summary["report"], indent=2), flush=True)
    sess.close()
    return summary


def main():
    tmp = tempfile.mkdtemp()
    flags, truth1, truth2 = make_parquets(tmp)
    keep_r = {i for i in range(N_REPORTS) if flags[i]}
    planted_thinned = {j for j in range(N_CANDS)
                       if any(truth1[(i, j)] for i in keep_r)}
    print(f"prediction: rows equal the CPU brute-force recombination "
          f"on both GPU counts; if the model matches the planted "
          f"truth, {len(planted_thinned)} of {N_CANDS} candidates "
          f"survive the barrier and stage 2 evaluates "
          f"{len(COLORS)} x {len(planted_thinned)} = "
          f"{len(COLORS) * len(planted_thinned)} tuples; a full "
          f"{N_CANDS}-candidate survivor set fails the run",
          flush=True)
    summary = {}
    summary["one_gpu"] = run_one(1, flags, truth1, truth2, tmp)
    summary["two_gpu"] = run_one(2, flags, truth1, truth2, tmp)
    same = (summary["one_gpu"]["rows"] == summary["two_gpu"]["rows"])
    summary["row_counts_match_across_gpu_counts"] = same
    print(f"row counts match across 1/2 GPUs: {same}", flush=True)
    Path("results").mkdir(exist_ok=True)
    with open("results/barrier_smoke.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("saved results/barrier_smoke.json")


if __name__ == "__main__":
    main()

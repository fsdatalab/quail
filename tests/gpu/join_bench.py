"""Join benchmark: a three-stage gate cell and a re-shard cell on planted color corpora."""

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
S = 1 / 6
COLORS = ("blue", "red", "green", "yellow", "purple", "orange")
FILLER = ("The projector hummed while the reel changed and nobody in "
          "the back row noticed the splice. ")

# gate cell sizes
N_B, N_P1, N_P2, N_P3 = 120, 4, 4, 120
# re-shard cell sizes
N_A, N_B2, N_C2 = 40, 60, 60


def _doc(rng, filler_reps, line):
    return FILLER * filler_reps + "\n\n" + line


def _color_table(path, ids, rng, n, filler_reps, line_fmt):
    colors = [COLORS[k] for k in rng.integers(0, len(COLORS), n)]
    pq.write_table(pa.table({
        "id": ids, "body": [_doc(rng, filler_reps, line_fmt.format(c))
                            for c in colors]}), path)
    return colors


def make_tables(tmp):
    rng = np.random.default_rng(SEED)
    t = {}
    t["bstar"] = _color_table(
        f"{tmp}/bstar.parquet", [f"b{i}" for i in range(N_B)], rng,
        N_B, 35, "The dominant color in this scene is {}.")
    t["p1"] = _color_table(
        f"{tmp}/p1.parquet", [f"x{i}" for i in range(N_P1)], rng,
        N_P1, 0, "The candidate color is {}.")
    t["p2"] = _color_table(
        f"{tmp}/p2.parquet", [f"y{i}" for i in range(N_P2)], rng,
        N_P2, 0, "The candidate color is {}.")
    t["p3"] = _color_table(
        f"{tmp}/p3.parquet", [f"z{i}" for i in range(N_P3)], rng,
        N_P3, 12, "The candidate color in this passage is {}.")
    t["along"] = _color_table(
        f"{tmp}/along.parquet", [f"a{i}" for i in range(N_A)], rng,
        N_A, 130, "The dominant color in this scene is {}.")
    t["bmid"] = _color_table(
        f"{tmp}/bmid.parquet", [f"b{i}" for i in range(N_B2)], rng,
        N_B2, 1, "The candidate color is {}.")
    t["cmid"] = _color_table(
        f"{tmp}/cmid.parquet", [f"c{i}" for i in range(N_C2)], rng,
        N_C2, 1, "The label names the color {}.")
    return t


M_LONG = ("Judge strictly from {0} whether it says its dominant "
          "color is the color named in {1}. Answer TRUE if it does, "
          "FALSE otherwise.\nANSWER=")
M_SHORT = ("Judge strictly whether {0} and {1} name the same color. "
           "Answer TRUE if they do, FALSE otherwise.\nANSWER=")


def gate_query(sess):
    return (sess.docs("bstar").alias("b")
            .ai_join(sess.docs("p1").alias("x"),
                     quail.prompt(M_LONG, quail.col("b.body"),
                                  quail.col("x.body")),
                     selectivity=S)
            .ai_join(sess.docs("p2").alias("y"),
                     quail.prompt(M_LONG, quail.col("b.body"),
                                  quail.col("y.body")),
                     selectivity=S)
            .ai_join(sess.docs("p3").alias("z"),
                     quail.prompt(M_LONG, quail.col("b.body"),
                                  quail.col("z.body")),
                     selectivity=S)
            .select("b.id", "x.id", "y.id", "z.id"))


def reshard_query(sess, anchors):
    return (sess.docs("along").alias("a")
            .ai_join(sess.docs("bmid").alias("b"),
                     quail.prompt(M_LONG, quail.col("a.body"),
                                  quail.col("b.body")),
                     selectivity=S, anchor=anchors[0])
            .ai_join(sess.docs("cmid").alias("c"),
                     quail.prompt(M_SHORT, quail.col("b.body"),
                                  quail.col("c.body")),
                     selectivity=S, anchor=anchors[1])
            .select("a.id", "b.id", "c.id"))


def stage_rows(res):
    return [s for s in res.report["stages"] if s["op"] == "join"]


def live_anchors(res, stage):
    """Anchors evaluated at one stage, from the raw answer rows."""
    return len(res.answer_rows["joins"][stage]["rows"])


def planted_gate(t):
    """Compute the planted gate survival counts and per-stage selectivities from the drawn corpus."""
    set1, set2 = set(t["p1"]), set(t["p2"])
    live1 = [c for c in t["bstar"] if c in set1]
    live2 = [c for c in live1 if c in set2]
    n1 = sum(1 for c in t["bstar"] for q in t["p1"] if c == q)
    n2 = sum(1 for c in live1 for q in t["p2"] if c == q)
    n3 = sum(1 for c in live2 for q in t["p3"] if c == q)
    return dict(
        live_after_gate1=len(live1), live_after_gate2=len(live2),
        stage1_selectivity=round(n1 / (N_B * N_P1), 4),
        stage2_selectivity=(round(n2 / (len(live1) * N_P2), 4)
                            if live1 else None),
        stage3_selectivity=(round(n3 / (len(live2) * N_P3), 4)
                            if live2 else None),
        p1_distinct_colors=len(set1), p2_distinct_colors=len(set2))


def planted_reshard(t):
    """Compute the planted stage selectivities and triple count for the re-shard corpus."""
    a, b, c = t["along"], t["bmid"], t["cmid"]
    s1 = sum(1 for x in a for y in b if x == y)
    s2 = sum(1 for y in b for z in c if y == z)
    rows = sum(1 for x in a for y in b for z in c if x == y == z)
    return dict(
        stage1_selectivity=round(s1 / (len(a) * len(b)), 4),
        stage2_selectivity=round(s2 / (len(b) * len(c)), 4),
        rows=rows)


def main():
    tmp = tempfile.mkdtemp()
    tables = make_tables(tmp)

    # store disabled: run order must not contaminate the comparison
    sess = quail.Session(EngineConfig(gpus=1, cpu_memory_gb=0))
    for name in ("bstar", "p1", "p2", "p3", "along", "bmid", "cmid"):
        sess.register(name, quail.DocumentProvider.from_parquet(
            f"{tmp}/{name}.parquet", id_col="id"))

    out = {}

    # ---------------- cell 1: the gate, predictions first
    surv1 = 1 - (1 - S) ** N_P1            # after gate 1
    surv2 = surv1 * (1 - (1 - S) ** N_P2)  # after gate 2
    pred = dict(
        live_after_gate1=round(N_B * surv1, 1),
        sigma1=round((N_B * surv1 * (1 - surv1)) ** 0.5, 1),
        live_after_gate2=round(N_B * surv2, 1),
        sigma2=round((N_B * surv2 * (1 - surv2)) ** 0.5, 1),
        stage3_tuples=round(N_B * surv2 * N_P3, 0),
        stage3_tuples_without_gates=N_B * N_P3)
    planted = planted_gate(tables)
    print("[gate] prediction (formula expectation):",
          json.dumps(pred), flush=True)
    print("[gate] prediction (conditional on the drawn corpus):",
          json.dumps(planted), flush=True)

    q = gate_query(sess)
    plan = q.plan()
    kinds = [n["op"] for n in plan.nodes]
    assert kinds.count("JoinGroup") == 1 and kinds.count("Barrier") == 0, kinds
    stages_plan = [st for n in plan.nodes if n["op"] == "JoinGroup"
                   for st in n["stages"]]
    assert [st["anchor"] for st in stages_plan] == ["b", "b", "b"]
    print(q.explain(), flush=True)
    res = q.run()
    js = stage_rows(res)
    out["gate"] = dict(
        prediction=pred,
        planted=planted,
        plan_expected_tuples=[st["expected_tuples"]
                              for st in stages_plan],
        measured_live=[live_anchors(res, i) for i in range(3)],
        measured_tuples=[s["tuples"] for s in js],
        observed_selectivity=[s["observed_selectivity"] for s in js],
        planted_selectivity=round(S, 4),
        rows=len(res.rows),
        wall_s=res.report["wall_s"], boot_s=res.report["boot_s"],
        fresh_tokens=res.report["fresh_tokens"])
    print("[gate] measured:", json.dumps(
        {k: v for k, v in out["gate"].items() if k != "prediction"}),
        flush=True)

    # ---------------- cell 2: the re-shard, predictions first
    # exact token arithmetic from the two plans' own stage counts is
    # printed with each plan; the headline prediction is the ratio
    planted_r = planted_reshard(tables)
    print("[reshard] prediction: the free plan anchors a (2 groups, "
          "1 barrier); baseline/free fresh-token ratio ~10x, wall "
          "ratio above 4x; planted truth:",
          json.dumps(planted_r), flush=True)

    # plan and assert BOTH configurations before running either, so a
    # plan surprise costs no GPU time
    base_q = reshard_query(sess, ("b", "b"))
    base_plan = base_q.plan()
    kinds = [n["op"] for n in base_plan.nodes]
    assert kinds.count("JoinGroup") == 1 and kinds.count("Barrier") == 0, kinds
    print(base_q.explain(), flush=True)

    free_q = reshard_query(sess, (None, None))
    free_plan = free_q.plan()
    kinds = [n["op"] for n in free_plan.nodes]
    free_stages = [st for n in free_plan.nodes if n["op"] == "JoinGroup"
                   for st in n["stages"]]
    print(free_q.explain(), flush=True)
    assert kinds.count("JoinGroup") == 2 and kinds.count("Barrier") == 1, kinds
    assert free_stages[0]["anchor"] == "a", free_stages

    base = base_q.run()
    free = free_q.run()

    # both runs answer the same planted question; rows must agree up
    # to model noise - report, do not gate
    out["reshard"] = dict(
        planted=planted_r,
        baseline=dict(
            anchors=[st["anchor"] for st in stage_rows(base)],
            plan_tuple_tokens=[st["tuple_tokens"] for st in
                               [s for n in base_plan.nodes
                                if n["op"] == "JoinGroup"
                                for s in n["stages"]]],
            wall_s=base.report["wall_s"],
            fresh_tokens=base.report["fresh_tokens"],
            tuples=[s["tuples"] for s in stage_rows(base)],
            observed_selectivity=[s["observed_selectivity"]
                                  for s in stage_rows(base)],
            rows=len(base.rows)),
        free=dict(
            anchors=[st["anchor"] for st in stage_rows(free)],
            plan_tuple_tokens=[st["tuple_tokens"] for st in free_stages],
            wall_s=free.report["wall_s"],
            fresh_tokens=free.report["fresh_tokens"],
            tuples=[s["tuples"] for s in stage_rows(free)],
            observed_selectivity=[s["observed_selectivity"]
                                  for s in stage_rows(free)],
            rows=len(free.rows)),
        rows_equal=sorted(base.rows) == sorted(free.rows),
        token_ratio=round(base.report["fresh_tokens"]
                          / max(1, free.report["fresh_tokens"]), 2),
        wall_ratio=round(base.report["wall_s"]
                         / max(0.01, free.report["wall_s"]), 2))
    print("[reshard] measured:", json.dumps(out["reshard"]),
          flush=True)

    sess.close()
    Path("results").mkdir(exist_ok=True)
    with open("results/join_bench.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved results/join_bench.json")


if __name__ == "__main__":
    main()

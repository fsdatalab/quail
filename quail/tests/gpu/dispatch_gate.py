"""The dispatch gate: the same queries on 1 GPU and on 2, through the
full Session path (plan -> shards -> container coordinator -> GPU
children -> merge -> sink).

PREDICTIONS, stated before the runs:
- filter (the milestone 10k five-filter corpus): gpus=1 near the
  measured 39.4 s driver wall; gpus=2 near half plus dispatch
  overhead (each child sees ~1.6M corpus tokens), so ~20-25 s.
- join (60 synthetic reports x 1,200 candidates = 72,000 pairs,
  ~2.6M pair tokens): gpus=2 near half of gpus=1 (anchors split by
  token count, partners replicated).
- answers identical across GPU counts up to knife-edge flips (the
  chunks pack differently per shard).

Run from the quail/ directory:

    uv run python tests/gpu/dispatch_gate.py 2>&1 | tee results/dispatch_gate.log
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import quail                                    # noqa: E402
from quail.planner.plan import EngineConfig     # noqa: E402

WORKLOAD_SEED = 20260731
FLAG_SEED = 424242
SELECTIVITY = (0.9, 0.9, 0.9, 0.8, 0.8)
FILLER = ("The projector hummed while the reel changed and nobody in "
          "the back row noticed the splice. ")
COLORS = ("blue", "red", "green", "yellow", "purple", "orange")


def build_filter_parquet(path, n_docs=10000):
    """The milestone corpus, built locally: same seeds, same planted
    flags, byte-identical bodies."""
    texts = []
    for split in ("train", "test"):
        f = hf_hub_download(
            "stanfordnlp/imdb",
            f"plain_text/{split}-00000-of-00001.parquet",
            repo_type="dataset")
        texts += pq.read_table(f, columns=["text"]).column(
            "text").to_pylist()
    rng = np.random.default_rng(WORKLOAD_SEED)
    idx = sorted(rng.choice(len(texts), size=10_000, replace=False))
    docs = [texts[i] for i in idx[:n_docs]]
    rng = np.random.default_rng(FLAG_SEED + 100)
    flags = (rng.random((n_docs, 5))
             < np.array(SELECTIVITY)[None, :]).astype(int)
    bodies = []
    for d, f in zip(docs, flags):
        line = " ".join(
            f"FLAG_{j+1}={'TRUE' if v else 'FALSE'}"
                        for j, v in enumerate(f))
        bodies.append(d + "\n\n[FLAGS] " + line)
    pq.write_table(pa.table({
        "id": [f"d{i}" for i in range(n_docs)],
        "body": bodies}), path)


def build_join_parquets(rpath, cpath, n_reports=60, n_cands=1200):
    reports = [FILLER * 110
               + f"\n\nThe dominant color in this scene is "
                 f"{COLORS[i % 6]}." for i in range(n_reports)]
    cands = [f"The candidate color is {COLORS[j % 6]}."
             for j in range(n_cands)]
    pq.write_table(pa.table({
        "id": [f"r{i}" for i in range(n_reports)],
        "report": reports}), rpath)
    pq.write_table(pa.table({
        "id": [f"c{j}" for j in range(n_cands)],
        "body": cands}), cpath)


def question(j):
    return (f"\n\nExample: if the line said [FLAGS] FLAG_9=FALSE, "
            f"then FLAG_9 has value FALSE.\nInstruction: output only the value "
            f"of FLAG_{j} from the [FLAGS] line above.\nFLAG_{j}=")


def run_pair(tmp, gpus):
    sess = quail.Session(EngineConfig(gpus=gpus))
    sess.register("docs", quail.DocumentProvider.from_parquet(
        f"{tmp}/docs.parquet", id_col="id"))
    sess.register("reports", quail.DocumentProvider.from_parquet(
        f"{tmp}/reports.parquet", id_col="id"))
    sess.register("cands", quail.DocumentProvider.from_parquet(
        f"{tmp}/cands.parquet", id_col="id"))

    conjuncts = "\n  AND ".join(
        f"AI_FILTER(PROMPT('{{0}}{question(j + 1)}', d.body), "
        f"{{'selectivity': {SELECTIVITY[j]}}})"
        for j in range(5))
    fq = sess.sql(f"SELECT d.id FROM docs d WHERE {conjuncts}")
    fres = fq.run()

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
    jres = jq.run()
    sess.close()
    return dict(
        gpus=gpus,
        filter=dict(wall_s=fres.report["wall_s"],
                    boot_s=fres.report["boot_s"],
                    rows=len(fres.rows),
                    fresh_tokens=fres.report["fresh_tokens"]),
        join=dict(wall_s=jres.report["wall_s"],
                  boot_s=jres.report["boot_s"],
                  rows=len(jres.rows),
                  fresh_tokens=jres.report["fresh_tokens"]),
        filter_ids=sorted(r[0] for r in fres.rows))


def main():
    tmp = tempfile.mkdtemp()
    build_filter_parquet(f"{tmp}/docs.parquet")
    build_join_parquets(f"{tmp}/reports.parquet", f"{tmp}/cands.parquet")

    print("[dispatch] prediction: filter gpus=1 ~40 s, gpus=2 "
          "~20-25 s; join gpus=2 near half of gpus=1; identical rows "
          "up to knife-edge flips", flush=True)
    one = run_pair(tmp, 1)
    print(json.dumps({k: v for k, v in one.items()
                      if k != "filter_ids"}, indent=2), flush=True)
    two = run_pair(tmp, 2)
    print(json.dumps({k: v for k, v in two.items()
                      if k != "filter_ids"}, indent=2), flush=True)

    overlap = len(set(one["filter_ids"]) & set(two["filter_ids"]))
    summary = dict(
        gpus1={k: v for k, v in one.items() if k != "filter_ids"},
        gpus2={k: v for k, v in two.items() if k != "filter_ids"},
        filter_speedup=round(one["filter"]["wall_s"]
                             / two["filter"]["wall_s"], 2),
        join_speedup=round(one["join"]["wall_s"]
                           / two["join"]["wall_s"], 2),
        filter_row_overlap=overlap,
        filter_rows=(len(one["filter_ids"]), len(two["filter_ids"])))
    print(json.dumps(summary, indent=2), flush=True)
    Path("results").mkdir(exist_ok=True)
    with open("results/dispatch_gate.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("saved results/dispatch_gate.json")


if __name__ == "__main__":
    main()

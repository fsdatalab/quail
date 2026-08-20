"""QUAIL-B: fifteen queries over five document sets, cold and warm.

The design (engine_design.md section 9): TPC-H's filter/join
skeletons over real text, planted predicates with known ground truth,
two scale knobs (SF scales document counts, LF scales document
length by concatenating real text), and a two-pass protocol in one
session - cold (store disabled, flushed) then warm (store enabled).
Provided vs observed selectivity is printed per stage, which is the
instrument check.

Measured caveat, stated up front: the 4B checkpoint answers YES to
essentially every constrained one-token equality judgment (measured
twice through a trivially-correct reference path). Content-style
predicates (the BioDEX shape) discriminate. So the join predicates
here are content questions wherever the design allows, and the
planted-key stages (B10) will show observed selectivity near 1.0 -
the gating machinery still executes, the instrument shows the bias,
and timing claims never depend on the model answering correctly.

Filter flags planted per set (rates fixed by seed):
    reviews  F1-F8 at .9 .9 .9 .8 .8 .2 .9 .8
             B2 asks F1-F5; B3 asks F1 F2 F6(.2) F7 F3; B13 asks
             F3 F4 F5 F7 F8 (new questions, same documents - that is
             what makes its warm restore honest)
    threads  T1-T3 at .05 .3 .9
    reports  R1-R2 at .5 .4

Build the data and run:

    uv run python -m quail.bench.quailb --sf 0.1 --lf 1
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FLAG_SEED = 424242
DATA_SEED = 20260818
CHARS_PER_TOKEN = 4.2

REVIEW_RATES = (0.9, 0.9, 0.9, 0.8, 0.8, 0.2, 0.9, 0.8)
THREAD_RATES = (0.05, 0.3, 0.9)
REPORT_RATES = (0.5, 0.4)

SETS = {
    # name: (docs at SF=1, mean tokens at LF=1, scales_with_sf)
    "reviews": (50_000, 400, True),
    "threads": (10_000, 1_200, True),
    "reports": (2_000, 3_000, True),
    "products": (1_000, 150, True),
    "terms": (2_560, 32, False),
}


# ---------------------------------------------------------- planting

def flags_line(bits, prefix="FLAG"):
    return ("\n\n[FLAGS] "
            + " ".join(f"{prefix}_{j+1}={'TRUE' if b else 'FALSE'}"
                       for j, b in enumerate(bits)))


def plant_flags(n_docs, rates, seed_offset):
    rng = np.random.default_rng(FLAG_SEED + seed_offset)
    return (rng.random((n_docs, len(rates)))
            < np.array(rates)[None, :]).astype(int)


def concat_to_chars(pool, target_chars, start):
    """Real text concatenated to a target length, cycling the pool.
    The counter tracks the joined length exactly (separators count
    only between parts)."""
    parts, total, i = [], -2, start
    while total < target_chars:
        t = pool[i % len(pool)]
        parts.append(t)
        total += len(t) + 2
        i += 1
    return "\n\n".join(parts), i


def flag_question(prefix, j):
    return f"\n\nIs {prefix}_{j} in the [FLAGS] line TRUE?"


# ------------------------------------------------------- set builders

def _imdb_pool():
    from huggingface_hub import hf_hub_download
    texts = []
    for split in ("train", "test"):
        f = hf_hub_download(
            "stanfordnlp/imdb",
            f"plain_text/{split}-00000-of-00001.parquet",
            repo_type="dataset")
        texts += pq.read_table(f, columns=["text"]).column(
            "text").to_pylist()
    rng = np.random.default_rng(DATA_SEED)
    rng.shuffle(texts)
    return texts


def _biodex_rows(n=2000):
    from datasets import load_dataset
    ds = load_dataset("BioDEX/BioDEX-Reactions", split="train",
                      streaming=True)
    rows = []
    for row in ds:
        text = str(row.get("fulltext_processed") or row.get("abstract"))
        terms = [t.strip() for t in
                 str(row.get("reactions", "")).split(",") if t.strip()]
        if len(text) >= 200 and terms:
            rows.append((text, terms))
        if len(rows) >= n:
            break
    return rows


def _abtbuy_products():
    # the matchbench repo is a legacy script dataset; its
    # auto-converted parquet lives on the convert branch
    from huggingface_hub import hf_hub_download
    f = hf_hub_download("matchbench/Abt-Buy",
                        "source/source/0000.parquet",
                        repo_type="dataset",
                        revision="refs/convert/parquet")
    table = pq.read_table(f)
    cols = {c.lower(): c for c in table.column_names}
    names = table.column(cols.get("name", "name")).to_pylist()
    descs = (table.column(cols["description"]).to_pylist()
             if "description" in cols else [""] * len(names))
    out = []
    for name, desc in zip(names, descs):
        text = f"{name or ''}. {desc or ''}".strip(". ")
        if text:
            out.append(text)
    return out


def _n_docs(name, sf):
    base, _, scales = SETS[name]
    return max(8, int(base * sf)) if scales else base


def build_sets(data_dir, sf, lf):
    """All five sets (plus the B7/B10 slices) as parquet files,
    cached by (sf, lf)."""
    d = Path(data_dir) / f"sf{sf}_lf{lf}"
    marker = d / "DONE"
    if marker.exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    imdb = _imdb_pool()
    bio = _biodex_rows()
    prods = _abtbuy_products()

    def write(name, ids, bodies, col="body"):
        pq.write_table(pa.table({"id": ids, col: bodies}),
                       d / f"{name}.parquet")

    # reviews: single IMDB texts padded to length by LF, 8 flags
    n = _n_docs("reviews", sf)
    target = int(SETS["reviews"][1] * lf * CHARS_PER_TOKEN)
    flags = plant_flags(n, REVIEW_RATES, seed_offset=1)
    bodies, cursor = [], 0
    for i in range(n):
        text, cursor = concat_to_chars(imdb, target, cursor)
        bodies.append(text + flags_line(flags[i], "FLAG"))
    write("reviews", [f"rv{i}" for i in range(n)], bodies)
    np.save(d / "reviews_flags.npy", flags)

    # threads: stacked IMDB, 3 flags, a planted key
    n = _n_docs("threads", sf)
    target = int(SETS["threads"][1] * lf * CHARS_PER_TOKEN)
    tflags = plant_flags(n, THREAD_RATES, seed_offset=2)
    bodies = []
    for i in range(n):
        text, cursor = concat_to_chars(imdb, target, cursor)
        bodies.append(text + flags_line(tflags[i], "T")
                      + f"\n[KEYS] X={i % 40}")
    write("threads", [f"th{i}" for i in range(n)], bodies,
          col="thread")
    np.save(d / "threads_flags.npy", tflags)

    # reports: BioDEX text to length, 2 flags
    n = _n_docs("reports", sf)
    target = int(SETS["reports"][1] * lf * CHARS_PER_TOKEN)
    rflags = plant_flags(n, REPORT_RATES, seed_offset=3)
    bio_texts = [t for t, _ in bio]
    bodies, bcur = [], 0
    for i in range(n):
        text, bcur = concat_to_chars(bio_texts, target, bcur)
        bodies.append(text + flags_line(rflags[i], "R"))
    write("reports", [f"rp{i}" for i in range(n)], bodies,
          col="report")
    np.save(d / "reports_flags.npy", rflags)

    # products: ABT-BUY descriptions (cycled if SF needs more)
    n = _n_docs("products", sf)
    bodies = [prods[i % len(prods)] for i in range(n)]
    write("products", [f"pr{i}" for i in range(n)], bodies,
          col="description")

    # terms: the reaction vocabulary, fixed size, never scales
    freq = {}
    for _, terms in bio:
        for t in terms:
            freq[t] = freq.get(t, 0) + 1
    vocab = [t for t, _ in sorted(freq.items(),
                                  key=lambda kv: (-kv[1], kv[0]))]
    vocab = vocab[:SETS["terms"][0]]
    write("terms", [f"tm{i}" for i in range(len(vocab))], vocab,
          col="term")

    # slices for B7 and B10
    rv = pq.read_table(d / "reviews.parquet")
    n7 = max(4, int(5_000 * sf))
    pq.write_table(rv.slice(0, min(n7, rv.num_rows)),
                   d / "reviews5k.parquet")
    n10 = max(4, int(2_000 * sf))
    pq.write_table(rv.slice(0, min(n10, rv.num_rows)),
                   d / "reviews2k.parquet")
    th = pq.read_table(d / "threads.parquet")
    pq.write_table(th.slice(0, min(n10, th.num_rows)),
                   d / "threads2k.parquet")
    marker.write_text("ok")
    return d


def register_sets(sess, data_dir):
    from quail.catalog import DocumentProvider
    for name, id_col in (("reviews", "id"), ("threads", "id"),
                         ("reports", "id"), ("products", "id"),
                         ("terms", "id"), ("reviews5k", "id"),
                         ("reviews2k", "id"), ("threads2k", "id")):
        sess.register(name, DocumentProvider.from_parquet(
            str(Path(data_dir) / f"{name}.parquet"), id_col=id_col))


# ---------------------------------------------------------- queries

def _filter_sql(alias, table, col, stages):
    conj = "\n  AND ".join(
        f"AI_FILTER(PROMPT('{{0}}{flag_question(p, j)}', "
        f"{alias}.{col}), {{'selectivity': {s}}})"
        for p, j, s in stages)
    return f"SELECT {alias}.id FROM {table} {alias} WHERE {conj}"


def queries(sess):
    """id -> (description, callable() -> Query). Fresh Query objects
    per call so each pass re-plans."""
    import quail

    def content_join(left, lcol, right, rcol, text, sel,
                     lfilters=(), rfilters=()):
        def make():
            lq = sess.docs(left).alias("a")
            for p, j, s in lfilters:
                lq = lq.ai_filter(
                    quail.prompt("{0}" + flag_question(p, j),
                                 quail.col(f"a.{lcol}")),
                    selectivity=s)
            rq = sess.docs(right).alias("b")
            for p, j, s in rfilters:
                rq = rq.ai_filter(
                    quail.prompt("{0}" + flag_question(p, j),
                                 quail.col(f"b.{rcol}")),
                    selectivity=s)
            return lq.ai_join(
                rq, quail.prompt(text, quail.col(f"a.{lcol}"),
                                 quail.col(f"b.{rcol}")),
                selectivity=sel).select("a.id", "b.id")
        return make

    REACTION = ("Does {0} describe the reaction named in {1} as "
                "something the patient experienced?")
    DISCUSS = "Does {0} discuss the product described in {1}?"
    MENTION = "Does {0} mention the medical term in {1}?"
    KEYEQ3 = ("Do the [FLAGS] or key values in {1} and in {2} both "
              "match the [KEYS] X value in {0}?")
    REACTION_DISCUSS = ("Does {0} describe the reaction named in {1} "
                        "as something the patient experienced and also "
                        "discuss the product described in {2}?")

    q = {}
    q["B1"] = ("1F reviews: the per-query floor", lambda: sess.sql(
        _filter_sql("r", "reviews", "body", [("FLAG", 1, 0.9)])))
    q["B2"] = ("5F reviews: the filter-chain anchor", lambda: sess.sql(
        _filter_sql("r", "reviews", "body",
                    [("FLAG", 1, 0.9), ("FLAG", 2, 0.9),
                     ("FLAG", 3, 0.9), ("FLAG", 4, 0.8),
                     ("FLAG", 5, 0.8)])))
    b3 = _filter_sql("r", "reviews", "body",
                     [("FLAG", 1, 0.9), ("FLAG", 2, 0.9),
                      ("FLAG", 6, 0.2), ("FLAG", 7, 0.9),
                      ("FLAG", 3, 0.9)])
    q["B3w"] = ("5F ordering, as written", lambda: sess.sql(
        b3, order="as_written"))
    q["B3c"] = ("5F ordering, by cost", lambda: sess.sql(
        b3, order="by_cost"))
    q["B4"] = ("2F reports: long documents", lambda: sess.sql(
        _filter_sql("r", "reports", "report",
                    [("R", 1, 0.5), ("R", 2, 0.4)])))
    q["B5"] = ("1J reports x terms: the BioDEX shape",
               content_join("reports", "report", "terms", "term",
                            REACTION, 0.05))
    q["B6"] = ("1F + 1J: pushdown deletes the pair list",
               content_join("threads", "thread", "products",
                            "description", DISCUSS, 0.05,
                            lfilters=[("T", 1, 0.05)]))
    q["B7"] = ("1J pure pair predicate",
               content_join("reviews5k", "body", "products",
                            "description", DISCUSS, 0.1))

    def b8():
        import quail as _q
        return (sess.docs("threads").alias("t")
                .ai_join(sess.docs("products").alias("s"),
                         _q.prompt(DISCUSS, _q.col("t.thread"),
                                   _q.col("s.description")),
                         selectivity=0.3, semantics="exists")
                .select("t.id"))
    q["B8"] = ("1J exists", b8)

    def b9():
        import quail as _q
        return (sess.docs("threads").alias("t")
                .ai_join(sess.docs("products").alias("s"),
                         _q.prompt(DISCUSS, _q.col("t.thread"),
                                   _q.col("s.description")),
                         selectivity=0.3, semantics="exists")
                .ai_join(sess.docs("terms").alias("m"),
                         _q.prompt(MENTION, _q.col("t.thread"),
                                   _q.col("m.term")),
                         selectivity=0.1, semantics="anti")
                .select("t.id"))
    q["B9"] = ("exists + anti", b9)

    def b10():
        import quail as _q
        # the 3-way join: one prompt holds all three documents; every
        # (b, a, c) tuple of the cross product is one model call. The
        # provided selectivity is per tuple (the old two stages at .2
        # and .1 pass together for about .02 of the triples).
        return (sess.docs("threads2k").alias("b")
                .ai_join([sess.docs("reviews2k").alias("a"),
                          sess.docs("products").alias("c")],
                         _q.prompt(KEYEQ3, _q.col("b.thread"),
                                   _q.col("a.body"),
                                   _q.col("c.description")),
                         selectivity=0.02, anchor="b")
                .select("a.id", "b.id", "c.id"))
    q["B10"] = ("3-way join, planted keys: one prompt per triple", b10)

    def b11():
        import quail as _q
        return (sess.docs("reports").alias("r")
                .ai_join([sess.docs("terms").alias("m"),
                          sess.docs("products").alias("p")],
                         _q.prompt(REACTION_DISCUSS,
                                   _q.col("r.report"),
                                   _q.col("m.term"),
                                   _q.col("p.description")),
                         selectivity=0.0025, anchor="r")
                .select("m.id", "r.id", "p.id"))
    q["B11"] = ("3-way join, long anchors: one prompt per triple", b11)
    q["B12"] = ("2F + 1J: two-sided pushdown",
                content_join("threads", "thread", "products",
                             "description", DISCUSS, 0.1,
                             lfilters=[("T", 2, 0.3)],
                             rfilters=[]))
    q["B13"] = ("B2 rerun, new questions: the scan restore",
                lambda: sess.sql(_filter_sql(
                    "r", "reviews", "body",
                    [("FLAG", 3, 0.9), ("FLAG", 4, 0.8),
                     ("FLAG", 5, 0.8), ("FLAG", 7, 0.9),
                     ("FLAG", 8, 0.8)])))
    q["B14"] = ("B5 rerun, new question: the anchor restore",
                content_join("reports", "report", "terms", "term",
                             REACTION.replace(
                                 "describe the reaction",
                                 "explicitly report the reaction"),
                             0.05))

    def b15():
        import quail as _q
        return (sess.docs("threads").alias("t")
                .ai_filter(_q.prompt("{0}" + flag_question("T", 3),
                                     _q.col("t.thread")),
                           selectivity=0.9)
                .ai_join(sess.docs("terms").alias("m"),
                         _q.prompt(MENTION, _q.col("t.thread"),
                                   _q.col("m.term")),
                         selectivity=0.1, semantics="exists")
                .ai_join(sess.docs("reports").alias("r"),
                         _q.prompt(
                             "Do {0} and {1} both discuss medicine?",
                             _q.col("t.thread"), _q.col("r.report")),
                         selectivity=0.2)
                .select("t.id", "r.id"))
    q["B15"] = ("everything at once", b15)
    return q


# ----------------------------------------------------------- driver

def run_suite(data_dir, sf=0.1, lf=1, gpus=1, only=None,
              out_path=None, cpu_memory_gb=80):
    """cpu_memory_gb defaults to what the 96 GB worker container
    holds: a 64 GB store (8 slabs). The corpus KV usually exceeds it,
    so the length threshold keeps the longest documents - partial
    restores are the capacity arithmetic working, not a bug."""
    import quail
    from quail.planner.plan import EngineConfig

    d = build_sets(data_dir, sf, lf)
    sess = quail.Session(EngineConfig(gpus=gpus,
                                      cpu_memory_gb=cpu_memory_gb))
    register_sets(sess, d)
    qdefs = queries(sess)
    ids = [i for i in qdefs if only is None or i in only]
    suite = dict(sf=sf, lf=lf, gpus=gpus, passes={})
    try:
        for pass_name in ("cold", "warm"):
            sess.set_store(pass_name == "warm")
            if pass_name == "cold":
                sess.flush_store()
            rows = []
            t_pass = time.time()
            for qid in ids:
                desc, make = qdefs[qid]
                print(f"[quailb] {pass_name} {qid}: {desc}",
                      flush=True)
                try:
                    res = make().run()
                    row = dict(query=qid, desc=desc,
                               wall_s=res.report["wall_s"],
                               boot_s=res.report["boot_s"],
                               boot_kind=res.report.get("boot_kind"),
                               boot=res.report.get("boot"),
                               fresh_tokens=res.report["fresh_tokens"],
                               rows=len(res.rows),
                               peak_gib=res.report.get("peak_gib"),
                               stages=res.report["stages"],
                               store=res.report.get("store"))
                except Exception as e:            # noqa: BLE001
                    row = dict(query=qid, desc=desc,
                               error=f"{type(e).__name__}: {e}")
                rows.append(row)
                print(f"[quailb] {row}", flush=True)
            suite["passes"][pass_name] = dict(
                queries=rows,
                pass_wall_s=round(time.time() - t_pass, 1))
    finally:
        sess.close()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(suite, f, indent=2)
        print(f"[quailb] saved {out_path}", flush=True)
    return suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=0.1)
    ap.add_argument("--lf", type=int, default=1)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--data-dir", default="results/quailb_data")
    ap.add_argument("--only", default=None,
                    help="comma-separated query ids")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    out = args.out or f"results/quailb_sf{args.sf}_lf{args.lf}.json"
    run_suite(args.data_dir, sf=args.sf, lf=args.lf, gpus=args.gpus,
              only=only, out_path=out)


if __name__ == "__main__":
    main()

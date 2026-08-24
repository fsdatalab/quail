"""Fit the cost-model constants from packing-sweep records (issue #25).

CPU only. Reads the raw per-chunk records the packing_sweep cell wrote
to the quail-results volume (pull them first, e.g.
`modal volume get quail-results ablations/packing_sweep_<model>_<tag>.json .`)
and writes the aggregated summary the report and plot script read:
results/packing_sweep.json. Raw records are never committed.

The model, in the calibrate convention (the causal /2 is absorbed in
the fitted coefficient, matching the a2 the length sweep fits):

    gpu_seconds(chunk) = a * T + a2c * Sc + a2x * Sx

    T    fresh tokens in the chunk
    Sc   sum over causal segments of n * L: a segment's tokens attend
         ~L/2 of their own segment (documents+question in filters,
         anchor prefixes and the suffix's own tokens in joins)
    Sx   sum over join suffixes of n * (anchor + frame): every suffix
         token reads the WHOLE kept anchor context, no /2

Filter chunks have Sx = 0, so the filter chunks alone pin (a, a2c) -
the two constants the calibration files carry. Join chunks then test
whether a2x is just the /2 bookkeeping (a2x = 2 * a2c) or the paged
cross read costs more per context token.

Usage, from quail/:

    uv run python ablations/packing_sweep_fit.py \
        <packing_sweep_*.json ...> --out results/packing_sweep.json

When two records carry the same query id (a re-run), the one listed
later wins - pass a warm re-run after the cold record so the fits and
the query table use the compile-free measurement.
"""

import argparse
import json
from collections import defaultdict

# The constants that were loaded before this sweep, for the
# prediction-vs-measured comparison in the report. 4B: the old
# exploration's anchor (custom-kernel ladder rung + old alpha sweep).
# 32B: the 2026-08-20 length-sweep fit, measured before the
# attention-path assignment and the 08-22 bug-fix rounds.
OLD = {
    "qwen3-4b-fp8": (8.261391217315875e-06, 4.933635085554017e-10),
    "qwen3-32b-fp8": (5.66366321319351e-05, 1.528474073628451e-09),
}
N_BINS = 12


def chunk_points(rec, qid):
    """Per-chunk (T, Sc, Sx, suffixes, gpu_s) for one query."""
    q = rec["queries"][qid]
    out = []
    if "body_tokens" in q:
        body = q["body_tokens"]
        stages = q.get("q_tokens_stages") or [q["q_tokens"]]
        p = q.get("shared_p", 0)
        for ch in q["chunks"]:
            T = Sc = Sx = 0.0
            n_suf = 0
            for doc, stage, fresh in ch["pieces"]:
                if stage == 0:
                    n = body[doc] + stages[0]
                    T += n
                    Sc += n * n
                else:
                    # a later stage's tail past the stages' shared
                    # preamble, against the document's kept KV
                    t = stages[stage] - p
                    T += t
                    Sc += t * t
                    Sx += t * (body[doc] + p)
                    n_suf += 1
            assert T == ch["tokens"], (qid, T, ch["tokens"])
            out.append(dict(T=T, Sc=Sc, Sx=Sx, suffixes=n_suf,
                            gpu_s=ch["gpu_ms"] / 1e3))
    else:
        pre, suf, fr = (q["prefix_tokens"], q["suffix_tokens"],
                        q["frame_tokens"])
        for ch in q["chunks"]:
            T = Sc = Sx = 0.0
            n_suf = 0
            for a, start, end, carried in ch["pieces"]:
                h0 = pre[a]
                if carried:
                    T += h0
                    Sc += h0 * h0
                if start == 0 and end > start:     # frame entry
                    T += fr
                    Sx += fr * h0
                    Sc += fr * fr
                for k in range(start, end):
                    n = suf[k]
                    T += n
                    Sc += n * n
                    Sx += n * (h0 + fr)
                n_suf += end - start
            assert T == ch["tokens"], (qid, T, ch["tokens"])
            out.append(dict(T=T, Sc=Sc, Sx=Sx, suffixes=n_suf,
                            gpu_s=ch["gpu_ms"] / 1e3))
    return out


def lstsq(ys, cols):
    """Ordinary least squares via normal equations (2-4 columns)."""
    k = len(cols[0])
    ata = [[sum(c[i] * c[j] for c in cols) for j in range(k)]
           for i in range(k)]
    atb = [sum(c[i] * y for c, y in zip(cols, ys)) for i in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            f = ata[j][i] / ata[i][i]
            for m in range(i, k):
                ata[j][m] -= f * ata[i][m]
            atb[j] -= f * atb[i]
    x = [0.0] * k
    for i in reversed(range(k)):
        x[i] = (atb[i] - sum(ata[i][j] * x[j]
                             for j in range(i + 1, k))) / ata[i][i]
    return x


def _inv(mat):
    """Gauss-Jordan inverse for the small normal-equation matrices."""
    k = len(mat)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(k)]
           for i, row in enumerate(mat)]
    for i in range(k):
        piv = max(range(i, k), key=lambda r: abs(aug[r][i]))
        aug[i], aug[piv] = aug[piv], aug[i]
        if abs(aug[i][i]) < 1e-30:
            return None
        f = aug[i][i]
        aug[i] = [v / f for v in aug[i]]
        for r in range(k):
            if r != i and aug[r][i]:
                fr = aug[r][i]
                aug[r] = [v - fr * w for v, w in zip(aug[r], aug[i])]
    return [row[k:] for row in aug]


def fit_se(points, cols_of):
    """OLS plus standard errors: SE_i = sqrt(RSS/(n-k) * (X'X)^-1_ii).
    A coefficient whose SE rivals its value is not identified by
    these chunks - that is a result, not a failure."""
    ys = [p["gpu_s"] for p in points]
    cols = [cols_of(p) for p in points]
    k, n = len(cols[0]), len(cols)
    x = lstsq(ys, cols)
    pred = [sum(ci * xi for ci, xi in zip(c, x)) for c in cols]
    rss = sum((y - q) ** 2 for y, q in zip(ys, pred))
    mean = sum(ys) / n
    r2 = 1 - rss / (sum((y - mean) ** 2 for y in ys) or 1e-12)
    ata = [[sum(c[i] * c[j] for c in cols) for j in range(k)]
           for i in range(k)]
    inv = _inv(ata)
    if inv is None or n <= k:
        return x, [None] * k, r2
    sigma2 = rss / (n - k)
    ses = [(sigma2 * inv[i][i]) ** 0.5 if inv[i][i] > 0 else None
           for i in range(k)]
    return x, ses, r2


def fit(points, cols_of):
    ys = [p["gpu_s"] for p in points]
    cols = [cols_of(p) for p in points]
    x = lstsq(ys, cols)
    pred = [sum(ci * xi for ci, xi in zip(c, x)) for c in cols]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
    mean = sum(ys) / len(ys)
    ss_tot = sum((y - mean) ** 2 for y in ys) or 1e-12
    return x, 1 - ss_res / ss_tot


def bins(points, n_bins=N_BINS):
    """Bin chunks by mean attended context per token, x = (Sc/2+Sx)/T.
    Committed in place of per-chunk records (house rule: aggregates
    only). Each bin: mean x, mean us/token, chunk count, token sum."""
    pts = sorted(points, key=lambda p: (p["Sc"] / 2 + p["Sx"]) / p["T"])
    size = max(1, -(-len(pts) // n_bins))
    out = []
    for i in range(0, len(pts), size):
        grp = pts[i:i + size]
        tok = sum(p["T"] for p in grp)
        out.append(dict(
            x_mean_ctx=round(sum((p["Sc"] / 2 + p["Sx"]) / p["T"]
                                 for p in grp) / len(grp), 1),
            us_per_token=round(sum(p["gpu_s"] for p in grp)
                               / tok * 1e6, 3),
            chunks=len(grp), tokens=tok))
    return out


def analyze(records):
    """records: list of raw record dicts for ONE model. Tags: "full"
    and "ext" carry per-chunk data (disjoint query sets, merged here);
    "rep*" records only feed the container table."""
    model = records[0]["model"]
    a_old, a2_old = OLD[model]
    main = [r for r in records if not r["tag"].startswith("rep")]
    full = next(r for r in main if r["tag"] == "full")
    pts, sources = {}, {}
    for r in main:
        for qid in r["queries"]:
            pts[qid] = chunk_points(r, qid)
            for p in pts[qid]:
                p["tag"] = r["tag"]
            sources[qid] = r

    # Each record is one container, and containers differ by a few
    # percent in plain rate (the container table below measures it on
    # repeats). Records beyond "full" therefore get an offset column
    # delta_tag on T, so a second container's rate lands in its
    # offset instead of bending the shared coefficients.
    tags = sorted({r["tag"] for r in main if r["tag"] != "full"})

    def cols(p, k):
        base = [p["T"], p["Sc"], p["Sx"], p["suffixes"]][:k]
        return base + [p["T"] if p["tag"] == t else 0.0 for t in tags]

    every = [p for ps in pts.values() for p in ps]
    # exclusion pass: a chunk more than 5% off its own GPU time under
    # the full model is excluded from every fit - ordinary scatter is
    # under 2%. Observed causes, both reported in the sweep report:
    # one-time kernel compiles (a cold container's first chunk; tiny
    # tail-chunk shapes no warmup covered, 0.7-0.9 s once), and the
    # ~20 ms per-chunk launch floor that dominates chunks under ~2k
    # tokens (a gated chain's trailing suffix chunks).
    x0, _ = fit(every, lambda p: cols(p, 4))
    for p in every:
        pred = sum(c * xi for c, xi in zip(cols(p, 4), x0))
        p["_drop"] = abs(p["gpu_s"] - pred) > 0.05 * p["gpu_s"]
    dropped = [p for p in every if p["_drop"]]

    # (a, a2c) come from the first record's causal chunks alone - the
    # committed-constants fit, on the same container as that record's
    # joins, so the shape terms below are anchored by chunks with no
    # container offset in them.
    filt = [p for p in every if not p["_drop"] and p["Sx"] == 0
            and p["tag"] == "full"]
    (a, a2c), r2f = fit(filt, lambda p: [p["T"], p["Sc"]])
    fits = dict(filter=dict(
        a_s_per_token=a, a2c_s_per_token2=a2c, r2=round(r2f, 5),
        n_chunks=len(filt),
        n_excluded=sum(1 for p in dropped
                       if p["Sx"] == 0 and p["tag"] == "full")))

    # everything else fits the residual past (a, a2c): the cross
    # coefficient, the per-suffix cost, and one rate offset per extra
    # record (= per container). The full record's join chunks carry
    # no offset column, so they anchor a2x and the per-suffix term;
    # the extra records' flat excess lands in their offsets.
    rest = [p for p in every if not p["_drop"]
            and not (p["Sx"] == 0 and p["tag"] == "full")]
    if rest:
        ys = [p["gpu_s"] - a * p["T"] - a2c * p["Sc"] for p in rest]

        def rcols(p):
            return [p["Sx"], float(p["suffixes"])] + \
                [p["T"] if p["tag"] == t else 0.0 for t in tags]

        xr = lstsq(ys, rcols_matrix := [rcols(p) for p in rest])
        a2x, suf_s = xr[0], xr[1]
        offs = dict(zip(tags, xr[2:]))
        # per-query residual past the full model, in us per token so
        # suffix-sparse chunks (a chain's mixed chunks) do not blow
        # the number up; comparable to the container band directly
        leftover = {}
        for qid in pts:
            cross = [p for p in pts[qid]
                     if p["Sx"] > 0 and not p["_drop"]]
            if not cross:
                continue
            lo = sum(p["gpu_s"] - a * p["T"] - a2c * p["Sc"]
                     - a2x * p["Sx"] - suf_s * p["suffixes"]
                     - offs.get(p["tag"], 0.0) * p["T"]
                     for p in cross)
            leftover[qid] = round(
                lo / sum(p["T"] for p in cross) * 1e6, 2)
        pred = [sum(c * x for c, x in zip(row, xr))
                for row in rcols_matrix]
        ss_res = sum((y - q) ** 2 for y, q in zip(ys, pred))
        mean = sum(ys) / len(ys)
        ss_tot = sum((y - mean) ** 2 for y in ys) or 1e-12
        fits["join"] = dict(
            a2x_s_per_token2=a2x, a2x_over_a2c=round(a2x / a2c, 2),
            per_suffix_s=suf_s,
            container_offset_us_per_token={
                t: round(d * 1e6, 3) for t, d in offs.items()},
            residual_us_per_token=leftover,
            n_excluded=sum(1 for p in dropped if p["Sx"] > 0),
            r2=round(1 - ss_res / ss_tot, 5))

    # Per-query fits, for comparison against the shared constants: each
    # query alone, no container offsets, only the columns its chunks
    # carry. Standard errors say which constants that query's packings
    # can actually pin down; the shared fit above exists because no
    # single query pins them all.
    per_query = {}
    for qid in pts:
        kq = [p for p in pts[qid] if not p["_drop"]]
        has_x = any(p["Sx"] > 0 for p in kq)
        names = ["a", "a2c"] + (["a2x", "per_suffix_s"] if has_x
                                else [])
        x, ses, r2q = fit_se(kq, lambda p: [p["T"], p["Sc"]] + (
            [p["Sx"], float(p["suffixes"])] if has_x else []))
        per_query[qid] = dict(
            tag=sources[qid]["tag"], n_chunks=len(kq),
            r2=round(r2q, 5),
            coefficients={nm: [v, ses[i]]
                          for i, (nm, v) in enumerate(zip(names, x))})
    fits["per_query"] = per_query

    queries = {}
    for qid in pts:
        q = sources[qid]["queries"][qid]
        p = pts[qid]
        T = sum(c["T"] for c in p)
        pred_old = sum(a_old * c["T"] + a2_old * (c["Sc"] + c["Sx"])
                       for c in p)
        keep_q = [c for c in p if not c["_drop"]]
        row = dict(
            kind=q["summary"]["kind"], chunks=q["summary"]["chunks"],
            fresh_tokens=q["summary"]["fresh_tokens"],
            wall_s=q["summary"]["wall_s"], gpu_s=q["summary"]["gpu_s"],
            us_per_token_wall=q["summary"]["us_per_token_wall"],
            us_per_token_gpu=q["summary"]["us_per_token_gpu"],
            predicted_us_old=round(pred_old / T * 1e6, 3),
            chunk_bins=bins(keep_q or p))
        if len(keep_q) < len(p):
            # the summary keeps the run's honest totals; only the
            # bins (and fits) drop the compile-stall chunks
            row["n_excluded"] = len(p) - len(keep_q)
        queries[qid] = row

    # deliberate same-query repeats only (full + rep*): an ext/ext2
    # pair re-runs a query to shed one-time kernel compiles, and that
    # delta is not container variation
    containers = defaultdict(dict)
    for r in records:
        if not (r["tag"] == "full" or r["tag"].startswith("rep")):
            continue
        for qid, q in r["queries"].items():
            containers[qid][r["tag"]] = q["summary"]["us_per_token_gpu"]
    spread = {}
    for qid, by_tag in containers.items():
        vals = sorted(by_tag.values())
        if len(vals) >= 2:
            spread[qid] = dict(
                by_container=by_tag,
                band_pct=round((vals[-1] / vals[0] - 1) * 100, 1))
    return dict(queries=queries, fits=fits, containers=spread,
                budget=full["budget"], boot_s=full["boot_s"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+")
    ap.add_argument("--out", default="results/packing_sweep.json")
    args = ap.parse_args()
    by_model = defaultdict(list)
    raw_paths = defaultdict(list)
    for path in args.records:
        rec = json.load(open(path))
        by_model[rec["model"]].append(rec)
        raw_paths[rec["model"]].append(
            f"/results/ablations/packing_sweep_{rec['model']}_"
            f"{rec['tag']}.json")
    out = dict(models={m: analyze(rs) for m, rs in by_model.items()},
               loaded_before={m: dict(a_s_per_token=a,
                                      a2_s_per_token2=a2)
                              for m, (a, a2) in OLD.items()},
               raw_volume_paths=dict(raw_paths))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {args.out}")
    for m, block in out["models"].items():
        f = block["fits"]["filter"]
        print(f"[{m}] filter a={f['a_s_per_token'] * 1e6:.3f} us/tok "
              f"a2c={f['a2c_s_per_token2']:.3e} R2={f['r2']}")
        if "join" in block["fits"]:
            j = block["fits"]["join"]
            print(f"[{m}] join a2x={j['a2x_s_per_token2']:.3e} "
                  f"(= {j['a2x_over_a2c']} x a2c) per-suffix "
                  f"{j['per_suffix_s'] * 1e6:.1f} us  offsets "
                  f"{j['container_offset_us_per_token']}  residual "
                  f"us/tok {j['residual_us_per_token']}")
        for qid, pq in block["fits"]["per_query"].items():
            def show(nm, scale, unit):
                c = pq["coefficients"].get(nm)
                if c is None:
                    return f"{nm}=-"
                se = "?" if c[1] is None else f"{c[1] * scale:.2f}"
                return f"{nm}={c[0] * scale:.2f}±{se}{unit}"
            print(f"[{m}]   {qid:7s} ({pq['tag']:4s} n={pq['n_chunks']:3d}) "
                  + "  ".join([show("a", 1e6, "us"),
                               show("a2c", 1e10, "e-10"),
                               show("a2x", 1e10, "e-10"),
                               show("per_suffix_s", 1e6, "us")]))


if __name__ == "__main__":
    main()

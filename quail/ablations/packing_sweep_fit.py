"""Fit the cost-model constants from packing-sweep records (issue #25).

CPU only. Reads the raw per-chunk records the packing_sweep cell wrote
to the quail-results volume (pull them first, e.g.
`modal volume get quail-results ablations/packing_sweep_<model>_<tag>.json .`)
and writes the aggregated summary the report and plot script read:
results/packing_sweep.json. Raw records are never committed.

The model (one set of constants for both causal and paged attention):

    gpu_seconds(chunk) = a * T + a2 * S + c + p * suffixes

    T          fresh tokens in the chunk
    S          total attention pairs: causal (n^2 per segment, the
               /2 absorbed into a2) plus cross-read (suffix_tokens *
               anchor_tokens)
    suffixes   paged-attention dispatch count (suffix reads against
               non-contiguous KV pages)

One a2 because the FLOPs per attention pair are the same regardless of
where the KV lives; the paged-attention overhead is a per-dispatch
cost (p), not a per-pair multiplier.

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

N_BINS = 12


def chunk_points(rec, qid):
    """Per-chunk (T, S, suffixes, gpu_s) for one query. S merges
    causal self-attention and cross-read pairs into a single column."""
    q = rec["queries"][qid]
    out = []
    if "body_tokens" in q:
        body = q["body_tokens"]
        stages = q.get("q_tokens_stages") or [q["q_tokens"]]
        p = q.get("shared_p", 0)
        for ch in q["chunks"]:
            T = S = 0.0
            n_suf = 0
            for doc, stage, fresh in ch["pieces"]:
                if stage == 0:
                    n = body[doc] + stages[0]
                    T += n
                    S += n * n
                else:
                    t = stages[stage] - p
                    T += t
                    S += t * t + t * (body[doc] + p)
                    n_suf += 1
            assert T == ch["tokens"], (qid, T, ch["tokens"])
            out.append(dict(T=T, S=S, suffixes=n_suf,
                            gpu_s=ch["gpu_ms"] / 1e3))
    else:
        pre, suf, fr = (q["prefix_tokens"], q["suffix_tokens"],
                        q["frame_tokens"])
        for ch in q["chunks"]:
            T = S = 0.0
            n_suf = 0
            for a, start, end, carried in ch["pieces"]:
                h0 = pre[a]
                if carried:
                    T += h0
                    S += h0 * h0
                if start == 0 and end > start:
                    T += fr
                    S += fr * fr + fr * h0
                for k in range(start, end):
                    n = suf[k]
                    T += n
                    S += n * n + n * (h0 + fr)
                n_suf += end - start
            assert T == ch["tokens"], (qid, T, ch["tokens"])
            out.append(dict(T=T, S=S, suffixes=n_suf,
                            gpu_s=ch["gpu_ms"] / 1e3))
    return out


def lstsq(ys, cols):
    """Ordinary least squares via normal equations (2-6 columns)."""
    k = len(cols[0])
    ata = [[sum(c[i] * c[j] for c in cols) for j in range(k)]
           for i in range(k)]
    atb = [sum(c[i] * y for c, y in zip(cols, ys)) for i in range(k)]
    for i in range(k):
        piv = max(range(i, k), key=lambda r: abs(ata[r][i]))
        if abs(ata[piv][i]) < 1e-30:
            ata[i][i] = 1e-30
            continue
        ata[i], ata[piv] = ata[piv], ata[i]
        atb[i], atb[piv] = atb[piv], atb[i]
        for j in range(i + 1, k):
            f = ata[j][i] / ata[i][i]
            for m in range(i, k):
                ata[j][m] -= f * ata[i][m]
            atb[j] -= f * atb[i]
    x = [0.0] * k
    for i in reversed(range(k)):
        if abs(ata[i][i]) < 1e-30:
            x[i] = 0.0
            continue
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
    """Bin chunks by mean attention intensity S/T.
    Each bin: mean S/T, mean us/token, chunk count, token sum."""
    pts = sorted(points, key=lambda p: p["S"] / p["T"])
    size = max(1, -(-len(pts) // n_bins))
    out = []
    for i in range(0, len(pts), size):
        grp = pts[i:i + size]
        tok = sum(p["T"] for p in grp)
        out.append(dict(
            x_mean_attn_per_tok=round(sum(p["S"] / p["T"]
                                          for p in grp) / len(grp), 1),
            us_per_token=round(sum(p["gpu_s"] for p in grp)
                               / tok * 1e6, 3),
            chunks=len(grp), tokens=tok))
    return out


def analyze(records):
    """records: list of raw record dicts for ONE model."""
    model = records[0]["model"]
    main = [r for r in records if not r["tag"].startswith("rep")]
    full = next(r for r in main if r["tag"] == "full")
    pts, sources = {}, {}
    for r in main:
        for qid in r["queries"]:
            pts[qid] = chunk_points(r, qid)
            for p in pts[qid]:
                p["tag"] = r["tag"]
            sources[qid] = r

    tags = sorted({r["tag"] for r in main if r["tag"] != "full"})

    def cols(p):
        base = [p["T"], p["S"], 1.0, float(p["suffixes"])]
        return base + [p["T"] if p["tag"] == t else 0.0 for t in tags]

    every = [p for ps in pts.values() for p in ps]

    # initial fit to identify outliers
    x0, _ = fit(every, cols)
    for p in every:
        pred = sum(c * xi for c, xi in zip(cols(p), x0))
        p["_drop"] = abs(p["gpu_s"] - pred) > 0.05 * p["gpu_s"]
    dropped = [p for p in every if p["_drop"]]

    # main fit on clean data
    clean = [p for p in every if not p["_drop"]]
    xc, r2 = fit(clean, cols)
    a, a2, c, p_coeff = xc[0], xc[1], xc[2], xc[3]
    offs = dict(zip(tags, xc[4:]))

    fits = dict(main=dict(
        a_s_per_token=a, a2_s_per_token2=a2,
        c_s_per_chunk=c, p_s_per_suffix=p_coeff,
        r2=round(r2, 5), n_chunks=len(clean),
        n_excluded=len(dropped),
        container_offset_us_per_token={
            t: round(d * 1e6, 3) for t, d in offs.items()}))

    # residual per query under the main model
    per_query_residual = {}
    for qid in pts:
        kq = [p for p in pts[qid] if not p["_drop"]]
        if not kq:
            continue
        lo = sum(p["gpu_s"] - a * p["T"] - a2 * p["S"]
                 - c - p_coeff * p["suffixes"]
                 - offs.get(p["tag"], 0.0) * p["T"]
                 for p in kq)
        per_query_residual[qid] = round(
            lo / sum(p["T"] for p in kq) * 1e6, 2)
    fits["main"]["residual_us_per_token"] = per_query_residual

    # per-query fits with standard errors
    per_query = {}
    for qid in pts:
        kq = [p for p in pts[qid] if not p["_drop"]]
        if not kq:
            continue
        has_suf = any(p["suffixes"] > 0 for p in kq)
        names = ["a", "a2", "c"] + (["p"] if has_suf else [])

        def qcols(pt, _has_suf=has_suf):
            base = [pt["T"], pt["S"], 1.0]
            return base + ([float(pt["suffixes"])] if _has_suf else [])

        x, ses, r2q = fit_se(kq, qcols)
        per_query[qid] = dict(
            tag=sources[qid]["tag"], n_chunks=len(kq),
            r2=round(r2q, 5),
            coefficients={nm: [v, ses[i]]
                          for i, (nm, v) in enumerate(zip(names, x))})
    fits["per_query"] = per_query

    # query summary table
    queries = {}
    for qid in pts:
        q = sources[qid]["queries"][qid]
        p_list = pts[qid]
        T = sum(ch["T"] for ch in p_list)
        keep_q = [ch for ch in p_list if not ch["_drop"]]
        row = dict(
            kind=q["summary"]["kind"], chunks=q["summary"]["chunks"],
            fresh_tokens=q["summary"]["fresh_tokens"],
            wall_s=q["summary"]["wall_s"], gpu_s=q["summary"]["gpu_s"],
            us_per_token_wall=q["summary"]["us_per_token_wall"],
            us_per_token_gpu=q["summary"]["us_per_token_gpu"],
            chunk_bins=bins(keep_q or p_list))
        if len(keep_q) < len(p_list):
            row["n_excluded"] = len(p_list) - len(keep_q)
        queries[qid] = row

    # container variation from deliberate same-query repeats
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
               raw_volume_paths=dict(raw_paths))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {args.out}")
    for m, block in out["models"].items():
        f = block["fits"]["main"]
        print(f"[{m}] a={f['a_s_per_token'] * 1e6:.3f} us/tok "
              f"a2={f['a2_s_per_token2']:.3e} "
              f"c={f['c_s_per_chunk'] * 1e3:.2f} ms/chunk "
              f"p={f['p_s_per_suffix'] * 1e6:.1f} us/suffix "
              f"R2={f['r2']}")
        for qid, pq in block["fits"]["per_query"].items():
            def show(nm, scale, unit):
                c = pq["coefficients"].get(nm)
                if c is None:
                    return f"{nm}=-"
                se = "?" if c[1] is None else f"{c[1] * scale:.2f}"
                return f"{nm}={c[0] * scale:.2f}+/-{se}{unit}"
            print(f"[{m}]   {qid:7s} ({pq['tag']:4s} n={pq['n_chunks']:3d}) "
                  + "  ".join([show("a", 1e6, "us"),
                               show("a2", 1e10, "e-10"),
                               show("c", 1e3, "ms"),
                               show("p", 1e6, "us")]))


if __name__ == "__main__":
    main()

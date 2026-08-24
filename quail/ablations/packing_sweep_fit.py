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

    filt = [p for p in every if not p["_drop"] and p["Sx"] == 0]
    join = [p for p in every if not p["_drop"] and p["Sx"] > 0]
    kept = filt + join
    xf, r2f = fit(filt, lambda p: cols(p, 2))
    a, a2c = xf[0], xf[1]
    offsets = {t: round(d * 1e6, 3) for t, d in zip(tags, xf[2:])}
    fits = dict(filter=dict(
        a_s_per_token=a, a2c_s_per_token2=a2c, r2=round(r2f, 5),
        n_chunks=len(filt),
        n_excluded=sum(1 for p in dropped if p["Sx"] == 0),
        container_offset_us_per_token=offsets))

    def base(p):
        d = dict(zip(tags, xf[2:]))
        return ((a + d.get(p["tag"], 0.0)) * p["T"] + a2c * p["Sc"])

    if join:
        resid = [p["gpu_s"] - base(p) for p in join]
        a2x = (sum(r * p["Sx"] for r, p in zip(resid, join))
               / sum(p["Sx"] ** 2 for p in join))
        leftover = {}
        for qid in pts:
            cross = [p for p in pts[qid]
                     if p["Sx"] > 0 and not p["_drop"]]
            n_suf = sum(p["suffixes"] for p in cross)
            if not n_suf:
                continue
            lo = sum(p["gpu_s"] - base(p) - a2x * p["Sx"]
                     for p in cross)
            leftover[qid] = round(lo / n_suf * 1e6, 1)
        x4, r24 = fit(kept, lambda p: cols(p, 4))
        fits["join"] = dict(
            a2x_s_per_token2=a2x, a2x_over_a2c=round(a2x / a2c, 2),
            leftover_us_per_suffix=leftover,
            n_excluded=sum(1 for p in dropped if p["Sx"] > 0),
            joint4=dict(a_s_per_token=x4[0], a2c_s_per_token2=x4[1],
                        a2x_s_per_token2=x4[2], per_suffix_s=x4[3],
                        container_offset_us_per_token={
                            t: round(d * 1e6, 3)
                            for t, d in zip(tags, x4[4:])},
                        r2=round(r24, 5)))

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

    containers = defaultdict(dict)
    for r in records:
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
                  f"(= {j['a2x_over_a2c']} x a2c) leftover/suffix "
                  f"{j['leftover_us_per_suffix']}")


if __name__ == "__main__":
    main()

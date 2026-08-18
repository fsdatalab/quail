"""Fits for the calibration sweeps.

Reads the rows the runner saved (experiments/modal_calibrate.py),
fits the three cost models, and writes one JSON with every fitted
constant, its producer, and the validation numbers:

    step time   T_step = b0 + f_B(B) + f_P(P) + bN * N
    host time   T_host = h0 + hN * N + hA * A + hR * R
    prefill     T_pre(h) = a1 * h + a2 * h * h   (a2 is attention)

plus t_read (per-token cached-read time), eps = t_read / a1, and the
offload-vs-recompute crossover per measured transfer tier.

f_B and f_P are monotone piecewise-linear: per-segment slopes fitted
nonnegative, so the curves never decrease but may bend either way.
Every fit uses nonnegative least squares - noise must not hand any
coefficient a negative sign.

numpy only. Run:

    python -m quail.plan.fit --rows results/engine/calibrate_all.json \
        --out results/engine/cost_model_fit.json
"""

import glob as globmod
import json

import numpy as np

from ..configs import QWEN3_4B_FP8
from .calib import resident_blocks

B_KNOTS = (0, 512, 1024, 2048, 4096, 8192, 16384, 32768)
P_KNOTS = (0, 1e5, 1e6, 4e6, 1.6e7, 6e7, 1.4e8)
RANK_MIN_GAP = 0.03
HOLDOUT_EVERY = 5

NO_R_TERM = (
    "a resident-bytes term cannot be fitted from these cells: with a "
    "shared per-request fresh-token count c, P = (c/m)*R + c(1-c)/2*N "
    "exactly, so the design is rank-deficient. Only C3 cells (wide "
    "spread in c at matched P) identify it, and none were run.")


# ---- nonnegative least squares ----------------------------------------

def nnls(X, y):
    """Nonnegative least squares, exact: the Lawson-Hanson active-set
    algorithm. A gradient method was tried first and could not
    converge on the quadratic prefill design [1, h, h*h], whose
    columns are nearly collinear over the swept lengths; the
    active-set solves each candidate support exactly instead. Columns
    are max-scaled for conditioning. Deterministic."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    scale = np.abs(X).max(axis=0)
    scale[scale == 0] = 1.0
    Xs = X / scale
    n = Xs.shape[1]
    passive = np.zeros(n, dtype=bool)
    theta = np.zeros(n)
    tol = 1e-10 * max(1.0, float(np.abs(Xs.T @ y).max()))
    for _ in range(3 * n + 30):
        w = Xs.T @ (y - Xs @ theta)
        w[passive] = -np.inf
        j = int(np.argmax(w))
        if w[j] <= tol:
            break
        passive[j] = True
        while True:
            s = np.zeros(n)
            s[passive] = np.linalg.lstsq(Xs[:, passive], y, rcond=None)[0]
            if s[passive].min() > 0:
                theta = s
                break
            # walk from theta toward s until the first passive
            # coefficient hits zero, then drop it from the support
            mask = passive & (s <= 0)
            steps = theta[mask] / (theta[mask] - s[mask])
            alpha = float(steps.min())
            theta = theta + alpha * (s - theta)
            passive &= theta > tol
            theta[~passive] = 0.0
    return theta / scale


# ---- monotone piecewise-linear basis ----------------------------------

def segment_basis(x, knots):
    """One column per segment: how far x has traveled inside it. A
    nonnegative coefficient per column is that segment's slope, so
    any fit is monotone nondecreasing with free curvature."""
    x = np.asarray(x, dtype=float)
    cols = [np.clip(x - knots[j], 0.0, knots[j + 1] - knots[j])
            for j in range(len(knots) - 1)]
    return np.stack(cols, axis=1)


def segment_eval(x, knots, slopes):
    return segment_basis(np.atleast_1d(x), knots) @ np.asarray(slopes)


# ---- rows -------------------------------------------------------------

def load_rows(patterns):
    rows = []
    for pattern in patterns:
        for path in sorted(globmod.glob(pattern)) or [pattern]:
            with open(path) as f:
                rows.extend(json.load(f))
    return rows


def cell_rows(rows, families):
    """Valid measured cells of the given families, deduplicated by
    cell name (the last occurrence wins, so a retried or re-run cell
    supersedes its earlier row)."""
    out = {}
    for r in rows:
        if r.get("family") in families and r.get("cell") \
                and r.get("exec_ms_median") is not None:
            if r.get("valid"):
                out[r["cell"]] = r
    return list(out.values())


def regime(row):
    cached = [q["cached"] for q in row["requests"]]
    if all(c == 0 for c in cached):
        return "prefill"
    if all(c > 0 for c in cached):
        return "cached"
    return "mixed"


def split_holdout(rows):
    """All c4 cells plus every HOLDOUT_EVERY-th c1/c2 cell (by sorted
    name, so the split is stable) are held out of the step-model
    fit."""
    train, held = [], []
    ordered = sorted(rows, key=lambda r: r["cell"])
    idx = {}
    for r in ordered:
        if r["family"] == "c4":
            held.append(r)
            continue
        if r["family"] in ("c1", "c2"):
            i = idx.setdefault(r["family"], 0)
            idx[r["family"]] = i + 1
            if i % HOLDOUT_EVERY == 0:
                held.append(r)
                continue
        train.append(r)
    return train, held


# ---- the step model ---------------------------------------------------

def has_c3_support(rows):
    """True only when real C3 decorrelation cells exist: same-N pairs
    with matched P (within 2 percent) whose suffix lengths differ by
    4x or more, the long side being a genuine long suffix (128 tokens
    or more) over a cached context. The plain C2 grid contains
    matched-P pairs by accident (c=16 at h=16K against c=64 at h=4K),
    but its suffixes stop at 64 tokens, so it never passes."""
    def suffix(row):
        cs = {q["new"] for q in row["requests"]}
        return max(cs) if len(cs) == 1 else None

    pairs = 0
    by_n = {}
    for r in rows:
        if r.get("p") and all(q["cached"] > 0 for q in r["requests"]):
            by_n.setdefault(r["n"], []).append(r)
    for group in by_n.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                ca, cb = suffix(a), suffix(b)
                if ca is None or cb is None:
                    continue
                lo, hi = min(ca, cb), max(ca, cb)
                close = abs(a["p"] - b["p"]) / max(a["p"], b["p"]) < 0.02
                if close and hi >= 128 and lo * 4 <= hi:
                    pairs += 1
    return pairs >= 2


def step_design(rows):
    B = np.array([r["b"] for r in rows], dtype=float)
    P = np.array([r["p"] for r in rows], dtype=float)
    N = np.array([r["n"] for r in rows], dtype=float)
    X = np.hstack([np.ones((len(rows), 1)),
                   segment_basis(B, B_KNOTS),
                   segment_basis(P, P_KNOTS),
                   N[:, None]])
    return X


def step_predict(fit, rows):
    B = np.array([r["b"] for r in rows], dtype=float)
    P = np.array([r["p"] for r in rows], dtype=float)
    N = np.array([r["n"] for r in rows], dtype=float)
    return (fit["b0_s"]
            + segment_eval(B, fit["b_knots"], fit["b_slopes"])
            + segment_eval(P, fit["p_knots"], fit["p_slopes"])
            + fit["beta_n_s"] * N)


def mape(pred, y):
    return float(np.mean(np.abs(pred - y) / y))


def ranking_accuracy(pred, y):
    """Over pairs whose measured times differ by more than
    RANK_MIN_GAP of the larger: the fraction the model orders the
    same way."""
    total = right = 0
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            if abs(y[i] - y[j]) <= RANK_MIN_GAP * max(y[i], y[j]):
                continue
            total += 1
            if (pred[i] - pred[j]) * (y[i] - y[j]) > 0:
                right += 1
    return float(right / total) if total else None


def fit_gpu(rows, fit_R=False):
    """The step model on the valid alpha + c1 + c2 cells, validated on
    the held-out cells (all of c4 plus every 5th c1/c2)."""
    if fit_R:
        if not has_c3_support(rows):
            raise ValueError(NO_R_TERM)
        raise NotImplementedError(
            "C3-style cells are present, but the R-term fit lands "
            "together with a real C3 sweep; this calibration skipped "
            "C3 on purpose.")
    usable = cell_rows(rows, ("alpha", "c1", "c2", "c4"))
    train, held = split_holdout(usable)
    y = np.array([r["exec_ms_median"] for r in train]) / 1e3
    theta = nnls(step_design(train), y)
    nb = len(B_KNOTS) - 1
    npk = len(P_KNOTS) - 1
    fit = dict(
        b0_s=float(theta[0]),
        b_knots=list(B_KNOTS), b_slopes=[float(t) for t in theta[1:1 + nb]],
        p_knots=list(P_KNOTS),
        p_slopes=[float(t) for t in theta[1 + nb:1 + nb + npk]],
        beta_n_s=float(theta[1 + nb + npk]),
        train_cells=len(train), held_cells=len(held))
    for name, part in (("train", train), ("held", held)):
        if not part:
            continue
        yy = np.array([r["exec_ms_median"] for r in part]) / 1e3
        pred = step_predict(fit, part)
        fit[f"{name}_mape"] = mape(pred, yy)
        fit[f"{name}_ranking"] = ranking_accuracy(pred, yy)
        for reg in ("prefill", "cached", "mixed"):
            sub = [r for r in part if regime(r) == reg]
            if sub:
                yr = np.array([r["exec_ms_median"] for r in sub]) / 1e3
                fit[f"{name}_mape_{reg}"] = mape(step_predict(fit, sub), yr)
    return fit


# ---- the host model ---------------------------------------------------

def fit_host(rows):
    """R (resident blocks) is load-bearing: at fixed N and A the c2
    cells' host time still rises sevenfold with h - the per-step
    block tables span each request's whole context - so a plane
    without R cannot fit prefill and cached cells at once."""
    usable = [r for r in cell_rows(rows, ("alpha", "c1", "c2"))
              if r.get("sched_ms_median") is not None
              and r.get("update_ms_median") is not None]
    y = np.array([r["sched_ms_median"] + r["update_ms_median"]
                  for r in usable]) / 1e3
    X = np.hstack([np.ones((len(usable), 1)),
                   np.array([r["n"] for r in usable], float)[:, None],
                   np.array([r["a"] for r in usable], float)[:, None],
                   np.array([resident_blocks(r["requests"])
                             for r in usable], float)[:, None]])
    theta = nnls(X, y)
    pred = X @ theta
    return dict(h0_s=float(theta[0]), h_n_s=float(theta[1]),
                h_a_s=float(theta[2]), h_r_s=float(theta[3]),
                cells=len(usable), mape=mape(pred, y))


# ---- prefill vs length ------------------------------------------------

def fit_alpha(rows):
    usable = cell_rows(rows, ("alpha",))
    h = np.array([r["b"] for r in usable], dtype=float)
    y = np.array([r["exec_ms_median"] for r in usable]) / 1e3
    X = np.stack([np.ones_like(h), h, h * h], axis=1)
    theta = nnls(X, y)
    pred = X @ theta
    lin = nnls(np.stack([np.ones_like(h), h], axis=1), y)
    lin_pred = np.stack([np.ones_like(h), h], axis=1) @ lin
    return dict(
        intercept_s=float(theta[0]),
        a1_s_per_token=float(theta[1]),
        a2_s_per_token2=float(theta[2]),
        bend_tokens=float(theta[1] / theta[2]) if theta[2] > 0 else None,
        mape=mape(pred, y),
        linear_only_mape=mape(lin_pred, y),
        cells=len(usable))


# ---- cached-read time -------------------------------------------------

T_READ_C = 32
T_READ_N = 32
T_READ_H = (8192, 16384)


def t_read_s(rows, gpu_fit):
    """Per-token cached-read time at the reference suffix c=32: the
    fitted f_P slope between the two reference c2 cells, times c.
    Cross-checked against the raw slope of the same two cells'
    measured medians; the two must agree within about 20 percent for
    the number to be trusted."""
    def p_of(h):
        return T_READ_N * (T_READ_C * h + T_READ_C * (T_READ_C + 1) // 2)

    p1, p2 = p_of(T_READ_H[0]), p_of(T_READ_H[1])
    f1, f2 = (float(segment_eval(p, gpu_fit["p_knots"],
                                 gpu_fit["p_slopes"])[0])
              for p in (p1, p2))
    fitted = (f2 - f1) / (p2 - p1) * T_READ_C

    names = {f"c2_c{T_READ_C}_h{h}_n{T_READ_N}": h for h in T_READ_H}
    meas = {}
    for r in cell_rows(rows, ("c2",)):
        if r["cell"] in names:
            meas[names[r["cell"]]] = r["exec_ms_median"] / 1e3
    raw = None
    if len(meas) == 2:
        dt = meas[T_READ_H[1]] - meas[T_READ_H[0]]
        dtok = T_READ_N * (T_READ_H[1] - T_READ_H[0])
        raw = dt / dtok
    agree = (fitted / raw) if raw else None
    return dict(t_read_s_per_token=fitted,
                t_read_raw_s_per_token=raw,
                fitted_over_raw=agree,
                reference=dict(c=T_READ_C, n=T_READ_N, h=list(T_READ_H)))


# ---- transfers and crossovers -----------------------------------------

def crossovers(rows, a1, a2):
    """Per measured tier: bytes per second, per-token transfer time
    for KV at kappa bytes per token, and where recomputation stops
    beating the transfer. a1 >= m/beta means the transfer wins at
    every length (crossover 0); otherwise the crossover is
    (m/beta - a1)/a2."""
    m = QWEN3_4B_FP8.kappa
    out = {}
    for r in rows:
        if r.get("family") != "c6" or not r.get("bytes_per_s"):
            continue
        beta = float(r["bytes_per_s"])
        per_token = m / beta
        if a1 >= per_token:
            h_star = 0.0
        elif a2 > 0:
            h_star = (per_token - a1) / a2
        else:
            h_star = None
        out[r["cell"]] = dict(bytes_per_s=beta,
                              s_per_token=per_token,
                              crossover_tokens=h_star)
    return out


# ---- assembly ---------------------------------------------------------

def fit_all(rows):
    gpu = fit_gpu(rows)
    host = fit_host(rows)
    alpha = fit_alpha(rows)
    tread = t_read_s(rows, gpu)
    a1 = alpha["a1_s_per_token"]
    eps = (tread["t_read_s_per_token"] / a1) if a1 > 0 else None
    unstable = [r["cell"] for r in rows
                if r.get("cell") and not r.get("stable", True)]
    invalid = [r["cell"] for r in rows
               if r.get("cell") and r.get("valid") is False]
    return dict(
        gpu=gpu, host=host, alpha=alpha, t_read=tread, eps=eps,
        crossovers=crossovers(rows, a1, alpha["a2_s_per_token2"]),
        kappa_bytes_per_token=QWEN3_4B_FP8.kappa,
        unstable_cells=sorted(set(unstable)),
        invalid_cells=sorted(set(invalid)))


def constants_block(fit):
    """The ready-to-paste block for quail/plan/cost.py."""
    # crossovers stay in the JSON record but not in the constants
    # block: they are derived from bandwidth and the alpha fit, and
    # cost.offload_crossover_tokens computes them at call time
    bw = {k: v["bytes_per_s"] for k, v in fit["crossovers"].items()}
    lines = [
        "# --- calibration fits (results/engine/cost_model_fit.json) ---",
        f"ALPHA1_S_PER_TOKEN = {fit['alpha']['a1_s_per_token']!r}",
        f"ALPHA2_S_PER_TOKEN2 = {fit['alpha']['a2_s_per_token2']!r}",
        f"T_READ_S_PER_TOKEN = {fit['t_read']['t_read_s_per_token']!r}",
        f"EPSILON_READ_OVER_PRE = {fit['eps']!r}",
        f"STEP_B0_S = {fit['gpu']['b0_s']!r}",
        f"STEP_BETA_N_S = {fit['gpu']['beta_n_s']!r}",
        f"HOST_H0_S = {fit['host']['h0_s']!r}",
        f"HOST_HN_S = {fit['host']['h_n_s']!r}",
        f"HOST_HA_S = {fit['host']['h_a_s']!r}",
        f"HOST_HR_S = {fit['host']['h_r_s']!r}",
        f"TRANSPORT_BW_BPS = {bw!r}",
    ]
    return "\n".join(lines)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", nargs="+", required=True,
                    help="row files (globs allowed)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    rows = load_rows(args.rows)
    fit = fit_all(rows)
    with open(args.out, "w") as f:
        json.dump(fit, f, indent=2)
    print(f"saved {args.out}")
    a = fit["alpha"]
    g = fit["gpu"]
    print(f"a1 {a['a1_s_per_token'] * 1e6:.2f} us/token, "
          f"a2 {a['a2_s_per_token2']:.3e} s/token^2, "
          f"bend {a['bend_tokens'] and round(a['bend_tokens'])} tokens")
    print(f"quadratic MAPE {a['mape']:.3f} vs linear-only "
          f"{a['linear_only_mape']:.3f} on the alpha cells")
    print(f"t_read {fit['t_read']['t_read_s_per_token'] * 1e9:.1f} "
          f"ns/token (raw check ratio "
          f"{fit['t_read']['fitted_over_raw']}), eps {fit['eps']:.2e}")
    print(f"b0 {g['b0_s'] * 1e3:.2f} ms, beta_N "
          f"{g['beta_n_s'] * 1e6:.1f} us/request; held-out MAPE "
          f"{g.get('held_mape')}, ranking {g.get('held_ranking')}")
    print()
    print(constants_block(fit))
    return fit


if __name__ == "__main__":
    main()

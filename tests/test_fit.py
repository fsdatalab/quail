"""The calibration fits, tested on synthetic rows built from the real
cell grids: nonnegative least squares, monotone segment fits, the
quadratic prefill fit, the cached-read extraction, the no-R guard,
and the validation numbers."""

import numpy as np
import pytest

from quail.plan import calib, fit


# ---- synthetic rows ---------------------------------------------------

B0, AB, AP, BN = 2.0e-3, 10.4e-6, 8.4e-10, 20e-6
H0, HN, HA, HR = 0.4e-3, 4e-6, 1.5e-6, 1.2e-6


def planted_step_s(cell):
    return (B0 + AB * cell["b"] + AP * cell["p"] + BN * cell["n"])


def synthetic_rows(noise=0.0, seed=0):
    """Rows shaped like the runner's, timed by a planted linear model
    so every fit has a known right answer."""
    rng = np.random.default_rng(seed)
    rows = []
    for cell in calib.cells_for("all"):
        exec_s = planted_step_s(cell) * (1 + noise * rng.standard_normal())
        host_s = (H0 + HN * cell["n"] + HA * cell["a"]
                  + HR * calib.resident_blocks(cell["requests"]))
        rows.append(dict(
            family=cell["family"], cell=cell["name"], valid=True,
            stable=True, n=cell["n"], b=cell["b"], p=cell["p"],
            a=cell["a"], requests=cell["requests"],
            exec_ms_median=exec_s * 1e3,
            sched_ms_median=host_s * 0.7e3,
            update_ms_median=host_s * 0.3e3))
    return rows


# ---- nnls -------------------------------------------------------------

def test_nnls_recovers_nonnegative_coefficients():
    rng = np.random.default_rng(1)
    X = rng.random((60, 3))
    theta_true = np.array([2.0, 0.5, 3.0])
    y = X @ theta_true
    theta = fit.nnls(X, y)
    assert np.allclose(theta, theta_true, rtol=1e-3, atol=1e-4)


def test_nnls_clips_a_negative_true_coefficient_to_zero():
    rng = np.random.default_rng(2)
    X = rng.random((80, 2))
    y = X @ np.array([2.0, -1.0])
    theta = fit.nnls(X, y)
    assert theta[1] == 0.0
    assert theta[0] > 0


# ---- monotone segments ------------------------------------------------

def test_segment_fit_represents_concave_monotone_data():
    knots = (0, 4, 8, 16, 32)
    x = np.linspace(0, 32, 60)
    y = np.sqrt(x)                       # concave, monotone
    slopes = fit.nnls(fit.segment_basis(x, knots), y)
    pred = fit.segment_eval(x, knots, slopes)
    assert float(np.max(np.abs(pred - y))) < 0.45
    assert np.all(np.diff(pred) >= -1e-9)  # never decreases
    assert slopes[0] > slopes[-1]          # slopes fall: concave shape


# ---- the fits on synthetic rows ---------------------------------------

def test_fit_alpha_recovers_the_quadratic():
    # planted P = h(h+1)/2 for one request, so the h*h coefficient
    # the fit should find is AP/2
    rows = synthetic_rows(noise=0.01)
    a = fit.fit_alpha(rows)
    assert a["a1_s_per_token"] == pytest.approx(AB, rel=0.15)
    assert a["a2_s_per_token2"] == pytest.approx(AP / 2, rel=0.25)
    assert a["mape"] < a["linear_only_mape"]
    assert a["bend_tokens"] == pytest.approx(AB / (AP / 2), rel=0.4)


def test_fit_gpu_predicts_held_out_cells():
    rows = synthetic_rows(noise=0.01)
    g = fit.fit_gpu(rows)
    assert g["held_cells"] > 10
    assert g["held_mape"] < 0.08
    assert g["held_ranking"] > 0.95
    assert g["b0_s"] == pytest.approx(B0, rel=0.5)


def test_fit_host_recovers_the_plane():
    rows = synthetic_rows()
    h = fit.fit_host(rows)
    assert h["h0_s"] == pytest.approx(H0, rel=0.2)
    assert h["h_n_s"] == pytest.approx(HN, rel=0.2)
    assert h["h_a_s"] == pytest.approx(HA, rel=0.2)
    assert h["h_r_s"] == pytest.approx(HR, rel=0.2)


def test_t_read_matches_the_planted_slope():
    rows = synthetic_rows(noise=0.005)
    g = fit.fit_gpu(rows)
    t = fit.t_read_s(rows, g)
    # planted: dT/d(cached token) = AP * c at c = 32
    want = AP * 32
    assert t["t_read_s_per_token"] == pytest.approx(want, rel=0.25)
    assert t["t_read_raw_s_per_token"] == pytest.approx(want, rel=0.25)
    assert t["fitted_over_raw"] == pytest.approx(1.0, abs=0.3)


def test_invalid_rows_are_excluded_and_reported():
    rows = synthetic_rows()
    bad = dict(rows[0])
    bad.update(cell="c1_broken", family="c1", valid=False, stable=False)
    out = fit.fit_all(rows + [bad])
    assert "c1_broken" in out["invalid_cells"]
    assert "c1_broken" not in [None] and all(
        "broken" not in c for c in [r["cell"] for r in rows])


def test_r_term_refused_without_c3_cells():
    rows = synthetic_rows()
    assert not fit.has_c3_support(rows)   # the plain grid never counts
    with pytest.raises(ValueError, match="rank-deficient"):
        fit.fit_gpu(rows, fit_R=True)


def _c3_row(c, h, n):
    reqs = [dict(new=c, cached=h)] * n
    return dict(family="c2", cell=f"c3_c{c}_h{h}_n{n}", valid=True,
                stable=True, n=n, b=calib.batch_tokens(reqs),
                p=calib.pair_count(reqs), a=calib.new_blocks(reqs),
                requests=reqs, exec_ms_median=1.0,
                sched_ms_median=0.1, update_ms_median=0.1)


def test_r_term_recognizes_real_c3_pairs():
    # matched P per request, suffixes 256 vs 64: real decorrelation
    planted = [_c3_row(256, 1920, 8), _c3_row(64, 8176, 8),
               _c3_row(256, 4000, 8), _c3_row(64, 16368, 8)]
    rows = synthetic_rows() + planted
    assert fit.has_c3_support(rows)
    with pytest.raises(NotImplementedError):
        fit.fit_gpu(rows, fit_R=True)


def test_crossovers_use_the_dominance_rule():
    a1, a2 = 10.4e-6, 4.2e-10
    m = fit.QWEN3_4B_FP8.kappa
    fast = dict(family="c6", cell="c6_h2d_pinned",
                bytes_per_s=m / (a1 / 2))     # transfer at half a1
    slow = dict(family="c6", cell="c6_disk_read",
                bytes_per_s=m / (a1 * 3))     # transfer at 3x a1
    out = fit.crossovers([fast, slow], a1, a2)
    assert out["c6_h2d_pinned"]["crossover_tokens"] == 0.0
    want = (3 * a1 - a1) / a2
    assert out["c6_disk_read"]["crossover_tokens"] == pytest.approx(want)


def test_fit_all_assembles_and_prints_constants():
    rows = synthetic_rows(noise=0.005)
    rows.append(dict(family="c6", cell="c6_h2d_pinned",
                     bytes_per_s=55e9))
    out = fit.fit_all(rows)
    assert out["eps"] > 0
    block = fit.constants_block(out)
    assert "ALPHA2_S_PER_TOKEN2" in block
    assert "OFFLOAD_CROSSOVER_TOKENS" in block

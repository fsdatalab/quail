"""The end-to-end check on fabricated reports with known answers."""

from quail.configs import DEVICES, MODELS
from quail.plan import validate
from quail.plan.cost import (quest_read_tokens, quest_token_count,
                             stage_survivals)

M = MODELS["Qwen3-4B-FP8"]
D = DEVICES["H100-SXM-80GB"]


def _report(rate, c0, n_filters=5, sel=(0.9, 0.9, 0.9, 0.8, 0.8)):
    """Rewind walls planted at exactly the container price, with the
    answered count planted so the effective selectivities equal the
    designed ones."""
    corpus = 3_000_000
    n = 10_000
    mean_doc = corpus / n
    step_tokens = 25_305
    q = quest_token_count(n, 1, n_filters, 46, 33, list(sel))
    r = quest_read_tokens(n, 1, n_filters, mean_doc, 33, list(sel))
    computed = corpus + q
    wall = validate.price(M, D, computed, r, step_tokens, c0=c0,
                          s_per_token=1.0 / rate)
    answered = round(n * sum(stage_survivals(n_filters, list(sel))))
    cells = [dict(arm="rewind", rep=i, wall=wall,
                  reads=round(computed / corpus, 3), c0_s=c0,
                  answered=answered) for i in range(3)]
    cells += [dict(arm="stock", rep=i, wall=wall + 2.0,
                   reads=round((computed + 2.0 * rate) / corpus, 3),
                   answered=answered) for i in range(3)]
    return dict(corpus_tokens=corpus, n_docs=n, n_filters=n_filters,
                selectivity=list(sel), step_tokens=step_tokens,
                probe=dict(rate_tok_s=rate), cells=cells)


def test_container_prediction_is_exact_on_planted_walls():
    out = validate.check(_report(rate=90_000.0, c0=1.5))
    rewind = next(r for r in out["rows"] if r["arm"] == "rewind")
    assert abs(rewind["container_effective"]["error"]) < 0.005


def test_effective_matches_designed_when_planted_consistent():
    out = validate.check(_report(rate=97_000.0, c0=0.5))
    rewind = next(r for r in out["rows"] if r["arm"] == "rewind")
    gap = abs(rewind["designed"]["predicted_s"]
              - rewind["effective"]["predicted_s"])
    assert gap < 0.01 * rewind["designed"]["predicted_s"]


def test_stock_checks_fleet_constants_only():
    out = validate.check(_report(rate=97_000.0, c0=2.0))
    stock = next(r for r in out["rows"] if r["arm"] == "stock")
    assert "container_effective" not in stock
    assert "measured reads" in stock["reads_basis"]
    assert stock["designed"]["predicted_s"] > 0

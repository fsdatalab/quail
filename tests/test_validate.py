"""The end-to-end check on fabricated reports with known answers."""

from quail.configs import DEVICES, MODELS
from quail.plan import validate


def _report(rate, c0, n_filters=5, sel=(0.9, 0.9, 0.9, 0.8, 0.8)):
    """Rewind walls planted at exactly tokens/rate + c0, stock walls
    3 seconds over with reads planted to match."""
    model = MODELS["Qwen3-4B-FP8"]
    device = DEVICES["H100-SXM-80GB"]
    corpus = 3_000_000
    quest = validate.question_tokens(model, device, 10_000, n_filters,
                                     list(sel))
    tokens = corpus + quest
    wall = tokens / rate + c0
    cells = [dict(arm="rewind", rep=i, wall=wall,
                  reads=round(tokens / corpus, 3), c0_s=c0)
             for i in range(3)]
    cells += [dict(arm="stock", rep=i, wall=wall + 3.0,
                   reads=round((tokens + 3.0 * rate) / corpus, 3))
              for i in range(3)]
    return dict(corpus_tokens=corpus, n_docs=10_000,
                n_filters=n_filters, selectivity=list(sel),
                probe=dict(rate_tok_s=rate), cells=cells)


def test_container_prediction_is_exact_on_planted_walls():
    out = validate.check(_report(rate=90_000.0, c0=1.5))
    rewind = next(r for r in out["rows"] if r["arm"] == "rewind")
    assert abs(rewind["container"]["error"]) < 0.005


def test_stock_row_checks_fleet_constants_only():
    out = validate.check(_report(rate=97_000.0, c0=2.0))
    stock = next(r for r in out["rows"] if r["arm"] == "stock")
    assert "container" not in stock
    assert "measured reads" in stock["reads_basis"]
    assert stock["fleet"]["predicted_s"] > 0

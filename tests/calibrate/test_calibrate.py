"""CPU checks for the hardware calibration package."""

import pytest

from calibrate.attention import points, summary_name
from calibrate.gemm import mamf_runs, parse_log
from calibrate.targets import TARGETS, gemm_shapes, head_layout
from calibrate.timing import window_stats
from calibrate.to_spec import (
    attention_rates,
    spec_text,
    sustained_fp8_flops,
    sustained_hbm_bw,
    unsteady_points,
)
from quail.specs import QWEN3_4B_FP8

# lines in the format of mamf-finder's README example output
MAMF_LOG_TEXT = """\
     #  MxNxK                mean median    max   best     W   MHz
    18  3072x2816x20480     828.4  828.6  831.6  828.4   506  1980  <- best mean
MAMF (max achievable,  boost burst): 842 TFLOPS @ 3072x2816x20480 (MxNxK)  172W 1980MHz
MSMF (max sustainable, saturated):   722 TFLOPS @ 3072x2816x8192 (MxNxK)  681W 1612MHz
"""


def test_parse_log_reads_headlines_and_scout_rows():
    parsed = parse_log(MAMF_LOG_TEXT)
    assert parsed["mamf"] == {"tflops": 842, "shape": "3072x2816x20480",
                              "watts": 172, "mhz": 1980, "celsius": None}
    assert parsed["msmf"]["tflops"] == 722 and parsed["msmf"]["mhz"] == 1612
    assert parsed["scouts"] == [{
        "m": 3072, "n": 2816, "k": 20480, "mean_tflops": 828.4,
        "median_tflops": 828.6, "max_tflops": 831.6, "watts": 506,
        "max_mhz": 1980}]
    assert parse_log("") == {"mamf": None, "msmf": None, "scouts": []}


def test_mamf_runs_time_published_shapes_or_search(tmp_path):
    shapes = {"qkv": (6144, 2560)}
    runs = mamf_runs(TARGETS["h100-sxm"].published_fp8, shapes, tmp_path)
    assert list(runs) == ["published_fp8", "qkv"]
    assert (tmp_path / "published_fp8_shapes.txt").read_text() == \
        "1280x16896x16384\n1536x2816x16384\n"
    assert runs["qkv"][-4:] == ["--n", "6144", "--k", "2560"]
    assert "131072" in runs["qkv"]
    assert list(mamf_runs(None, {}, tmp_path)) == ["search"]


def test_model_shapes_match_qwen3_4b():
    assert gemm_shapes(QWEN3_4B_FP8) == {
        "qkv": (6144, 2560), "o": (2560, 4096),
        "gate_up": (19456, 2560), "down": (2560, 9728)}
    assert head_layout(QWEN3_4B_FP8) == (32, 8, 128)
    assert summary_name((32, 8, 128)) == "attention_32_8_128.json"


def test_sweep_has_copy_documents_and_cached_reads():
    sweep = points(smoke=False)
    kinds = [point["kind"] for point in sweep]
    assert kinds[0] == "copy"
    assert kinds.count("packed") == 9 and kinds.count("cached") == 56
    assert [point["kind"] for point in points(smoke=True)] == \
        ["copy", "packed", "cached", "cached"]


def test_window_is_steady_only_without_drop_or_spread():
    # FlashAttention 3 over 32k-token documents on an H100: no slowdown,
    # but the sub-windows differ by 11%
    drop, spread, steady = window_stats([696, 701, 656, 626, 686, 695, 666, 629])
    assert drop < 1.0
    assert spread == pytest.approx(10.93, abs=0.01)
    assert not steady
    assert window_stats([195.0] * 8) == (0.0, 0.0, True)


def _measured(tflops, steady=True):
    return {"tflops": tflops, "gbps": 0.0, "steady": steady,
            "spread_pct": 11.0 if not steady else 1.0, "drop_pct": 0.1}


def _attention_summary():
    rows = [{"kind": "copy", "copy": {"tflops": 0.0, "gbps": 3000.0,
                                      "steady": True}}]
    for length, fa3, fi in ((128, 100, 90), (512, 286, 301)):
        rows.append({"kind": "packed", "length": length,
                     "fa3": _measured(fa3), "flashinfer": _measured(fi)})
    for q_rows in (16, 64):
        for context in (1024, 8192):
            rows.append({"kind": "cached", "query_rows": q_rows,
                         "context": context, "pages": "private",
                         "fa3": _measured(q_rows * 10 + context / 1024,
                                          steady=q_rows == 16),
                         "flashinfer": {"error": "RuntimeError: no kernel"}})
            rows.append({"kind": "cached", "query_rows": q_rows,
                         "context": context, "pages": "shared",
                         "fa3": _measured(9999.0, steady=False)})
    return {"stamp": "20261012T235536Z", "model": "qwen3-4b-fp8",
            "layout": [32, 8, 128], "gpu": {"name": "NVIDIA H100 80GB HBM3"},
            "points": rows}


def test_attention_rates_take_the_faster_library_on_private_pages():
    rates = attention_rates(_attention_summary())
    assert rates.causal_tokens == (128, 512)
    assert rates.causal == (100e12, 301e12)
    assert rates.read_rows == (16, 64) and rates.read_contexts == (1024, 8192)
    assert rates.read == ((161e12, 168e12), (641e12, 648e12))
    assert sustained_hbm_bw(_attention_summary()) == 3000e9
    broken = _attention_summary()
    broken["points"][1]["fa3"] = {"error": "x"}
    broken["points"][1]["flashinfer"] = {"error": "x"}
    with pytest.raises(ValueError, match="no library measured"):
        attention_rates(broken)


def test_spec_fields_use_the_best_msmf_and_flag_unsteady_points():
    gemm = {"stamp": "20261012T222632Z", "runs": {
        "published_fp8": {"msmf": {"tflops": 1352}},
        "qkv": {"msmf": {"tflops": 1287}},
        "search": {"msmf": None}}}
    assert sustained_fp8_flops(gemm) == 1352e12
    text = spec_text(gemm, _attention_summary())
    assert "sustained_fp8_flops=1.352e+15" in text
    assert "attention_rates=(((32, 8, 128), QWEN3_4B_FP8_ATTENTION),)" in text
    unsteady = unsteady_points(_attention_summary())
    assert len(unsteady) == 2
    assert all("'query_rows': 64" in line for line in unsteady)

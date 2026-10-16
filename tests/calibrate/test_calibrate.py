"""CPU checks for the hardware calibration package."""

import json
from dataclasses import asdict

import pytest

from calibrate import gemm, timing
from calibrate.attention import AttentionSweep, parse_heads
from calibrate.gemm import ProjectionSweep, fetch_mamf_finder, mamf_runs, parse_log
from calibrate.targets import TARGETS, call_sizes, gemm_shapes, head_layouts
from calibrate.timing import Window, window_stats
from calibrate.to_spec import (
    attention_rates,
    attention_text,
    load,
    spec_text,
    sustained_fp8_flops,
    sustained_hbm_bw,
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
    published = TARGETS["h100-sxm"].published_fp8
    assert list(mamf_runs(published, None, tmp_path)) == ["published_fp8"]
    assert (tmp_path / "published_fp8_shapes.txt").read_text() == \
        "1280x16896x16384\n1536x2816x16384\n"
    assert list(mamf_runs(None, None, tmp_path)) == ["search"]
    sweep = ProjectionSweep("qwen3-4b-fp8", {"qkv": (6144, 2560)})
    runs = mamf_runs(published, sweep, tmp_path)
    assert list(runs) == ["published_fp8", "qkv"]
    assert runs["qkv"][-4:] == ["--n", "6144", "--k", "2560"]
    assert "131072" in runs["qkv"]


def test_gemm_run_writes_under_the_given_root(tmp_path, monkeypatch):
    # stands in for mamf-finder: writes the README example to --output_file
    fake = tmp_path / "fake_mamf.py"
    fake.write_text(
        "import sys\n"
        "out = sys.argv[sys.argv.index('--output_file') + 1]\n"
        f"open(out, 'w').write({MAMF_LOG_TEXT!r})\n")
    monkeypatch.setattr(gemm, "gpu_info", lambda: {"name": "test"})
    saves = []
    root = tmp_path / "results"
    summary = gemm.run(root, "h100-sxm", "20261013T000000Z",
                       TARGETS["h100-sxm"].published_fp8, None, fake,
                       on_save=lambda: saves.append(1))
    assert summary["runs"]["published_fp8"]["msmf"]["tflops"] == 722
    assert (root / "h100-sxm" / "gemm" / "20261013T000000Z"
            / "published_fp8.txt").exists()
    saved = root / "h100-sxm" / "gemm" / "20261013T000000Z" / "summary.json"
    assert json.loads(saved.read_text()) == summary
    assert not (root / "h100-sxm" / "gemm.json").exists()
    assert len(saves) == 1


def test_fetch_mamf_finder_rejects_a_wrong_file(tmp_path):
    (tmp_path / gemm.MAMF_NAME).write_text("print('not mamf-finder')\n")
    with pytest.raises(ValueError, match="sha256"):
        fetch_mamf_finder(tmp_path)


def test_model_shapes_match_qwen3_4b():
    assert gemm_shapes(QWEN3_4B_FP8) == {
        "qkv": (6144, 2560), "o": (2560, 4096),
        "gate_up": (19456, 2560), "down": (2560, 9728)}
    assert parse_heads("32,8,128") == (32, 8, 128)


def test_layouts_and_call_sizes_come_from_the_priced_models():
    # DiffusionGemma has sliding layers, so the cost model never uses its rates
    assert head_layouts() == [(16, 8, 128), (32, 8, 128), (64, 8, 128)]
    chunk, kv = call_sizes((32, 8, 128), "h100-sxm")
    assert chunk == 110376 and kv > 300_000
    with pytest.raises(ValueError, match="no measured model"):
        call_sizes((16, 8, 256), "h100-sxm")


def test_sweep_sizes_calls_like_quail():
    sweep = AttentionSweep(chunk_tokens=110376, kv_tokens=362240)
    kinds = [point["kind"] for point in sweep.points()]
    assert kinds[0] == "copy"
    assert kinds.count("packed") == 9 + 2 and kinds.count("cached") == 28 + 2
    # the chunk's rows bound short contexts, the KV arena bounds long ones
    assert sweep.sequences({"query_rows": 1024, "context": 1024, "size": 1.0}) == 107
    assert sweep.sequences({"query_rows": 16, "context": 32768, "size": 1.0}) == 11
    assert sweep.sequences({"query_rows": 64, "context": 8192, "size": 0.5}) == 22
    assert sweep.documents({"length": 128, "size": 1.0}) == 862
    assert AttentionSweep(41943, 106368).documents({"length": 32768, "size": 1.0}) == 1
    assert AttentionSweep.from_dict(json.loads(json.dumps(asdict(sweep)))) == sweep
    smoke = AttentionSweep.smoke(110376, 362240)
    assert [p["kind"] for p in smoke.points()] == ["copy", "packed", "cached",
                                                   "packed", "cached"]
    assert smoke.window.repeats == 1


def test_window_is_steady_only_without_drop_or_spread():
    # FlashAttention 3 over 32k-token documents on an H100: no slowdown,
    # but the sub-windows differ by 11%
    subs = [696, 701, 656, 626, 686, 695, 666, 629]
    drop, spread, steady = window_stats(subs, Window())
    assert drop < 1.0
    assert spread == pytest.approx(11.09, abs=0.01)
    assert not steady
    assert window_stats([195.0] * 8, Window()) == (0.0, 0.0, True)
    assert window_stats(subs, Window(max_spread_pct=12.0))[2]


def test_measure_retries_until_enough_windows_are_steady(monkeypatch):
    outcomes = iter([(100.0, False), (90.0, True), (95.0, True), (300.0, False),
                     (92.0, True)])

    def fake_window(*args):
        tflops, steady = next(outcomes)
        return {"tflops": tflops, "gbps": 0.0, "steady": steady}

    monkeypatch.setattr(timing, "_call_seconds", lambda torch, run: 0.001)
    monkeypatch.setattr(timing, "settle", lambda *args: {"settled": True})
    monkeypatch.setattr(timing, "time_window", fake_window)

    class FakeTorch:
        int32 = None

        @staticmethod
        def empty(*args, **kwargs):
            return None

    result = timing.measure(FakeTorch, lambda: None, 1.0, 0.0, False, None, Window())
    assert len(result["windows"]) == 5
    assert result["steady"] and result["steady_windows"] == 3
    assert result["tflops"] == 92.0
    lonely = iter([(100.0, False)] * 6)
    monkeypatch.setattr(timing, "time_window", lambda *a: dict(
        zip(("tflops", "steady"), next(lonely)), gbps=0.0))
    result = timing.measure(FakeTorch, lambda: None, 1.0, 0.0, False, None, Window())
    assert not result["steady"] and len(result["windows"]) == 6


def _measured(tflops, steady=True):
    return {"tflops": tflops, "gbps": 0.0, "steady": steady}


def _attention_summary(uuid, scale=1.0, check=False):
    rows = [{"kind": "copy", "copy": {"tflops": 0.0, "gbps": 3000.0 * scale,
                                      "steady": True}}]
    for length, fa3, fi in ((128, 100, 90), (512, 286, 301), (1024, 350, 340)):
        rows.append({"kind": "packed", "length": length, "size": 1.0,
                     "fa3": _measured(fa3 * scale),
                     "flashinfer": _measured(fi * scale)})
    rows.append({"kind": "packed", "length": 1024, "size": 0.5,
                 "fa3": _measured(300 * scale), "flashinfer": _measured(1.0)})
    for q_rows in (16, 64):
        for context in (1024, 8192):
            rows.append({"kind": "cached", "query_rows": q_rows,
                         "context": context, "size": 1.0,
                         "fa3": _measured((q_rows * 10 + context / 1024) * scale),
                         "flashinfer": {"error": "RuntimeError: no kernel"}})
    return {"stamp": f"20261014T000000Z-{uuid}", "heads": [32, 8, 128],
            "gpu": {"name": "NVIDIA H100 80GB HBM3", "uuid": uuid},
            "sweep": {"chunk_tokens": 110376, "kv_tokens": 362240,
                      "size_length": 1024, "size_read": [64, 8192]},
            "check": check, "points": rows}


def test_attention_rates_take_the_fastest_steady_gpu_and_library():
    slow, fast = _attention_summary("a"), _attention_summary("b", scale=1.1)
    rates = attention_rates([slow, fast])
    assert rates.causal_tokens == (128, 512, 1024)
    assert rates.causal == pytest.approx((110e12, 331.1e12, 385e12))
    assert rates.read_rows == (16, 64) and rates.read_contexts == (1024, 8192)
    assert rates.read[0][0] == pytest.approx(161 * 1.1e12)
    assert sustained_hbm_bw([slow, fast]) == pytest.approx(3300e9)
    # an unsteady window on the fast GPU leaves the slow GPU's steady rate
    fast["points"][1]["fa3"]["steady"] = False
    fast["points"][1]["flashinfer"]["steady"] = False
    assert attention_rates([slow, fast]).causal[0] == pytest.approx(100e12)
    slow["points"][1]["fa3"]["steady"] = False
    slow["points"][1]["flashinfer"]["steady"] = False
    with pytest.raises(ValueError, match="no steady measurement"):
        attention_rates([slow, fast])


def test_spec_text_combines_gpus_and_skips_smoke(tmp_path):
    for name, summary in (("a", _attention_summary("a")),
                          ("b", _attention_summary("b", scale=1.1)),
                          ("s", _attention_summary("s", scale=9.0, check=True))):
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.json").write_text(json.dumps(summary))
    gemm_runs = [{"stamp": "g0", "gpu": {"uuid": "a"}, "runs": {
        "published_fp8": {"mamf": {"tflops": 1543}},
        "search": {"mamf": None}}},
        {"stamp": "g1", "gpu": {"uuid": "b"}, "runs": {
            "published_fp8": {"mamf": {"tflops": 1629}}}}]
    for i, summary in enumerate(gemm_runs):
        (tmp_path / f"g{i}").mkdir()
        (tmp_path / f"g{i}" / "summary.json").write_text(json.dumps(summary))
    gemm_summaries, attention = load([tmp_path])
    assert len(gemm_summaries) == 2 and len(attention) == 2
    assert sustained_fp8_flops(gemm_summaries) == 1629e12
    assert "sustained_fp8_flops=1.629e+15" in spec_text(gemm_summaries)
    text = attention_text(attention)
    assert "over 2 GPUs" in text and "spread up to 9.1%" in text
    assert "# ATTENTION_32_8_128 = AttentionRates(" in text
    assert "('packed', 1024) at 0.5 of the call: 330 TFLOPS" in text

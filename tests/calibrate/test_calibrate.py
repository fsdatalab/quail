"""CPU checks for the hardware calibration package."""

import json

import pytest

from calibrate import gemm
from calibrate.gemm import fetch_mamf_finder, mamf_runs, parse_log
from calibrate.targets import TARGETS
from calibrate.to_spec import load, mamf_fp8_flops, spec_text

# lines in the format of mamf-finder's README example output
MAMF_LOG_TEXT = """\
     #  MxNxK                mean median    max   best     W   MHz
    18  3072x2816x20480     828.4  828.6  831.6  828.4   506  1980  <- best mean
MAMF (max achievable,  boost burst): 842 TFLOPS @ 3072x2816x20480 (MxNxK)  172W 1980MHz
MSMF (max sustainable, saturated):   722 TFLOPS @ 3072x2816x8192 (MxNxK)  681W 1612MHz
"""


def test_parse_log_reads_the_headlines():
    parsed = parse_log(MAMF_LOG_TEXT)
    assert parsed["mamf"] == {"tflops": 842, "shape": "3072x2816x20480",
                              "watts": 172, "mhz": 1980, "celsius": None}
    assert parsed["msmf"]["tflops"] == 722 and parsed["msmf"]["mhz"] == 1612
    assert parse_log("") == {"mamf": None, "msmf": None}


def test_mamf_runs_time_published_shapes_or_search(tmp_path):
    published = TARGETS["h100-sxm"].published_fp8
    assert list(mamf_runs(published, tmp_path)) == ["published_fp8"]
    assert (tmp_path / "published_fp8_shapes.txt").read_text() == \
        "1280x16896x16384\n1536x2816x16384\n"
    assert mamf_runs(None, tmp_path) == {"search": ["--search", "auto"]}


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
                       TARGETS["h100-sxm"].published_fp8, fake,
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


def test_spec_text_takes_the_best_mamf_over_gpus(tmp_path):
    gemm_runs = [{"stamp": "g0", "gpu": {"uuid": "a"}, "runs": {
        "published_fp8": {"mamf": {"tflops": 1543}},
        "search": {"mamf": None}}},
        {"stamp": "g1", "gpu": {"uuid": "b"}, "runs": {
            "published_fp8": {"mamf": {"tflops": 1629}}}}]
    for i, summary in enumerate(gemm_runs):
        (tmp_path / f"g{i}").mkdir()
        (tmp_path / f"g{i}" / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "other.json").write_text("{}")
    summaries = load([tmp_path])
    assert len(summaries) == 2
    assert mamf_fp8_flops(summaries) == 1629e12
    text = spec_text(summaries)
    assert "mamf_fp8_flops=1.629e+15" in text and "over 2 GPUs" in text

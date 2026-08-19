"""CPU tests for stock boot timing helpers (no GPU, no LLM)."""

import logging

from baselines.stock_boot import (warm_boot_dict, _StartupStamp,
                                  _WEIGHT_DONE, _KV_MARK)


def test_warm_boot_dict_zeros():
    b = warm_boot_dict()
    assert b["kind"] == "warm"
    assert b["boot_s"] == 0.0
    assert b["llm_init_s"] == 0.0
    assert b["weight_load_s"] is None
    assert b["kv_profile_s"] is None


def test_startup_stamp_weight_and_kv_markers():
    stamp = _StartupStamp(t0=0.0)
    # fabricate elapsed by patching perf is awkward; call emit with
    # a logger record and check the regex path sets fields when
    # messages match (elapsed will be wall since process start -
    # just assert they become non-None).
    log = logging.getLogger("test.vllm.stamp")
    log.addHandler(stamp)
    log.setLevel(logging.INFO)
    log.info("Loading model weights done")
    assert stamp.weight_done_s is not None
    log.info("GPU KV cache size: 100 tokens")
    assert stamp.kv_mark_s is not None
    log.removeHandler(stamp)


def test_marker_regexes():
    assert _WEIGHT_DONE.search("Finished loading model weights")
    assert _KV_MARK.search("Available KV cache memory: 12.3 GiB")

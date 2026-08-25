"""CPU tests for stock boot timing helpers (no GPU, no LLM)."""

import logging

from baselines.stock_boot import _StartupStamp, warm_boot_dict


def test_startup_stamp_weight_and_kv_markers():
    assert warm_boot_dict() == {
        "kind": "warm",
        "llm_init_s": 0.0,
        "weight_load_s": None,
        "kv_profile_s": None,
        "boot_s": 0.0,
    }
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

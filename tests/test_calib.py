"""Cell arithmetic for the calibration sweeps, tested without an
engine: grids, feasibility, pair counts, unique-content sequences, and
the step checks the runner applies to every measured round."""

import pytest

from quail.plan import calib


# ---- grids ------------------------------------------------------------

def test_family_sizes():
    assert len(calib.family_c1()) == 25
    assert len(calib.family_alpha()) == 11
    assert len(calib.family_c2()) == 54
    assert len(calib.family_c4()) == 6
    assert len(calib.family_c5()) == 4


def test_c1_caps_at_the_boot_budget():
    cells = calib.family_c1()
    assert max(c["b"] for c in cells) == 32_768
    assert all(c["b"] <= calib.BOOT["max_num_batched_tokens"]
               for c in cells)
    # the paper's N=512 survives only at c=64
    big = [c for c in cells if c["n"] == 512]
    assert len(big) == 1 and big[0]["requests"][0]["new"] == 64


def test_every_length_is_a_whole_number_of_blocks():
    for cell in calib.cells_for("all"):
        for r in cell["requests"]:
            assert r["new"] % calib.BLOCK == 0
            assert r["cached"] % calib.BLOCK == 0
        assert all(h % calib.BLOCK == 0 for h in cell["warm"])


def test_alpha_reaches_16k_in_one_step():
    cells = calib.family_alpha()
    assert cells[-1]["b"] == 16_384
    assert all(c["n"] == 1 and c["b"] <= 32_768 for c in cells)


def test_c5_cells_share_b_p_n():
    cells = calib.family_c5()
    assert len({(c["b"], c["p"], c["n"]) for c in cells}) == 1


def test_cells_for_rejects_unknown_family():
    with pytest.raises(ValueError):
        calib.cells_for("c1,nope")


def test_cells_for_all_matches_union():
    all_names = {c["name"] for c in calib.cells_for("all")}
    union = set()
    for fam in ("alpha", "c1", "c2", "c4", "c5"):
        union |= {c["name"] for c in calib.cells_for(fam)}
    assert all_names == union
    assert len(all_names) == 25 + 11 + 54 + 6 + 4


# ---- step statistics --------------------------------------------------

def test_pair_count_matches_brute_force():
    reqs = [dict(new=3, cached=32), dict(new=5, cached=0)]
    # request 1: each of 3 new tokens sees 32 cached, plus the causal
    # triangle over the new tokens themselves (1+2+3); request 2: only
    # its own triangle (1+2+3+4+5).
    assert calib.pair_count(reqs) == (3 * 32 + 6) + 15
    assert calib.batch_tokens(reqs) == 8
    assert calib.resident_tokens(reqs) == 40


def test_new_blocks_counts_only_the_fresh_span():
    # cached lengths are whole blocks, so a request allocates exactly
    # its fresh tokens' blocks
    assert calib.new_blocks([dict(new=32, cached=4096)]) == 2
    assert calib.new_blocks([dict(new=512, cached=0)]) == 32
    assert calib.new_blocks([dict(new=16, cached=16)]) == 1


def test_resident_blocks_counts_the_whole_context():
    # cached plus fresh, in whole blocks - the span the block tables cover
    assert calib.resident_blocks([dict(new=32, cached=4096)]) == 258
    assert calib.resident_blocks([dict(new=512, cached=0)]) == 32
    assert calib.resident_blocks(
        [dict(new=16, cached=16), dict(new=32, cached=0)]) == 4


def test_feasibility_drops_oversized_cells():
    ok = calib._cell("c2", "fits", [dict(new=32, cached=2048)] * 4,
                     warm=[2048] * 4)
    # 50 x 16,384 = 819,200 warm tokens, past the 0.75 cap on a
    # 946,800-token pool (710,100)
    fat = calib._cell("c2", "too_much_kv",
                      [dict(new=32, cached=16_384)] * 50,
                      warm=[16_384] * 50)
    kept, dropped = calib.feasible([ok, fat], pool_tokens=946_800)
    assert [c["name"] for c in kept] == ["fits"]
    assert dropped == ["too_much_kv"]


def test_all_planned_cells_fit_the_measured_pool():
    kept, dropped = calib.feasible(calib.cells_for("all"),
                                   pool_tokens=946_800)
    assert dropped == []


# ---- unique content ---------------------------------------------------

def test_nonce_ids_are_unique_and_deterministic():
    alphabet = [11, 22, 33, 44]
    seen = set()
    for counter in range(4096):
        ids = calib.make_nonce_ids(counter, alphabet)
        assert len(ids) == calib.NONCE_TOKENS
        assert set(ids) <= set(alphabet)
        seen.add(tuple(ids))
    assert len(seen) == 4096
    assert calib.make_nonce_ids(7, alphabet) == calib.make_nonce_ids(
        7, alphabet)


def test_nonce_rejects_bad_inputs():
    with pytest.raises(ValueError):
        calib.make_nonce_ids(0, [5])
    with pytest.raises(ValueError):
        calib.make_nonce_ids(-1, [1, 2])


def test_build_sequence_exact_length_and_wrap():
    pool = list(range(100, 130))       # 30 tokens
    nonce = calib.make_nonce_ids(3, [1, 2])
    ids, cursor = calib.build_sequence(pool, 25, 64, nonce)
    assert len(ids) == 64
    assert ids[:16] == nonce
    # 48 body tokens from cursor 25: 5 to the end, then wraps
    assert ids[16:21] == pool[25:]
    assert ids[21:51] == pool
    assert ids[51:64] == pool[:13]
    assert cursor == 13


def test_build_sequence_rejects_partial_blocks():
    with pytest.raises(ValueError):
        calib.build_sequence(list(range(50)), 0, 100,
                             calib.make_nonce_ids(0, [1, 2]))


# ---- the runner's checks ----------------------------------------------

def _cell():
    return calib._cell("c2", "c2_c32_h2048_n2",
                       [dict(new=32, cached=2048)] * 2,
                       warm=[2048, 2048])


def test_check_step_accepts_the_requested_step():
    cell = _cell()
    rec = dict(tokens=64, seqs=2, shapes=[[32, 2048], [32, 2048]])
    ok, why = calib.check_step(cell, [rec])
    assert ok, why


def test_check_step_rejects_splits_and_wrong_shapes():
    cell = _cell()
    half = dict(tokens=32, seqs=1)
    assert not calib.check_step(cell, [half, half])[0]
    assert not calib.check_step(cell, [dict(tokens=64, seqs=3)])[0]
    assert not calib.check_step(cell, [dict(tokens=48, seqs=2)])[0]
    bad = dict(tokens=64, seqs=2, shapes=[[32, 2048], [32, 1024]])
    assert not calib.check_step(cell, [bad])[0]


def test_check_step_ignores_empty_schedules():
    cell = _cell()
    rec = dict(tokens=64, seqs=2)
    ok, _ = calib.check_step(cell, [dict(tokens=0, seqs=0), rec])
    assert ok


def test_check_cached_wants_designed_counts():
    cell = _cell()
    assert calib.check_cached(cell, [2048, 2048])[0]
    assert not calib.check_cached(cell, [2048, 2032])[0]
    assert not calib.check_cached(cell, [2048])[0]

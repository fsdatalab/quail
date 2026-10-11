"""The state slot pool beside the KV pools: one page currency and slot lifetimes."""

import pytest

from quail.backends.quail.executor.arena import KVArena

torch = pytest.importorskip("torch")


def state_arena(pages=64, slots=5):
    # layer 0 runs linear attention and keeps no KV; layer 1 keeps
    # every token. A slot prices at the pages over the usable slots:
    # 64 / 4 = 16 pages with five slots.
    return KVArena(n_layers=2, n_pages=pages, page_tokens=16, n_kv=1, d_head=2,
                   dtype=torch.float32, device="cpu", layer_kv=[(0, 0), (1, 2)],
                   state_layers=(0,), n_state_slots=slots,
                   state_shape=(8, 8), conv_shape=(4, 3))


def test_slot_pages_and_the_free_page_currency():
    arena = state_arena()
    assert arena.has_state
    assert arena.slot_pages == 16
    assert arena.n_state_slots == 5
    # slot 0 is the zero state, so four slots are usable
    assert arena.admission_pages == 64
    assert arena.free_pages == 64
    state, conv = arena.state_pools(0)
    assert state.shape == (5, 8, 8) and state.dtype == torch.float32
    assert conv.shape == (5, 4, 3) and conv.dtype == torch.float32
    assert not state.any() and not conv.any()
    with pytest.raises(ValueError, match="state shape"):
        KVArena(n_layers=1, n_pages=4, page_tokens=16, n_kv=1, d_head=2,
                dtype=torch.float32, device="cpu", state_layers=(0,),
                n_state_slots=4)
    with pytest.raises(ValueError, match="two slots"):
        state_arena(slots=1)

    plain = KVArena(n_layers=1, n_pages=8, page_tokens=16, n_kv=1, d_head=2,
                    dtype=torch.float32, device="cpu")
    assert not plain.has_state
    assert plain.slot_pages == 0 and plain.admission_pages == 8
    assert plain.free_pages == 8
    assert plain.activate(("d", 0), 20, capacity_tokens=40, slots=0)
    assert plain.held_cost(("d", 0)) == 3


def test_activate_reserves_slots_and_prices_them_as_pages():
    arena = state_arena()
    key = ("d", 0)
    assert arena.activate(key, 20, capacity_tokens=40, base_tokens=20, slots=2)
    assert arena.state.owned_count(key) == 2
    # the key costs the larger of its 3 pages and its slots' 32
    assert arena.held_cost(key) == 32
    assert arena.free_pages == min(64 - 3, 2 * 16)
    assert arena.growth_cost(key, 40, slots=3) == 16
    assert arena.growth_cost(key, 40) == 0
    # a resident key grows its reservation
    assert arena.activate(key, 20, capacity_tokens=40, slots=3) is not None
    assert arena.state.owned_count(key) == 3
    assert arena.free_pages == 16
    # one slot left: a key asking for two is refused and takes no pages
    other = ("d", 1)
    assert arena.activate(other, 16, slots=2) is None
    assert not arena.is_resident(other)
    assert arena.free_pages == 16
    assert arena.activate(other, 16, slots=1)
    assert arena.free_pages == 0
    assert arena.free_key(other) == 16
    assert arena.free_key(key) == 48
    assert arena.free_pages == 64


def test_claims_rewind_shares_and_borrowing():
    arena = state_arena(slots=6)
    key = ("d", 0)
    assert arena.activate(key, 100, capacity_tokens=120, base_tokens=100, slots=3)
    share = arena.claim_state(key, 64, "share")
    base = arena.claim_state(key, 100, "base")
    kept = arena.claim_state(key, 105, "kept")
    assert len({share, base, kept}) == 3 and 0 not in (share, base, kept)
    assert arena.state_slot_at(key, 64) == share
    assert arena.state_slot_at(key, 100) == base
    assert arena.state_slot_at(key, 105) == kept
    # a child may borrow only a prefix whose state the parent saved
    assert arena.can_borrow(key, 64)
    assert not arena.can_borrow(key, 32)
    # a held key keeps its share slot through retention
    arena.hold(key, 1)
    arena.retain(key, 100)
    assert arena.state_slot_at(key, 105) is None
    assert arena.state_slot_at(key, 64) == share
    assert arena.state.owned_count(key) == 2
    # the retained size is the larger of 7 pages of KV and 2 slots
    # at 64 / 5 = 13 pages each
    assert arena.retained_pages == 26
    # the last borrower is admitted: the share slot goes
    arena.release(key)
    assert arena.state_slot_at(key, 64) is None
    assert not arena.can_borrow(key, 64)
    assert arena.state.owned_count(key) == 1
    assert arena.free_key(key) == 13


def test_eviction_frees_slots_when_the_pool_is_slot_bound():
    arena = state_arena(slots=3)
    first, second, third = ("d", 0), ("d", 1), ("d", 2)
    for key in (first, second):
        assert arena.activate(key, 16, slots=1)
        arena.claim_state(key, 16, "base")
        arena.retain(key, 16)
    assert arena.free_pages == 0
    assert arena.activate(third, 16, slots=1)
    assert arena.evicted_keys == 1
    assert not arena.is_resident(first)
    # the evicted key's slot, 64 / 2 = 32 pages, is what it freed
    assert arena.evicted_pages == 32
    assert arena.state.owned_count(third) == 1


def test_resize_rebuilds_the_state_pool():
    arena = state_arena()
    key = ("d", 0)
    assert arena.activate(key, 16, slots=1)
    arena.state_pools(0)[0][arena.claim_state(key, 16, "base")] = 1.0
    arena.resize(32, 0, 7, free_resident=True)
    assert arena.n_pages == 32 and arena.n_state_slots == 7
    assert arena.state_pools(0)[0].shape == (7, 8, 8)
    assert not arena.state_pools(0)[0].any()
    assert arena.free_pages == min(32, 6 * 6)
    # a resize that names no slot count keeps the pool
    arena.resize(16)
    assert arena.n_state_slots == 7

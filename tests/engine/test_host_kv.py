"""The host store keeps a freed document prefix and restores it into new pages."""

import pytest

from quail.backends.quail.executor.arena import KVArena
from quail.backends.quail.executor.host_kv import (
    PENDING,
    HostKV,
    _Ring,
    prefix_hash,
)

torch = pytest.importorskip("torch")

PAGE = 4


def make_arena(*, window=None, slots=0, pages=8):
    extra = {}
    if window is not None:
        extra = dict(layer_kv=[(1, 2), (1, 2)], sliding_layers=(1,),
                     sliding_window=window, num_sliding_pages=pages)
    if slots:
        # a recurrent layer holds no KV, as a hybrid model's do
        extra.update(num_state_slots=slots, gdn_state_shape=(1, 2, 2),
                     conv_state_shape=(1, 3), layer_kv=[(1, 2), (0, 0)])
    return KVArena(num_layers=2, num_pages=pages, page_tokens=PAGE,
                   num_kv_heads=1, d_head=2, dtype=torch.float32, device="cpu",
                   **extra)


def attach(arena, entries=2, tokens=6):
    """A store holding `entries` prefixes of `tokens` rows each."""
    row_bytes = sum(2 * heads * dim * 4 for heads, dim in arena.shapes)
    host = HostKV(arena, entries * tokens * row_bytes, torch, pinned=False)
    arena.host = host
    return host


def write_prefix(arena, key, tokens, seed):
    """Fill the key's first `tokens` rows of every layer with known values."""
    for layer, (heads, dim) in enumerate(arena.shapes):
        if not heads * dim:
            continue
        k, v = arena.layer_kv(layer)
        rows = (arena.capacity_rows_sliding(key) if layer in arena.sliding_layers
                else arena.capacity_rows(key))
        start = arena.sliding_start(key) if layer in arena.sliding_layers else 0
        for row in range(start, tokens):
            k[rows[row - start]] = seed + 100 * layer + row
            v[rows[row - start]] = -(seed + 100 * layer + row)


def read_prefix(arena, key, first, last):
    """The key's rows first..last of every layer, as (layer, side, row, values)."""
    out = {}
    for layer, (heads, dim) in enumerate(arena.shapes):
        if not heads * dim:
            continue
        sliding = layer in arena.sliding_layers
        rows = arena.capacity_rows_sliding(key) if sliding else arena.capacity_rows(key)
        start = arena.sliding_start(key) if sliding else 0
        for side, tensor in enumerate(arena.layer_kv(layer)):
            for row in range(max(first, start), last):
                out[(layer, side, row)] = tensor[rows[row - start]].tolist()
    return out


def test_the_ring_drops_the_oldest_entries_in_a_new_entry_way():
    ring = _Ring(10)
    assert ring.take(4) == (0, ())
    assert ring.take(4) == (4, ())
    # the run does not fit before the end: the tail rows go unused
    # and the ring wraps, dropping the entry at the start
    start, evicted = ring.take(3)
    assert start == 0 and evicted == ((0, 4),)
    start, evicted = ring.take(3)
    assert start == 3 and evicted == ((4, 4),)
    assert ring.take(11) == (None, ())
    assert [entry for entry in ring.entries] == [(0, 3), (3, 3)]


def test_a_freed_prefix_is_saved_and_read_back_into_new_pages():
    arena = make_arena()
    host = attach(arena)
    key, prefix = ("d", 0), [1, 2, 3, 4, 5, 6]
    digest = host.register(key, prefix, len(prefix))
    assert arena.reserve(key, 6, capacity_tokens=8)
    write_prefix(arena, key, 6, seed=1000)
    host.mark_written(key)
    before = read_prefix(arena, key, 0, 6)
    arena.free_key(key)
    assert host.holds(digest) and arena.free_pages == 8
    assert host.saved_keys == 1 and host.saved_tokens == 6
    for layer in range(2):
        for tensor in arena.layer_kv(layer):
            tensor.zero_()
    # the same document under the same head is a hit, wherever its pages are
    assert arena.reserve(("e", 3), 6, capacity_tokens=8)
    assert arena.reserve(key, 6, capacity_tokens=8)
    assert host.restore(key, digest)
    assert read_prefix(arena, key, 0, 6) == before
    assert host.hits == 1 and host.restored_tokens == 6
    # a restored key is written: freed again, it is saved again once
    # the entry is gone
    arena.free_key(key)
    assert host.saved_keys == 1


def test_a_prefix_nobody_computed_is_not_saved():
    arena = make_arena()
    host = attach(arena)
    key = ("d", 0)
    digest = host.register(key, [1, 2, 3], 3)
    assert arena.reserve(key, 3, capacity_tokens=4)
    arena.free_key(key)
    assert not host.holds(digest) and host.saved_keys == 0
    # a key renamed to another prefix holds nothing until written again
    assert arena.reserve(key, 3, capacity_tokens=4)
    host.mark_written(key)
    renamed = host.register(key, [7, 8, 9, 10], 4)
    arena.free_key(key)
    assert not host.holds(renamed) and not host.holds(digest)
    # a key rewound below its prefix holds only part of it
    assert arena.reserve(key, 4, capacity_tokens=8)
    host.mark_written(key)
    arena.retain(key, 3)
    arena.free_key(key)
    assert not host.holds(renamed)


def test_a_full_store_keeps_a_query_first_documents_and_drops_older_queries():
    arena = make_arena(pages=16)
    host = attach(arena, entries=2, tokens=6)
    digests = []

    def save(document, seed):
        key = ("d", document)
        digests.append(host.register(key, [seed] * 6, 6))
        assert arena.reserve(key, 6, capacity_tokens=8)
        write_prefix(arena, key, 6, seed=seed)
        host.mark_written(key)
        arena.free_key(key)

    for document in range(3):
        save(document, 10 * document)
    # the third save would drop the first, saved by this query: it is skipped
    assert [host.holds(digest) for digest in digests] == [True, True, False]
    assert host.skipped_keys == 1 and host.report()["host_kv_entries"] == 2
    key = ("d", 1)
    assert arena.reserve(key, 6, capacity_tokens=8)
    assert host.restore(key, digests[1])
    assert read_prefix(arena, key, 0, 6)[(1, 0, 5)] == [[115.0, 115.0]]
    arena.free_key(key)
    assert not host.restore(("d", 2), digests[2]) and host.misses == 1
    # the next query's saves drop the last query's entries, oldest first
    host.begin_query()
    save(3, 30)
    assert [host.holds(digest) for digest in digests] == [False, True, False, True]


def test_a_sliding_prefix_restores_its_window_rows_trimmed():
    arena = make_arena(window=4)
    host = attach(arena)
    key = ("d", 0)
    digest = host.register(key, list(range(10)), 10)
    assert arena.reserve(key, 10, capacity_tokens=12)
    write_prefix(arena, key, 10, seed=500)
    host.mark_written(key)
    arena.trim_window(key)
    assert arena.sliding_start(key) == 4
    before = read_prefix(arena, key, 0, 10)
    arena.free_key(key)
    assert host.holds(digest)
    for layer in range(2):
        for tensor in arena.layer_kv(layer):
            tensor.zero_()
    assert arena.reserve(key, 10, capacity_tokens=12)
    assert host.restore(key, digest)
    assert arena.sliding_start(key) == 4
    assert read_prefix(arena, key, 0, 10) == before


def test_a_hybrid_prefix_restores_its_checkpoint_with_its_rows():
    arena = make_arena(slots=2)
    host = attach(arena)
    key = ("d", 0)
    digest = host.register(key, [1, 2, 3, 4, 5, 6], 6)
    assert arena.reserve(key, 6, capacity_tokens=8)
    write_prefix(arena, key, 6, seed=7)
    slot = arena.checkpoint_slot(key)
    arena.gdn_state[slot].fill_(2.5)
    arena.conv_state[slot].fill_(-1.0)
    arena.record_checkpoint(key, 6)
    host.mark_written(key)
    before = read_prefix(arena, key, 0, 6)
    arena.free_key(key)
    assert host.holds(digest) and host.num_slots >= 1
    arena.gdn_state.zero_()
    arena.conv_state.zero_()
    assert arena.reserve(key, 6, capacity_tokens=8)
    assert host.restore(key, digest)
    assert arena.checkpoint_tokens(key) == 6
    slot = arena.accounting.slots[key]
    assert torch.all(arena.gdn_state[slot] == 2.5)
    assert torch.all(arena.conv_state[slot] == -1.0)
    assert read_prefix(arena, key, 0, 6) == before


def test_the_hash_names_the_tokens_and_their_count():
    assert prefix_hash([1, 2, 3]) == prefix_hash([1, 2, 3])
    assert prefix_hash([1, 2, 3]) != prefix_hash([1, 2, 4])
    assert prefix_hash([1, 2, 3]) != prefix_hash([1, 2, 3, 0])


class _Event:
    """A save's event: fired when the test says so."""

    def __init__(self):
        self.fired = False
        self.waited = False

    def query(self):
        return self.fired

    def synchronize(self):
        self.waited = True
        self.fired = True


def test_a_freed_key_keeps_its_pages_until_its_save_has_landed():
    arena = make_arena()
    events = []

    class Store:
        def before_free(self, key):
            events.append(_Event())
            return events[-1]

        def flush_pending(self, wait=False, until=None):
            return {}

    arena.host = Store()
    key = ("d", 0)
    assert arena.reserve(key, 6, capacity_tokens=8)
    assert arena.free_key(key) == 0
    assert arena.is_resident(key) and arena.free_pages == 6
    # a second free is a no-op; the pages go once the save has read them
    assert arena.free_key(key) == 0 and arena.reclaim() == 0
    events[0].fired = True
    assert arena.reclaim() == 2 and not arena.is_resident(key)
    assert arena.free_pages == 8
    # the key used again keeps its pages: the arena waits for the save
    assert arena.reserve(key, 6, capacity_tokens=8)
    arena.free_key(key)
    assert arena.reserve(key, 6, capacity_tokens=8)
    assert events[1].waited and not arena._draining
    assert arena.free_key(key) == 0
    assert arena.reserve(("d", 1), 6, capacity_tokens=8)
    arena.free_key(("d", 1))
    # the oldest save alone, or every save, is waited for
    assert arena.reclaim_oldest() == 2 and events[2].waited and not events[3].waited
    assert arena.reclaim(wait=True) == 2 and events[3].waited and not arena._draining


def test_a_save_with_no_staging_block_waits_in_the_queue_with_its_pages():
    arena = make_arena()
    events = {}
    queue = []
    blocks = [0]    # staging blocks free for the next flush

    class Store:
        def before_free(self, key):
            queue.append(key)
            return PENDING

        def flush_pending(self, wait=False, until=None):
            flushed = {}
            while queue and (wait or blocks[0]):
                key = queue.pop(0)
                blocks[0] = max(0, blocks[0] - 1)
                events[key] = _Event()
                flushed[key] = events[key]
                if key == until:
                    break
            return flushed

        def cancel_pending(self, key):
            queue.remove(key)

    arena.host = Store()
    keys = [("d", index) for index in range(3)]
    for key in keys:
        assert arena.reserve(key, 4, capacity_tokens=4)
        arena.free_key(key)
    assert arena.free_pages == 5 and queue == keys
    # a reclaim runs what fits and frees nothing until a save lands
    blocks[0] = 1
    assert arena.reclaim() == 0 and queue == keys[1:]
    events[keys[0]].fired = True
    assert arena.reclaim() == 1 and queue == keys[1:]
    # a key used again leaves the queue with its rows
    assert arena.reserve(keys[2], 4, capacity_tokens=4)
    assert queue == [keys[1]] and keys[2] not in arena._draining
    arena.free_key(keys[2])
    # the oldest save in flight is run patiently and waited for
    assert arena.reclaim_oldest() == 1 and events[keys[1]].waited
    assert arena.reclaim(wait=True) == 1 and not arena._draining
    assert arena.free_pages == 8

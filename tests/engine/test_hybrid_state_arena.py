"""Checkpoint allocation, retention, and memory accounting."""


from helpers.fakes import bare_arena

from quail.backends.quail.executor.arena import KVArena, PageArena
from quail.common.retention import RetentionPolicy


def checkpoint_arena(slots, pages=16):
    arena = bare_arena(KVArena.__new__(KVArena), pages)
    arena.accounting = PageArena(pages, 16, num_slots=slots)
    arena.accounting.configure_retention(
        RetentionPolicy(1.0, 0.0, {"a": (1, 0)}, checkpoint_pages=98.25), pages)
    return arena


def keep(arena, key, tokens):
    assert arena.accounting.alloc(key, tokens) is not None
    arena._rows[key] = arena._capacity_rows[key] = None
    arena.checkpoint_slot(key)
    arena.retain(key, tokens)


def test_a_full_pool_evicts_the_lowest_priority_document_with_its_slot():
    arena = checkpoint_arena(slots=2)
    keep(arena, ("a", 0), 32)
    keep(arena, ("a", 1), 64)
    assert arena.has_checkpoint(("a", 0)) and arena.has_checkpoint(("a", 1))
    arena.accounting.alloc(("a", 2), 32)
    arena._rows[("a", 2)] = arena._capacity_rows[("a", 2)] = None
    assert arena.checkpoint_slot(("a", 2)) is not None
    # per page the 32-token document is worth the same as the 64-token one;
    # the tie goes to the earlier document
    assert not arena.is_resident(("a", 0))
    assert arena.is_resident(("a", 1))
    assert arena.evicted_keys == 1
    assert arena.num_state_slots == 2


def test_row_class_warmup_skips_an_arena_with_too_few_checkpoint_slots():
    from types import SimpleNamespace

    from quail.backends.quail.executor.warmup import _warm_row_classes

    # mammoth-1 on an H100 has about 25 checkpoint slots; 17 readers
    # need 2 each and the filler 1
    for slots in (1, 34):
        activated = []
        arena = SimpleNamespace(
            num_state_slots=slots,
            activate=lambda key, *_, found=activated, **__: found.append(key))
        _warm_row_classes(None, arena, pipeline=None, attention_mode="unified")
        assert activated == []

"""Checkpoint allocation, retention, and memory accounting."""


from types import SimpleNamespace

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


def test_row_class_warmup_fits_an_arena_with_few_checkpoint_slots(monkeypatch):
    import pytest
    pytest.importorskip("torch")
    from quail.backends.quail.executor import chunk, stage_executor, warmup

    resident, slots = set(), set()

    def free_key(key):
        resident.discard(key)
        slots.difference_update({("checkpoint", key), ("extension", key)})

    arena = SimpleNamespace(
        num_state_slots=10,
        activate=lambda key, *_, **__: resident.add(key),
        is_resident=lambda key: key in resident,
        free_key=free_key)
    held = []

    def pack_chunk(torch, arena, groups, attention_mode):
        for group in groups:
            slots.add(("checkpoint", group["key"]))
            if group.get("write_suffix_tokens"):
                slots.add(("extension", group["key"]))
            if len(slots) > arena.num_state_slots:
                raise chunk.ArenaFullError("no free checkpoint slot")
        held.append(len(slots))

    monkeypatch.setattr(chunk, "pack_chunk", pack_chunk)
    monkeypatch.setattr(stage_executor, "forward_chunk", lambda *_: None)
    warmup._warm_row_classes(None, arena, None, "tree")
    assert max(held) == 9 and not resident and not slots

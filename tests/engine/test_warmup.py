"""Boot-time kernel warmup without a GPU."""

from types import SimpleNamespace

from quail.backends.quail.executor.warmup import _warm_row_classes


def test_row_class_warmup_skips_an_arena_with_too_few_checkpoint_slots():
    # mammoth-1 on an H100 has about 25 checkpoint slots; 17 readers
    # need 2 each and the filler 1
    for slots in (1, 34):
        activated = []
        arena = SimpleNamespace(
            num_state_slots=slots,
            activate=lambda key, *_, found=activated, **__: found.append(key))
        _warm_row_classes(None, arena, pipeline=None, attention_mode="unified")
        assert activated == []

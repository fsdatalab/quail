"""Boot-time kernel warmup without a GPU."""

from types import SimpleNamespace

from quail.backends.quail.executor.warmup import _warm_row_classes


def test_row_class_warmup_skips_an_arena_with_too_few_checkpoint_slots():
    activated = []
    # a 27B bf16 hybrid on an H100 has one checkpoint slot
    arena = SimpleNamespace(num_state_slots=1,
                            activate=lambda key, *_, **__: activated.append(key))
    _warm_row_classes(None, arena, pipeline=None, attention_mode="unified")
    assert activated == []

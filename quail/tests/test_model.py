"""CPU tests for the single-rank group stub load_model installs."""

from quail.executor.model import _SingleRank


class _Torch:
    class device:
        def __init__(self, name):
            self.name = name


def test_single_rank_collectives_are_identity():
    group = _SingleRank(_Torch)
    assert group.world_size == 1
    assert group.is_first_rank
    assert group.is_last_rank
    x = object()
    assert group.all_reduce(x) is x
    assert group.all_gather(x) is x
    assert group.broadcast(x) is x
    assert group.broadcast_object("ok") == "ok"

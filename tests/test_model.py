"""CPU tests for load_model's single-rank stub and the head move."""

from types import SimpleNamespace

from quail.executor.model import _SingleRank, move_untied_head_to_host


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


class _Tensor:
    """The few tensor methods move_untied_head_to_host touches."""

    def __init__(self, ptr, device_type="cuda", numel=6,
                 element_size=2):
        self.ptr = ptr
        self.device = SimpleNamespace(type=device_type)
        self._numel = numel
        self._element_size = element_size

    def data_ptr(self):
        return self.ptr

    def numel(self):
        return self._numel

    def element_size(self):
        return self._element_size

    def detach(self):
        return self

    def to(self, device):
        return _Tensor(self.ptr, device_type=device, numel=self._numel,
                       element_size=self._element_size)


def _fake_torch(calls):
    return SimpleNamespace(
        nn=SimpleNamespace(
            Parameter=lambda data, requires_grad=False: data),
        cuda=SimpleNamespace(
            empty_cache=lambda: calls.append("empty_cache")))


def _model(head_weight, embed_weight):
    return SimpleNamespace(
        lm_head=SimpleNamespace(weight=head_weight),
        model=SimpleNamespace(
            embed_tokens=SimpleNamespace(weight=embed_weight)))


def test_untied_head_moves_to_cpu():
    calls = []
    head = _Tensor(ptr=1, numel=151_936 * 8, element_size=2)
    model = _model(head, _Tensor(ptr=2))
    freed = move_untied_head_to_host(_fake_torch(calls), model)
    assert freed == 151_936 * 8 * 2
    assert model.lm_head.weight.device.type == "cpu"
    assert calls == ["empty_cache"]
    # a second call finds the weight on the CPU and does nothing
    assert move_untied_head_to_host(_fake_torch(calls), model) == 0
    assert calls == ["empty_cache"]


def test_tied_head_stays_on_gpu():
    calls = []
    shared = _Tensor(ptr=7)
    model = _model(shared, _Tensor(ptr=7))
    freed = move_untied_head_to_host(_fake_torch(calls), model)
    assert freed == 0
    assert model.lm_head.weight is shared
    assert calls == []

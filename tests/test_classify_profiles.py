"""CPU checks for the classification profile cell's window and summaries."""

from types import SimpleNamespace

from experiments.cells import classify_profiles as cell


def test_window_arms_after_offset_and_closes_after_seconds_at_boundaries():
    window = cell.Window(offset=10.0, seconds=5.0)
    assert window.boundary(100.0) is None       # no classification stage yet
    window.phase(100.0)
    assert window.boundary(105.0) is None       # 5 s in, before the offset
    assert window.boundary(112.0) == "start"    # first boundary past 10 s
    assert window.boundary(114.0) is None
    assert window.boundary(116.0) is None
    assert window.boundary(117.5) == "stop"     # first boundary past 5 s
    assert window.boundary(119.0) is None       # one window per query
    assert window.finish(130.0) is None
    assert window.summary(100.0) == {
        "offset_s": 10.0, "seconds": 5.0, "armed": True, "start_s": 12.0,
        "end_s": 17.5, "passes": 2, "cut": False}


def test_window_counts_only_the_first_phase_start():
    window = cell.Window(offset=10.0, seconds=5.0)
    window.phase(100.0)
    window.phase(108.0)                          # a second classification
    assert window.boundary(111.0) == "start"


def test_window_closed_by_finish_is_marked_cut():
    window = cell.Window(offset=0.0, seconds=5.0)
    window.phase(100.0)
    assert window.boundary(100.0) == "start"
    assert window.finish(102.0) == "stop"
    summary = window.summary(100.0)
    assert summary["cut"] is True
    assert summary["end_s"] == 2.0


def test_window_never_armed_reports_so():
    window = cell.Window(offset=10.0, seconds=5.0)
    window.phase(100.0)
    assert window.boundary(105.0) is None
    assert window.finish(106.0) is None
    summary = window.summary(100.0)
    assert summary["armed"] is False
    assert summary["start_s"] is None and summary["end_s"] is None


def test_gpu_activity_unions_device_events_and_lists_the_longest_gaps():
    from torch.autograd import DeviceType

    def event(start, end, name, device=DeviceType.CUDA):
        return SimpleNamespace(
            device_type=device, name=name,
            time_range=SimpleNamespace(start=start, end=end))

    events = [
        event(0, 1_000_000, "launch", DeviceType.CPU),   # host op, ignored
        event(0, 2_000_000, "gemm"),
        event(1_000_000, 3_000_000, "attention"),         # overlaps gemm
        event(4_000_000, 5_000_000, "norm"),              # after a 1 s gap
        event(5_500_000, 6_000_000, "copy"),              # after a 0.5 s gap
    ]
    activity = cell.gpu_activity(events, top=1)
    assert activity["span_s"] == 6.0
    assert activity["busy_s"] == 4.5
    assert activity["idle_s"] == 1.5
    assert activity["gaps"] == [
        {"ms": 1000.0, "at_ms": 3000.0, "after": "attention", "before": "norm"}]
    assert cell.gpu_activity([events[0]], top=1) == {}


def test_top_kernels_rank_by_device_time_with_shares():
    averages = [
        SimpleNamespace(key="gemm", count=4, self_device_time_total=3_000_000.0),
        SimpleNamespace(key="aten::copy_", count=9, self_device_time_total=0.0),
        SimpleNamespace(key="attention", count=2,
                        self_device_time_total=1_000_000.0),
    ]
    assert cell.top_kernels(averages, top=5) == [
        {"name": "gemm", "calls": 4, "device_s": 3.0, "share": 0.75},
        {"name": "attention", "calls": 2, "device_s": 1.0, "share": 0.25},
    ]
    assert cell.top_kernels(averages, top=1)[0]["name"] == "gemm"

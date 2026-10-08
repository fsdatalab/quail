"""CPU checks for the classification profile cell's windows and summaries."""

from types import SimpleNamespace

from experiments.cells import classify_profiles as cell


def test_windows_arm_after_offset_and_close_after_seconds_at_boundaries():
    windows = cell.Windows(offset=10.0, every=0.0, seconds=5.0)
    assert windows.boundary(100.0) is None       # model execution not started
    windows.phase(100.0)
    assert windows.boundary(105.0) is None       # 5 s in, before the offset
    assert windows.boundary(112.0) == "start"    # first boundary past 10 s
    assert windows.boundary(114.0) is None
    assert windows.boundary(116.0) is None
    assert windows.boundary(117.5) == "stop"     # first boundary past 5 s
    assert windows.boundary(300.0) is None       # every=0: one window
    assert windows.finish(300.0) is None
    phases = [{"stage": "filter", "start_s": 0.0, "end_s": 13.0},
              {"stage": "classify", "start_s": 13.0, "end_s": 40.0},
              {"stage": "later", "start_s": 20.0, "end_s": 30.0}]
    assert windows.summary(100.0, phases) == [
        {"index": 0, "start_s": 12.0, "end_s": 17.5, "passes": 2, "cut": False,
         "stages": ["filter", "classify"]}]


def test_windows_repeat_every_seconds_after_the_previous_start():
    windows = cell.Windows(offset=0.0, every=60.0, seconds=5.0)
    windows.phase(100.0)
    assert windows.boundary(100.0) == "start"
    assert windows.boundary(106.0) == "stop"
    assert windows.boundary(150.0) is None
    assert windows.boundary(170.0) == "start"    # 60 s after the first start
    assert windows.boundary(176.0) == "stop"
    # a late boundary shifts the grid instead of chaining windows
    assert windows.boundary(400.0) == "start"
    assert windows.boundary(406.0) == "stop"
    assert windows.boundary(420.0) is None
    assert [w["index"] for w in windows.windows] == [0, 1, 2]


def test_windows_counts_only_the_first_phase_start():
    windows = cell.Windows(offset=10.0, every=0.0, seconds=5.0)
    windows.phase(100.0)
    windows.phase(108.0)                          # a later node
    assert windows.boundary(111.0) == "start"


def test_window_closed_by_finish_is_marked_cut():
    windows = cell.Windows(offset=0.0, every=0.0, seconds=5.0)
    windows.phase(100.0)
    assert windows.boundary(100.0) == "start"
    assert windows.finish(102.0) == "stop"
    [summary] = windows.summary(100.0, [{"stage": "open", "start_s": 0.0}])
    assert summary["cut"] is True
    assert summary["end_s"] == 2.0
    assert summary["stages"] == ["open"]          # a stage still running


def test_windows_never_armed_report_nothing():
    windows = cell.Windows(offset=10.0, every=0.0, seconds=5.0)
    windows.phase(100.0)
    assert windows.boundary(105.0) is None
    assert windows.finish(106.0) is None
    assert windows.summary(100.0, []) == []


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
        event(3_000_000, 4_000_000, "Command Buffer Full"),  # queue marker
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


def test_top_kernels_rank_device_entries_by_device_time_with_shares():
    from torch.autograd import DeviceType

    def average(key, count, device_us, device=DeviceType.CUDA):
        return SimpleNamespace(key=key, count=count, device_type=device,
                               self_device_time_total=device_us)

    averages = [
        average("gemm", 4, 3_000_000.0),
        # the host operator that launched gemm: its device time is gemm's
        average("aten::mm", 4, 3_000_000.0, DeviceType.CPU),
        average("aten::copy_", 9, 0.0, DeviceType.CPU),
        average("Command Buffer Full", 7, 2_000_000.0),
        average("attention", 2, 1_000_000.0),
    ]
    assert cell.top_kernels(averages, top=5) == [
        {"name": "gemm", "calls": 4, "device_s": 3.0, "share": 0.75},
        {"name": "attention", "calls": 2, "device_s": 1.0, "share": 0.25},
    ]
    assert cell.top_kernels(averages, top=1)[0]["name"] == "gemm"


def test_timed_records_stage_names_and_starts_the_phase():
    windows = cell.Windows(offset=0.0, every=0.0, seconds=5.0)
    capture = SimpleNamespace(windows=windows)
    phases = []
    run = cell._timed(capture, phases, 0.0, lambda *a, label=None: f"stages: {label}",
                      lambda *a, label=None: len(a), False)
    assert windows.phase_started is None
    assert run(1, 2, label="join (2 stages)") == 2
    assert phases[0]["stage"] == "stages: join (2 stages)"
    assert phases[0]["end_s"] >= phases[0]["start_s"]
    node = cell._timed(capture, phases, 0.0, lambda self, node, inputs: node.name,
                       lambda self, node, inputs: inputs, True)
    assert node(None, SimpleNamespace(name="quail.ai_filter"), {"x": 1}) == {"x": 1}
    assert windows.phase_started is not None
    assert phases[1]["stage"] == "quail.ai_filter"

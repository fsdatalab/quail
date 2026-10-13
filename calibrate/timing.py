"""Sustained-rate timing of one GPU call, as mamf-finder times MSMF.

A call is timed with GPU events after the L2 cache is flushed, over a
window split into sub-windows, after the clock has settled at full
load. Window holds the lengths and the steadiness limits.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    """How long to time a call, and when its rate counts as steady.

    The defaults are mamf-finder's MSMF settings: 2 s in 8 sub-windows,
    steady when the second half is at most 1% slower than the first
    (msmf_max_drop) and the sub-windows differ by at most 10% of their
    median (msmf_max_spread).

    Attributes:
        seconds: Timed seconds.
        warm_seconds: Untimed seconds of the same call before the timing.
        sub_windows: Equal parts of the timed calls.
        max_drop_pct: Largest slowdown of the second half against the first.
        max_spread_pct: Largest gap between sub-windows, against their median.
        l2_flush_bytes: Size of the buffer zeroed before each call; at
            least the GPU's L2 size.
    """

    seconds: float = 2.0
    warm_seconds: float = 1.0
    sub_windows: int = 8
    max_drop_pct: float = 1.0
    max_spread_pct: float = 10.0
    l2_flush_bytes: int = 256 * 2**20


class Telemetry:
    """NVML power, SM clock, and temperature of GPU 0."""

    def __init__(self):
        import pynvml
        pynvml.nvmlInit()
        self.nvml = pynvml
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def power(self) -> float:
        """Return instantaneous power in watts, or the 1 s average where unsupported."""
        # NVML field 186 is instantaneous power; older drivers lack it
        value = self.nvml.nvmlDeviceGetFieldValues(self.handle, [186])[0]
        if value.nvmlReturn == 0:
            return value.value.uiVal / 1000.0
        return self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0

    def clock(self) -> float:
        """Return the SM clock in MHz."""
        return float(self.nvml.nvmlDeviceGetClockInfo(
            self.handle, self.nvml.NVML_CLOCK_SM))

    def temp(self) -> float:
        """Return the die temperature in degrees C."""
        return float(self.nvml.nvmlDeviceGetTemperature(
            self.handle, self.nvml.NVML_TEMPERATURE_GPU))

    def info(self) -> dict:
        """Return the name, power limit, top SM clock, and driver version."""
        nvml, h = self.nvml, self.handle
        return {"name": nvml.nvmlDeviceGetName(h),
                "power_limit_w": nvml.nvmlDeviceGetPowerManagementLimit(h) / 1000,
                "max_sm_mhz": nvml.nvmlDeviceGetMaxClockInfo(h, nvml.NVML_CLOCK_SM),
                "driver": nvml.nvmlSystemGetDriverVersion()}


def settle_clock(torch, telem: Telemetry, seconds: float = 20.0) -> float:
    """Run a bf16 matmul at full load until the SM clock stops dropping.

    Args:
        torch: The torch module.
        telem: The GPU's telemetry.
        seconds: The longest to wait.

    Returns:
        The last SM clock read, in MHz.
    """
    a = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
    c = torch.empty_like(a)
    end, prev, stable = time.time() + seconds, None, 0
    while time.time() < end and stable < 3:
        for _ in range(100):
            torch.mm(a, a, out=c)
        torch.cuda.synchronize()
        clk = telem.clock()
        stable = stable + 1 if prev is not None and clk >= prev - 5 else 0
        prev = clk
    return prev


def window_stats(subs: list[float], window: Window) -> tuple[float, float, bool]:
    """Return a window's drop and spread in percent, and whether it was steady.

    Args:
        subs: The rate of each sub-window, in order.
        window: The window's steadiness limits.
    """
    half = len(subs) // 2
    first, second = sum(subs[:half]) / half, sum(subs[half:]) / half
    drop = (1 - second / first) * 100
    spread = (max(subs) - min(subs)) / sorted(subs)[half] * 100
    return (drop, spread,
            drop <= window.max_drop_pct and spread <= window.max_spread_pct)


def measure_window(torch, op, flops: float, nbytes: float, flush: bool,
                   telem: Telemetry, window: Window) -> dict:
    """Time op back to back over a sustained window.

    Args:
        torch: The torch module.
        op: The call to time.
        flops: FLOPs one call does; 0 rates the window in bytes.
        nbytes: Bytes one call must move.
        flush: Zero an L2-sized buffer before each call, outside the timing.
        telem: The telemetry sampled every 20 ms.
        window: The window's lengths and steadiness limits.

    Returns:
        TFLOPS and GB/s over the window, the rate of each sub-window,
        the drop and spread in percent, whether it was steady, and the
        mean power, clock, and temperature.
    """
    l2 = torch.empty(window.l2_flush_bytes // 4, dtype=torch.int32, device="cuda")

    def run(count, starts=None, ends=None):
        for i in range(count):
            if flush:
                l2.zero_()
            if starts is not None:
                starts[i].record()
            op()
            if ends is not None:
                ends[i].record()

    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    run(2)
    s.record()
    run(5)
    e.record()
    torch.cuda.synchronize()
    per_call = max(s.elapsed_time(e) / 5 / 1000, 1e-6)
    run(max(1, int(window.warm_seconds / per_call)))
    n = max(window.sub_windows, int(window.seconds / per_call))
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    torch.cuda.synchronize()
    samples, stop = [], threading.Event()

    def sample():
        while not stop.is_set():
            samples.append((telem.power(), telem.clock(), telem.temp()))
            stop.wait(0.02)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        run(n, starts, ends)
        torch.cuda.synchronize()
    finally:
        stop.set()
        thread.join()
    times = [a.elapsed_time(b) / 1000 for a, b in zip(starts, ends)]
    size = n // window.sub_windows
    work, scale = (flops, 1e12) if flops else (nbytes, 1e9)
    subs = [work * size / sum(times[i * size:(i + 1) * size]) / scale
            for i in range(window.sub_windows)]
    drop, spread, steady = window_stats(subs, window)
    total = sum(times)

    def mean(i):
        return sum(x[i] for x in samples) / len(samples) if samples else None

    return {"tflops": flops * n / total / 1e12, "gbps": nbytes * n / total / 1e9,
            "ms_per_call": total / n * 1000, "calls": n, "subs": subs,
            "drop_pct": drop, "spread_pct": spread, "steady": steady,
            "watts": mean(0), "mhz": mean(1), "celsius": mean(2)}

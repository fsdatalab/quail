"""Sustained-rate timing of one GPU call, after mamf-finder's MSMF.

measure() runs a call until the SM clock stops falling, then times it
over a window split into sub-windows, and repeats until enough windows
are steady. Each window is timed with GPU events, with the L2 cache
flushed before each call, outside the timing. Window holds the lengths,
the limits, and the repeat counts.
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass

# NVML clock event reason bits (nvmlClocksEventReason*)
CLOCK_REASONS = {0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x20: "sw_thermal",
                 0x40: "hw_thermal", 0x80: "hw_power_brake"}


@dataclass(frozen=True)
class Window:
    """How to settle, time, and accept the rate of one call.

    The window defaults are mamf-finder's MSMF settings: 2 s in 8
    sub-windows, steady when the second half is at most 1% slower than
    the first (msmf_max_drop) and (max - min) / median of the
    sub-windows is at most 10% (msmf_max_spread). The settle defaults
    are its settle_clock: the clock is settled when 3 readings in a row
    are each at most 5 MHz below the reading before, within 20 s.

    Attributes:
        seconds: Timed seconds per window.
        sub_windows: Equal parts of each window's calls.
        max_drop_pct: Largest slowdown of the second half against the first.
        max_spread_pct: Largest (max - min) / median of the sub-windows.
        settle_min_s: Fewest seconds the call runs before each window.
        settle_max_s: Most seconds the call runs before each window.
        settle_tick_s: Seconds of calls between clock readings.
        settle_mhz: Largest fall between readings that counts as settled.
        settle_ticks: Settled readings in a row needed.
        repeats: Steady windows wanted per call.
        max_tries: Most windows timed per call.
        l2_flush_bytes: Size of the buffer zeroed before each call; at
            least the GPU's L2 size.
    """

    seconds: float = 2.0
    sub_windows: int = 8
    max_drop_pct: float = 1.0
    max_spread_pct: float = 10.0
    settle_min_s: float = 1.0
    settle_max_s: float = 20.0
    settle_tick_s: float = 0.25
    settle_mhz: float = 5.0
    settle_ticks: int = 3
    repeats: int = 3
    max_tries: int = 6
    l2_flush_bytes: int = 256 * 2**20


class Telemetry:
    """NVML power, SM clock, temperature, and clock event reasons of GPU 0."""

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

    def reasons(self) -> int:
        """Return the bitmask of reasons the clock is below its maximum."""
        nvml = self.nvml
        read = (getattr(nvml, "nvmlDeviceGetCurrentClocksEventReasons", None)
                or nvml.nvmlDeviceGetCurrentClocksThrottleReasons)
        return int(read(self.handle))

    def info(self) -> dict:
        """Return the name, UUID, power limit, top SM clock, and driver version."""
        nvml, h = self.nvml, self.handle
        return {"name": nvml.nvmlDeviceGetName(h),
                "uuid": nvml.nvmlDeviceGetUUID(h),
                "power_limit_w": nvml.nvmlDeviceGetPowerManagementLimit(h) / 1000,
                "max_sm_mhz": nvml.nvmlDeviceGetMaxClockInfo(h, nvml.NVML_CLOCK_SM),
                "driver": nvml.nvmlSystemGetDriverVersion()}


def reason_names(mask: int) -> list[str]:
    """Return the names of the clock event reasons set in mask."""
    return [name for bit, name in CLOCK_REASONS.items() if mask & bit]


def window_stats(subs: list[float], window: Window) -> tuple[float, float, bool]:
    """Return a window's drop and spread in percent, and whether it was steady.

    Args:
        subs: The rate of each sub-window, in order.
        window: The window's steadiness limits.
    """
    half = len(subs) // 2
    first, second = sum(subs[:half]) / half, sum(subs[half:]) / half
    drop = (1 - second / first) * 100
    spread = (max(subs) - min(subs)) / statistics.median(subs) * 100
    return (drop, spread,
            drop <= window.max_drop_pct and spread <= window.max_spread_pct)


def _call_seconds(torch, run) -> float:
    """Return the GPU seconds of one call, from 5 calls after 2 untimed."""
    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    run(2)
    s.record()
    run(5)
    e.record()
    torch.cuda.synchronize()
    return max(s.elapsed_time(e) / 5 / 1000, 1e-6)


def settle(torch, run, per_call: float, telem: Telemetry, window: Window) -> dict:
    """Run the call until the SM clock stops falling.

    Args:
        torch: The torch module.
        run: Runs the call a given number of times, flushing L2 first.
        per_call: Seconds of one call.
        telem: The GPU's telemetry.
        window: The settle limits.

    Returns:
        The seconds run, the last clock reading in MHz, and whether the
        clock settled before settle_max_s.
    """
    batch = max(1, int(window.settle_tick_s / per_call))
    start, prev, stable = time.time(), None, 0
    while True:
        run(batch)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        clk = telem.clock()
        held = prev is not None and clk >= prev - window.settle_mhz
        stable = stable + 1 if held else 0
        prev = clk
        settled = stable >= window.settle_ticks
        if elapsed >= (window.settle_min_s if settled else window.settle_max_s):
            return {"seconds": elapsed, "mhz": clk, "settled": settled}


def time_window(torch, run, per_call: float, flops: float, nbytes: float,
                telem: Telemetry, window: Window) -> dict:
    """Time the call back to back for one window.

    Returns:
        TFLOPS and GB/s over the window, the rate of each sub-window,
        the drop and spread in percent, whether it was steady, the mean
        power, clock, and temperature, and the clock event reasons seen.
    """
    n = max(window.sub_windows, int(window.seconds / per_call))
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    torch.cuda.synchronize()
    samples, stop = [], threading.Event()

    def sample():
        while not stop.is_set():
            samples.append((telem.power(), telem.clock(), telem.temp(),
                            telem.reasons()))
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

    mask = 0
    for x in samples:
        mask |= x[3]
    return {"tflops": flops * n / total / 1e12, "gbps": nbytes * n / total / 1e9,
            "ms_per_call": total / n * 1000, "calls": n, "subs": subs,
            "drop_pct": drop, "spread_pct": spread, "steady": steady,
            "watts": mean(0), "mhz": mean(1), "celsius": mean(2),
            "clock_reasons": reason_names(mask)}


def measure(torch, op, flops: float, nbytes: float, flush: bool,
            telem: Telemetry, window: Window) -> dict:
    """Return the sustained rate of a call, from repeated steady windows.

    Before each window the call runs until the SM clock stops falling.
    Windows are timed until window.repeats are steady or window.max_tries
    have run.

    Args:
        torch: The torch module.
        op: The call to time.
        flops: FLOPs one call does; 0 rates the windows in bytes.
        nbytes: Bytes one call must move.
        flush: Zero an L2-sized buffer before each call, outside the timing.
        telem: The telemetry sampled every 20 ms.
        window: The settle and window settings.

    Returns:
        The median TFLOPS and GB/s of the steady windows (of every window
        when none was steady), whether window.repeats windows were steady,
        and each window with the settle before it.
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

    per_call = _call_seconds(torch, run)
    windows = []
    while len(windows) < window.max_tries:
        settled = settle(torch, run, per_call, telem, window)
        windows.append({**time_window(torch, run, per_call, flops, nbytes,
                                      telem, window), "settle": settled})
        if sum(w["steady"] for w in windows) >= window.repeats:
            break
    kept = [w for w in windows if w["steady"]] or windows
    return {"tflops": statistics.median(w["tflops"] for w in kept),
            "gbps": statistics.median(w["gbps"] for w in kept),
            "steady": sum(w["steady"] for w in windows) >= window.repeats,
            "steady_windows": sum(w["steady"] for w in windows),
            "ms_per_call": per_call * 1000, "windows": windows}

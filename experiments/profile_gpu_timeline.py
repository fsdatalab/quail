"""Extract GPU active intervals from a saved PyTorch trace."""

import gzip
import struct

from experiments.sglang_profile_analysis import merge_intervals


def read_gpu_intervals(path, window_unix_ns):
    """Return merged GPU intervals in microseconds after the join starts."""
    import ijson

    with gzip.open(path, "rb") as stream:
        base_ns = next(ijson.items(stream, "baseTimeNanoseconds"))
    left, right = ((value - base_ns) / 1000 for value in window_unix_ns)
    intervals = []
    with gzip.open(path, "rb") as stream:
        for event in ijson.items(stream, "traceEvents.item", use_float=True):
            if event.get("ph") != "X" or event.get("cat") not in (
                "kernel", "gpu_memcpy", "gpu_memset",
            ):
                continue
            start = max(left, event["ts"])
            end = min(right, event["ts"] + event["dur"])
            if end > start:
                intervals.append((start - left, end - left))
    return merge_intervals(intervals)


def encode_intervals(intervals):
    """Encode interval endpoints as gzip-compressed little-endian doubles."""
    return gzip.compress(b"".join(struct.pack("<dd", *interval) for interval in intervals), mtime=0)

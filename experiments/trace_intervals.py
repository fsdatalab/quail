"""Merge elapsed-time intervals read from PyTorch traces."""

from __future__ import annotations


def merge_intervals(intervals):
    """Return the union of elapsed-time intervals."""
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged

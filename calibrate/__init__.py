"""Measure GEMM MAMF for Quail's device specifications."""

from datetime import datetime, timezone


def utc_stamp() -> str:
    """Return the current UTC time as a run directory name."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

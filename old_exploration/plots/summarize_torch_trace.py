"""Stream a vLLM torch-profiler trace into a compact JSON summary."""

import argparse
import gzip
import json
from pathlib import Path

import ijson


GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def kernel_class(name: str) -> str:
    name = name.lower()
    if "gemm" in name or "cutlass" in name or "nvjet" in name:
        return "gemm"
    if "quant" in name or "scale" in name or "cast" in name:
        return "quantize"
    if "norm" in name or "rms" in name:
        return "norm"
    if ("attn" in name or "attention" in name or "flash" in name
            or "fmha" in name):
        return "attention"
    if ("silu" in name or "gelu" in name or "add" in name
            or "mul" in name or "residual" in name):
        return "elementwise"
    return "other"


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def summarize(path: Path) -> dict:
    intervals = []
    classes = {
        "gemm": 0.0,
        "quantize": 0.0,
        "norm": 0.0,
        "elementwise": 0.0,
        "attention": 0.0,
        "other": 0.0,
    }
    step_durations_ms = []
    n_kernels = 0
    total_kernel_us = 0.0

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as trace:
        for event in ijson.items(trace, "traceEvents.item"):
            duration = float(event.get("dur", 0))
            if duration <= 0:
                continue
            category = event.get("cat")
            name = event.get("name", "")
            if category in GPU_CATEGORIES:
                start = float(event["ts"])
                intervals.append((start, start + duration))
                total_kernel_us += duration
                n_kernels += 1
                classes[kernel_class(name)] += duration
            elif category == "user_annotation" and name.startswith(
                    "execute_context"):
                step_durations_ms.append(duration / 1000)

    intervals.sort()
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    span_us = merged[-1][1] - merged[0][0]
    busy_us = sum(end - start for start, end in merged)
    step_durations_ms.sort()

    result = {
        "n_kernels": n_kernels,
        "total_kernel_us": round(total_kernel_us, 3),
        "window_s": round(span_us / 1e6, 3),
        "gpu_busy_frac": round(busy_us / span_us, 4),
        "n_steps": len(step_durations_ms),
        "step_mean_ms": round(
            sum(step_durations_ms) / len(step_durations_ms), 3),
        "step_median_ms": round(percentile(step_durations_ms, 0.5), 3),
        "step_p10_ms": round(percentile(step_durations_ms, 0.1), 3),
        "step_p90_ms": round(percentile(step_durations_ms, 0.9), 3),
    }
    for name, duration in classes.items():
        result[f"{name}_us"] = round(duration, 3)
        result[f"{name}_frac"] = round(duration / total_kernel_us, 4)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = summarize(args.trace)
    if args.batch_size is not None:
        result = {"B": args.batch_size, **result}
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()

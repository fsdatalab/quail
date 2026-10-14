"""Turn calibrate's saved summaries into a device's DeviceSpec fields.

Pass the GEMM and attention summaries, or directories holding them, from
any number of GPUs. For example, after a Modal run:

    modal volume get quail-results calibrate/h100-sxm .
    uv run python -m calibrate.to_spec h100-sxm

Each rate is the highest steady rate over the GPUs measured, so the
speed-of-light estimate stays a lower bound on every GPU of the type:

- sustained_fp8_flops is the best MSMF of any mamf-finder run.
- sustained_hbm_bw is the best copy bandwidth.
- each attention point is the faster library's rate.

A point without a steady measurement on any GPU is an error; smoke
summaries are skipped.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from quail.specs import AttentionRates

LIBRARIES = ("fa3", "flashinfer")


def load(paths) -> tuple[list[dict], list[dict]]:
    """Return the GEMM and attention summaries under paths, smoke runs left out.

    Attention summaries from before the sweep was recorded, and smoke
    summaries, which check against torch, are left out.

    Args:
        paths: Summary files, or directories searched for summary.json.
    """
    gemm, attention = [], []
    for path in map(Path, paths):
        files = sorted(path.rglob("summary.json")) if path.is_dir() else [path]
        for file in files:
            summary = json.loads(file.read_text())
            if "runs" in summary:
                gemm.append(summary)
            elif ("points" in summary and "sweep" in summary
                  and not summary.get("check") and not summary.get("smoke")):
                attention.append(summary)
    return gemm, attention


def gpu_id(summary: dict) -> str:
    """Return the GPU's UUID, or the run's stamp when none was recorded."""
    return summary.get("gpu", {}).get("uuid") or summary["stamp"]


def best_msmf(summary: dict) -> float:
    """Return the best MSMF of one GEMM summary's runs, in TFLOPS."""
    return max((run["msmf"]["tflops"] for run in summary["runs"].values()
                if run.get("msmf")), default=0.0)


def sustained_fp8_flops(gemm: list[dict]) -> float:
    """Return the best MSMF over every GEMM summary, in FLOP/s."""
    return max(best_msmf(s) for s in gemm) * 1e12


def _key(row: dict) -> tuple:
    if row["kind"] == "packed":
        return ("packed", row["length"])
    return ("cached", row["query_rows"], row["context"])


def best_steady(row: dict) -> float | None:
    """Return the faster library's steady TFLOPS at one point, or None."""
    rates = [row[lib]["tflops"] for lib in LIBRARIES
             if isinstance(row.get(lib), dict) and row[lib].get("steady")]
    return max(rates) if rates else None


def point_rates(summaries: list[dict]) -> dict[tuple, dict[str, float]]:
    """Return each full-size point's steady rate on each GPU.

    Returns:
        Point key -> GPU id -> TFLOPS, with no entry for a GPU whose
        measurements of the point were not steady.
    """
    out = defaultdict(dict)
    for summary in summaries:
        for row in summary["points"]:
            if row["kind"] == "copy" or row.get("size", 1.0) != 1.0:
                continue
            out[_key(row)]
            rate = best_steady(row)
            if rate is not None:
                gpu = gpu_id(summary)
                out[_key(row)][gpu] = max(rate, out[_key(row)].get(gpu, 0.0))
    return dict(out)


def attention_rates(summaries: list[dict]) -> AttentionRates:
    """Return the highest steady rate over the GPUs at each point of one layout.

    Raises:
        ValueError: If a point has no steady measurement on any GPU.
    """
    rates = point_rates(summaries)
    missing = sorted(key for key, by_gpu in rates.items() if not by_gpu)
    if missing:
        raise ValueError(f"no steady measurement at {missing}")
    best = {key: max(by_gpu.values()) * 1e12 for key, by_gpu in rates.items()}
    causal = sorted((k[1], v) for k, v in best.items() if k[0] == "packed")
    read = {k[1:]: v for k, v in best.items() if k[0] == "cached"}
    rows = sorted({r for r, _ in read})
    contexts = sorted({c for _, c in read})
    return AttentionRates(
        causal_tokens=tuple(n for n, _ in causal),
        causal=tuple(v for _, v in causal),
        read_rows=tuple(rows),
        read_contexts=tuple(contexts),
        read=tuple(tuple(read[r, c] for c in contexts) for r in rows))


def sustained_hbm_bw(attention: list[dict]) -> float:
    """Return the best steady copy bandwidth over every summary, in bytes/s.

    Raises:
        ValueError: If no copy was steady.
    """
    rates = [row["copy"]["gbps"] for s in attention for row in s["points"]
             if row["kind"] == "copy" and row["copy"].get("steady")]
    if not rates:
        raise ValueError("no steady copy measurement")
    return max(rates) * 1e9


def by_layout(attention: list[dict]) -> dict[tuple, list[dict]]:
    """Return the attention summaries grouped by head layout."""
    out = defaultdict(list)
    for summary in attention:
        out[tuple(summary["heads"])].append(summary)
    return dict(sorted(out.items()))


def gpu_spread(summaries: list[dict]) -> float:
    """Return the largest (max - min) / max over GPUs of any point, in percent."""
    worst = 0.0
    for by_gpu in point_rates(summaries).values():
        if len(by_gpu) > 1:
            high, low = max(by_gpu.values()), min(by_gpu.values())
            worst = max(worst, (high - low) / high * 100)
    return worst


def size_lines(summaries: list[dict]) -> list[str]:
    """Return the size points' best steady rate per call-size fraction."""
    rates = defaultdict(list)
    for summary in summaries:
        for row in summary["points"]:
            if row["kind"] == "copy" or (rate := best_steady(row)) is None:
                continue
            sizes = summary["sweep"]
            if row["kind"] == "packed" and row["length"] == sizes["size_length"]:
                rates["packed", row["length"], row.get("size", 1.0)].append(rate)
            elif (row["kind"] == "cached"
                  and [row["query_rows"], row["context"]] == list(sizes["size_read"])):
                rates["cached", row["query_rows"], row["context"],
                      row.get("size", 1.0)].append(rate)
    return [f"{key[:-1]} at {key[-1]:g} of the call: {max(v):.0f} TFLOPS"
            for key, v in sorted(rates.items())]


def _tflops(values) -> str:
    return ", ".join(f"{value / 1e12:.1f}e12" for value in values)


def table_text(heads: tuple, summaries: list[dict]) -> str:
    """Return one layout's AttentionRates as Python source."""
    rates = attention_rates(summaries)
    gpus = sorted({gpu_id(s) for s in summaries})
    sweep = summaries[0]["sweep"]
    name = "ATTENTION_" + "_".join(map(str, heads))
    read_rows = ",\n          ".join(f"({_tflops(line)})" for line in rates.read)
    notes = "".join(f"# {line}\n" for line in size_lines(summaries))
    return (
        f"# Faster of FlashAttention 3 and FlashInfer at each point, highest\n"
        f"# steady rate over {len(gpus)} GPUs (spread up to "
        f"{gpu_spread(summaries):.1f}%); calls of\n"
        f"# {sweep['chunk_tokens']} tokens, cached reads capped at "
        f"{sweep['kv_tokens']} KV tokens: calibrate runs\n"
        f"# {', '.join(s['stamp'] for s in summaries)}\n"
        f"{notes}"
        f"{name} = AttentionRates(\n"
        f"    causal_tokens={rates.causal_tokens},\n"
        f"    causal=({_tflops(rates.causal)}),\n"
        f"    read_rows={rates.read_rows},\n"
        f"    read_contexts={rates.read_contexts},\n"
        f"    read=({read_rows}),\n"
        f")\n")


def spec_text(gemm: list[dict], attention: list[dict]) -> str:
    """Return the DeviceSpec fields and one attention table per layout."""
    layouts = by_layout(attention)
    tables = "\n".join(table_text(heads, s) for heads, s in layouts.items())
    entries = ", ".join(
        f"(({', '.join(map(str, h))}), ATTENTION_{'_'.join(map(str, h))})"
        for h in layouts)
    msmf = ", ".join(f"{best_msmf(s):.0f}" for s in gemm)
    return (
        f"{tables}\n"
        f"    sustained_fp8_flops={sustained_fp8_flops(gemm):.4g},"
        f"  # best MSMF over {len({gpu_id(s) for s in gemm})} GPUs: {msmf}\n"
        f"    sustained_hbm_bw={sustained_hbm_bw(attention):.4g},"
        f"  # best 4 GB copy, held 2 s\n"
        f"    attention_rates=({entries},),\n")


def main(argv=None) -> None:
    """Print the DeviceSpec fields for the summaries under the given paths."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path,
                        help="summary.json files or directories holding them")
    args = parser.parse_args(argv)
    gemm, attention = load(args.paths)
    print(spec_text(gemm, attention))


if __name__ == "__main__":
    main()

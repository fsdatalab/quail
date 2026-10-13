"""Turn calibrate's saved summaries into a device's DeviceSpec fields.

Print the fields to paste into quail/specs/<device>.py from the two
summaries. For example, after a run on Modal, pull them from the
results volume first:

    modal volume get quail-results calibrate/h100-sxm/gemm.json .
    modal volume get quail-results calibrate/h100-sxm/attention_32_8_128.json .
    uv run python -m calibrate.to_spec gemm.json attention_32_8_128.json

After a local run, pass the files under the results directory instead.

sustained_fp8_flops is the best MSMF of any run: the GPU's best shape,
so the speed of light estimate stays a bound for every model's shapes.
sustained_hbm_bw is the copy's bandwidth. The attention rates take the
faster library at each point, on private pages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quail.specs import AttentionRates


def sustained_fp8_flops(gemm: dict) -> float:
    """Return the best MSMF of any mamf-finder run, in FLOP/s."""
    return max(run["msmf"]["tflops"] for run in gemm["runs"].values()
               if run.get("msmf")) * 1e12


def _best(row: dict) -> dict | None:
    """Return the faster library's measurement of one point, or None."""
    found = [row[lib] for lib in ("fa3", "flashinfer")
             if isinstance(row.get(lib), dict) and "tflops" in row[lib]]
    return max(found, key=lambda m: m["tflops"]) if found else None


def sustained_hbm_bw(attention: dict) -> float:
    """Return the copy's sustained bandwidth, in bytes/s."""
    copy = next(row["copy"] for row in attention["points"]
                if row["kind"] == "copy")
    return copy["gbps"] * 1e9


def attention_rates(attention: dict) -> AttentionRates:
    """Return the faster library's rate at each measured point.

    Args:
        attention: A full attention sweep's summary.

    Raises:
        ValueError: If a point has no measurement from either library.
    """
    causal = {}
    read = {}
    for row in attention["points"]:
        if row["kind"] == "copy" or row.get("pages") == "shared":
            continue
        best = _best(row)
        if best is None:
            raise ValueError(f"no library measured {json.dumps(row)[:200]}")
        if row["kind"] == "packed":
            causal[row["length"]] = best["tflops"] * 1e12
        else:
            read[row["query_rows"], row["context"]] = best["tflops"] * 1e12
    rows = sorted({r for r, _ in read})
    contexts = sorted({c for _, c in read})
    return AttentionRates(
        causal_tokens=tuple(sorted(causal)),
        causal=tuple(causal[n] for n in sorted(causal)),
        read_rows=tuple(rows),
        read_contexts=tuple(contexts),
        read=tuple(tuple(read[r, c] for c in contexts) for r in rows))


def unsteady_points(attention: dict) -> list[str]:
    """Return the points the spec uses whose faster measurement was not steady."""
    out = []
    for row in attention["points"]:
        if row.get("pages") == "shared":
            continue
        best = row.get("copy") if row["kind"] == "copy" else _best(row)
        if best and not best.get("steady", True):
            shape = {k: row[k] for k in ("kind", "length", "query_rows",
                                         "context", "pages") if k in row}
            out.append(f"{shape}: spread {best['spread_pct']:.1f}%, "
                       f"drop {best['drop_pct']:.2f}%")
    return out


def _tflops(values) -> str:
    return ", ".join(f"{value / 1e12:.1f}e12" for value in values)


def spec_text(gemm: dict, attention: dict) -> str:
    """Return the DeviceSpec fields and attention table as Python source."""
    rates = attention_rates(attention)
    n_q, n_kv, d_head = attention["heads"]
    table = f"ATTENTION_{n_q}_{n_kv}_{d_head}"
    read_rows = ",\n          ".join(f"({_tflops(line)})" for line in rates.read)
    return (
        f"# Faster of FlashAttention 3 and FlashInfer at each point, each held\n"
        f"# 2 s; cached reads on private 16-token pages: calibrate run\n"
        f"# {attention['stamp']} on {attention['gpu']['name']}\n"
        f"{table} = AttentionRates(\n"
        f"    causal_tokens={rates.causal_tokens},\n"
        f"    causal=({_tflops(rates.causal)}),\n"
        f"    read_rows={rates.read_rows},\n"
        f"    read_contexts={rates.read_contexts},\n"
        f"    read=({read_rows}),\n"
        f")\n\n"
        f"    sustained_fp8_flops={sustained_fp8_flops(gemm):.4g},"
        f"  # best MSMF, calibrate run {gemm['stamp']}\n"
        f"    sustained_hbm_bw={sustained_hbm_bw(attention):.4g},"
        f"  # 4 GB copy held 2 s\n"
        f"    attention_rates=((({n_q}, {n_kv}, {d_head}), {table}),),\n")


def main(argv=None) -> None:
    """Print the DeviceSpec fields for two saved summaries."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("gemm", type=Path, help="calibrate/<device>/gemm.json")
    parser.add_argument("attention", type=Path,
                        help="calibrate/<device>/attention_<q>_<kv>_<d>.json")
    args = parser.parse_args(argv)
    gemm = json.loads(args.gemm.read_text())
    attention = json.loads(args.attention.read_text())
    print(spec_text(gemm, attention))
    for line in unsteady_points(attention):
        print(f"# not steady: {line}")


if __name__ == "__main__":
    main()

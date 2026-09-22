"""Replace the launch blog's QUAIL-B metrics table from saved results."""

from __future__ import annotations

import argparse
from pathlib import Path

from quailb_results import PIPELINE_QUERIES, QUERY_ORDER, load_results

HERE = Path(__file__).resolve().parent
START = "::: {.metrics-table .benchmark-metrics}\n"
END = ":::\n"


def _measured_row(query: str, label: str, row: dict) -> str:
    regret = (
        "not measured"
        if row["regret_tokens"] is None
        else f"{row['regret_tokens']}"
    )
    return (
        f"| {query} | {label} | {row['input_tokens_per_second']:.2f} "
        f"| {regret} | {row['cost_usd']:.4f} |"
    )


def render_table(workdir: Path) -> str:
    """Return the current QUAIL-B metrics table."""
    rows, sol = load_results(workdir)
    lines = [
        "| query | method | tok_per_sec | kv_regret | cost_usd |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for query in QUERY_ORDER:
        lines.append(_measured_row(query, "Quail", rows["quail"][query]))
        lines.append(
            _measured_row(query, "Stock vLLM", rows["stock_vllm"][query])
        )
        if query in PIPELINE_QUERIES:
            lines.append(
                _measured_row(
                    query,
                    "Pipelined vLLM",
                    rows["pipelined_vllm"][query],
                )
            )
        estimate = sol[query]
        throughput = estimate["input_tokens"] / estimate["runtime_s"]
        cost = estimate["runtime_s"] / 3600 * 3.9492
        lines.append(
            f"| {query} | SoL | {throughput:.2f} | 0 | {cost:.4f} |"
        )
    return "\n".join(lines)


def main(workdir: Path, blogpost: Path) -> None:
    """Write the generated table into the first metrics-table block."""
    text = blogpost.read_text()
    start = text.index(START) + len(START)
    end = text.index(END, start)
    replacement = render_table(workdir) + "\n"
    blogpost.write_text(text[:start] + replacement + text[end:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    parser.add_argument(
        "--blogpost",
        type=Path,
        default=HERE / "blogpost.md",
    )
    arguments = parser.parse_args()
    main(arguments.workdir, arguments.blogpost)

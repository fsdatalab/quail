"""Quail on 1×L40S and 8×L40S for three SF 0.1 queries.

One model copy per GPU. Documents are split across workers. Join
partners are copied to every worker. There is no tensor-parallel
weight sharding. Quail only: stock vLLM, pipelined vLLM, and SGLang
are off.

    uv run python experiments/cells/l40s_scale.py
    mkdir -p results/benchmark
    log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-l40s-scale.log"
    uv run python experiments/cells/l40s_scale.py --submit \
      2>&1 | tee "$log"

Each ``--submit`` job is one ``bench.quailb_parallel`` detach on
``quail-milestone1``. Results stay on quail-results. After a run:

    modal volume get quail-results /benchmarks/quailb/<run_id>

``cost_usd`` in ``quail/<family>/run.json`` is ``$/query``: query
hours times the GPU count times the L40S hourly price, startup
excluded. Speedup is the 1-GPU ``runtime_s`` divided by the 8-GPU
``runtime_s``. Scaling efficiency is that speedup divided by 8.
The Quail answer tables in the two run directories must match.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from quail.specs import L40S

SF = 0.1
MODEL = "qwen3-4b-fp8"
DEVICE = "l40s"
GPU_COUNTS = (1, 8)

# sf 0.1: reviews are 5,000 (50,000 * 0.1), aspects are the 12 fixed
# strings, contracts are 45 (round(449 * 0.1)).
QUERIES = (
    ("IMDB-1", "large filter, 5,000 reviews"),
    ("IMDB-2", "join, 5,000 reviews x 12 aspects"),
    ("CUAD-5", "small filter, 45 contracts"),
)

PREDICTION_TEXT = (
    "One Qwen3 4B copy per L40S, no tensor parallel. "
    "IMDB-1 splits 5,000 reviews, so 8 GPUs are at most 8x the "
    "1-GPU query time. IMDB-2 copies 12 aspects to every worker, "
    "so the join scales less than the filter. CUAD-5 has 45 "
    "contracts, so 8 GPUs have little work each. Answers match "
    "the 1-GPU run."
)


@dataclass(frozen=True)
class Job:
    """One Modal spawn: one query, one GPU count, Quail only."""

    query: str
    note: str
    gpus: int

    @property
    def modal_gpu(self) -> str:
        if self.gpus == 1:
            return L40S.modal_gpu
        return f"{L40S.modal_gpu}:{self.gpus}"


def jobs(*, gpu_counts=GPU_COUNTS, queries=QUERIES):
    """Yield one spawn per (query, GPU count).

    Counts run in the given order, and the default is 1 then 8, so
    every 1-GPU job finishes before an 8-GPU request starts waiting.
    """
    for gpus in gpu_counts:
        for query, note in queries:
            yield Job(query=query, note=note, gpus=gpus)


def modal_argv(job: Job) -> list[str]:
    """The ``modal run --detach`` argv for one job."""
    return [
        sys.executable, "-m", "modal", "run", "--detach",
        "-m", "bench.quailb_parallel",
        "--model", MODEL,
        "--sf", str(SF),
        "--device", DEVICE,
        "--gpus", str(job.gpus),
        "--query", job.query,
        "--no-include-baselines",
        "--no-include-sglang",
        "--timeout-s", "86400",
    ]


def print_suite(selected: list[Job]) -> None:
    """Print the three queries and the spawns that will run them."""
    print(f"PREDICTION: {PREDICTION_TEXT}")
    counts = []
    for job in selected:
        if job.gpus not in counts:
            counts.append(job.gpus)
    print(f"sf={SF} model={MODEL} device={DEVICE} gpus={counts}")
    print("one model copy per GPU; documents split; join partners copied")
    print()
    print("after each pair of run.json files:")
    print("  speedup = runtime_s(1 GPU) / runtime_s(8 GPU)")
    print("  scaling efficiency = speedup / 8")
    print(
        "  $/query = cost_usd = runtime hours x GPUs x "
        f"{L40S.usd_per_hour} (startup excluded)"
    )
    print("  Quail answer tables must be identical")
    print()
    print(f"{len(selected)} spawn(s):")
    for job in selected:
        print(f"  {job.modal_gpu} {job.query}  {job.note}")
        print(f"    {' '.join(modal_argv(job))}")


def submit(selected: list[Job], log_dir: Path) -> None:
    """Detach one ``quailb_parallel`` run per job and print the log path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"PREDICTION: {PREDICTION_TEXT}", flush=True)
    for job in selected:
        log = log_dir / f"{stamp}-l40s-g{job.gpus}-{job.query}.log"
        argv = modal_argv(job)
        print(f"tee={log}", flush=True)
        print(" ".join(argv), flush=True)
        with log.open("w") as handle:
            handle.write(f"PREDICTION: {PREDICTION_TEXT}\n")
            handle.write(" ".join(argv) + "\n")
            handle.flush()
            subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT,
                           check=True)
        print(log.read_text(), end="", flush=True)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submit", action="store_true",
        help="detach each job on Modal; default only prints the matrix")
    parser.add_argument(
        "--gpus", type=int, action="append", dest="gpu_counts",
        choices=GPU_COUNTS,
        help="restrict to these GPU counts; default 1 and 8")
    parser.add_argument(
        "--query", action="append", dest="queries",
        choices=[query for query, _note in QUERIES],
        help="restrict to these query ids; default all three")
    parser.add_argument(
        "--log-dir", type=Path, default=Path("results/benchmark"),
        help="directory for teed spawn logs")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Print the matrix, or detach the selected jobs."""
    args = _parse_args(argv)
    wanted = set(args.queries or [query for query, _note in QUERIES])
    selected = list(jobs(
        gpu_counts=tuple(args.gpu_counts or GPU_COUNTS),
        queries=tuple(
            item for item in QUERIES if item[0] in wanted),
    ))
    if not selected:
        raise SystemExit("no jobs match --gpus / --query")
    if args.submit:
        submit(selected, args.log_dir)
        return
    print_suite(selected)


if __name__ == "__main__":
    main()

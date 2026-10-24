r"""One-GPU quail-b SF 0.1 suite for one of H100, L40S, or B200.

One invocation is one device, one model copy, no tensor parallel.
Qwen3 4B runs every IMDB, BIO, FEV, LEP, and AGENT query. Qwen3 32B
runs one query per dataset, including on L40S, where its weights are
34.4 GB of 48 GB. DiffusionGemma runs every query on H100 and B200.
REL-AGENT is not in the issue list.

L40S does not run DiffusionGemma. The planner refuses it because one
65,536-token chunk's sliding KV needs 13.63 GB and 5.82 GB remains
(``sliding_kv_needs_more_memory``). A smaller chunk does not make the
model run. Five of the thirty layers keep every token with 512-wide
heads. FlashAttention 2, which L40S selects for compute capability
8.9, stops at head dimension 256, so those layers go through
FlashAttention 4. FlashAttention 4 does not run there. A plan at
32,768 tokens still died in kernel warmup with ``paged KV not
supported on SM 8.0``
(/results/benchmarks/quailb/20261015T120134Z-1aa1a30d). See
``docs/device-leftover-plan.md``.

Configurations are Quail, stock vLLM operator-at-a-time, and
pipelined vLLM. SGLang is off.

    uv run python experiments/cells/sf01_device_suite.py \
      --only-device h100-sxm
    mkdir -p results/benchmark
    log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-sf01-h100.log"
    uv run python experiments/cells/sf01_device_suite.py --submit \
      --only-device h100-sxm \
      2>&1 | tee "$log"

Each ``--submit`` job is one ``bench.quailb_parallel`` detach
on ``quail-milestone1``. Results stay on quail-results. After a run:

    modal volume get quail-results /benchmarks/quailb/<run_id>

Put ``runtime_s``, ``input_tokens_per_second``, ``cost_usd``,
``fresh_tokens``, ``regret_tokens``, and accuracy, precision, and
recall beside the H100 cite. ``cost_usd`` is ``$/query``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from quail.specs import DEVICES, DIFFUSION_GEMMA_26B_FP8, QWEN3_32B_FP8

SF = 0.1
GPUS = 1
OUTPUT_DIR = "/results/benchmarks/quailb"

# IMDB-1+FEV-1 slice only. After the H100 spawn finishes, point this
# at that run so the other devices can sit beside a full suite.
H100_CITE = "/results/benchmarks/quailb/20261014T030723Z-43af63b2"

DATASETS = {
    "IMDB": tuple(f"IMDB-{i}" for i in range(1, 16)),
    "BIO": tuple(f"BIO-{i}" for i in range(1, 7)),
    "FEV": tuple(f"FEV-{i}" for i in range(1, 12)),
    "LEP": tuple(f"LEP-{i}" for i in range(1, 7)),
    "AGENT": tuple(f"AGENT-{i}" for i in range(1, 6)),
}
EVERY_QUERY = tuple(qid for ids in DATASETS.values() for qid in ids)
# one small query per dataset so 32B stays short
ONE_PER_DATASET = ("IMDB-1", "BIO-1", "FEV-1", "LEP-1", "AGENT-1")

FULL_MODELS = ("qwen3-4b-fp8", "diffusion-gemma-26b-a4b-fp8")
SHORT_MODEL = "qwen3-32b-fp8"
RUN_DEVICES = ("h100-sxm", "l40s", "b200")
CONFIGS = ("quail", "stock_vllm", "pipelined_vllm")

PREDICTION_TEXT = (
    "Compute-bound prefill. B200 at most 2.3x H100 (4.5 / 1.98 PFLOPS). "
    "L40S at most 0.37x H100 (0.73 / 1.98). "
    "Qwen3 4B reserves 6 GiB. Qwen3 32B reserves 10 GiB, so B200 "
    "stays at 0.91 and H100 and L40S drop below it. "
    "L40S does not run DiffusionGemma."
)


@dataclass(frozen=True)
class Job:
    """One Modal spawn: one device, one model, one GPU, one query list."""

    device: str
    model: str
    queries: tuple[str, ...]
    note: str

    @property
    def configs(self) -> tuple[str, ...]:
        return CONFIGS


def queries_for(model: str) -> tuple[str, ...]:
    """Every listed query, or one per dataset on 32B."""
    if model == SHORT_MODEL:
        return ONE_PER_DATASET
    if model in FULL_MODELS:
        return EVERY_QUERY
    raise ValueError(f"model {model!r} is not in the #181 suite")


def diffusion_on_l40s(device: str, model: str) -> bool:
    """Return whether this pair is the DiffusionGemma run L40S skips."""
    return device == "l40s" and model == DIFFUSION_GEMMA_26B_FP8.name


def jobs(*, devices=RUN_DEVICES, models=(*FULL_MODELS, SHORT_MODEL)):
    """Yield one spawn per model on this device.

    L40S yields Qwen3 4B and Qwen3 32B. DiffusionGemma is omitted there.
    """
    for device in devices:
        if device not in DEVICES:
            raise ValueError(f"unknown device {device!r}")
        for model in models:
            if diffusion_on_l40s(device, model):
                continue
            note = "Quail, stock vLLM operator-at-a-time, pipelined vLLM"
            if model == SHORT_MODEL:
                w_gb = QWEN3_32B_FP8.w_mem_bytes / 1e9
                note += f"; one query per dataset ({w_gb:.1f} GB weights)"
            yield Job(
                device=device, model=model, queries=queries_for(model),
                note=note)


def modal_argv(job: Job) -> list[str]:
    """The ``modal run --detach`` argv for one job."""
    return [
        sys.executable, "-m", "modal", "run", "--detach",
        "-m", "bench.quailb_parallel",
        "--model", job.model,
        "--scale-factor", str(SF),
        "--device", job.device,
        "--gpus", str(GPUS),
        "--query", ",".join(job.queries),
        "--no-include-sglang",
    ]


def print_suite(selected: list[Job]) -> None:
    """Print the issue components and the jobs that will spawn."""
    print(f"PREDICTION: {PREDICTION_TEXT}")
    print(f"sf={SF} gpus={GPUS} (one model copy per GPU, no TP)")
    print(f"H100 cite (replace after the H100 spawn): {H100_CITE}")
    print()
    print("datasets (every query on 4B and DiffusionGemma):")
    for name, ids in DATASETS.items():
        print(f"  {name}: {len(ids)}  {','.join(ids)}")
    print(f"32B one per dataset: {','.join(ONE_PER_DATASET)}")
    print(
        "configs: Quail, stock vLLM operator-at-a-time, "
        "pipelined vLLM"
    )
    for device in dict.fromkeys(job.device for job in selected):
        if any(diffusion_on_l40s(device, model) for model in FULL_MODELS):
            print(
                "skipped on l40s: diffusion-gemma-26b-a4b-fp8"
            )
            print(
                "  sliding KV for the 65536-token chunk does not fit; "
                "512-wide heads need FlashAttention 4"
            )
    print()
    print("report columns after each run.json lands:")
    print(
        "  runtime_s, input_tokens_per_second, $/query, "
        "fresh_tokens, regret_tokens,"
    )
    print(
        "  accuracy / precision / recall vs the reference model "
        "in the suite, H100 from the cite"
    )
    print()
    print(f"{len(selected)} spawn(s):")
    for job in selected:
        print(
            f"  {job.device} {job.model}  "
            f"{len(job.queries)} queries  "
            f"configs={','.join(job.configs)}"
        )
        print(f"    {job.note}")
        print(f"    {' '.join(modal_argv(job))}")


def submit(selected: list[Job], log_dir: Path) -> None:
    """Detach one ``quailb_parallel`` run per job and print the log path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"PREDICTION: {PREDICTION_TEXT}", flush=True)
    print(f"H100 cite: {H100_CITE}", flush=True)
    for job in selected:
        slug = f"{job.device}-{job.model.replace('.', '-')}"
        log = log_dir / f"{stamp}-sf01-{slug}.log"
        argv = modal_argv(job)
        print(f"tee={log}", flush=True)
        print(" ".join(argv), flush=True)
        with log.open("w") as handle:
            handle.write(f"PREDICTION: {PREDICTION_TEXT}\n")
            handle.write(f"H100 cite: {H100_CITE}\n")
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
        "--only-device", dest="device", required=True, choices=RUN_DEVICES,
        help="run this one device; one invocation is one device")
    parser.add_argument(
        "--only-model", action="append", dest="models",
        choices=(*FULL_MODELS, SHORT_MODEL),
        help="restrict to these models; default all three")
    parser.add_argument(
        "--log-dir", type=Path, default=Path("results/benchmark"),
        help="directory for teed spawn logs")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Print the suite, or detach the selected jobs."""
    args = _parse_args(argv)
    selected = list(jobs(
        devices=(args.device,),
        models=tuple(args.models or (*FULL_MODELS, SHORT_MODEL)),
    ))
    if not selected:
        raise SystemExit("no jobs match --only-device / --only-model")
    if args.submit:
        submit(selected, args.log_dir)
        return
    print_suite(selected)


if __name__ == "__main__":
    main()

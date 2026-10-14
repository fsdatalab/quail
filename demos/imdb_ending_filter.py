"""Filter all 100,000 IMDB reviews with two questions, both required."""

import argparse
import os
import time

import pyarrow as pa
import pyarrow.dataset as ds

import quail
from quail.specs import DEVICES, MODAL_GPU_USD_PER_HOUR, MODELS

os.environ.setdefault("HF_HUB_OFFLINE", "1")

IMDB_REVISION = "e6281661ce1c48d982bc483cf8a173c1bbeb5d31"

SQL = """
    SELECT r.review_id
    FROM reviews AS r
    WHERE AI.IF(
        PROMPT(
            'Does this review discuss the ending of the movie?\\n\\n{0}',
            r.review
        ),
        {'selectivity': 0.25}
    )
    AND AI.IF(
        PROMPT(
            'Does the reviewer recommend watching the movie?\\n\\n{0}',
            r.review
        ),
        {'selectivity': 0.5}
    )
"""


def load_reviews() -> ds.Dataset:
    """Return every IMDB review as an Arrow dataset with one id column."""
    from datasets import concatenate_datasets, load_dataset
    try:
        imdb = load_dataset("stanfordnlp/imdb", revision=IMDB_REVISION)
    except Exception:
        os.environ.pop("HF_HUB_OFFLINE", None)
        imdb = load_dataset("stanfordnlp/imdb", revision=IMDB_REVISION)
    all_reviews = concatenate_datasets([
        imdb["train"],
        imdb["test"],
        imdb["unsupervised"],
    ])
    # take the Arrow column as is; indexing the dataset would copy every
    # string through Python
    return ds.dataset(pa.table({
        "review_id": pa.array(f"review-{i}" for i in range(len(all_reviews))),
        "review": all_reviews.data.table.column("text"),
    }))


def run(model: str = "qwen3-4b-fp8", device: str = "h100-sxm",
        gpus: int = 1, gpu_usd_per_hour: float | None = None) -> dict:
    """Run the two-question filter over every review and print its report.

    Args:
        model: A name in ``quail.specs.MODELS``.
        device: A name in ``quail.specs.DEVICES``.
        gpus: GPUs used by the query.
        gpu_usd_per_hour: Hourly price per GPU; Modal pricing by default.

    Returns:
        The query's timing, token counts, stage selectivities, and cost.
    """
    price_source = "custom" if gpu_usd_per_hour is not None else "Modal"
    if gpu_usd_per_hour is None:
        gpu_usd_per_hour = MODAL_GPU_USD_PER_HOUR[device]
    t0 = time.perf_counter()
    reviews = load_reviews()
    n_docs = reviews.count_rows()
    print(f"reviews: {n_docs}, loaded in {time.perf_counter() - t0:.1f} s",
          flush=True)

    with quail.Session(
        config=quail.EngineConfig(
            gpus=gpus,
            model=model,
            backend="quail",
            device=device,
        ),
    ) as session:
        session.register(
            "reviews",
            quail.DocumentProvider.from_dataset(reviews, id_col="review_id"),
        )
        query = session.sql(SQL, dialect="bq")
        print(query.explain(), flush=True)

        result = query.run()
        table = result.collect()
        report = result.report
    wall_s = report["wall_s"]
    boot_s = report.get("boot_s") or 0.0
    input_tokens = report["fresh_tokens"] + (report.get("cached_tokens") or 0)
    cost_per_second = gpus * gpu_usd_per_hour / 3600
    summary = {
        "model": model,
        "device": device,
        "gpus": gpus,
        "reviews": n_docs,
        "matching_reviews": table.num_rows,
        "stages": [
            {"evaluated": stage["evaluated"],
             "observed_selectivity": stage["observed_selectivity"]}
            for stage in report.get("stages", ())
            if stage.get("op") == "filter"],
        "boot_s": boot_s,
        "boot_kind": report.get("boot_kind"),
        "token_wait_s": report.get("token_wait_s"),
        "wall_s": wall_s,
        "estimated_seconds": report.get("estimated_seconds"),
        "fresh_tokens": report["fresh_tokens"],
        "cached_tokens": report.get("cached_tokens"),
        "input_tokens": input_tokens,
        "input_tokens_per_second": input_tokens / wall_s,
        "documents_per_second": n_docs / wall_s,
        "gpu_usd_per_hour": gpu_usd_per_hour,
        "usd_per_query": wall_s * cost_per_second,
        "startup_usd": boot_s * cost_per_second,
    }
    print(f"matching reviews: {table.num_rows} of {n_docs}")
    for stage in summary["stages"]:
        print(f"  stage evaluated {stage['evaluated']} reviews, "
              f"{stage['observed_selectivity']:.3f} passed")
    print(f"boot_s: {boot_s} ({summary['boot_kind']})")
    print(f"token_wait_s: {summary['token_wait_s']}")
    print(f"wall_s: {wall_s} (planner estimate {summary['estimated_seconds']})")
    print(f"fresh_tokens: {summary['fresh_tokens']}, "
          f"cached_tokens: {summary['cached_tokens']}")
    print(f"input tokens/second: {summary['input_tokens_per_second']:.0f}")
    print(f"documents/second: {summary['documents_per_second']:.1f}")
    print(f"GPU price: ${gpu_usd_per_hour:.4f}/GPU-hour ({price_source})")
    print(f"GPU cost/query: ${summary['usd_per_query']:.4f}")
    print(f"GPU startup cost: ${summary['startup_usd']:.4f}")
    return summary


def main() -> None:
    """Parse arguments and run the query."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS),
                        default="qwen3-4b-fp8")
    parser.add_argument("--device", choices=sorted(DEVICES), default="h100-sxm")
    parser.add_argument("--gpus", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--gpu-usd-per-hour", type=float,
                        help="hourly price per GPU; defaults to Modal pricing")
    args = parser.parse_args()
    run(args.model, args.device, args.gpus, args.gpu_usd_per_hour)


if __name__ == "__main__":
    main()

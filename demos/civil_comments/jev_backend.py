"""Run the Civil Comments comparison query with Jev.

This follows the SQL text's join-then-filter order with the same sample,
label cutoff, and question text as the Quail backend. Jev's ``noul``
probability is TRUE when it is at least 0.5.

Set the API key without putting it on the command line, then run:

    read -s TYPESAFE_API_KEY
    export TYPESAFE_API_KEY
    uv run python demos/civil_comments/jev_backend.py \
      --limit 10000 --concurrency 256 \
      2>&1 | tee /tmp/civil-comments-jev.log

The script checkpoints answers under ``--output`` and resumes missing
requests. Do not commit that directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from collections import Counter
from pathlib import Path

import httpx

from demos.civil_comments import (
    DATASET,
    DATASET_REVISION,
    FIELDS,
    JOIN_PROMPT_INSTRUCTIONS,
    LABEL_CUTOFF,
    TOXICITY_PROMPT_INSTRUCTIONS,
    accuracy_summary,
    load_comments,
    requested_input_tokens,
)

MODEL = "jev-1.13.0"
URL = "https://api.typesafe.ai/v1/systemone"
USD_PER_MILLION_INPUT_TOKENS = 0.042

FILTER_QUESTIONS = {
    "toxicity": {
        "type": "noul",
        "instructions": TOXICITY_PROMPT_INSTRUCTIONS,
        "criteria": {
            "true": "Toxic or very toxic",
            "false": "Not toxic, only slightly toxic, or hard to say",
        },
    },
}

JOIN_QUESTIONS = {
    field: {
        "type": "noul",
        "instructions": (
            f"{JOIN_PROMPT_INSTRUCTIONS}\n\n"
            f"DOCUMENT 1:\nThe comment {statement}."
        ),
    }
    for field, statement in FIELDS.items()
}


def read_answers(path: Path) -> dict[str, dict]:
    """Read completed responses from a checkpoint."""
    rows = {}
    if not path.exists():
        return rows
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["comment_id"]] = row
    return rows


async def request(
    client: httpx.AsyncClient,
    api_key: str,
    comment_id: str,
    state,
    questions: dict,
) -> dict:
    """Send one request, retrying rate limits and temporary errors."""
    payload = {"model": MODEL, "state": state, "questions": questions}
    delay = 0.25
    for attempt in range(12):
        try:
            response = await client.post(
                URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
        except httpx.HTTPError:
            if attempt == 11:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, 8)
            continue
        if response.status_code == 429 or response.status_code >= 500:
            wait = float(response.headers.get("Retry-After", delay))
            await asyncio.sleep(wait)
            delay = min(delay * 2, 8)
            continue
        response.raise_for_status()
        body = response.json()
        return {
            "comment_id": comment_id,
            "model": body["model"],
            "answers": {
                name: float(answer["noul"])
                for name, answer in body["answers"].items()
            },
            "input_tokens": int(body["usage"]["input_tokens"]),
        }
    raise RuntimeError(f"request retries exhausted for {comment_id}")


async def run_pass(
    name: str,
    items: list[tuple[str, object]],
    questions: dict,
    checkpoint: Path,
    concurrency: int,
) -> tuple[dict[str, dict], float]:
    """Run one checkpointed parallel API pass."""
    timing_path = checkpoint.with_suffix(".timing.json")
    elapsed_total = 0.0
    if timing_path.exists():
        elapsed_total = float(json.loads(timing_path.read_text())["wall_s"])
    api_key = os.environ["TYPESAFE_API_KEY"]
    for round_index in range(4):
        completed = read_answers(checkpoint)
        pending = [
            (comment_id, state)
            for comment_id, state in items
            if comment_id not in completed
        ]
        print(
            f"{name}: {len(items)} requests, {len(completed)} already done, "
            f"{len(pending)} left, concurrency {concurrency}",
            flush=True,
        )
        if not pending:
            return completed, elapsed_total

        semaphore = asyncio.Semaphore(concurrency)
        write_lock = asyncio.Lock()
        started = time.perf_counter()
        finished = 0
        input_tokens = 0
        limits = httpx.Limits(
            max_connections=concurrency + 16,
            max_keepalive_connections=concurrency,
        )
        async with httpx.AsyncClient(
            http2=True,
            limits=limits,
            timeout=httpx.Timeout(60.0),
        ) as client:

            async def worker(comment_id, state):
                nonlocal finished, input_tokens
                async with semaphore:
                    row = await request(
                        client,
                        api_key,
                        comment_id,
                        state,
                        questions,
                    )
                async with write_lock:
                    with checkpoint.open("a") as handle:
                        handle.write(json.dumps(row) + "\n")
                    finished += 1
                    input_tokens += row["input_tokens"]
                    if finished % 250 == 0 or finished == len(pending):
                        elapsed = time.perf_counter() - started
                        rate = finished / elapsed
                        print(
                            f"  {finished}/{len(pending)} at {rate:.1f} "
                            f"requests/s; {input_tokens:,} API input tokens",
                            flush=True,
                        )

            results = await asyncio.gather(
                *(worker(comment_id, state) for comment_id, state in pending),
                return_exceptions=True,
            )
        errors = [
            result for result in results if isinstance(result, Exception)
        ]
        elapsed = time.perf_counter() - started
        elapsed_total += elapsed
        timing_path.write_text(json.dumps({"wall_s": elapsed_total}))
        print(
            f"{name}: {len(pending) - len(errors)} completed, "
            f"{len(errors)} failed, {elapsed:.2f} s",
            flush=True,
        )
        if errors and round_index < 3:
            print(f"{name}: retrying failed requests", flush=True)
    completed = read_answers(checkpoint)
    missing = [comment_id for comment_id, _ in items if comment_id not in completed]
    if missing:
        raise RuntimeError(
            f"{name} is missing {len(missing)} responses; rerun to resume"
        )
    return completed, elapsed_total


async def evaluate(
    limit: int,
    concurrency: int,
    output: Path,
) -> dict:
    """Run Jev and return its comparison summary."""
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    comments = load_comments(None if limit == 0 else limit)
    ids = comments["comment_id"].to_pylist()
    texts = comments["text"].to_pylist()
    output.mkdir(parents=True, exist_ok=True)
    (output / "inputs.json").write_text(
        json.dumps(
            {
                "backend": "jev",
                "dataset": DATASET,
                "dataset_revision": DATASET_REVISION,
                "model": MODEL,
                "comments": len(ids),
                "label_cutoff": LABEL_CUTOFF,
                "concurrency": concurrency,
            },
            indent=2,
        )
    )

    all_comments = list(zip(ids, texts))
    join_rows, join_s = await run_pass(
        "join",
        [
            (comment_id, {"DOCUMENT 0": text})
            for comment_id, text in all_comments
        ],
        JOIN_QUESTIONS,
        output / "join.jsonl",
        concurrency,
    )
    joined_ids = {
        comment_id
        for comment_id, row in join_rows.items()
        if any(
            probability >= LABEL_CUTOFF
            for probability in row["answers"].values()
        )
    }
    filter_rows, filter_s = await run_pass(
        "filter",
        [
            (comment_id, text)
            for comment_id, text in all_comments
            if comment_id in joined_ids
        ],
        FILTER_QUESTIONS,
        output / "filter.jsonl",
        concurrency,
    )
    toxic_found = {
        comment_id
        for comment_id, row in filter_rows.items()
        if row["answers"]["toxicity"] >= LABEL_CUTOFF
    }
    query_s = filter_s + join_s
    pairs_found = {
        (comment_id, field)
        for comment_id, row in join_rows.items()
        if comment_id in toxic_found
        for field, probability in row["answers"].items()
        if probability >= LABEL_CUTOFF
    }
    accuracy = accuracy_summary(comments, toxic_found, pairs_found)
    api_input_tokens = sum(
        row["input_tokens"]
        for row in [*filter_rows.values(), *join_rows.values()]
    )
    logical_input_tokens = requested_input_tokens(
        comments,
        filter_ids=joined_ids,
        join_ids=set(ids),
    )
    summary = {
        "backend": "jev",
        "model": MODEL,
        "plan_order": "join_then_filter",
        "comments": len(ids),
        "concurrency": concurrency,
        "wall_s": query_s,
        "filter_wall_s": filter_s,
        "join_wall_s": join_s,
        "input_tokens": logical_input_tokens,
        "input_tokens_per_second": logical_input_tokens / query_s,
        "api_input_tokens": api_input_tokens,
        "api_input_tokens_per_second": api_input_tokens / query_s,
        "api_cost_usd": (
            api_input_tokens
            * USD_PER_MILLION_INPUT_TOKENS
            / 1_000_000
        ),
        "evaluated_pairs": len(ids) * len(FIELDS),
        "filter_evaluated_comments": len(joined_ids),
        "accuracy": accuracy,
        "field_counts": dict(Counter(field for _, field in pairs_found)),
        "result_path": str(output),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    """Parse arguments and run Jev."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp") / f"civil-comments-jev-{uuid.uuid4().hex}",
    )
    args = parser.parse_args()
    if args.limit < 0 or args.concurrency < 1:
        parser.error("limit must be >= 0 and concurrency must be >= 1")
    asyncio.run(evaluate(args.limit, args.concurrency, args.output))


if __name__ == "__main__":
    main()

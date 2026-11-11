"""Run the position-bias experiment against the Jev API.

Jev answers a `noul` question about a state with P(TRUE). The state
is the document Quail would score (one record, or RECORD 1 and RECORD
2), and the instructions are the AI.SCORE template with the document
placeholder removed.

    read -s TYPESAFE_API_KEY
    export TYPESAFE_API_KEY
    uv run python -m experiments.position_bias.jev --tasks sales,support,legal

Records: results/position_bias/jev-1.13.0/{task}.json.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx

from experiments.position_bias.experiment import run_task
from experiments.position_bias.tasks import TASKS, load_task

MODEL = "jev-1.13.0"
URL = "https://api.typesafe.ai/v1/systemone"
LOCAL_DIR = Path("results/position_bias") / MODEL
QUESTION = "answer"


def jev_instructions(template: str) -> str:
    """The template's text around the document placeholder."""
    return template.replace("\n\n{0}\n\n", "\n\n")


async def _request(client, semaphore, api_key, state, instructions) -> tuple:
    payload = {"model": MODEL, "state": state,
               "questions": {QUESTION: {"type": "noul",
                                        "instructions": instructions}}}
    delay = 0.25
    async with semaphore:
        for attempt in range(12):
            try:
                response = await client.post(
                    URL, headers={"Authorization": f"Bearer {api_key}"},
                    json=payload)
            except httpx.HTTPError:
                if attempt == 11:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, 8)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                await asyncio.sleep(float(response.headers.get("Retry-After",
                                                               delay)))
                delay = min(delay * 2, 8)
                continue
            response.raise_for_status()
            body = response.json()
            return (float(body["answers"][QUESTION]["noul"]),
                    int(body["usage"]["input_tokens"]))
    raise RuntimeError("Jev request retries exhausted")


class JevScorer:
    """Score documents with Jev's P(TRUE), many requests at a time.

    Attributes:
        input_tokens: Input tokens Jev billed.
    """

    def __init__(self, concurrency: int):
        self.concurrency = concurrency
        self.api_key = os.environ["TYPESAFE_API_KEY"]
        self.input_tokens = 0

    def __call__(self, template: str, documents) -> list[float]:
        """Return P(TRUE) for each document under the template."""
        return asyncio.run(self._score(jev_instructions(template), documents))

    async def _score(self, instructions, documents) -> list[float]:
        semaphore = asyncio.Semaphore(self.concurrency)
        limits = httpx.Limits(max_connections=self.concurrency)
        async with httpx.AsyncClient(timeout=120, limits=limits) as client:
            answers = await asyncio.gather(*(
                _request(client, semaphore, self.api_key, document, instructions)
                for document in documents))
        self.input_tokens += sum(tokens for _, tokens in answers)
        return [probability for probability, _ in answers]


def main():
    """Run Jev on each task and save one record per task."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", default="sales,support,legal")
    parser.add_argument("--concurrency", type=int, default=64)
    args = parser.parse_args()
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.tasks.split(","):
        scorer = JevScorer(args.concurrency)
        record = run_task(load_task(TASKS[name]), scorer,
                          log=lambda line: print(f"[{MODEL}] {line}", flush=True))
        record.update(model=MODEL, input_tokens=scorer.input_tokens)
        (LOCAL_DIR / f"{name}.json").write_text(json.dumps(record))


if __name__ == "__main__":
    main()

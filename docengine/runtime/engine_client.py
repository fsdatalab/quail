"""Blocked streaming client scheduler (scheduler plan step three).

Runs a chain of yes or no filters over a document corpus against an
unmodified serving engine, combining the three wins measured in steps
one and two. Prompts are pre-tokenized once, so the engine never
re-converts the same document text. Admission is controlled by a token
budget sized to the engine's KV pool, so a document is admitted only
when its notes can stay resident, and because each document streams
through all of its filters the moment each answer arrives, it finishes
its whole chain while hot and returns its budget quickly. This yields
the blocked execution order without any stage barriers: the fastest
document never waits for the slowest.

Optional lookahead submits the next `lookahead` questions of a document
concurrently, paying for wasted questions when an earlier one fails.
Under streaming there are no stage barriers left for speculation to
hide, so its expected value is limited to filling the admission tail.

The engine object must expose the vLLM AsyncLLM interface:
`generate(prompt, sampling_params, request_id)` returning an async
generator whose final item has `prompt_token_ids`, `num_cached_tokens`,
and `outputs[0].text`.
"""

import asyncio
import time


def _yes(out):
    return 1 if out.outputs[0].text.strip().upper().startswith("Y") else 0


async def run_filter_chain(engine, sampling_params, body_ids, q_ids,
                           budget_tokens, lookahead=1, tag="q"):
    """Run every document through the filter chain; return timings,
    counters, per-call answers keyed (doc, stage) with stages 1-indexed,
    and the surviving document ids."""
    n = len(q_ids)
    q_cost = sum(len(q) for q in q_ids) + n
    used = 0
    cond = asyncio.Condition()
    counters = dict(requests=0, prompt_tokens=0, cached_tokens=0)
    answers = {}
    survivors = []

    async def ask(ids, rid):
        final = None
        async for out in engine.generate({"prompt_token_ids": ids},
                                         sampling_params, rid):
            final = out
        counters["requests"] += 1
        counters["prompt_tokens"] += len(final.prompt_token_ids)
        counters["cached_tokens"] += (getattr(final, "num_cached_tokens", 0)
                                      or 0)
        return _yes(final)

    async def chain(i, cost):
        nonlocal used
        try:
            j = 0
            while j < n:
                kk = min(lookahead, n - j)
                got = await asyncio.gather(*[
                    ask(body_ids[i] + q_ids[j + off], f"{tag}-{i}-{j + off}")
                    for off in range(kk)])
                passes = 0
                for off in range(kk):
                    answers[(i, j + off + 1)] = got[off]
                    if got[off] and passes == off:
                        passes += 1
                if passes < kk:
                    return
                j += kk
            survivors.append(i)
        finally:
            async with cond:
                used -= cost
                cond.notify_all()

    t0 = time.time()
    tasks = []
    for i in range(len(body_ids)):
        cost = len(body_ids[i]) + q_cost
        async with cond:
            while used + cost > budget_tokens and used > 0:
                await cond.wait()
            used += cost
        tasks.append(asyncio.create_task(chain(i, cost)))
    await asyncio.gather(*tasks)
    wall = time.time() - t0
    return dict(wall=wall, survivors=sorted(survivors), answers=answers,
                **counters)

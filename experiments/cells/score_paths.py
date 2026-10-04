"""AI.SCORE on H100: the decision model's head score, and tree against unified.

decision_side: Decision-2.0-Kai-0.6B scores 2,000 QUAIL-B IMDB reviews
(one document) and 200 reviews x 12 aspects (pairs) with AI.SCORE,
and answers the same prompts with AI_FILTER.

reranker_side: Qwen3 Reranker 4B scores the same inputs with AI.SCORE,
forced onto unified then tree attention, twice each.

Prediction: a decision score above 0.5 agrees with the AI_FILTER answer
on at least 99% of rows and pairs. Reranker scores differ by less than
0.02 between the two paths on every row, and the paths' times are
within 5% of each other.

    uv run modal run experiments/cells/score_paths.py \
        2>&1 | tee /tmp/score_paths.log

Summaries are written to /results/score_paths/<run>_<side>.json on the
quail-results volume.
"""

import json
import time

import modal

try:
    from quail.bench.images import gpu_image
    image = gpu_image()
except ImportError:    # a container without the local quail package
    image = None

N_REVIEWS = 2000
N_PAIR_REVIEWS = 200
SINGLE = "Is this movie review positive? {0}"
PAIR = ("Does the review in DOCUMENT {0} discuss the movie aspect in "
        "DOCUMENT {1}?")

app = modal.App("quail-milestone1")
VOLUMES = {
    "/root/.cache/huggingface": modal.Volume.from_name(
        "quail-hf-cache", create_if_missing=True),
    "/root/.cache/kernels": modal.Volume.from_name(
        "quail-kernel-cache", create_if_missing=True),
    "/results": modal.Volume.from_name("quail-results", create_if_missing=True),
}


def _session(model):
    import pyarrow.parquet as pq

    import quail

    data = "/results/quailb_data/sf0.1"
    reviews = pq.read_table(f"{data}/reviews.parquet").slice(0, N_REVIEWS)
    session = quail.Session(quail.EngineConfig(model=model, device="h100-sxm"))
    session.register("reviews", quail.DocumentProvider.from_table(
        reviews, id_col="id"))
    session.register("few", quail.DocumentProvider.from_table(
        reviews.slice(0, N_PAIR_REVIEWS), id_col="id"))
    session.register("aspects", quail.DocumentProvider.from_parquet(
        f"{data}/aspects.parquet", id_col="id"))
    return session


SCORE_SQL = (f"SELECT r.id, AI.SCORE(PROMPT('{SINGLE}', r.body)) AS s "
             "FROM reviews r")
PAIR_SQL = (f"SELECT r.id, a.id, AI.SCORE(PROMPT('{PAIR}', r.body, a.aspect)) "
            "AS s FROM few r CROSS JOIN aspects a")


def _timed(query):
    t0 = time.perf_counter()
    result = query.run()
    return time.perf_counter() - t0, result


def _save(run, side, summary):
    import os

    path = f"/results/score_paths/{run}_{side}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    VOLUMES["/results"].commit()
    return path


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def decision_side(run: str) -> dict:
    """AI.SCORE above 0.5 against AI_FILTER answers on the same prompts."""
    session = _session("decision-2.0-kai-0.6b-bf16")
    summary = {}
    _timed(session.sql(SCORE_SQL))    # boot and warm-up
    seconds, scores = _timed(session.sql(SCORE_SQL))
    score = {r[0]: r[1] for r in scores.to_rows()}
    seconds_f, passed = _timed(session.sql(
        f"SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('{SINGLE}', r.body))"))
    passed = {r[0] for r in passed.to_rows()}
    summary["single"] = {
        "rows": len(score), "score_seconds": seconds, "filter_seconds": seconds_f,
        "agreement": sum((s > 0.5) == (i in passed) for i, s in score.items())
        / len(score),
        "filter_true": len(passed), "score_above_half": sum(
            s > 0.5 for s in score.values())}
    seconds, scores = _timed(session.sql(PAIR_SQL))
    score = {(r[0], r[1]): r[2] for r in scores.to_rows()}
    seconds_f, pairs = _timed(session.sql(
        f"SELECT r.id, a.id FROM few r JOIN aspects a ON AI_FILTER(PROMPT("
        f"'{PAIR}', r.body, a.aspect))"))
    pairs = set(pairs.to_rows())
    summary["pair"] = {
        "rows": len(score), "score_seconds": seconds, "filter_seconds": seconds_f,
        "agreement": sum((s > 0.5) == (k in pairs) for k, s in score.items())
        / len(score),
        "filter_true": len(pairs), "score_above_half": sum(
            s > 0.5 for s in score.values())}
    session.close()
    return {"path": _save(run, "decision", summary), **summary}


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def reranker_side(run: str) -> dict:
    """AI.SCORE forced onto unified and tree attention, alternating."""
    import quail.backends.quail.executor.score as score_module

    run_join = score_module.run_join
    path = {"mode": None}

    def forced(*args, **kwargs):
        return run_join(*args, attention_mode=path["mode"], **kwargs)

    score_module.run_join = forced
    session = _session("qwen3-reranker-4b-bf16")
    _timed(session.sql(SCORE_SQL))    # boot and warm-up
    summary = {}
    for name, sql in (("single", SCORE_SQL), ("pair", PAIR_SQL)):
        rounds, scores = [], {}
        for mode in ("unified", "tree", "unified", "tree"):
            path["mode"] = mode
            seconds, result = _timed(session.sql(sql))
            rounds.append({"attention": mode, "seconds": seconds})
            scores.setdefault(mode, {r[:-1]: r[-1] for r in result.to_rows()})
        gaps = [abs(scores["tree"][k] - v) for k, v in scores["unified"].items()]
        summary[name] = {"rows": len(gaps), "rounds": rounds,
                         "max_abs_gap": max(gaps),
                         "mean_abs_gap": sum(gaps) / len(gaps)}
    session.close()
    return {"path": _save(run, "reranker", summary), **summary}


@app.local_entrypoint()
def main():
    run = time.strftime("%Y%m%d-%H%M%S")
    calls = {"decision": decision_side.spawn(run),
             "reranker": reranker_side.spawn(run)}
    for side, call in calls.items():
        print(f"{side}_side function call id: {call.object_id}", flush=True)
    for side, call in calls.items():
        print(json.dumps(call.get(), indent=2), flush=True)

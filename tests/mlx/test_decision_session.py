"""Decision-2.0-Kai-0.6B through a Session on the Apple GPU, against mlx-lm.

The session scores each review, then runs a two-filter chain whose
second filter reads the KV the first one kept. mlx-lm runs every whole
prompt on the same weights. Both run bf16, so the yardstick for the
session's scores is mlx-lm's own bf16 error against float32.

Runs on Apple silicon when the checkpoint is in the Hugging Face cache;
skipped otherwise, so it never downloads the weights.
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("vllm_metal")
pytest.importorskip("mlx_lm")

from test_decision_forward import (  # noqa: E402
    QUESTIONS,
    REVIEWS,
    SPEC,
    TIE,
    WEIGHTS,
    reference_scores,
)

import quail  # noqa: E402
from quail.backends.quail import worker  # noqa: E402
from quail.execution import execute  # noqa: E402
from quail.logical import bind_prompt, render_filter_prompt_ids  # noqa: E402
from quail.planner.plan import EngineConfig  # noqa: E402

pytestmark = pytest.mark.skipif(
    not isinstance(WEIGHTS, str),
    reason=f"{SPEC.hf_name} is not in the Hugging Face cache")


def test_a_session_scores_and_filters_as_mlx_lm_does(tmp_path, monkeypatch):
    # the model and its KV pools are released when the test ends
    state = {}
    monkeypatch.setattr(execute, "_BACKEND_STATE", state)
    path = str(Path(WEIGHTS).parents[1])
    pq.write_table(pa.table({"id": list(range(len(REVIEWS))), "body": REVIEWS}),
                   tmp_path / "reviews.parquet")
    try:
        with quail.Session(
                EngineConfig(model=SPEC.name, device="apple-gpu")) as session:
            session.register("reviews", quail.DocumentProvider.from_parquet(
                str(tmp_path / "reviews.parquet"), id_col="id"))
            scored = session.sql(
                f"SELECT r.id, AI.SCORE(PROMPT('{QUESTIONS[0]}', r.body)) AS s "
                "FROM reviews r").run()
            chain = session.sql(
                "SELECT r.id FROM reviews r "
                f"WHERE AI_FILTER(PROMPT('{QUESTIONS[0]}', r.body)) "
                f"AND AI_FILTER(PROMPT('{QUESTIONS[1]}', r.body))").run()
            # the weights, the KV pools, and the buffers MLX keeps
            held = mx.get_active_memory() + mx.get_cache_memory()
            budget = session.device.mem_bytes
            ids = session.tokenizer
            prompts = [[render_filter_prompt_ids(
                bind_prompt(question, ("body",), ids, turn=SPEC.turn,
                            layout=SPEC.prompt_layout), ids(review), ids)
                for review in REVIEWS] for question in QUESTIONS]
        (loaded,) = state.values()
        head, offsets = loaded.decision_head, loaded.decision_offsets
        margins = {}
        for dtype in (mx.float32, mx.bfloat16):
            margins[dtype] = [np.diff(reference_scores(
                path, dtype, head, offsets, whole))[:, 0] for whole in prompts]
    finally:
        worker.release_booted_models(state)

    assert scored.report["boot_kind"] == "cold"
    assert chain.report["boot_kind"] == "warm"
    # answers of the last query are still alive beside the pools
    assert 0.5 * budget < held <= 1.01 * budget
    exact, half = margins[mx.float32], margins[mx.bfloat16]
    got = dict(scored.to_rows())
    probability = np.array([got[r] for r in range(len(REVIEWS))])
    ours = np.abs(np.log(probability) - np.log1p(-probability) - exact[0]).max()
    theirs = np.abs(half[0] - exact[0]).max()
    assert ours < 2 * theirs + 0.02, (ours, theirs)

    survivors = {row[0] for row in chain.to_rows()}
    decided = [r for r in range(len(REVIEWS))
               if abs(exact[0][r]) > TIE and abs(exact[1][r]) > TIE]
    assert len(decided) >= len(REVIEWS) // 2
    for r in decided:
        assert (r in survivors) == (exact[0][r] > 0 and exact[1][r] > 0), r
    # the second filter packed only its question after each kept review
    assert chain.report["fresh_tokens"] < sum(
        len(whole) for question in prompts for whole in question)

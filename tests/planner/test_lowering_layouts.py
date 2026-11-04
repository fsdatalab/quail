"""Prompt lowering in Rust matches Python's on every layout's model.

tests/conftest.py repeats each prompt_lowering call in Rust. The suite
plans on ai-if and d1 models; these queries also plan classifications
on the decision2-noul model, whose option blocks choice_token_parts
lowers, and classify joined rows on it.
"""

import pyarrow as pa
import pytest
from helpers.sessions import fake_tok

import quail
from quail.api.config import EngineConfig
from quail.physical.plan import Refusal

pytest.importorskip("_quail_native")

REVIEWS = pa.table({"id": ["r0", "r1"], "body": ["good film", "slow film"]})
ASPECTS = pa.table({"id": ["a0", "a1"], "aspect": ["ending", "pacing"]})
QUERIES = (
    "SELECT r.id, AI.CLASSIFY(PROMPT('Tone of {0}?', r.body), ['good', 'bad']) "
    "AS s FROM reviews r",
    "SELECT r.id, AI.CLASSIFY(PROMPT('{0}', r.body), "
    "[('good', 'liked it'), ('bad', 'did not')]) AS s FROM reviews r "
    "WHERE AI_FILTER(PROMPT('q1: {0}', r.body))",
    "SELECT r.id, a.id FROM reviews r JOIN aspects a "
    "ON AI_FILTER(PROMPT('Does {0} discuss {1}?', r.body, a.aspect))",
    "SELECT r.id, a.id, AI.CLASSIFY(PROMPT('How does {0} treat {1}?', r.body, "
    "a.aspect), ['well', 'badly']) AS s FROM reviews r JOIN aspects a "
    "ON AI_FILTER(PROMPT('Does {0} discuss {1}?', r.body, a.aspect))",
)


@pytest.mark.parametrize("model", ["decision-2.0-kai-0.6b-bf16", "d1-2b-fp8"])
@pytest.mark.parametrize("text", QUERIES)
def test_each_layout_plans_with_the_rust_lowering(model, text):
    config = EngineConfig(model=model, device="h100-sxm")
    with quail.Session(config, tokenizer=fake_tok) as session:
        session.register("reviews",
                         quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
        session.register("aspects",
                         quail.DocumentProvider.from_table(ASPECTS, id_col="id"))
        plan = session.sql(text).plan()
    assert not isinstance(plan, Refusal), plan

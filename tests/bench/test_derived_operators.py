"""Every QUAIL-B region derives the binder's operators from its optimized plan.

Rust derives a model region's operators by walking the region's plan
after DataFusion's rules; the binder builds them from the plan as
DataFusion's SQL planner bound it. Until the engine plans from the
derived operators, each region is checked to give the same operators,
or the same refusal, both ways.
"""

import _quail_native
from helpers.quailb_data import standin_sets
from helpers.sessions import TOKENIZER

import quail
from bench.quailb import queries, register_tables
from quail.api.config import EngineConfig


def test_every_quailb_region_derives_the_binders_operators(tmp_path):
    standin_sets(tmp_path)
    with quail.Session(
        EngineConfig(gpus=1, model="qwen3-4b-fp8", device="h100-sxm"),
        tokenizer=TOKENIZER,
    ) as session:
        register_tables(session, tmp_path)
        _quail_native.take_shadow_log()
        num_regions = sum(build().num_regions
                          for _, build in queries(session).values())
        num_comparisons, _, regions, differences = _quail_native.take_shadow_log()
    assert not differences, "\n\n".join(differences)
    assert num_comparisons == num_regions >= len(regions) > 80

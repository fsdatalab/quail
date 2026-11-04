"""Put the project root on sys.path so tests run without an install.

When the ``_quail_native`` module is built, every SELECT the SQL front
end compiles is also bound by DataFusion and the Rust binder, every SELECT
it binds is also built by the Rust planner, and every prompt it binds is
also bound in Rust (tests/helpers/rust_builder.py). A test fails where the
two disagree. Set QUAIL_RUST_BINDER_LOG to a file to log each query's
outcome.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def _rust_builder(monkeypatch):
    try:
        import _quail_native
    except ImportError:
        return
    from helpers.rust_builder import shadow_place_relation, shadow_prompt_binder

    import quail.frontend.sql as frontend
    from quail.frontend.prompts import PromptBinder

    monkeypatch.setattr(PromptBinder, "bind", shadow_prompt_binder(
        _quail_native, PromptBinder.bind))
    monkeypatch.setattr(frontend, "place_relation", shadow_place_relation(
        _quail_native, frontend.place_relation))
    from helpers.rust_builder import (
        LOWERING_FUNCTIONS,
        checked_compile_sql,
        shadow_logical_alternatives,
        shadow_lowering,
    )

    import quail.api.query as query_module
    import quail.api.session as session_module
    import quail.planner.optimize as optimize_module
    import quail.planner.prompt_lowering as lowering_module

    for name in LOWERING_FUNCTIONS:
        original = getattr(lowering_module, name)
        shadow = shadow_lowering(_quail_native, name, original)
        for module in [module for key, module in list(sys.modules.items())
                       if key.startswith("quail.") and module is not None]:
            if getattr(module, name, None) is original:
                monkeypatch.setattr(module, name, shadow)
    from helpers.rust_builder import shadow_column_spec

    from quail.planner.operators.extract import ExtractPlanner
    from quail.planner.operators.map import MapPlanner
    from quail.planner.operators.sae_features import SaeFeaturesPlanner

    for planner in (ExtractPlanner, MapPlanner, SaeFeaturesPlanner):
        monkeypatch.setattr(planner, "spec", shadow_column_spec(
            _quail_native, planner.spec))
    from helpers.rust_builder import (
        shadow_classify,
        shadow_filter_costs,
        shadow_join_specs,
        shadow_score_spec,
    )

    import quail.planner.operators.filter as filter_module
    import quail.planner.operators.join as join_module
    import quail.planner.operators.score as score_module
    from quail.planner.operators.classify import (
        ClassifyBuilder,
        ClassifyPlanner,
        ClassifyScoring,
    )

    for owner, name, replacement in shadow_classify(
            _quail_native, ClassifyPlanner, ClassifyScoring, ClassifyBuilder):
        monkeypatch.setattr(owner, name, replacement)
    from helpers.rust_builder import shadow_operator_order

    import quail.planner.search.operator_order as order_module

    for owner, name, make in ((order_module, "choose_operator_order",
                               shadow_operator_order),
                              (score_module, "score_spec", shadow_score_spec),
                              (filter_module, "prepare_filter_costs",
                               shadow_filter_costs),
                              (join_module, "join_specs", shadow_join_specs)):
        original = getattr(owner, name)
        shadow = make(_quail_native, original)
        for module in [module for key, module in list(sys.modules.items())
                       if key.startswith("quail.") and module is not None]:
            if getattr(module, name, None) is original:
                monkeypatch.setattr(module, name, shadow)
    from helpers.rust_builder import shadow_plan_region

    import quail.planner.lowering as region_module

    original = region_module.plan_region
    shadow = shadow_plan_region(_quail_native, original)
    for module in [module for key, module in list(sys.modules.items())
                   if key.startswith("quail.") and module is not None]:
        if getattr(module, "plan_region", None) is original:
            monkeypatch.setattr(module, "plan_region", shadow)
    from helpers.rust_builder import shadow_search_plan

    monkeypatch.setattr(optimize_module, "search_plan", shadow_search_plan(
        _quail_native, optimize_module.search_plan))
    from helpers.rust_builder import shadow_speed_of_light

    import quail
    import quail.planner.sol.estimate as estimate_module

    shadow = shadow_speed_of_light(
        _quail_native, estimate_module.speed_of_light_estimate)
    for module in (quail, estimate_module):
        monkeypatch.setattr(module, "speed_of_light_estimate", shadow)
    for module in (query_module, optimize_module):
        monkeypatch.setattr(module, "logical_alternatives", shadow_logical_alternatives(
            _quail_native, module.logical_alternatives))
    monkeypatch.setattr(session_module, "compile_sql", checked_compile_sql(
        _quail_native, session_module.compile_sql,
        os.environ.get("QUAIL_RUST_BINDER_LOG")))

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
        shadow_filter_costs,
        shadow_join_specs,
        shadow_score_spec,
    )

    import quail.planner.operators.filter as filter_module
    import quail.planner.operators.join as join_module
    import quail.planner.operators.score as score_module

    for owner, name, make in ((score_module, "score_spec", shadow_score_spec),
                              (filter_module, "prepare_filter_costs",
                               shadow_filter_costs),
                              (join_module, "join_specs", shadow_join_specs)):
        original = getattr(owner, name)
        shadow = make(_quail_native, original)
        for module in [module for key, module in list(sys.modules.items())
                       if key.startswith("quail.") and module is not None]:
            if getattr(module, name, None) is original:
                monkeypatch.setattr(module, name, shadow)
    for module in (query_module, optimize_module):
        monkeypatch.setattr(module, "logical_alternatives", shadow_logical_alternatives(
            _quail_native, module.logical_alternatives))
    monkeypatch.setattr(session_module, "compile_sql", checked_compile_sql(
        _quail_native, session_module.compile_sql,
        os.environ.get("QUAIL_RUST_BINDER_LOG")))

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
    from helpers.rust_builder import checked_compile_sql

    import quail.api.session as session_module

    monkeypatch.setattr(session_module, "compile_sql", checked_compile_sql(
        _quail_native, session_module.compile_sql,
        os.environ.get("QUAIL_RUST_BINDER_LOG")))

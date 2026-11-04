"""Put the project root on sys.path so tests run without an install.

When the ``_quail_native`` module is built, every SELECT the SQL front
end binds, and every prompt it binds, is also built by the Rust planner
(tests/helpers/rust_builder.py), and a test fails where the two disagree.
"""

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

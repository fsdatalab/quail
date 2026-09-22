"""Check the headline figure's percent-of-SoL calculation."""

import importlib.util
import json
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "blogs/intro/make_headline_tok_per_sec.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("headline_tok", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tokens_per_second_label_uses_suffixes():
    module = _module()
    assert module._tokens_per_second_label(11_487_790) == "11.5M"
    assert module._tokens_per_second_label(1_422_504) == "1.42M"
    assert module._tokens_per_second_label(705_447) == "705k"


def test_percent_of_sol_divides_means():
    module = _module()
    assert module.percent_of_sol([10, 30], [40, 60]) == 40


def test_bio_remake_keeps_saved_sol_requested_tokens(tmp_path):
    module = _module()
    comparison = {
        "rows": {"quail": {}, "pipelined_vllm": {}},
        "sol": {
            "BIO-1": {"requested_tokens": 1000, "sol_s": 10},
            "BIO-3": {"requested_tokens": 1386176311, "sol_s": 42.78387005732039},
        },
    }
    (tmp_path / "comparison.json").write_text(json.dumps(comparison))
    quail_queries = [
        {
            "id": "BIO-1",
            "runtime_s": 2,
            "metrics": {"input_tokens": 9999},
        },
        {
            "id": "BIO-3",
            "runtime_s": 104.81,
            "metrics": {"input_tokens": 1749336759},
        },
        {
            "id": "BIO-4",
            "runtime_s": 69.39,
            "metrics": {"input_tokens": 995835615},
        },
    ]
    vllm_queries = [
        {"id": "BIO-1", "runtime_s": 4, "metrics": {"input_tokens": 1}},
        {"id": "BIO-3", "runtime_s": 883.96, "metrics": {"input_tokens": 1}},
        {"id": "BIO-4", "runtime_s": 494.69, "metrics": {"input_tokens": 1}},
    ]
    (tmp_path / "bio-remake-quail.json").write_text(
        json.dumps({"queries": quail_queries})
    )
    (tmp_path / "bio-remake-vllm.json").write_text(
        json.dumps({"queries": vllm_queries})
    )
    (tmp_path / "bio4-sf01-sol.json").write_text(
        json.dumps({"estimate": {"sol_s": 34.84490838706049}})
    )

    rows = {
        (row["query"], row["method"]): row for row in module.collect_rows(tmp_path)
    }
    # BIO-3 keeps the SoL requested-token total, not the Quail input total.
    assert rows[("BIO-3", "Quail")]["shared_tokens"] == 1386176311
    assert round(rows[("BIO-3", "vLLM")]["tok_per_sec"], 2) == 1568143.71
    # BIO-4 has no saved requested-token total, so it uses Quail's input tokens.
    assert rows[("BIO-4", "Quail")]["shared_tokens"] == 995835615
    assert round(rows[("BIO-4", "Quail")]["tok_per_sec"], 2) == 14351284.26
    assert round(rows[("BIO-4", "SoL")]["tok_per_sec"], 2) == 28579085.47


def test_deleted_lepard_queries_are_omitted(tmp_path):
    module = _module()
    queries = [f"LEP-{number}" for number in range(1, 9)]
    comparison = {
        "rows": {
            "quail": {
                query: {"runtime_s": 1, "requested_tokens": 100} for query in queries
            },
            "pipelined_vllm": {
                query: {"runtime_s": 2, "requested_tokens": 100} for query in queries
            },
        },
        "sol": {
            query: {"requested_tokens": 100, "sol_s": 0.5} for query in queries
        },
    }
    (tmp_path / "comparison.json").write_text(json.dumps(comparison))
    kept = {row["query"] for row in module.collect_rows(tmp_path)}
    assert kept == {"LEP-1", "LEP-2", "LEP-3", "LEP-4", "LEP-5"}

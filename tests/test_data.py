"""CPU checks for the QUAIL-B document sets."""

from quailb.data import (
    AGENT_TRACE_DOCUMENTS,
    AGENT_TRACE_MAX_TOKENS,
    AGENT_TRACE_TURN_INTERVAL,
    ASPECTS,
    LEPARD_POSITIVE_PAIRS,
    SCENARIOS,
    SETS,
    _agent_snapshot_boundaries,
    _agent_trace_rows,
    _lepard_documents,
    _n_agent_documents,
    _n_lepard_pairs,
    _sample_lepard_pairs,
)


def test_set_table_matches_design():
    assert SETS == {
        "reviews": 50_000,
        "reports": 5_000,
        "claims": 5_000,
        "agent_traces": AGENT_TRACE_DOCUMENTS,
        "policies": 1_000_000,
    }
    assert LEPARD_POSITIVE_PAIRS == 5_000
    assert _n_lepard_pairs(0.1) == 500
    assert len(ASPECTS) == 12
    assert len(SCENARIOS) == 100
    assert AGENT_TRACE_TURN_INTERVAL == 5
    assert AGENT_TRACE_MAX_TOKENS == 24_000
    assert _n_agent_documents(1.0) == 17_718
    assert _n_agent_documents(0.1) == 1_772


def test_agent_snapshots_include_every_fifth_turn_and_following_tool():
    messages = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "issue"},
    ]
    for turn in range(1, 12):
        messages.append({"role": "assistant", "content": f"action {turn}"})
        messages.append({"role": "tool", "content": f"result {turn}"})

    text, boundaries = _agent_snapshot_boundaries(messages)
    snapshots = [(turn, text[:end]) for turn, end in boundaries]

    assert [turn for turn, _snapshot in snapshots] == [5, 10]
    assert snapshots[0][1].endswith("[TOOL]\nresult 5")
    assert snapshots[1][1].startswith(snapshots[0][1])
    assert snapshots[1][1].endswith("[TOOL]\nresult 10")


class _WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()


def test_agent_trace_rows_have_unique_ids_and_cumulative_text():
    messages = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "issue"},
    ]
    for turn in range(1, 11):
        messages.append({"role": "assistant", "content": f"action {turn}"})
        messages.append({"role": "tool", "content": f"result {turn}"})

    rows = _agent_trace_rows(7, messages, _WordTokenizer())

    assert [row["id"] for row in rows] == ["at0007-t005", "at0007-t010"]
    assert [row["turn_index"] for row in rows] == [5, 10]
    assert rows[1]["trace"].startswith(rows[0]["trace"])
    assert rows[0]["token_count"] < rows[1]["token_count"]


def test_agent_trace_rows_require_a_user_issue():
    messages = [
        {"role": "system", "content": "Solve the issue."},
        {"role": "assistant", "content": "action"},
    ]

    assert _agent_trace_rows(7, messages, _WordTokenizer()) == []


def test_lepard_samples_pairs_before_deduplicating_documents():
    context_a = "A" * 60
    context_b = "B" * 60
    context_c = "C" * 60
    rows = [
        ("d1", context_a, "p1"),
        ("d1", context_a, "p1"),
        ("d1", context_a, "p2"),
        ("d2", context_b, "p1"),
        ("d3", context_c, "missing"),
    ]
    passages = {"p1": "shared passage", "p2": "shared passage"}

    pairs = _sample_lepard_pairs(rows, passages, 10)
    contexts, passage_rows = _lepard_documents(pairs)

    assert len(pairs) == 3
    assert {row["destination_context"]: row["cited_passage_ids"]
            for row in contexts} == {
        context_a: ["p1", "p2"],
        context_b: ["p1"],
    }
    assert passage_rows == [
        {"id": "lp0", "passage_text": "shared passage",
         "passage_ids": ["p1", "p2"]},
    ]


def test_lepard_pair_sample_is_stable_and_nested():
    rows = [(f"d{i}", f"context {i} " + "x" * 50, f"p{i}")
            for i in range(20)]
    passages = {f"p{i}": f"passage {i}" for i in range(20)}

    small = _sample_lepard_pairs(rows, passages, 5)
    large = _sample_lepard_pairs(reversed(rows), passages, 10)

    assert small == large[:5]

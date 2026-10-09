"""Trace tables and the CPU compile of trace analytics queries."""

import json

import pyarrow as pa
import pytest
from fakes import letter_tokens

import quail
from quail.logical import CompileError
from quail.planner.plan import EngineConfig, Refusal
from quail.traces import message_text, register_traces, trace_tables, transcript

CALL = {"id": "call-1", "type": "function",
        "function": {"name": "find_order", "arguments": '{"order": "#W1"}'}}

TRACES = [
    {
        "id": "t1", "task_id": "task-7", "trial": 0, "reward": 1.0,
        "messages": [
            {"role": "system", "content": "Policy: be brief."},
            {"role": "user", "content": "I want to return my order."},
            {"role": "assistant", "content": None, "tool_calls": [CALL]},
            {"role": "tool", "tool_call_id": "call-1", "content": "shipped"},
            {"role": "assistant", "content": "It has shipped; returns open "
                                             "after delivery."},
            {"role": "user", "content": "That is not what I asked."},
            {"role": "assistant", "content": "Sorry. I can cancel it now."},
        ],
    },
    {
        "id": "t2", "task_id": "task-7", "trial": 1, "reward": 0.0,
        "messages": [
            {"role": "user", "content": "I want to return my order."},
            {"role": "assistant", "content": "Done."},
        ],
    },
]


def test_message_text_renders_content_and_tool_calls():
    assert message_text({"role": "assistant", "content": None,
                         "tool_calls": [CALL]}) == (
        '[tool call] find_order({"order": "#W1"})')
    assert message_text({"role": "assistant", "content": "Looking.",
                         "tool_calls": [CALL]}).startswith("Looking.\n[tool")
    assert message_text({"role": "user", "content": [{"type": "text"}]}) == (
        '[{"type":"text"}]')


def test_transcript_keeps_roles_and_cuts_long_messages():
    messages = TRACES[0]["messages"]
    text = transcript(messages)
    assert text.startswith("[USER]\nI want to return my order.\n\n[ASSISTANT]")
    assert "Policy" not in text
    assert "[TOOL]\nshipped" in text
    assert "[SYSTEM]\nPolicy" in transcript(messages, roles=("system",))
    cut = transcript(messages, max_message_chars=5)
    assert "[USER]\nI wan\n[cut 21 characters]" in cut


def test_trace_tables_link_each_message_to_the_ones_before_it():
    tables = trace_tables(TRACES, keep=("task_id", "trial", "reward"))
    traces = tables["traces"].to_pylist()
    assert [row["id"] for row in traces] == ["t1", "t2"]
    assert traces[0]["request"] == "I want to return my order."
    assert traces[0]["message_count"] == 7
    assert traces[0]["task_id"] == "task-7" and traces[0]["reward"] == 1.0
    assert traces[0]["transcript"].startswith("[USER]")
    assert tables["traces"].schema.field("trial").type == pa.int64()

    messages = tables["messages"].to_pylist()
    assert [row["id"] for row in messages[:3]] == ["t1/0", "t1/1", "t1/2"]
    assert [row["role"] for row in messages[:7]] == [
        "system", "user", "assistant", "tool", "assistant", "user",
        "assistant"]
    pushback = messages[5]
    assert pushback["content"] == "That is not what I asked."
    assert pushback["prev_id"] == "t1/4"
    assert pushback["prev_assistant_id"] == "t1/4"
    assert pushback["prev_user_id"] == "t1/1"
    assert messages[0]["prev_id"] is None
    assert messages[3]["tool_call_id"] == "call-1"
    assert messages[7]["id"] == "t2/0" and messages[7]["prev_id"] is None
    assert tables["messages"].schema.field("turn_index").type == pa.int32()


def test_trace_tables_read_arrow_tables_and_json_messages():
    table = pa.table({
        "trace": ["a", "b"],
        "trial": pa.array([0, 1], pa.int32()),
        "history": [json.dumps(row["messages"]) for row in TRACES],
    })
    tables = trace_tables(table, id_col="trace", messages_col="history",
                          keep=("trial",))
    assert tables["traces"].column("id").to_pylist() == ["a", "b"]
    assert tables["traces"].schema.field("trial").type == pa.int32()
    assert tables["messages"].num_rows == 9


def test_trace_tables_reject_bad_input():
    with pytest.raises(CompileError, match="not in trace table schema"):
        trace_tables(pa.table({"id": ["a"]}))
    with pytest.raises(CompileError, match="lacks"):
        trace_tables([{"id": "a"}])
    with pytest.raises(CompileError, match="repeats"):
        trace_tables([{"id": "a", "messages": []},
                      {"id": "a", "messages": []}])
    with pytest.raises(CompileError, match="'role'"):
        trace_tables([{"id": "a", "messages": [{"content": "x"}]}])
    with pytest.raises(CompileError, match="list of messages"):
        trace_tables([{"id": "a", "messages": {"role": "user"}}])
    with pytest.raises(CompileError, match="clashes"):
        trace_tables(TRACES, keep=("request",))


INTENT = ("What is the customer asking the agent to do?\n\n{0}")
PUSHBACK = ("The agent said:\n{0}\n\nThe customer replied:\n{1}\n\n"
            "Does the customer disagree with or correct the agent?")
DISAGREEMENT = ("The agent said:\n{0}\n\nThe customer replied:\n{1}\n\n"
                "What is the disagreement about?")
DIFFERENT = ("Run A:\n{0}\n\nRun B:\n{1}\n\nDo the two runs take different "
             "approaches to the same request?")
APPROACH = "Which approach does the agent take in this run?\n\n{0}"

ANALYTICS_SQL = {
    "intents": f"""
        SELECT AI.CLASSIFY(PROMPT('{INTENT}', t.request),
                           ARRAY['return', 'exchange', 'cancel', 'other'])
                   AS intent,
               COUNT(*) AS n
        FROM support_traces t
        GROUP BY intent
        ORDER BY n DESC
    """,
    "pushback": f"""
        SELECT u.id, a.id
        FROM support_messages u
        JOIN support_messages a
          ON u.prev_assistant_id = a.id
         AND u.role = 'user'
         AND AI.IF(PROMPT('{PUSHBACK}', a.content, u.content))
    """,
    "disagreement_kinds": f"""
        SELECT u.id,
               AI.CLASSIFY(PROMPT('{DISAGREEMENT}', a.content, u.content),
                           ARRAY['policy', 'wrong item', 'tone', 'other'])
                   AS kind
        FROM support_messages u
        JOIN support_messages a
          ON u.prev_assistant_id = a.id
         AND u.role = 'user'
         AND AI.IF(PROMPT('{PUSHBACK}', a.content, u.content))
    """,
    "different_runs": f"""
        SELECT r1.id, r2.id
        FROM support_traces r1
        JOIN support_traces r2
          ON r1.task_id = r2.task_id
         AND AI.IF(PROMPT('{DIFFERENT}', r1.transcript, r2.transcript))
    """,
    "effective_approaches": f"""
        SELECT AI.CLASSIFY(PROMPT('{APPROACH}', t.transcript),
                           ARRAY['refund', 'replace', 'escalate']) AS approach,
               AVG(t.reward) AS success,
               COUNT(*) AS n
        FROM support_traces t
        GROUP BY approach
        HAVING n >= 2
    """,
}


@pytest.mark.parametrize("name", sorted(ANALYTICS_SQL))
def test_trace_analytics_queries_compile_and_plan(name):
    with quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
                       tokenizer=letter_tokens) as session:
        tables = register_traces(session, TRACES, prefix="support_",
                                 keep=("task_id", "trial", "reward"))
        assert set(session.catalog.providers) == {
            "support_traces", "support_messages"}
        assert tables["messages"].num_rows == 9
        query = session.sql(ANALYTICS_SQL[name], dialect="bq")
        plan = query.plan()
        assert not isinstance(plan, Refusal), plan
        assert "physical:" in query.explain()

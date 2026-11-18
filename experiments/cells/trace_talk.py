"""Run the "Trace Analytics in Quail" talk demos over SWE-chat coding-agent sessions.

The download step copies the SALT-NLP/SWE-chat tables to the quail-results
volume under /talk/swe-chat/, using the hf-token Modal secret; the dataset is
gated, so the token's account must accept its terms first. The run step
flattens the raw tables with plain SQL in Quail, then runs each demo query on
one H100 and saves its rows under /talk/swe-chat/demo_results_v2/<model>/.

    uv run modal run -m experiments.cells.trace_talk --download-tables
    uv run modal run --detach -m experiments.cells.trace_talk \
      --only q1,q6 --model qwen3.8-27b-fp8 2>&1 | tee "$run_log"

Demos q1 to q3 are user analytics, q4 to q7 agent behavior analytics; a name
ending in _count or _rows is a companion query of the same demo.
"""

from __future__ import annotations

import json
import time

import modal

from quail.deploy.images import cpu_image, gpu_image
from quail.specs import modal_gpu_type

app = modal.App("quail-milestone1")
image = gpu_image()
results_vol = modal.Volume.from_name("quail-results")
VOLUMES = {
    "/root/.cache/huggingface": modal.Volume.from_name(
        "quail-hf-cache", create_if_missing=True),
    "/root/.cache/kernels": modal.Volume.from_name(
        "quail-kernel-cache", create_if_missing=True),
    "/results": results_vol,
}
RAW = "/results/talk/swe-chat"
OUT = "/results/talk/swe-chat/demo_results_v2"
TABLES = ("sessions.parquet", "conversations.parquet")
PUSHBACK = "('correction', 'failure_report', 'rejection')"
INTENTS = ("['fix a bug', 'build a feature', 'refactor', 'review code', "
           "'plan or design', 'explain code', 'git or release', 'set up tools', "
           "'write tests', 'other']")

# plain SQL over the raw tables; each result is registered under its name
FLATTEN_SQL = {
    "demo_sessions": ("session_id", f"""
SELECT s.session_id, s.agent, s.created_at, s.agent_percentage,
       COALESCE(p.pushbacks, 0) AS pushbacks
FROM sessions s
LEFT JOIN (SELECT session_id, COUNT(*) AS pushbacks FROM conversations
           WHERE prompt_pushback IN {PUSHBACK} GROUP BY session_id) p
  ON p.session_id = s.session_id
WHERE s.agent IN ('Claude Code', 'Codex') AND s.agent_percentage IS NOT NULL
  AND s.created_at >= TIMESTAMP '2026-05-01T00:00:00Z'
ORDER BY right(s.session_id, 6)
LIMIT 1000"""),
    "messages": ("turn_id", """
SELECT c.turn_id, c.session_id, c.turn_number, c.content_block_index AS block,
       CASE WHEN c.role = 'user' THEN 'user' ELSE 'agent' END AS speaker,
       c.prompt_intent, left(c.content, 4000) AS content
FROM conversations c JOIN demo_sessions d ON d.session_id = c.session_id
WHERE (c.role = 'user' AND c.turn_type = 'user_prompt')
   OR (c.role = 'assistant' AND c.turn_type = 'assistant_response')"""),
    "tool_calls": ("turn_id", """
SELECT c.turn_id, c.session_id, c.turn_number, c.content_block_index AS block,
       c.tool_name, c.tool_call_id, c.bash_category,
       left(c.command, 2000) AS command, left(c.content, 2000) AS content
FROM conversations c JOIN demo_sessions d ON d.session_id = c.session_id
WHERE c.role = 'tool_use'"""),
    "tool_results": ("turn_id", """
SELECT c.turn_id, c.session_id, c.turn_number, c.tool_call_id,
       right(c.content, 2000) AS output
FROM conversations c JOIN demo_sessions d ON d.session_id = c.session_id
WHERE c.role = 'tool_result'"""),
}


USER_REPLIES_TEMPLATE = """ordered AS (
  SELECT session_id, speaker, turn_number,
         CASE WHEN content LIKE '%My request for Codex:%'
              THEN split_part(content, 'My request for Codex:', 2)
              ELSE content END AS content,
         LAG(content) OVER (PARTITION BY session_id
                            ORDER BY turn_number, block) AS previous_message,
         LAG(speaker) OVER (PARTITION BY session_id
                            ORDER BY turn_number, block) AS previous_speaker
  FROM messages
  WHERE right(session_id, 1) IN ({sample})),
user_replies AS (
  SELECT session_id, previous_message, content,
         ROW_NUMBER() OVER (PARTITION BY session_id
                            ORDER BY turn_number) AS reply_index
  FROM ordered
  WHERE speaker = 'user' AND previous_speaker = 'agent'
    AND content NOT LIKE '%AGENTS.md%' AND content NOT LIKE '<%'
    AND content NOT LIKE '/%' AND content NOT LIKE '$%')"""


def user_replies_sql(sample: str) -> str:
    """Return the CTE of each user reply with the agent message before it.

    Args:
        sample: Quoted last characters of the session ids to keep, such
            as "'0', '8'" for 1 in 8 sessions.
    """
    return USER_REPLIES_TEMPLATE.format(sample=sample)


EIGHTH = "'0', '8'"
HALF = "'0', '2', '4', '6', '8', 'a', 'c', 'e'"
FRUSTRATED_PROMPT = ("'Agent: {0} User: {1} Is the user frustrated or angry "
                     "with the agent?'")
CORRECTS_PROMPT = ("'Agent: {0} User: {1} Does the user correct the agent or say "
                   "the agent got something wrong?'")
TAKEOVER_PROMPT = ("'Agent: {0} User: {1} Does the user say they changed the code "
                   "themselves, or will change it themselves, instead of the agent? "
                   "Merging, pushing, or setting tokens does not count.'")
CLAIMS_PASS_PROMPT = ("'Does the agent say all the tests pass or the build "
                      "succeeded? {0}'")
ENDING_PROMPT = ("'In one sentence, say what the agent did last and whether "
                 "the task looks finished.'")
REPRODUCES_PROMPT = ("'Bug report: {0} Commands the agent ran before its first "
                     "edit: {1} Does any command reproduce the bug?'")
STUCK_PROMPT = ("'Previous tool call: {0} Current tool call: {1} Does the current "
                "call retry the previous call without a new approach?'")
FAILING_SQL = ("regexp_like(right(output, 300), '(--- FAIL|(?m)^FAIL|FAILED|"
               "Traceback|BUILD FAILED|[(]fail[)]| [1-9][0-9]* failed|"
               "error.E[0-9]|error:)')")
TEST_PAIRS_SQL = """test_outputs AS (
  SELECT r.session_id, r.turn_number, r.output
  FROM tool_calls u JOIN tool_results r ON r.tool_call_id = u.tool_call_id
  WHERE u.bash_category = 'test/build'),
replies AS (
  SELECT t.session_id, t.turn_number, t.output, MIN(m.turn_number) AS reply_turn
  FROM test_outputs t
  JOIN messages m ON m.session_id = t.session_id AND m.speaker = 'agent'
   AND m.turn_number > t.turn_number
  GROUP BY t.session_id, t.turn_number, t.output),
pairs AS (
  SELECT r.session_id, r.output, m.content AS reply
  FROM replies r
  JOIN messages m ON m.session_id = r.session_id AND m.turn_number = r.reply_turn
   AND m.speaker = 'agent' AND m.block = 0),
unique_pairs AS (
  SELECT MIN(session_id) AS session_id, output, reply
  FROM pairs GROUP BY output, reply)"""

DEMO_SQL = {
    "q1": f"""
WITH {user_replies_sql(EIGHTH)}
SELECT session_id, right(previous_message, 200) AS agent_said,
       left(content, 250) AS user_replied
FROM user_replies
WHERE AI.IF(PROMPT({FRUSTRATED_PROMPT}, previous_message, content))""",
    "q1_count": f"""
WITH {user_replies_sql(EIGHTH)}
SELECT COUNT(*) AS user_replies, COUNT(DISTINCT session_id) AS sessions,
       SUM(CASE WHEN AI.IF(PROMPT({FRUSTRATED_PROMPT}, previous_message, content))
                THEN 1 ELSE 0 END) AS frustrated
FROM user_replies""",
    "q2": f"""
WITH {user_replies_sql(EIGHTH)},
first_prompts AS (
  SELECT session_id, content,
         ROW_NUMBER() OVER (PARTITION BY session_id
                            ORDER BY turn_number) AS prompt_index
  FROM messages
  WHERE speaker = 'user' AND right(session_id, 1) IN ({EIGHTH})
    AND content NOT LIKE '%AGENTS.md%' AND content NOT LIKE '<%'
    AND content NOT LIKE 'Caveat:%'),
corrections AS (
  SELECT session_id, COUNT(*) AS num_corrections FROM user_replies
  WHERE AI.IF(PROMPT({CORRECTS_PROMPT}, previous_message, content))
  GROUP BY session_id),
intents AS (
  SELECT session_id, AI.CLASSIFY(content, {INTENTS}) AS intent
  FROM first_prompts WHERE prompt_index = 1)
SELECT i.intent, COUNT(*) AS sessions,
       AVG(COALESCE(c.num_corrections, 0)) AS corrections_per_session,
       AVG(s.pushbacks) AS dataset_pushbacks_per_session
FROM intents i
LEFT JOIN corrections c ON c.session_id = i.session_id
JOIN demo_sessions s ON s.session_id = i.session_id
GROUP BY i.intent ORDER BY corrections_per_session DESC""",
    "q2_rows": f"""
WITH {user_replies_sql(EIGHTH)}
SELECT session_id, right(previous_message, 200) AS agent_said,
       left(content, 200) AS user_replied
FROM user_replies
WHERE AI.IF(PROMPT({CORRECTS_PROMPT}, previous_message, content))
LIMIT 15""",
    "q3": f"""
WITH {user_replies_sql(HALF)},
takeovers AS (
  SELECT session_id, COUNT(*) AS takeover_replies FROM user_replies
  WHERE length(content) < 400
    AND AI.IF(PROMPT({TAKEOVER_PROMPT}, right(previous_message, 1500), content))
  GROUP BY session_id),
sampled AS (SELECT * FROM demo_sessions WHERE right(session_id, 1) IN ({HALF}))
SELECT t.session_id IS NOT NULL AS took_over, COUNT(*) AS sessions,
       median(s.agent_percentage) AS median_agent_share
FROM sampled s LEFT JOIN takeovers t ON t.session_id = s.session_id
GROUP BY t.session_id IS NOT NULL""",
    "q3_rows": f"""
WITH {user_replies_sql(HALF)}
SELECT session_id, right(previous_message, 200) AS agent_said,
       left(content, 250) AS user_replied
FROM user_replies
WHERE length(content) < 400
  AND AI.IF(PROMPT({TAKEOVER_PROMPT}, right(previous_message, 1500), content))""",
    "q4": f"""
WITH last_replies AS (
  SELECT session_id, content,
         ROW_NUMBER() OVER (PARTITION BY session_id
                            ORDER BY turn_number DESC, block DESC) AS rn
  FROM messages WHERE speaker = 'agent')
SELECT s.session_id, s.agent, s.agent_percentage, s.pushbacks,
       AI.MAP(l.content, {ENDING_PROMPT}, 40) AS ending
FROM demo_sessions s JOIN last_replies l ON l.session_id = s.session_id AND l.rn = 1
WHERE s.agent_percentage < 20 AND s.pushbacks >= 3""",
    "q5": f"""
WITH bug_reports AS (
  SELECT session_id, content AS request,
         ROW_NUMBER() OVER (PARTITION BY session_id
                            ORDER BY turn_number) AS prompt_index
  FROM messages WHERE speaker = 'user' AND prompt_intent = 'debug'),
first_edits AS (
  SELECT session_id, MIN(turn_number) AS first_edit FROM tool_calls
  WHERE tool_name IN ('Edit', 'edit', 'Write', 'write', 'apply_patch')
  GROUP BY session_id),
before_edit AS (
  SELECT c.session_id,
         left(string_agg(left(c.command, 200), ' ; '
                         ORDER BY c.turn_number), 2000) AS commands
  FROM tool_calls c JOIN first_edits f ON f.session_id = c.session_id
  WHERE c.command IS NOT NULL AND c.turn_number < f.first_edit
  GROUP BY c.session_id),
judged AS (
  SELECT b.session_id,
         AI.IF(PROMPT({REPRODUCES_PROMPT},
                      b.request, COALESCE(e.commands, 'none'))) AS reproduced_first
  FROM bug_reports b
  JOIN first_edits f ON f.session_id = b.session_id
  LEFT JOIN before_edit e ON e.session_id = b.session_id
  WHERE b.prompt_index = 1)
SELECT j.reproduced_first, COUNT(*) AS sessions,
       median(s.agent_percentage) AS median_agent_share,
       AVG(s.pushbacks) AS pushbacks_per_session
FROM judged j JOIN demo_sessions s ON s.session_id = j.session_id
GROUP BY j.reproduced_first""",
    "q6": f"""
WITH {TEST_PAIRS_SQL}
SELECT session_id, right(output, 200) AS output_tail, left(reply, 300) AS reply
FROM unique_pairs
WHERE {FAILING_SQL} AND AI.IF(PROMPT({CLAIMS_PASS_PROMPT}, reply))""",
    "q6_count": f"""
WITH {TEST_PAIRS_SQL}
SELECT COUNT(*) AS failing_outputs,
       SUM(CASE WHEN AI.IF(PROMPT({CLAIMS_PASS_PROMPT}, reply))
                THEN 1 ELSE 0 END) AS claims_success
FROM unique_pairs
WHERE {FAILING_SQL}""",
    "q7": f"""
WITH numbered AS (
  SELECT session_id, content,
         ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY turn_number, block) AS step
  FROM tool_calls WHERE right(session_id, 1) IN ({EIGHTH})),
calls AS (
  SELECT session_id, step, content,
         LAG(content) OVER (PARTITION BY session_id ORDER BY step) AS previous_content
  FROM numbered WHERE step <= 30),
stuck AS (
  SELECT session_id, step FROM calls
  WHERE previous_content IS NOT NULL
    AND AI.IF(PROMPT({STUCK_PROMPT},
                     previous_content, content))),
onsets AS (
  SELECT session_id, MIN(step) AS first_stuck_step, COUNT(*) AS stuck_calls
  FROM stuck GROUP BY session_id),
lengths AS (SELECT session_id, MAX(step) AS num_calls FROM calls GROUP BY session_id)
SELECT s.agent_percentage >= 50 AS agent_wrote_most, COUNT(*) AS sessions,
       COUNT(o.session_id) AS sessions_stuck,
       median(o.first_stuck_step) AS median_first_stuck_step,
       AVG(COALESCE(o.stuck_calls, 0)) AS stuck_calls_per_session
FROM lengths l
JOIN demo_sessions s ON s.session_id = l.session_id
LEFT JOIN onsets o ON o.session_id = l.session_id
GROUP BY s.agent_percentage >= 50""",
}


@app.function(image=cpu_image(), secrets=[modal.Secret.from_name("hf-token")],
              volumes={"/results": results_vol}, timeout=3600)
def download() -> str:
    """Copy the SWE-chat tables the demos read to the quail-results volume.

    Returns:
        One line per table with its size.
    """
    import os

    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    lines = []
    for name in TABLES:
        path = hf_hub_download("SALT-NLP/SWE-chat", name, repo_type="dataset",
                               token=token, local_dir=RAW)
        lines.append(f"{name}: {os.path.getsize(path) / 1e9:.2f} GB")
    results_vol.commit()
    return "\n".join(lines)


@app.function(image=image, gpu=modal_gpu_type("h100-sxm"), memory=131072,
              timeout=7200, volumes=VOLUMES)
def run_demos(names: list[str], model: str) -> str:
    """Flatten the raw tables in Quail and run each named demo query.

    Args:
        names: Keys of DEMO_SQL to run, in order.
        model: The model spec name, such as qwen3.8-27b-fp8.

    Returns:
        Each step's time and row count, and up to 30 rows per query.
    """
    import os
    import traceback

    import pyarrow.parquet as pq

    import quail

    os.makedirs(f"{OUT}/{model}", exist_ok=True)
    lines = []
    config = quail.EngineConfig(model=model, device="h100-sxm")
    with quail.Session(config=config) as session:
        session.register("sessions", quail.DocumentProvider.from_parquet(
            f"{RAW}/sessions.parquet", id_col="session_id"))
        session.register("conversations", quail.DocumentProvider.from_parquet(
            f"{RAW}/conversations.parquet", id_col="turn_id"))
        for name, (id_col, sql) in FLATTEN_SQL.items():
            start = time.perf_counter()
            table = session.sql(sql, dialect="bq").collect()
            # Quail names each output column with its table alias
            table = table.rename_columns([c.split(".")[-1] for c in table.column_names])
            session.register(name, quail.DocumentProvider.from_table(
                table, id_col=id_col, identity=name))
            lines.append(f"flatten {name}: {time.perf_counter() - start:.1f} s, "
                         f"{table.num_rows} rows")
        start = time.perf_counter()
        session.sql("SELECT turn_id FROM messages WHERE AI.IF(PROMPT("
                    "'Is the text a question? {0}', content)) LIMIT 1",
                    dialect="bq").collect()
        lines.append(f"model startup: {time.perf_counter() - start:.1f} s")
        for name in names:
            start = time.perf_counter()
            try:
                result = session.sql(DEMO_SQL[name], dialect="bq").run()
                table = result.collect()
            except Exception:
                lines.append(f"\n== {name} FAILED after "
                             f"{time.perf_counter() - start:.1f} s\n"
                             + traceback.format_exc()[-2000:])
                continue
            seconds = time.perf_counter() - start
            pq.write_table(table, f"{OUT}/{model}/{name}.parquet")
            report = {k: result.report.get(k)
                      for k in ("boot_s", "evaluated_documents", "fresh_tokens")}
            lines.append(f"\n== {name}: {seconds:.1f} s, {table.num_rows} rows; "
                         f"{json.dumps(report)}")
            for row in table.slice(0, 30).to_pylist():
                preview = {k: str(v)[:220] for k, v in row.items()}
                lines.append("  " + json.dumps(preview))
    results_vol.commit()
    return "\n".join(lines)


@app.local_entrypoint()
def main(only: str = ",".join(DEMO_SQL), model: str = "qwen3.8-27b-fp8",
         download_tables: bool = False) -> None:
    """Download the tables, or run the demos and print the report.

    Args:
        only: Comma-separated demo names to run.
        model: The model spec name.
        download_tables: Copy the SWE-chat tables to the volume instead.
    """
    if download_tables:
        print(download.remote())
        return
    call = run_demos.spawn(only.split(","), model)
    print(f"function call id: {call.object_id}")
    print(call.get())

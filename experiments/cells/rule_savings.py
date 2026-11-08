r"""Measure what each rewrite rule saves on QUAIL-B tables, with it on and off.

Each focused query uses one rule: R6 push_down_case_guards, R7
push_down_or_guards, R4 push_down_substring_prechecks, R13
push_down_semi_join_fetch, R14 push_down_count_threshold, R5
push_down_projection_into_ai_nodes, R17 merge_repeated_ai_calls, and the
physical rule should_stop_answers_early. A group of rules runs on its own
H100: one Session with every rule on runs the group's queries, then one
Session per rule, with that rule off, runs that rule's queries. Each
Session first runs a warm-up query on copies of 64 reviews whose first
line differs, so no KV it writes serves a timed query.

    uv run modal run --detach experiments/cells/rule_savings.py::compare \
        --scale-factor 0.1 2>&1 | tee "$run_log"

Per query and configuration the cell records seconds from building the
query to its collected rows (planning included, startup excluded), the
engine's wall seconds, questions (the rows each AI operator asked the
model about, summed over operators), fresh and cached tokens, map and
extraction answer tokens, $/query on H100, and a hash of the rows. A
query of several model regions sums each measure over its regions.
Results go to /results/ablations/rules/<run id>/<group>.json.
"""

import hashlib
import json
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from bench.quailb_parallel import DATA_DIR, VOLUMES, app, ensure_data, results_vol
from quail.deploy.images import gpu_image

OUTPUT_DIR = "/results/ablations/rules"
WARMUP_DOCUMENTS = 64
SENTIMENT = "ARRAY['positive', 'negative', 'mixed']"
GOVERNING_LAW = "Which state or country has its law govern this agreement?"
ENDING = "PROMPT('Does this review discuss the ending of the movie?\\n\\n{0}', r.body)"
RECOMMEND = "PROMPT('Does the reviewer recommend the movie?\\n\\n{0}', r.body)"
ACTING = "PROMPT('Does the review praise the acting?\\n\\n{0}', r.body)"
CANCEL = ("PROMPT('Does the customer ask to cancel, return, or exchange "
          "something?\\n\\n{0}', m.content)")
# name -> (rule, SQL, how rows compare: "rows" whole, or "ids" the first column)
QUERIES = {
    "case classify": ("push_down_case_guards", (
        "SELECT r.id, CASE WHEN LENGTH(r.body) > 2000 THEN "
        f"AI.CLASSIFY(r.body, {SENTIMENT}) END AS sentiment FROM reviews r"),
        "rows"),
    "case map": ("push_down_case_guards", (
        "SELECT r.id, CASE WHEN LENGTH(r.body) > 2000 THEN AI.MAP(r.body, "
        "'Name the film the review is about.', 16) END AS film FROM reviews r"),
        "nulls"),
    "case extract": ("push_down_case_guards", (
        "SELECT c.id, CASE WHEN c.token_count < 6000 THEN AI.EXTRACT(c.text, "
        f"{{'law': '{GOVERNING_LAW}'}}) END AS e FROM contracts c"), "rows"),
    "column or filter": ("push_down_or_guards", (
        f"SELECT r.id FROM reviews r WHERE LENGTH(r.body) > 1500 "
        f"OR AI.IF({ENDING})"), "rows"),
    "filter or filter": ("push_down_or_guards", (
        f"SELECT r.id FROM reviews r WHERE AI.IF({RECOMMEND}) "
        f"OR AI.IF({ACTING})"), "rows"),
    "extract like contracts": ("push_down_substring_prechecks", (
        f"SELECT c.id, AI.EXTRACT(c.text, {{'law': '{GOVERNING_LAW}'}}) AS e "
        "FROM contracts c WHERE e.law LIKE '%Delaware%'"), "rows"),
    "extract like reviews": ("push_down_substring_prechecks", (
        "SELECT r.id, AI.EXTRACT(r.body, {'director': 'Which director does the "
        "review name?'}) AS e FROM reviews r "
        "WHERE e.director LIKE '%Hitchcock%'"), "rows"),
    "in subquery": ("push_down_semi_join_fetch", (
        "SELECT t.id FROM support_traces t WHERE t.id IN (SELECT m.trace_id "
        f"FROM support_messages m WHERE m.role = 'user' AND AI.IF({CANCEL}))"),
        "rows"),
    "correlated exists": ("push_down_semi_join_fetch", (
        "SELECT t.id FROM support_traces t WHERE EXISTS (SELECT 1 FROM "
        "support_messages m WHERE m.trace_id = t.id AND m.role = 'user' "
        f"AND AI.IF({CANCEL}))"), "rows"),
    "count threshold": ("push_down_count_threshold", (
        "SELECT m.trace_id FROM support_messages m WHERE m.role = 'user' "
        f"AND AI.IF({CANCEL}) GROUP BY m.trace_id HAVING COUNT(*) >= 2"),
        "rows"),
    "one of three fields": ("push_down_projection_into_ai_nodes", (
        "SELECT x.id, x.e.law AS law FROM (SELECT c.id, AI.EXTRACT(c.text, "
        f"{{'law': '{GOVERNING_LAW}', "
        "'parties': 'Who are the parties to the agreement?', "
        "'term': 'How long does the agreement last?'}) AS e "
        "FROM contracts c) x"), "rows"),
    "repeat across derived table": ("merge_repeated_ai_calls", (
        f"SELECT d.id, d.t, AI.CLASSIFY(d.body, {SENTIMENT}) AS u FROM "
        f"(SELECT r.id, r.body, AI.CLASSIFY(r.body, {SENTIMENT}) AS t "
        "FROM reviews r) d"), "rows"),
    "map prefix test": ("should_stop_answers_early", (
        "SELECT r.id FROM reviews r WHERE AI.MAP(r.body, 'Does the reviewer "
        "recommend the movie? Start with Yes or No, then give the reason in "
        "one sentence.', 64) LIKE 'Yes%'"), "rows"),
    "extract equality test": ("should_stop_answers_early", (
        f"SELECT c.id, AI.EXTRACT(c.text, {{'law': '{GOVERNING_LAW}'}}) AS e "
        "FROM contracts c WHERE e.law = 'Delaware'"), "rows"),
}
# each group runs on its own H100
GROUPS = {
    "guards": ("push_down_case_guards", "push_down_or_guards"),
    "keyed": ("push_down_semi_join_fetch", "push_down_count_threshold",
              "merge_repeated_ai_calls"),
    "extractions": ("push_down_substring_prechecks",
                    "push_down_projection_into_ai_nodes"),
    "answer stops": ("should_stop_answers_early",),
}
image = gpu_image()


def registry_without(rule: str | None):
    """Return the built-in registry with one rule left out, or every rule."""
    from quail.builtins import LOGICAL_RULES, PHYSICAL_RULES, built_in_registry

    return built_in_registry(
        logical_rules=[name for name in LOGICAL_RULES if name != rule],
        physical_rules=[name for name in PHYSICAL_RULES if name != rule])


def row_keys(table, compare: str) -> list[str]:
    """Return one comparable key per row.

    Args:
        table: The query's rows.
        compare: "rows" keys whole rows, "nulls" keys the first column and
            whether each other column is NULL, since map text may vary.
    """
    rows = table.to_pylist()
    if compare == "nulls":
        rows = [{key: value if key == table.column_names[0] else value is None
                 for key, value in row.items()} for row in rows]
    return [json.dumps(row, sort_keys=True, default=str) for row in rows]


def measure(session, sql: str, compare: str, usd_per_hour: float) -> dict:
    """Run one query and return its measurements and row keys."""
    started = time.perf_counter()
    query = session.sql(sql)
    plan_text = query.explain()
    result = query.run()
    table = result.collect()
    seconds = time.perf_counter() - started
    # a query of several model regions reports each region apart
    regions = result.report.get("regions") or [result.report]
    node_metrics = {
        (name if len(regions) == 1 else f"region {index}/{name}"): metrics
        for index, region in enumerate(regions, 1)
        for name, metrics in region.get("node_metrics", {}).items()}
    nodes = node_metrics.values()
    keys = row_keys(table, compare)
    fresh = sum(region.get("fresh_tokens", 0) for region in regions)
    cached = sum(region.get("cached_tokens", 0) for region in regions)
    return {
        "seconds": round(seconds, 3),
        "num_regions": len(regions),
        "wall_s": round(sum(region.get("wall_s", 0.0) for region in regions), 3),
        "questions": sum(node.get("evaluated_documents", 0) for node in nodes),
        "fresh_tokens": fresh, "cached_tokens": cached,
        "input_tokens_per_second": round((fresh + cached) / seconds, 1),
        "answer_tokens": sum(node.get("answer_tokens", 0) for node in nodes),
        "answers_stopped_early": sum(
            node.get("answers_stopped_early", 0) for node in nodes),
        "usd_per_query": seconds / 3600 * usd_per_hour,
        "num_rows": table.num_rows,
        "rows_sha256": hashlib.sha256(
            "\n".join(sorted(keys)).encode()).hexdigest(),
        "node_metrics": node_metrics,
        "plan": plan_text,
        "keys": keys,
    }


def warm_up(session, reviews) -> None:
    """Load the model and capture CUDA graphs on copies of the last reviews."""
    import pyarrow.compute as pc

    import quail

    rows = reviews.slice(len(reviews) - WARMUP_DOCUMENTS)
    rows = rows.set_column(
        rows.schema.get_field_index("body"), "body",
        pc.binary_join_element_wise("Warm-up copy.\n", rows["body"], ""))
    session.register("warmup", quail.DocumentProvider.from_table(rows, id_col="id"))
    session.sql(f"SELECT w.id, AI.CLASSIFY(w.body, {SENTIMENT}) AS s, "
                "AI.MAP(w.body, 'Name the film.', 16) AS f FROM warmup w "
                "WHERE AI.IF(PROMPT('Is it long?\\n\\n{0}', w.body))").run().collect()


@app.function(image=image, gpu="H100!", memory=98304, timeout=4 * 3600,
              volumes=VOLUMES)
def run_group(group: str, scale_factor: float, run_dir: str) -> str:
    """Run a group's queries with every rule on, then each rule off, and save them.

    Returns:
        The group's record as JSON, without row keys.
    """
    import traceback

    import pyarrow.parquet as pq

    import quail
    from quail.specs import H100_USD_PER_HOUR

    directory = Path(DATA_DIR) / f"sf{scale_factor}"
    tables = {path.stem: pq.read_table(path)
              for path in sorted(directory.glob("*.parquet"))}
    rules = GROUPS[group]
    names = [name for name, (rule, _, _) in QUERIES.items() if rule in rules]
    config = quail.EngineConfig(model="qwen3-4b-fp8", device="h100-sxm")
    record = {"group": group, "scale_factor": scale_factor,
              "usd_per_hour": H100_USD_PER_HOUR, "queries": {}}
    for configuration in ("on", *rules):
        off = None if configuration == "on" else configuration
        with quail.Session(config, registry=registry_without(off)) as session:
            warm_up(session, tables["reviews"])
            for name, table in tables.items():
                session.register(name, quail.DocumentProvider.from_table(
                    table, id_col="id"))
            for name in names:
                rule, sql, compare = QUERIES[name]
                if off is not None and rule != off:
                    continue
                entry = record["queries"].setdefault(
                    name, {"rule": rule, "sql": sql})
                label = "on" if off is None else "off"
                try:
                    entry[label] = measure(session, sql, compare, H100_USD_PER_HOUR)
                except Exception:
                    entry[label] = {"error": traceback.format_exc()}
                print(group, name, label, {key: value for key, value in
                      entry[label].items()
                      if key not in ("keys", "node_metrics", "plan")}, flush=True)
    for entry in record["queries"].values():
        on, off = entry.get("on", {}), entry.get("off", {})
        if "keys" in on and "keys" in off:
            left, right = Counter(on.pop("keys")), Counter(off.pop("keys"))
            entry["rows_equal"] = left == right
            entry["num_rows_differing"] = sum(((left - right) + (right - left))
                                              .values())
    path = Path(run_dir) / f"{group.replace(' ', '_')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, default=str))
    results_vol.commit()
    return json.dumps(record, default=str)


@app.local_entrypoint()
def compare(scale_factor: float = 0.1, groups: str = ",".join(GROUPS)):
    # these QUAIL-B queries bring the reviews, contracts, and support tables
    ensure_data.remote(scale_factor, ["IMDB-20", "CUAD-21", "SUPPORT-1",
                                      "SUPPORT-4"], "")
    run_id = (f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
              f"{uuid.uuid4().hex[:8]}")
    run_dir = f"{OUTPUT_DIR}/{run_id}"
    print(f"run directory: {run_dir}", flush=True)
    calls = [run_group.spawn(group, scale_factor, run_dir)
             for group in filter(None, groups.split(","))]
    for group, call in zip(groups.split(","), calls):
        print(f"function call id: {call.object_id} ({group})", flush=True)
    for call in calls:
        print(call.get(), flush=True)

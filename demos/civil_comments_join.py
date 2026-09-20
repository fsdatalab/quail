"""Filter Civil Comments for toxicity, then join each with 31 fields.

The data is the Jigsaw Civil Comments release mirrored on Hugging Face
as pietrolesci/civilcomments-wilds (config "raw"). A score is the
fraction of annotators who applied the label, so a score of at least
0.5 is the label: a majority chose Toxic or Very Toxic, not Slightly
toxic or hard to say. The filter prompt uses the ConversationAI rater
definitions and their calibration examples.

The query keeps the toxic comments, then joins each one with a table of
31 statements: six toxicity subtypes, twenty-four identity groups, and
moderator rejection. Each comment is read once. The filter question and
the 31 statements are short suffixes on that comment's KV.

Run DiffusionGemma 26B-A4B fp8 on Modal H100 GPUs from the repository
root. --limit 0 uses every comment.

    uv run modal run --detach demos/civil_comments_join.py \
      --limit 20000 \
      2>&1 | tee /tmp/quail-civil-comments.log

Inputs, answers, and timing are saved under
/results/demos/civil-comments/<run-id> on the quail-results volume.
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from pathlib import Path

import modal
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from quail.bench.images import gpu_image

DATASET = "pietrolesci/civilcomments-wilds"
DATASET_REVISION = "c227534cc0a34cf21db6a0bf0edd0f9050c8b305"
DATASET_FILE = "raw/train-00000-of-00001.parquet"
MODEL = "diffusion-gemma-26b-a4b-fp8"
DEVICE = "h100-sxm"

FILTER_PROMPT = (
    "Rate the comment with these rater definitions. Very toxic: a very "
    "hateful, aggressive, or disrespectful comment, or otherwise very "
    "likely to make a user leave a discussion or give up on sharing "
    "their perspective. Toxic: a rude, disrespectful, or unreasonable "
    "comment, or otherwise somewhat likely to make a user leave a "
    "discussion or give up on sharing their perspective. Not toxic: a "
    "civil, polite, or merely opinionated comment that is unlikely to "
    "make someone leave.\n\n"
    "Answer TRUE only if the comment is toxic or very toxic. Answer "
    "FALSE if it is not toxic, only slightly toxic, or hard to say.\n\n"
    "Example: Peter is an idiot.\nTRUE\n"
    "Example: That is fucking amazing! thanks for sharing.\nFALSE\n"
    "Example: Agree\nFALSE\n\n"
    "{0}\n\n"
    "Instruction: answer TRUE or FALSE.")

FIELDS = {
    "severe_toxicity": "is very hateful, aggressive, or disrespectful, or "
                       "otherwise very likely to make a user leave",
    "obscene": "contains swear words, curse words, or other obscene or "
               "profane language",
    "threat": "describes a wish or intention for pain, injury, or "
              "violence against a person or group",
    "insult": "is an insulting, inflammatory, or negative comment toward "
              "a person or a group of people",
    "identity_attack": "is a negative, discriminatory, or hateful comment "
                       "about people based on identity",
    "sexual_explicit": "contains references to sexual acts or body parts "
                       "in a sexual way, or other lewd content",
    "male": "explicitly mentions men or boys",
    "female": "explicitly mentions women or girls",
    "transgender": "explicitly mentions transgender people",
    "other_gender": "names a specific gender identity that is not male, "
                    "female, or transgender",
    "heterosexual": "explicitly mentions heterosexual people",
    "homosexual_gay_or_lesbian": "explicitly mentions gay or lesbian people",
    "bisexual": "explicitly mentions bisexual people",
    "other_sexual_orientation": "names a specific sexual orientation that "
                                "is not heterosexual, gay, lesbian, or "
                                "bisexual",
    "christian": "explicitly mentions Christians",
    "jewish": "explicitly mentions Jewish people",
    "muslim": "explicitly mentions Muslims",
    "hindu": "explicitly mentions Hindus",
    "buddhist": "explicitly mentions Buddhists",
    "atheist": "explicitly mentions atheists",
    "other_religion": "names a specific religion that is not Christianity, "
                      "Judaism, Islam, Hinduism, Buddhism, or atheism",
    "black": "explicitly mentions Black people",
    "white": "explicitly mentions white people",
    "asian": "explicitly mentions Asian people",
    "latino": "explicitly mentions Latino people",
    "other_race_or_ethnicity": "names a specific race or ethnicity that is "
                               "not Black, white, Asian, or Latino",
    "physical_disability": "explicitly mentions people with a physical "
                           "disability",
    "intellectual_or_learning_disability": "explicitly mentions people "
                                           "with an intellectual or "
                                           "learning disability",
    "psychiatric_or_mental_illness": "explicitly mentions people with a "
                                     "psychiatric or mental illness",
    "other_disability": "names a specific disability that is not physical, "
                        "intellectual, or psychiatric",
    "rejected": "would actually be removed by a news-site moderator, not "
                "merely because it is rude or unpopular",
}

# Measured on the full table: 11.3% of comments are toxic, and among
# those a joined field is true 6.7% of the time on average.
JOIN_PROMPT = (
    "Judge only the statement in DOCUMENT {1} about the comment in "
    "DOCUMENT {0}. A toxic comment is not automatically an insult, a "
    "threat, or an identity attack. Answer TRUE only if that statement "
    "is specifically true. Answer FALSE if it is only loosely related "
    "or is a stronger claim than the comment supports.")

app = modal.App("quail-milestone1")
results_volume = modal.Volume.from_name("quail-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache", create_if_missing=True)
image = gpu_image(("demos", "/root/demos")).add_local_python_source("demos")


def build_sql(filter_only: bool = False) -> str:
    """Return the query, or the toxicity filter alone."""
    join = "" if filter_only else f"""
    JOIN fields f
      ON AI_FILTER(PROMPT('{JOIN_PROMPT}', c.text, f.statement),
                   {{'selectivity': 0.067}})"""
    return f"""
    SELECT c.comment_id{"" if filter_only else ", f.field"}
    FROM comments c{join}
    WHERE AI_FILTER(PROMPT('{FILTER_PROMPT}', c.text), {{'selectivity': 0.113}})
"""


def own_answers(result):
    """Return an answer(prompt, assignment) callable replaying a run's answers."""
    filters, joins = {}, {}
    for (alias, _), table in result.answer_tables["filters"].items():
        for row in table.to_pylist():
            filters[(alias, row[alias])] = bool(row["answer"])
    for table in result.answer_tables["joins"].values():
        aliases = [name for name in table.column_names if name != "answer"]
        for row in table.to_pylist():
            key = tuple(sorted((alias, row[alias]) for alias in aliases))
            joins[key] = bool(row["answer"])

    def answer(prompt, assignment):
        if len(prompt.args) == 1:
            alias = prompt.args[0].alias
            return filters.get((alias, assignment[alias]), False)
        return joins.get(tuple(sorted(assignment.items())), False)

    return answer


def load_comments(limit: int | None) -> pa.Table:
    """Return the comments with a string id and a 0/1 rejected score."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, DATASET_FILE, repo_type="dataset",
                           revision=DATASET_REVISION)
    table = pq.read_table(path)
    table = table.filter(pc.is_valid(table["comment_text"]))
    if limit is not None:
        # the file is not shuffled: its first rows are mostly toxic
        rows = np.random.default_rng(0).choice(table.num_rows, limit,
                                               replace=False)
        table = table.take(np.sort(rows))
    return (table
            .append_column("comment_id", pc.cast(table["id"], pa.string()))
            .append_column("text", table["comment_text"])
            .append_column("rejected", pc.cast(
                pc.equal(table["rating"], "rejected"), pa.float64())))


def labeled_pairs(comments: pa.Table) -> set:
    """(comment_id, field) pairs whose toxicity and field scores are >= 0.5."""
    ids = comments["comment_id"].to_pylist()
    toxic = np.asarray(comments["toxicity"].to_pylist()) >= 0.5
    pairs = set()
    for field in FIELDS:
        scores = np.asarray(comments[field].to_pylist()) >= 0.5
        for index in np.flatnonzero(toxic & scores):
            pairs.add((ids[int(index)], field))
    return pairs


def requested_input_tokens(session, query, result, filter_only: bool) -> int:
    """Sum full prompt lengths of every evaluated filter and join pair."""
    comment_lengths = np.asarray(session.token_lengths("comments", "text"))
    canvas = session.model.canvas_tokens
    filter_prompt = query.logical.operators().filters["c"][0].prompt
    total = int((filter_prompt.preamble_tokens + comment_lengths
                 + filter_prompt.tail_tokens + canvas).sum())
    if filter_only:
        return total
    join_prompt = query.logical.operators().joins[0].prompt
    labels = {alias: (label, frame)
              for alias, label, frame in join_prompt.labels}
    field_lengths = np.asarray(session.token_lengths("fields", "statement"))
    answers = result.answer_tables["filters"][("c", 0)]
    survivors = answers.filter(answers["answer"])["c"].to_numpy()
    pair_fixed = (join_prompt.preamble_tokens + labels["c"][1]
                  + labels["f"][0] + join_prompt.tail_tokens + canvas)
    field_sum = int(field_lengths.sum())
    n_fields = len(field_lengths)
    for row in survivors:
        total += n_fields * (pair_fixed + int(comment_lengths[int(row)]))
        total += field_sum
    return total


def json_ready(value):
    """Convert numpy scalars so a summary can be written as JSON."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate_tables(directory: Path, limit: int | None, gpus: int,
                    filter_only: bool) -> dict:
    """Run the query and write answers, the plan, and the timing report."""
    import quail
    from quail.specs import H100_USD_PER_HOUR

    comments = load_comments(limit)
    fields = pa.table({
        "field": list(FIELDS),
        "statement": [f"The comment {s}." for s in FIELDS.values()],
    })
    labeled = labeled_pairs(comments)
    ids = comments["comment_id"].to_pylist()
    (directory / "inputs.json").write_text(json.dumps({
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "model": MODEL,
        "device": DEVICE,
        "gpus": gpus,
        "limit": limit,
        "filter_only": filter_only,
        "comments": len(ids),
        "labeled_pairs": len(labeled),
        "sql": build_sql(filter_only),
    }, indent=2))
    pq.write_table(comments.select(["comment_id", "text"]),
                   directory / "comments.parquet")
    with quail.Session(config=quail.EngineConfig(
            model=MODEL, device=DEVICE, gpus=gpus, gpu_timing=True)) as session:
        session.register("comments", quail.DocumentProvider.from_table(
            comments, id_col="comment_id"))
        session.register("fields", quail.DocumentProvider.from_table(
            fields, id_col="field"))
        query = session.sql(build_sql(filter_only))
        explanation = query.explain()
        print(explanation, flush=True)
        (directory / "plan.txt").write_text(explanation)
        result = query.run()
        table = result.collect()
        pq.write_table(table, directory / "retained.parquet")
        report = dict(result.report)
        input_tokens = requested_input_tokens(
            session, query, result, filter_only)
        ideal = quail.speed_of_light_estimate(query, own_answers(result))
    answers = result.answer_tables["filters"][("c", 0)]
    pq.write_table(answers, directory / "filter_answers.parquet")
    toxic_found = {ids[int(i)] for i, yes in zip(
        answers["c"].to_pylist(), answers["answer"].to_pylist()) if yes}
    toxic_labeled = {ids[int(i)] for i in np.flatnonzero(
        np.asarray(comments["toxicity"].to_pylist()) >= 0.5)}
    toxic_hits = len(toxic_found & toxic_labeled)
    found = set() if filter_only else {
        (row["c.comment_id"], row["f.field"]) for row in table.to_pylist()}
    pair_hits = len(found & labeled)
    wall_s = report["wall_s"]
    boot_s = report.get("boot_s", 0.0)
    pairs = 0
    for stage in report.get("stages", ()):
        if stage.get("op") == "filter":
            print(f"  filter evaluated {stage['evaluated']} comments, "
                  f"{stage['observed_selectivity']:.3f} passed", flush=True)
        elif stage.get("op") == "join":
            pairs += stage["tuples"]
            print(f"  join evaluated {stage['tuples']} pairs, "
                  f"{stage['observed_selectivity']:.3f} passed", flush=True)
    field_counts = Counter(field for _, field in found)
    summary = {
        **report,
        "model": MODEL,
        "device": DEVICE,
        "gpus": gpus,
        "comments": len(ids),
        "limit": limit,
        "filter_only": filter_only,
        "input_tokens": input_tokens,
        "input_tokens_per_second": (
            input_tokens / wall_s if wall_s else None),
        "evaluated_pairs": pairs,
        "gpu_cost_usd": wall_s * gpus * H100_USD_PER_HOUR / 3600,
        "gpu_startup_cost_usd": boot_s * gpus * H100_USD_PER_HOUR / 3600,
        "filter_precision": toxic_hits / max(1, len(toxic_found)),
        "filter_recall": toxic_hits / max(1, len(toxic_labeled)),
        "filter_found": len(toxic_found),
        "filter_labeled": len(toxic_labeled),
        "join_precision": None if filter_only else pair_hits / max(1, len(found)),
        "join_recall": None if filter_only else pair_hits / max(1, len(labeled)),
        "join_found": len(found),
        "labeled_pairs": len(labeled),
        "field_counts": dict(field_counts),
        "sol_s": ideal.seconds,
        "sol_fresh_tokens": ideal.fresh_tokens,
        "wall_over_sol": wall_s / ideal.seconds if ideal.seconds else None,
        "sol": ideal.as_dict(),
        "result_volume_path": str(directory),
    }
    print(f"filter precision against the toxicity labels: "
          f"{summary['filter_precision']:.3f}", flush=True)
    print(f"filter recall against the toxicity labels: "
          f"{summary['filter_recall']:.3f}", flush=True)
    if not filter_only:
        print(f"(comment, field) pairs found: {len(found)}", flush=True)
        for field, count in field_counts.most_common():
            print(f"  {field}: {count}", flush=True)
        print(f"precision against the labels: "
              f"{summary['join_precision']:.3f}", flush=True)
        print(f"recall against the labels: "
              f"{summary['join_recall']:.3f}", flush=True)
    print(f"boot_s: {boot_s} ({report.get('boot_kind')})", flush=True)
    print(f"wall_s: {wall_s}", flush=True)
    if "gpu_s" in report:
        idle = wall_s - report["gpu_s"]
        print(f"gpu_s: {report['gpu_s']} busy, {idle:.2f} idle "
              f"({100 * idle / wall_s:.1f}% of wall_s)", flush=True)
    print(f"speed of light for this run's answers: {ideal.seconds:.2f} s, "
          f"{ideal.fresh_tokens:,.0f} fresh tokens; measured wall is "
          f"{summary['wall_over_sol']:.2f}x ideal", flush=True)
    print(f"fresh_tokens: {report.get('fresh_tokens')}", flush=True)
    print(f"input_tokens: {input_tokens}", flush=True)
    if wall_s:
        print(f"input_tokens_per_second: "
              f"{summary['input_tokens_per_second']:.1f}", flush=True)
    print(f"GPU cost/query: ${summary['gpu_cost_usd']:.4f}", flush=True)
    print(f"GPU startup cost: ${summary['gpu_startup_cost_usd']:.4f}",
          flush=True)
    summary = json_ready(summary)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"result volume path: {directory}", flush=True)
    return summary


@app.function(
    image=image, gpu="H100!", timeout=86_400, memory=98_304,
    volumes={
        "/results": results_volume,
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/kernels": kernel_cache,
    },
)
def run(limit: int, gpus: int, filter_only: bool) -> dict:
    """Load the comments and evaluate the query on a Modal GPU."""
    results_volume.reload()
    hf_cache.reload()
    directory = Path("/results/demos/civil-comments") / uuid.uuid4().hex
    directory.mkdir(parents=True)
    sample = None if limit == 0 else limit
    summary = evaluate_tables(directory, sample, gpus, filter_only)
    results_volume.commit()
    hf_cache.commit()
    return summary


@app.local_entrypoint()
def main(limit: int = 20_000, gpus: int = 1, filter_only: bool = False):
    """Run DiffusionGemma on one or more H100s. --limit 0 uses every comment."""
    if limit < 0 or gpus not in (1, 2, 4, 8):
        raise ValueError("limit must be >= 0; gpus must be 1, 2, 4, or 8")
    gpu = "H100!" if gpus == 1 else f"H100!:{gpus}"
    call = run.with_options(gpu=gpu).spawn(limit, gpus, filter_only)
    print(f"function call id: {call.object_id}", flush=True)
    summary = call.get()
    print(json.dumps({
        "wall_s": summary["wall_s"],
        "sol_s": summary["sol_s"],
        "wall_over_sol": summary["wall_over_sol"],
        "fresh_tokens": summary.get("fresh_tokens"),
        "input_tokens": summary["input_tokens"],
        "input_tokens_per_second": summary["input_tokens_per_second"],
        "gpu_cost_usd": summary["gpu_cost_usd"],
        "filter_precision": summary["filter_precision"],
        "filter_recall": summary["filter_recall"],
        "join_precision": summary["join_precision"],
        "join_recall": summary["join_recall"],
        "result_volume_path": summary["result_volume_path"],
    }, indent=2), flush=True)

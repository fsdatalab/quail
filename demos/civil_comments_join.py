"""Filter 448,000 Civil Comments for toxicity, then join each with 31 fields.

The data is the Jigsaw Civil Comments release mirrored on Hugging Face
as pietrolesci/civilcomments-wilds (config "raw"). It has every column
of the Kaggle train file: the text, seven toxicity scores, twenty-four
identity scores, the moderator rating, reader reactions, and thread
ids. A score is the fraction of annotators who applied the label, so a
score of at least 0.5 is the label. The google/civil_comments mirror
keeps only the seven toxicity scores.

The query keeps the toxic comments, then joins each one with a table of
31 statements, one per remaining semantic field: six toxicity subtypes,
twenty-four identity groups, and moderator rejection. Each comment is
read by the model once; the filter question and the 31 statements are
short suffixes on that comment's KV.

Run on a machine with a CUDA GPU:

    uv run python demos/civil_comments_join.py --limit 20000

--parquet takes any file with the same columns instead of the download.
"""

import argparse
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import quail
from quail.specs import DEVICES, MODAL_GPU_USD_PER_HOUR

DATASET = "pietrolesci/civilcomments-wilds"
DATASET_REVISION = "c227534cc0a34cf21db6a0bf0edd0f9050c8b305"
DATASET_FILE = "raw/train-00000-of-00001.parquet"

# Score column -> the statement the model judges. Toxicity is the filter.
TOXIC = ("is rude, disrespectful, or unreasonable enough to make someone "
         "leave the discussion")
FIELDS = {
    "severe_toxicity": "is very hateful, aggressive, or disrespectful, far "
                       "beyond ordinary rudeness",
    "obscene": "uses swear words, curse words, or other obscene language",
    "threat": "states an intention to inflict pain, injury, or violence on "
              "a person or group",
    "insult": "is insulting, inflammatory, or negative toward a person or "
              "group",
    "identity_attack": "is negative or hateful toward people because of "
                       "their identity",
    "sexual_explicit": "refers to sexual acts, body parts, or other lewd "
                       "content",
    "male": "mentions or refers to men or boys",
    "female": "mentions or refers to women or girls",
    "transgender": "mentions or refers to transgender people",
    "other_gender": "mentions or refers to people of another gender identity",
    "heterosexual": "mentions or refers to heterosexual people",
    "homosexual_gay_or_lesbian": "mentions or refers to gay or lesbian people",
    "bisexual": "mentions or refers to bisexual people",
    "other_sexual_orientation": "mentions or refers to people of another "
                                "sexual orientation",
    "christian": "mentions or refers to Christians",
    "jewish": "mentions or refers to Jewish people",
    "muslim": "mentions or refers to Muslims",
    "hindu": "mentions or refers to Hindus",
    "buddhist": "mentions or refers to Buddhists",
    "atheist": "mentions or refers to atheists",
    "other_religion": "mentions or refers to people of another religion",
    "black": "mentions or refers to Black people",
    "white": "mentions or refers to white people",
    "asian": "mentions or refers to Asian people",
    "latino": "mentions or refers to Latino people",
    "other_race_or_ethnicity": "mentions or refers to people of another "
                               "race or ethnicity",
    "physical_disability": "mentions or refers to people with a physical "
                           "disability",
    "intellectual_or_learning_disability": "mentions or refers to people "
                                           "with an intellectual or "
                                           "learning disability",
    "psychiatric_or_mental_illness": "mentions or refers to people with a "
                                     "psychiatric or mental illness",
    "other_disability": "mentions or refers to people with another "
                        "disability",
    "rejected": "would be rejected by the news site moderators",
}

# Measured on the full table: 11.3% of comments are toxic, and among
# those a joined field is true 6.7% of the time on average.
SQL = f"""
    SELECT c.comment_id, f.field
    FROM comments c
    JOIN fields f
      ON AI_FILTER(
           PROMPT('Is the statement in DOCUMENT {{1}} true of DOCUMENT {{0}}?',
                  c.text, f.statement),
           {{'selectivity': 0.067}})
    WHERE AI_FILTER(
            PROMPT('{{0}}

Instruction: answer TRUE if the comment {TOXIC}, FALSE otherwise.', c.text),
            {{'selectivity': 0.113}})
"""


def load_comments(parquet: str | None, limit: int | None) -> pa.Table:
    """Return the comments with a string id and a 0/1 rejected score."""
    if parquet is None:
        from huggingface_hub import hf_hub_download
        parquet = hf_hub_download(DATASET, DATASET_FILE, repo_type="dataset",
                                  revision=DATASET_REVISION)
    table = pq.read_table(parquet)
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
    df = comments.to_pandas()
    toxic = df[df["toxicity"] >= 0.5]
    return {(comment_id, field) for field in FIELDS
            for comment_id in toxic.loc[toxic[field] >= 0.5, "comment_id"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", help="a file with the Kaggle columns")
    parser.add_argument("--limit", type=int, help="random sample of N comments")
    parser.add_argument("--device", choices=sorted(DEVICES), default="h100-sxm")
    parser.add_argument("--gpus", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--gpu-usd-per-hour", type=float,
                        help="hourly price per GPU; defaults to Modal pricing")
    args = parser.parse_args()
    hourly_price = args.gpu_usd_per_hour
    price_source = "custom" if hourly_price is not None else "Modal"
    if hourly_price is None:
        hourly_price = MODAL_GPU_USD_PER_HOUR[args.device]

    t0 = time.perf_counter()
    comments = load_comments(args.parquet, args.limit)
    n_docs = comments.num_rows
    print(f"comments: {n_docs}, loaded in {time.perf_counter() - t0:.1f} s",
          flush=True)
    labeled = labeled_pairs(comments)
    print(f"labeled (comment, field) pairs: {len(labeled)}")

    with quail.Session(
        config=quail.EngineConfig(gpus=args.gpus, device=args.device),
    ) as session:
        session.register("comments", quail.DocumentProvider.from_table(
            comments, id_col="comment_id"))
        session.register("fields", quail.DocumentProvider.from_table(
            pa.table({"field": list(FIELDS),
                      "statement": [f"The comment {s}." for s in FIELDS.values()]}),
            id_col="field"))
        query = session.sql(SQL)
        print(query.explain(), flush=True)

        result = query.run()
        table = result.collect()
        report = result.report
        wall_s = report["wall_s"]
        boot_s = report.get("boot_s", 0.0)
        found = {(row["c.comment_id"], row["f.field"]) for row in table.to_pylist()}
        print(f"(comment, field) pairs found: {len(found)}")
        print(table.to_pandas()["f.field"].value_counts().to_string())
        hits = len(found & labeled)
        print(f"precision against the labels: {hits / max(1, len(found)):.3f}")
        print(f"recall against the labels: {hits / max(1, len(labeled)):.3f}")
        pairs = 0
        for stage in report.get("stages", ()):
            if stage.get("op") == "filter":
                print(f"  filter evaluated {stage['evaluated']} comments, "
                      f"{stage['observed_selectivity']:.3f} passed")
            elif stage.get("op") == "join":
                pairs += stage["tuples"]
                print(f"  join evaluated {stage['tuples']} pairs, "
                      f"{stage['observed_selectivity']:.3f} passed")
        print(f"boot_s: {boot_s} ({report.get('boot_kind')})")
        print(f"wall_s: {wall_s}")
        print(f"total_s: {wall_s + boot_s:.2f} (boot + query)")
        print(f"fresh_tokens: {report.get('fresh_tokens')}")
        print(f"document pairs/second: {pairs / wall_s:.1f}")
        cost_per_second = args.gpus * hourly_price / 3600
        print(f"GPU price: ${hourly_price:.4f}/GPU-hour ({price_source})")
        print(f"GPU cost/query: ${wall_s * cost_per_second:.4f}")
        print(f"GPU startup cost: ${boot_s * cost_per_second:.4f}")


if __name__ == "__main__":
    main()

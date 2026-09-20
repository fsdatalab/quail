"""Shared Civil Comments workload for the Quail and Jev demos."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DATASET = "pietrolesci/civilcomments-wilds"
DATASET_REVISION = "c227534cc0a34cf21db6a0bf0edd0f9050c8b305"
DATASET_FILE = "raw/train-00000-of-00001.parquet"
LABEL_CUTOFF = 0.5

TOXICITY_PROMPT_INSTRUCTIONS = (
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
    "Example: Agree\nFALSE"
)

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

JOIN_PROMPT_INSTRUCTIONS = (
    "Judge only the statement in DOCUMENT {1} about the comment in "
    "DOCUMENT {0}. A toxic comment is not automatically an insult, a "
    "threat, or an identity attack. Answer TRUE only if that statement "
    "is specifically true. Answer FALSE if it is only loosely related "
    "or is a stronger claim than the comment supports."
)

FILTER_PROMPT = (
    "{0}\n\n"
    f"{TOXICITY_PROMPT_INSTRUCTIONS}\n\n"
    "Instruction: answer TRUE or FALSE."
)
JOIN_PROMPT = JOIN_PROMPT_INSTRUCTIONS


def load_comments(limit: int | None) -> pa.Table:
    """Load the deterministic Civil Comments sample."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        DATASET,
        DATASET_FILE,
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    table = pq.read_table(path)
    table = table.filter(pc.is_valid(table["comment_text"]))
    if limit is not None:
        rows = np.random.default_rng(0).choice(
            table.num_rows, limit, replace=False
        )
        table = table.take(np.sort(rows))
    return (
        table
        .append_column("comment_id", pc.cast(table["id"], pa.string()))
        .append_column("text", table["comment_text"])
        .append_column(
            "rejected",
            pc.cast(pc.equal(table["rating"], "rejected"), pa.float64()),
        )
    )


def fields_table() -> pa.Table:
    """Return the 31 field statements."""
    return pa.table({
        "field": list(FIELDS),
        "statement": [f"The comment {value}." for value in FIELDS.values()],
    })


def labeled_outputs(comments: pa.Table) -> tuple[set[str], set[tuple[str, str]]]:
    """Return toxicity labels and labeled comment-field pairs."""
    ids = comments["comment_id"].to_pylist()
    toxic = np.asarray(comments["toxicity"].to_pylist()) >= LABEL_CUTOFF
    toxic_ids = {ids[int(index)] for index in np.flatnonzero(toxic)}
    pairs = set()
    for field in FIELDS:
        scores = np.asarray(comments[field].to_pylist()) >= LABEL_CUTOFF
        for index in np.flatnonzero(toxic & scores):
            pairs.add((ids[int(index)], field))
    return toxic_ids, pairs


def classification_metrics(found: set, labeled: set) -> dict[str, float | int]:
    """Return counts, precision, recall, and F1."""
    hits = len(found & labeled)
    precision = hits / len(found) if found else 0.0
    recall = hits / len(labeled) if labeled else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "found": len(found),
        "labeled": len(labeled),
        "hits": hits,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def accuracy_summary(
    comments: pa.Table,
    toxic_found: set[str],
    pairs_found: set[tuple[str, str]],
) -> dict:
    """Score filter and join outputs against labels at the 0.5 cutoff."""
    toxic_labeled, pairs_labeled = labeled_outputs(comments)
    return {
        "label_cutoff": LABEL_CUTOFF,
        "filter": classification_metrics(toxic_found, toxic_labeled),
        "join": classification_metrics(pairs_found, pairs_labeled),
    }


def requested_input_tokens(
    comments: pa.Table,
    filter_ids: set[str],
    join_ids: set[str],
    model_name: str = "RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic",
) -> int:
    """Count full Quail prompt tokens for evaluated predicates.

    This is the comparison throughput numerator. It counts the complete
    prompt for every evaluated question, including a comment's shared
    prefix each time. Jev's billed tokens are reported separately.
    """
    from gigatoken import Tokenizer

    from quail.logical.prompts import bind_join_prompt, bind_prompt
    from quail.specs.diffusion_gemma_26b import DIFFUSION_GEMMA_26B_FP8

    class Ref:
        def __init__(self, alias):
            self.alias = alias

    tokenizer = Tokenizer(model_name).encode
    model = DIFFUSION_GEMMA_26B_FP8
    filter_prompt = bind_prompt(
        FILTER_PROMPT,
        (Ref("c"),),
        tokenizer,
        turn=model.turn,
    )
    join_prompt = bind_join_prompt(
        JOIN_PROMPT,
        (Ref("c"), Ref("f")),
        tokenizer,
        turn=model.turn,
    )
    texts = comments["text"].to_pylist()
    ids = comments["comment_id"].to_pylist()
    comment_lengths = np.asarray([len(tokenizer(text)) for text in texts])
    labels = {
        alias: (label, frame) for alias, label, frame in join_prompt.labels
    }
    field_lengths = np.asarray([
        len(tokenizer(f"The comment {value}."))
        for value in FIELDS.values()
    ])
    total = 0
    pair_fixed = (
        join_prompt.preamble_tokens
        + labels["c"][1]
        + labels["f"][0]
        + join_prompt.tail_tokens
        + model.canvas_tokens
    )
    field_sum = int(field_lengths.sum())
    for index, comment_id in enumerate(ids):
        if comment_id in filter_ids:
            total += (
                filter_prompt.preamble_tokens
                + int(comment_lengths[index])
                + filter_prompt.tail_tokens
                + model.canvas_tokens
            )
        if comment_id in join_ids:
            total += len(field_lengths) * (
                pair_fixed + int(comment_lengths[index])
            )
            total += field_sum
    return total

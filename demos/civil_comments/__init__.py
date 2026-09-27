"""Shared Civil Comments workload for the Quail and Jev demos."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class FieldPrompt:
    """Define one field and its calibration examples."""

    definition: str
    true_example: str
    false_example: str


FIELDS = {
    "severe_toxicity": FieldPrompt(
        "is extremely hateful, aggressive, or disrespectful, or very likely "
        "to drive someone from the discussion; an ordinary insult alone is "
        "not severe toxicity",
        "You are subhuman filth, and everyone would be better off if you died.",
        "Your argument is foolish and wrong.",
    ),
    "obscene": FieldPrompt(
        "contains a swear word, curse word, or other profane language",
        "This fucking policy is bullshit.",
        "This policy is terrible.",
    ),
    "threat": FieldPrompt(
        "expresses the speaker's wish or intention to cause pain, injury, or "
        "violence; merely reporting violence is not a threat",
        "I will find you and break your legs.",
        "The article reports that the victim was attacked.",
    ),
    "insult": FieldPrompt(
        "directly insults or uses inflammatory negative language toward a "
        "person or group; criticism of an idea alone is not an insult",
        "You are an ignorant fool.",
        "Your estimate is incorrect.",
    ),
    "identity_attack": FieldPrompt(
        "attacks a person or group because of race, religion, gender, "
        "nationality, disability, age, or sexual orientation; a neutral "
        "identity mention is not an attack",
        "Muslims are disgusting parasites.",
        "The mosque serves Muslim families.",
    ),
    "sexual_explicit": FieldPrompt(
        "describes sexual acts or sexual body parts in a sexual or lewd way; "
        "a non-graphic mention of sexual misconduct is not enough",
        "He graphically described grabbing her breasts during sex.",
        "The article discusses sexual harassment policy.",
    ),
    "male": FieldPrompt(
        "mentions male people or the male gender, including men and boys",
        "The men and boys waited outside.",
        "The women waited outside.",
    ),
    "female": FieldPrompt(
        "mentions female people or the female gender, including women and "
        "girls",
        "The women and girls waited outside.",
        "The men waited outside.",
    ),
    "transgender": FieldPrompt(
        "mentions transgender people or a transgender identity",
        "A transgender woman addressed the council.",
        "A gay man addressed the council.",
    ),
    "other_gender": FieldPrompt(
        "mentions a nonbinary, gender-fluid, intersex, or another gender "
        "identity outside male, female, and transgender",
        "A nonbinary person addressed the council.",
        "A transgender person addressed the council.",
    ),
    "heterosexual": FieldPrompt(
        "mentions heterosexual or straight people or orientation",
        "The survey included straight couples.",
        "The survey included lesbian couples.",
    ),
    "homosexual_gay_or_lesbian": FieldPrompt(
        "mentions gay or lesbian people or orientation",
        "Gay and lesbian couples joined the event.",
        "Straight couples joined the event.",
    ),
    "bisexual": FieldPrompt(
        "mentions bisexual people or orientation",
        "She publicly identified as bisexual.",
        "She publicly identified as lesbian.",
    ),
    "other_sexual_orientation": FieldPrompt(
        "mentions an orientation outside heterosexual, gay, lesbian, and "
        "bisexual, such as asexual or pansexual",
        "The speaker identified as asexual.",
        "The speaker identified as bisexual.",
    ),
    "christian": FieldPrompt(
        "mentions Christians or a Christian denomination, including Catholic "
        "or Protestant",
        "The Catholic church held a service.",
        "The synagogue held a service.",
    ),
    "jewish": FieldPrompt(
        "mentions Jewish people, Judaism, or a Jewish institution",
        "Jewish families attended the synagogue.",
        "Christian families attended the church.",
    ),
    "muslim": FieldPrompt(
        "mentions Muslims, Islam, or an Islamic institution",
        "Muslim families attended the mosque.",
        "Hindu families attended the temple.",
    ),
    "hindu": FieldPrompt(
        "mentions Hindus or Hinduism",
        "The Hindu community opened a temple.",
        "The Buddhist community opened a temple.",
    ),
    "buddhist": FieldPrompt(
        "mentions Buddhists or Buddhism",
        "A Buddhist monk spoke at the event.",
        "A Hindu priest spoke at the event.",
    ),
    "atheist": FieldPrompt(
        "mentions atheists, atheism, or people explicitly described as "
        "nonbelievers",
        "Several atheists joined the debate.",
        "Several Christians joined the debate.",
    ),
    "other_religion": FieldPrompt(
        "mentions a religion outside Christianity, Judaism, Islam, Hinduism, "
        "Buddhism, and atheism, such as Sikhism or Mormonism",
        "A Sikh family opened the community center.",
        "A Muslim family opened the community center.",
    ),
    "black": FieldPrompt(
        "mentions Black people or Black racial identity",
        "Black residents organized the meeting.",
        "They bought a black car.",
    ),
    "white": FieldPrompt(
        "mentions white people or white racial identity",
        "White residents organized the meeting.",
        "The White House issued a statement.",
    ),
    "asian": FieldPrompt(
        "mentions Asian people or an Asian ethnicity or nationality, such as "
        "Chinese, Japanese, Korean, or Indian",
        "Chinese and Korean families joined the event.",
        "They served Asian pears at the event.",
    ),
    "latino": FieldPrompt(
        "mentions Latino, Latina, Hispanic, or Latin American people",
        "Latina and Hispanic voters attended.",
        "The author quoted a Latin phrase.",
    ),
    "other_race_or_ethnicity": FieldPrompt(
        "mentions a race or ethnicity outside Black, white, Asian, and "
        "Latino, such as Arab, Indigenous, or Pacific Islander",
        "Native American leaders attended the hearing.",
        "Black leaders attended the hearing.",
    ),
    "physical_disability": FieldPrompt(
        "mentions people with a physical or mobility disability",
        "The building needs access for wheelchair users.",
        "The clinic treats depression.",
    ),
    "intellectual_or_learning_disability": FieldPrompt(
        "mentions an intellectual, developmental, or learning disability, "
        "such as autism, dyslexia, or Down syndrome",
        "The school provides support for students with dyslexia.",
        "The student struggled to learn the new software.",
    ),
    "psychiatric_or_mental_illness": FieldPrompt(
        "mentions a psychiatric condition or mental illness, such as "
        "depression, schizophrenia, or bipolar disorder",
        "The clinic treats people with bipolar disorder.",
        "That policy is crazy.",
    ),
    "other_disability": FieldPrompt(
        "mentions disabled people or a disability that is not identified as "
        "physical, intellectual, learning, psychiatric, or mental illness",
        "The program offers services for people with disabilities.",
        "The program offers ramps for wheelchair users.",
    ),
}

JOIN_PROMPT_INSTRUCTIONS = (
    "DOCUMENT {1} defines one field and gives labeled calibration examples. "
    "Judge only whether the comment in DOCUMENT {0} meets the criterion. "
    "Words that appear only in the examples are not part of the comment. "
    "A toxic comment is not automatically an insult, threat, or identity "
    "attack. Answer TRUE only if the criterion is specifically true. "
    "Answer FALSE if it is loosely related or stronger than the comment."
)

FILTER_PROMPT = (
    "{0}\n\n"
    f"{TOXICITY_PROMPT_INSTRUCTIONS}\n\n"
    "Instruction: answer TRUE or FALSE."
)
JOIN_PROMPT = JOIN_PROMPT_INSTRUCTIONS


def field_statement(spec: FieldPrompt) -> str:
    """Render one field definition and its calibration examples."""
    return (
        f"Criterion: the comment {spec.definition}.\n"
        f"TRUE example: {spec.true_example}\n"
        f"FALSE example: {spec.false_example}"
    )


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
    )


def fields_table() -> pa.Table:
    """Return the 30 semantic field statements."""
    return pa.table({
        "field": list(FIELDS),
        "statement": [field_statement(spec) for spec in FIELDS.values()],
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
        len(tokenizer(field_statement(spec)))
        for spec in FIELDS.values()
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

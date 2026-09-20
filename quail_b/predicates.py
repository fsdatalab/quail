"""The QUAIL-B predicates and the identity of their reference labels.

A predicate is one TRUE or FALSE question over one table column
(filter) or two (join). Its version hashes everything that can change
the model's answer; the label-set id hashes the version, the corpus,
and the judge, so a label file's path says exactly what produced it.
The pass that writes the labels lives in Quail's repository.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from quail_b import data, prompts, rendering
from quail_b.rendering import SHARED_PRE

SCHEMA_VERSION = 1
MODEL_REPO = "Qwen/Qwen3-32B-FP8"
MODEL_REVISION = "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
MODEL_NAME = "qwen3-32b-fp8"
MAX_MODEL_LEN = 32_768


@dataclass(frozen=True)
class PredicateSpec:
    key: str
    workload: str
    slug: str
    kind: str
    template: str
    left_role: str
    left_table: str
    left_column: str
    right_role: str | None = None
    right_table: str | None = None
    right_column: str | None = None
    source_policy: str = "qwen3_32b"


PREDICATES = (
    PredicateSpec(
        "quailb.imdb.review.mentions_positive_aspect", "imdb",
        "review_mentions_positive_aspect", "filter", prompts.F1,
        "review", "reviews", "body"),
    PredicateSpec(
        "quailb.imdb.review.discusses_ending", "imdb",
        "review_discusses_ending", "filter", prompts.F4,
        "review", "reviews", "body"),
    PredicateSpec(
        "quailb.imdb.review.mentions_named_actor", "imdb",
        "review_mentions_named_actor", "filter", prompts.F5,
        "review", "reviews", "body"),
    PredicateSpec(
        "quailb.imdb.review.discusses_aspect", "imdb",
        "review_discusses_aspect", "join", prompts.DISCUSS_ASPECT,
        "review", "reviews", "body", "aspect", "aspects", "aspect"),
    PredicateSpec(
        "quailb.imdb.review.positive_sentiment_about_aspect",
        "imdb",
        "review_positive_sentiment_about_aspect", "join",
        prompts.ASPECT_SENTIMENT,
        "review", "reviews", "body", "aspect", "aspects", "aspect"),
    PredicateSpec(
        "quailb.biodex.report.describes_serious_adverse_event", "biodex",
        "report_describes_serious_adverse_event", "filter",
        prompts.SERIOUS_ADVERSE_EVENT,
        "report", "reports", "report"),
    PredicateSpec(
        "quailb.biodex.report.experienced_reaction", "biodex",
        "report_experienced_reaction", "join", prompts.REACTION,
        "report", "reports", "report", "reaction", "terms", "term"),
    PredicateSpec(
        "quailb.biodex.reaction.is_neurological", "biodex",
        "reaction_is_neurological", "filter", prompts.NEUROLOGICAL_REACTION,
        "reaction", "terms", "term"),
    PredicateSpec(
        "quailb.biodex.reaction.is_cardiovascular", "biodex",
        "reaction_is_cardiovascular", "filter", prompts.CARDIOVASCULAR_REACTION,
        "reaction", "terms", "term"),
    PredicateSpec(
        "quailb.fever.claim.about_person", "fever",
        "claim_about_person", "filter", prompts.F11,
        "claim", "claims", "claim"),
    PredicateSpec(
        "quailb.fever.claim.contains_date", "fever",
        "claim_contains_date", "filter", prompts.F12,
        "claim", "claims", "claim"),
    PredicateSpec(
        "quailb.fever.passage.about_person", "fever",
        "passage_about_person", "filter", prompts.F13,
        "passage", "evidence", "text"),
    PredicateSpec(
        "quailb.fever.passage.supports_claim", "fever",
        "passage_supports_claim", "join", prompts.SUPPORT,
        "claim", "claims", "claim", "passage", "evidence", "text",
        "fever_annotation_then_qwen3_32b"),
    PredicateSpec(
        "quailb.fever.passage.refutes_claim", "fever",
        "passage_refutes_claim", "join", prompts.REFUTE,
        "claim", "claims", "claim", "passage", "evidence", "text"),
    PredicateSpec(
        "quailb.lepard.excerpt.reasoning_does_not_apply", "lepard",
        "excerpt_reasoning_does_not_apply", "filter", prompts.LEP1,
        "excerpt", "citation_contexts", "destination_context"),
    PredicateSpec(
        "quailb.lepard.excerpt.procedural_or_jurisdictional", "lepard",
        "excerpt_procedural_or_jurisdictional", "filter",
        prompts.LEP2, "excerpt", "citation_contexts", "destination_context"),
    PredicateSpec(
        "quailb.lepard.excerpt.treats_passage_as_binding", "lepard",
        "excerpt_treats_passage_as_binding", "filter", prompts.LEP3,
        "excerpt", "citation_contexts", "destination_context"),
    PredicateSpec(
        "quailb.lepard.excerpt.supports_liability_or_guilt", "lepard",
        "excerpt_supports_liability_or_guilt", "filter",
        prompts.LEP4, "excerpt", "citation_contexts", "destination_context"),
    PredicateSpec(
        "quailb.lepard.excerpt.acknowledges_court_disagreement", "lepard",
        "excerpt_acknowledges_court_disagreement", "filter",
        prompts.LEP5, "excerpt", "citation_contexts", "destination_context"),
    PredicateSpec(
        "quailb.lepard.passage.states_general_rule", "lepard",
        "passage_states_general_rule", "filter", prompts.LEPS1,
        "passage", "citation_passages", "passage_text"),
    PredicateSpec(
        "quailb.lepard.excerpt.cites_passage", "lepard",
        "excerpt_cites_passage", "join", prompts.LEPJOIN,
        "excerpt", "citation_contexts", "destination_context",
        "passage", "citation_passages", "passage_text",
        "lepard_citation_edge"),
    PredicateSpec(
        "quailb.agent.trace.recovered_after_unsuccessful_approach",
        "agent", "recovered_after_unsuccessful_approach",
        "filter", prompts.AGENT_RECOVERED,
        "agent_trace", "agent_traces", "trace"),
    PredicateSpec(
        "quailb.agent.trace.implemented_plausible_fix",
        "agent", "implemented_plausible_fix",
        "filter", prompts.AGENT_IMPLEMENTED_FIX,
        "agent_trace", "agent_traces", "trace"),
)

PREDICATE_BY_KEY = {p.key: p for p in PREDICATES}

WORKLOADS = tuple(dict.fromkeys(spec.workload for spec in PREDICATES))


def workload_specs(workload: str) -> tuple:
    return tuple(p for p in PREDICATES if p.workload == workload)


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _full_hash(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _named_id(prefix: str, full_hash: str) -> str:
    return f"{prefix}_{full_hash[:32]}"


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def predicate_payload(spec: PredicateSpec) -> dict:
    render = ("filter_document_then_question_v1" if spec.kind == "filter"
              else "join_arg0_anchor_then_arg1_v1")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "predicate_key": spec.key,
        "kind": spec.kind,
        "template": spec.template,
        "left_role": spec.left_role,
        "left_table": spec.left_table,
        "left_column": spec.left_column,
        "right_role": spec.right_role,
        "right_table": spec.right_table,
        "right_column": spec.right_column,
        "render": render,
        "shared_preamble": SHARED_PRE,
    }
    # Published raw label hashes omit these fields for the original layout.
    raw_instruction = "Evaluate TRUE or FALSE for the following question: "
    if rendering.TASK_INSTRUCTION != raw_instruction:
        payload["task_instruction"] = rendering.TASK_INSTRUCTION
    if rendering.ANSWER_CUE != "\nANSWER:":
        payload["answer_cue"] = rendering.ANSWER_CUE
    if rendering.PROMPT_FORMAT != "raw-v1":
        payload["prompt_format"] = rendering.PROMPT_FORMAT
    return payload


def predicate_version(spec: PredicateSpec) -> tuple[str, str]:
    full = _full_hash(predicate_payload(spec))
    return _named_id("pv", full), full


# Only fields that can change the model's answer belong in this hash:
# it flows into JUDGE_ID, then label_set_id, then every label path, so
# any field added here invalidates all existing labels. Scheduler
# capacity knobs (max_num_batched_tokens, max_num_seqs,
# gpu_memory_utilization) change throughput and memory, not the token
# a greedy 1-token decode picks, so they stay out. They used to be in
# here, which made an out-of-memory fix cost a full relabel.
JUDGE_SPEC = {
    "model_repo": MODEL_REPO,
    "model_revision": MODEL_REVISION,
    "tokenizer_revision": MODEL_REVISION,
    "temperature": 0.0,
    "max_tokens": 1,
    "min_tokens": 1,
    "seed": data.DATA_SEED,
    "allowed_answers": ["TRUE", "FALSE"],
    "prefix_caching": True,
    "max_model_len": MAX_MODEL_LEN,
}
JUDGE_FULL_HASH = _full_hash(JUDGE_SPEC)
JUDGE_ID = _named_id("j", JUDGE_FULL_HASH)

# The judge Quail's labeling pass runs: the same model through Quail's
# own executor, answering by comparing the TRUE and FALSE token logits
# after one forward pass. A different engine can flip a borderline
# answer, so its labels get their own ids.
QUAIL_JUDGE_SPEC = {
    "engine": "quail",
    "model": MODEL_NAME,
    "model_repo": MODEL_REPO,
    "model_revision": MODEL_REVISION,
    "answer": "argmax over the TRUE and FALSE token logits",
    "max_model_len": MAX_MODEL_LEN,
}

SOURCE_SPECS = {
    "fever_annotation": {
        "dataset": "fever/fever",
        "revision": data.SOURCE_REVISIONS["fever/fever"],
        "rule": "matching evidence_wiki_url; SUPPORTS is true; REFUTES false",
    },
    "lepard_citation_edge": {
        "dataset": "rmahari/LePaRD",
        "revision": data.SOURCE_REVISIONS["rmahari/LePaRD"],
        "rule": ("anchor cited_passage_ids intersects candidate "
                 "passage_ids"),
    },
}


def label_sources(spec: PredicateSpec, judge: dict = JUDGE_SPEC) -> list[dict]:
    """The sources of one predicate's labels, the judge first when it has one."""
    sources = []
    if "qwen3_32b" in spec.source_policy:
        full = _full_hash(judge)
        sources.append({"id": _named_id("j", full), "full_hash": full,
                        "spec": judge})
    if spec.source_policy.startswith("fever_annotation"):
        payload = SOURCE_SPECS["fever_annotation"]
        full = _full_hash(payload)
        sources.append({"id": _named_id("s", full), "full_hash": full,
                        "spec": payload})
    if spec.source_policy == "lepard_citation_edge":
        payload = SOURCE_SPECS["lepard_citation_edge"]
        full = _full_hash(payload)
        sources.append({"id": _named_id("s", full), "full_hash": full,
                        "spec": payload})
    return sources


def label_set_identity(spec: PredicateSpec, corpus_id: str,
                       corpus_full_hash: str,
                       judge: dict = JUDGE_SPEC) -> dict:
    """The id of one predicate's labels over one corpus by one judge."""
    pversion, pfull = predicate_version(spec)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": corpus_id,
        "corpus_full_hash": corpus_full_hash,
        "predicate_key": spec.key,
        "predicate_version": pversion,
        "predicate_full_hash": pfull,
        "sources": label_sources(spec, judge),
    }
    full = _full_hash(payload)
    return {**payload, "label_set_id": _named_id("ls", full),
            "label_set_full_hash": full}


def example_identity(corpus_id: str, operands: list[dict]) -> tuple[str, str]:
    full = _full_hash({"corpus_id": corpus_id, "operands": operands})
    return _named_id("ex", full), full


def judgment_identity(label_set_id: str, example_full_hash: str) -> str:
    full = _full_hash({"label_set_id": label_set_id,
                       "example_full_hash": example_full_hash})
    return _named_id("jd", full)


def render_filter_prompt(spec: PredicateSpec, document: str) -> str:
    return rendering.render_filter_prompt(spec.template, document)


def render_join_prompt(spec: PredicateSpec, left: str, right: str) -> str:
    return rendering.render_join_prompt(spec.template, (left, right), anchor=0)

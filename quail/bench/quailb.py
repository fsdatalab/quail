"""Build, run, and score QUAIL-B queries with Quail."""

import argparse
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import pyarrow as pa

import quail
import quail_b as benchmark
from quail.specs import H100_USD_PER_HOUR
from quail_b.queries import (
    SELECTIVITY_ESTIMATE_COLLECTION,
    SELECTIVITY_ESTIMATE_CORPUS,
    SELECTIVITY_ESTIMATE_SCALE_FACTOR,
    QuerySpec,
)
from quail_b.queries import queries as query_specs
from quail_b.scoring import RunOutput, reference_answer

DEFAULT_SETS = ("reviews", "aspects", "reports", "terms",
                "claims", "evidence", "citation_contexts",
                "citation_passages", "agent_traces")


def register_sets(sess, data_dir):
    """Register the default document sets from their Parquet files."""
    for name in DEFAULT_SETS:
        sess.register(name, quail.DocumentProvider.from_parquet(
            str(Path(data_dir) / f"{name}.parquet"), id_col="id"))


def register_privacy_sets(sess, data_dir):
    """Register policies and scenarios tables for PRIV queries.

    Separate from register_sets so the privacy policy queries do not
    run unless the corpus was built.
    """
    for name in ("policies", "scenarios"):
        sess.register(name, quail.DocumentProvider.from_parquet(
            str(Path(data_dir) / f"{name}.parquet"), id_col="id"))


def build_query(sess, spec: QuerySpec):
    """Build the Quail query of one specification on one session."""

    def with_filters(builder, alias_spec):
        column = quail.col(f"{alias_spec.alias}.{alias_spec.column}")
        for template in alias_spec.filters:
            builder = builder.ai_filter(
                quail.prompt(template, column),
                selectivity=spec.filter_selectivity(template))
        return builder

    base = spec.aliases[0]
    query = with_filters(sess.docs(base.table).alias(base.alias), base)
    for join, partner in zip(spec.joins, spec.aliases[1:]):
        partner_query = with_filters(
            sess.docs(partner.table).alias(partner.alias), partner)
        columns = [
            quail.col(f"{alias}.{spec.alias(alias).column}")
            for alias in join.aliases
        ]
        query = query.ai_join(
            partner_query,
            quail.prompt(join.template, *columns),
            selectivity=spec.join_selectivity(join.template))
    return query.select(*spec.select, order=spec.order)


def queries(sess):
    """Id -> (description, callable() -> Query), for every registered set.

    Fresh Query objects per call so each pass re-plans.
    """
    return {
        spec.id: (spec.description,
                  lambda spec=spec: build_query(sess, spec))
        for spec in query_specs(
            include_privacy="policies" in sess.catalog).values()
    }


def canonical_templates(ground_truth) -> dict[str, str]:
    """Map each predicate's template as Quail binds it to the written one."""
    canonical = {}
    for labels in ground_truth.predicates.values():
        predicate = labels.predicate
        left = quail.ColumnRef(
            "left", predicate["left_table"], predicate["left_column"])
        if predicate["kind"] == "filter":
            bound = quail.bind_prompt(predicate["template"], (left,))
        else:
            right = quail.ColumnRef(
                "right", predicate["right_table"], predicate["right_column"])
            bound = quail.bind_join_prompt(
                predicate["template"], (left, right))
        canonical[bound.template] = predicate["template"]
    return canonical


def _document_ids(rows):
    if isinstance(rows, pa.Table):
        return rows.column("id").to_pylist()
    return [row["id"] for row in rows]


def answer_oracle(ground_truth, corpus_rows):
    """Return the `answer(prompt, assignment)` callable Quail's estimate takes."""
    templates = canonical_templates(ground_truth)
    ids_by_provider = {
        name: [str(row_id) for row_id in _document_ids(rows)]
        for name, rows in corpus_rows.items()
    }

    def answer(prompt, assignment):
        ids = tuple(
            ids_by_provider[arg.provider][assignment[arg.alias]]
            for arg in prompt.args
        )
        return reference_answer(ground_truth, templates[prompt.template], ids)

    return answer


def run_output(result, spec: QuerySpec, corpus_rows) -> RunOutput:
    """Translate a Quail result's row indices into benchmark ids."""
    ids = {
        alias_spec.alias: [
            str(row_id) for row_id in _document_ids(corpus_rows[alias_spec.table])]
        for alias_spec in spec.aliases
    }

    def id_column(alias, indices):
        return pa.array(
            [ids[alias][int(index)] for index in indices], type=pa.string())

    filter_answers = {}
    for (alias, written_pos), table in result.answer_tables["filters"].items():
        filter_answers[(alias, written_pos)] = pa.table({
            alias: id_column(alias, table.column(alias).to_pylist()),
            "answer": table.column("answer"),
        })
    join_answers = {}
    for written_pos, table in result.answer_tables["joins"].items():
        join = spec.joins[written_pos]
        join_answers[written_pos] = pa.table({
            **{alias: id_column(alias, table.column(alias).to_pylist())
               for alias in join.aliases},
            "answer": table.column("answer"),
        })
    for name in spec.select:
        if name.split(".", 1)[1] != "id":
            raise NotImplementedError(
                "QUAIL-B output accuracy needs id columns in the select list")
    started = time.perf_counter()
    rows = result.collect()
    collection_s = time.perf_counter() - started
    rows = rows.rename_columns([name.split(".", 1)[0] for name in spec.select])
    return RunOutput(
        filter_answers, join_answers, rows, result.report["wall_s"],
        dict(result.report, collection_s=collection_s))


def run_query(session, spec, tables):
    """Execute one query and return benchmark IDs, answers, and measurements."""
    for name, table in tables.items():
        if name not in session.catalog:
            session.register(
                name, quail.DocumentProvider.from_table(table, id_col="id"))
    result = build_query(session, spec).run()
    return run_output(result, spec, tables)


def run_suite(only=None, *, sf=0.1, config=None, data_dir=None,
              ground_truth_collection=None, output_dir,
              h100_usd_per_hour=H100_USD_PER_HOUR):
    """Run Quail queries through QUAIL-B and save the benchmark report."""
    config = config or quail.EngineConfig()
    with quail.Session(config) as session:
        return benchmark.run(
            partial(run_query, session), queries=only, scale_factor=sf,
            output_dir=output_dir, data_dir=data_dir,
            collection_id=ground_truth_collection,
            gpu_count=config.gpus, gpu_hourly_rate_usd=h100_usd_per_hour,
            metadata={
                "engine": config.backend, "model": config.model,
                "configuration": asdict(config),
                "warmup": "engine startup and kernel warmup excluded",
                "cache_reuse": "one session per backend and query family",
                "planning": {
                    "collection_id": SELECTIVITY_ESTIMATE_COLLECTION,
                    "corpus_id": SELECTIVITY_ESTIMATE_CORPUS,
                    "scale_factor": SELECTIVITY_ESTIMATE_SCALE_FACTOR,
                },
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sf", type=float, choices=(0.1, 0.5, 1.0), default=0.1)
    parser.add_argument("--only", help="comma-separated query IDs")
    parser.add_argument("--model", default="qwen3-4b-fp8")
    parser.add_argument("--backend", default="quail")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--data-dir", help="directory containing input Parquet files")
    parser.add_argument("--ground-truth-collection")
    parser.add_argument("--output-dir", required=True, help="new run directory")
    args = parser.parse_args()
    run_suite(
        [value.strip() for value in args.only.split(",")] if args.only else None,
        sf=args.sf,
        config=quail.EngineConfig(
            model=args.model, backend=args.backend, gpus=args.gpus),
        data_dir=args.data_dir,
        ground_truth_collection=args.ground_truth_collection,
        output_dir=args.output_dir)
    print(f"saved {args.output_dir}")


if __name__ == "__main__":
    main()

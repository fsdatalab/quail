"""Run QUAIL-B on Quail and the request backends it hosts.

The benchmark itself (document sets, prompts, query specifications,
labels, scoring) is the `quailb` package. This module is Quail's
runner: it registers the document sets with a session, builds a Quail
query from each `QuerySpec`, runs it, and hands the answers back to the
benchmark's scoring as a `RunOutput`.

    uv run python -m quail.bench.quailb --sf 0.1 --model qwen3-4b-fp8 --gpus 1
"""

import argparse
import json
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa

import quail
from quailb.data import (
    DATA_SEED,
    SOURCE_REVISIONS,
    _ids,
    build_sets,
    corpus_identity,
    read_corpus,
)
from quailb.labels import (
    ModalVolumeFiles,
    load_ground_truth,
    load_ground_truth_workload,
)
from quailb.queries import (
    SELECTIVITY_ESTIMATE_COLLECTION,
    SELECTIVITY_ESTIMATE_CORPUS,
    SELECTIVITY_ESTIMATE_SCALE_FACTOR,
    QuerySpec,
)
from quailb.queries import queries as query_specs
from quailb.scoring import (
    Evaluator,
    RunOutput,
    add_query_metrics,
    summarize_queries,
)

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


def answer_oracle(evaluator: Evaluator):
    """Return the `answer(prompt, assignment)` callable Quail's estimate takes."""
    templates = canonical_templates(evaluator.ground_truth)
    ids_by_provider = {
        name: [str(row_id) for row_id in _ids(rows)]
        for name, rows in evaluator.corpus_rows.items()
    }

    def answer(prompt, assignment):
        ids = tuple(
            ids_by_provider[arg.provider][assignment[arg.alias]]
            for arg in prompt.args
        )
        return evaluator.answer(templates[prompt.template], ids)

    return answer


def run_output(result, spec: QuerySpec, corpus_rows) -> RunOutput:
    """Translate a Quail result's row indices into benchmark ids."""
    ids = {
        alias_spec.alias: [
            str(row_id) for row_id in _ids(corpus_rows[alias_spec.table])]
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
    rows = result.collect()
    rows = rows.rename_columns([name.split(".", 1)[0] for name in spec.select])
    return RunOutput(filter_answers, join_answers, rows)


# ----------------------------------------------------------- driver

def _artifact_stem(started, sf, lf, model, backend="quail"):
    timestamp = started.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-quailb-sf{sf}-lf{lf}-{model}-{backend}"


def run_suite(data_dir, sf=0.1, lf=1, gpus=1, only=None,
              out_path=None, model="qwen3-4b-fp8",
              backend="quail",
              accuracy=True, ground_truth_collection=None,
              ground_truth_workload=None,
              h100_usd_per_hour=3.9492, ground_truth_files=None,
              prediction=None, artifact_stem=None,
              compute_provider=None):
    """Run all (or selected) QUAIL-B queries through the engine."""
    from quail.specs import H100_PRICE_SOURCE

    d = build_sets(data_dir, sf, lf)
    corpus_rows = read_corpus(d)
    corpus = corpus_identity(
        corpus_rows, sf, DATA_SEED, SOURCE_REVISIONS)
    evaluator = None
    truth = None
    result_files = ground_truth_files or ModalVolumeFiles()
    if accuracy:
        if ground_truth_workload:
            truth = load_ground_truth_workload(
                result_files,
                scale_factor=sf,
                corpus_id=corpus["corpus_id"],
                corpus_full_hash=corpus["corpus_full_hash"],
                workload=ground_truth_workload,
            )
        else:
            truth = load_ground_truth(
                result_files, scale_factor=sf,
                corpus_id=corpus["corpus_id"],
                collection_id=ground_truth_collection)
        if truth.corpus_id != corpus["corpus_id"]:
            raise ValueError(
                f"benchmark corpus {corpus['corpus_id']} does not match "
                f"ground truth {truth.corpus_id}")
        evaluator = Evaluator(truth, corpus_rows)
    sess = quail.Session(quail.EngineConfig(
        gpus=gpus,
        model=model,
        backend=backend,
    ), compute_provider=compute_provider)
    register_sets(sess, d)
    specs = query_specs()
    if only is None:
        ids = list(specs)
    elif isinstance(only, (set, frozenset)):
        ids = [query_id for query_id in specs if query_id in only]
    else:
        ids = [query_id for query_id in only if query_id in specs]
    started = datetime.now(timezone.utc)
    artifact_stem = artifact_stem or _artifact_stem(
        started, sf, lf, model, backend)
    run_id = (f"qb_{started.strftime('%Y%m%dT%H%M%SZ')}_"
              f"{uuid.uuid4().hex[:8]}")
    raw_root = f"benchmarks/quailb/runs/{run_id}"
    aggregate_volume_path = f"{raw_root}/{artifact_stem}.json"
    suite = dict(
        run_id=run_id,
        artifact_stem=artifact_stem,
        started_at=started.isoformat(),
        prediction=prediction,
        sf=sf, lf=lf, gpus=gpus, model=model, backend=backend,
        corpus_id=corpus["corpus_id"],
        selectivity_estimates=dict(
            source_collection=SELECTIVITY_ESTIMATE_COLLECTION,
            source_corpus=SELECTIVITY_ESTIMATE_CORPUS,
            source_scale_factor=SELECTIVITY_ESTIMATE_SCALE_FACTOR,
            method=("TRUE labels divided by all labels, fixed across "
                    "scale factors")),
        raw_volume_path=f"/results/{raw_root}",
        aggregate_volume_path=f"/results/{aggregate_volume_path}",
        pricing=dict(
            gpu="H100!",
            h100_usd_per_hour=h100_usd_per_hour,
            gpu_count=gpus,
            price_source=H100_PRICE_SOURCE,
            method=("query runtime in hours multiplied by the H100 hourly "
                    "price and GPU count"),
        ),
        metric_definitions=dict(
            runtime_s=("GPU worker query runtime; corpus construction, "
                       "ground truth loading, and local evaluation are "
                       "excluded"),
            runtime_with_boot_s=("GPU worker query runtime plus model load "
                                 "and warmup; ground truth loading is "
                                 "excluded"),
            pass_wall_s=("host time for the query loop; ground truth "
                         "loading is excluded"),
            tokens_processed=("sum of fresh tokens sent through model "
                              "forward calls; tokens read from KV are not "
                              "counted again"),
            regret_tokens=("per document KV regret: fresh tokens spent "
                           "recomputing a document's own prefix after an "
                           "earlier request of the query computed it"),
            shared_prefix_tokens=("scanned-document tokens that are a "
                                  "prefix another scanned document also "
                                  "has, within one alias or across "
                                  "aliases of one column; an execution "
                                  "that computes each distinct prefix "
                                  "once never computes them"),
            cross_row_cached_tokens=("cached tokens inside a document's "
                                     "own tokens that another document's "
                                     "request computed; cached preamble, "
                                     "label, question, or block rounding "
                                     "tokens do not count; null when the "
                                     "run did not record it"),
            regret_distinct_tokens=("distinct prefix KV regret: "
                                    "regret_tokens plus "
                                    "shared_prefix_tokens minus "
                                    "cross_row_cached_tokens"),
            input_document_rows=("sum of input table rows for every query "
                                 "alias; a self join counts the table once "
                                 "per alias"),
            documents_per_second=("input_document_rows divided by query "
                                  "runtime_s"),
            inference_cost_per_token_usd=("inference_cost_usd divided by "
                                          "tokens_processed"),
            answer_accuracy=("agreement with saved labels on model calls "
                             "that the query evaluated"),
            output_accuracy=("precision, recall, and F1 for final returned "
                             "rows against rows derived from saved labels"),
        ),
        ground_truth=(
            dict(collection_id=truth.collection_id,
                 reference_model=truth.reference_model,
                 workload=ground_truth_workload)
            if truth else None),
        passes={})
    try:
        pass_name = "single"
        rows = []
        t_pass = time.time()
        for qid in ids:
            spec = specs[qid]
            desc = spec.description
            print(f"[quailb] {qid}: {desc}", flush=True)
            try:
                query = build_query(sess, spec)
                res = query.run()
                result_rows = res.count()
                row = dict(query=qid, desc=desc,
                           backend=backend,
                           wall_s=res.report["wall_s"],
                           boot_s=res.report["boot_s"],
                           boot_kind=res.report.get("boot_kind"),
                           boot=res.report.get("boot"),
                           fresh_tokens=res.report["fresh_tokens"],
                           cached_tokens=res.report.get("cached_tokens"),
                           regret_tokens=res.report.get("regret_tokens"),
                           rows=result_rows,
                           peak_gib=res.report.get("peak_gib"),
                           stages=res.report["stages"],
                           backend_metrics=res.report.get(
                               "backend_metrics"
                           ),
                           shared_prefix_tokens=res.report[
                               "shared_prefix_tokens"],
                           cross_row_cached_tokens=res.report[
                               "cross_row_cached_tokens"],
                           regret_distinct_tokens=res.report[
                               "regret_distinct_tokens"])
                if evaluator is not None:
                    evaluation = evaluator.evaluate(
                        spec, run_output(res, spec, corpus_rows))
                    add_query_metrics(
                        row, evaluation, h100_usd_per_hour, gpus)
                    raw_path = f"{raw_root}/{pass_name}/{qid}.json"
                    answer_paths = {"filters": {}, "joins": {}}
                    for (alias, written_pos), table in \
                            res.answer_tables["filters"].items():
                        path = (
                            f"{raw_root}/{pass_name}/{qid}/answers/"
                            f"filter-{alias}-{written_pos}.parquet")
                        result_files.write_parquet(path, table)
                        answer_paths["filters"][
                            f"{alias}:{written_pos}"] = f"/results/{path}"
                    for written_pos, table in \
                            res.answer_tables["joins"].items():
                        path = (
                            f"{raw_root}/{pass_name}/{qid}/answers/"
                            f"join-{written_pos}.parquet")
                        result_files.write_parquet(path, table)
                        answer_paths["joins"][str(written_pos)] = \
                            f"/results/{path}"
                    result_files.write_json(raw_path, {
                        "run_id": run_id,
                        "pass": pass_name,
                        "query": qid,
                        "description": desc,
                        "columns": res.columns,
                        "result": {
                            "schema": str(res.schema),
                            "rows": result_rows,
                            "materialized": False,
                        },
                        "answer_tables": answer_paths,
                        "engine_report": res.report,
                        "accuracy": row["accuracy"],
                    })
                    row["raw_volume_path"] = f"/results/{raw_path}"
            except Exception as e:            # noqa: BLE001
                row = dict(query=qid, desc=desc,
                           error=f"{type(e).__name__}: {e}",
                           traceback=traceback.format_exc())
            rows.append(row)
            print(f"[quailb] {row}", flush=True)
        passed = dict(
            queries=rows,
            pass_wall_s=round(time.time() - t_pass, 1))
        if evaluator is not None:
            passed["summary"] = summarize_queries(
                rows, h100_usd_per_hour, gpus)
        suite["passes"][pass_name] = passed
    finally:
        sess.close()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(suite, f, indent=2)
        print(f"[quailb] saved {out_path}", flush=True)
    result_files.write_json(aggregate_volume_path, suite)
    print(
        f"[quailb] saved /results/{aggregate_volume_path} on quail-results",
        flush=True)
    return suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=0.1)
    ap.add_argument("--lf", type=int, default=1,
                    help="load factor, unused for now (see build_sets)")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--data-dir", default="results/quailb_data")
    ap.add_argument("--only", default=None,
                    help=("comma-separated query ids; default runs all "
                          "queries"))
    ap.add_argument(
        "--out", default=None,
        help=("JSON path; default uses a UTC timestamp under "
              "results/benchmark/"))
    ap.add_argument("--model", default="qwen3-4b-fp8",
                    help="registered ModelSpec name, see quail.specs.MODELS")
    ap.add_argument(
        "--backend",
        default="quail",
        choices=(
            "quail",
            "stock_vllm",
            "pipelined_vllm",
            "pipelined_sglang",
        ),
    )
    ap.add_argument(
        "--accuracy", action=argparse.BooleanOptionalAction, default=True,
        help="compare answers and output rows with the Modal ground truth")
    ap.add_argument("--ground-truth-collection", default=None,
                    help="collection id; default is the one matching the corpus")
    ap.add_argument("--ground-truth-workload", default=None,
                    help="load labels for one workload from the current corpus")
    ap.add_argument("--h100-usd-per-hour", type=float, default=3.9492,
                    help="H100 price used for query cost estimates")
    ap.add_argument("--prediction", default=None,
                    help="prediction stated before this benchmark run")
    ap.add_argument(
        "--report", action=argparse.BooleanOptionalAction, default=True,
        help=("write a Markdown report under results/benchmark/ and a PNG "
              "plot under reports/plots/benchmark/"))
    ap.add_argument("--report-path", default=None,
                    help="Markdown path; default uses the run UTC timestamp")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    started = datetime.now(timezone.utc)
    artifact_stem = _artifact_stem(
        started, args.sf, args.lf, args.model, args.backend)
    out = args.out or f"results/benchmark/{artifact_stem}.json"
    suite = run_suite(
        args.data_dir, sf=args.sf, lf=args.lf, gpus=args.gpus,
        only=only, out_path=out, model=args.model,
        backend=args.backend,
        accuracy=args.accuracy,
        ground_truth_collection=args.ground_truth_collection,
        ground_truth_workload=args.ground_truth_workload,
        h100_usd_per_hour=args.h100_usd_per_hour,
        prediction=args.prediction,
        artifact_stem=artifact_stem)
    if args.report:
        if not args.accuracy:
            raise ValueError("the evaluation report requires accuracy")
        report_path = args.report_path or (
            f"results/benchmark/{suite['artifact_stem']}.md")
        script = Path(__file__).resolve().parents[2] / "reports" \
            / "make_quailb_eval_plots.py"
        subprocess.run(
            [sys.executable, str(script), "--input", out,
             "--report", report_path], check=True)


if __name__ == "__main__":
    main()

"""Run QUAIL-B queries with Quail: the adapter `quail_b.run` calls.

`run_query(session, spec, tables)` builds one query's Substrait plan
on the session, runs it, and returns the answers keyed by the plan's
operator ids, as QUAIL-B scores them.

Runs on any machine with a supported GPU and saves to a local directory;
no Modal or volume is involved:

    uv run python -m quail.bench.quailb --sf 0.1 --only IMDB-4 \
      --model qwen3-4b-fp8 --device h100-sxm \
      --output-dir results/quailb/imdb-4
"""

import argparse
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import quail
import quail_b as benchmark
from quail.bench import substrait
from quail.bench.substrait import QueryPlan, Relation, read_plan
from quail.planner.plan import Refusal
from quail.specs import H100_USD_PER_HOUR, MODELS
from quail_b.cuad import parse_document_reference, resolve_documents
from quail_b.data import FILE_TABLES
from quail_b.queries import (
    FILTER_SELECTIVITY_ESTIMATES,
    JOIN_SELECTIVITY_ESTIMATES,
    SELECTIVITY_ESTIMATE_COLLECTION,
    SELECTIVITY_ESTIMATE_CORPUS,
    SELECTIVITY_ESTIMATE_SCALE_FACTOR,
    QuerySpec,
)
from quail_b.queries import queries as query_specs
from quail_b.scoring import RunOutput, reference_answer

# the fixed planner inputs, by prompt; a predicate without an
# estimate here gets the planner's default selectivity
SELECTIVITY = {**FILTER_SELECTIVITY_ESTIMATES, **JOIN_SELECTIVITY_ESTIMATES}


# A PDF provider forms these columns from the files it opens, so the
# benchmark's copies are dropped rather than carried beside them.
PDF_FORMED_COLUMNS = ("document", "page_count", "page_number")


def pdf_provider(rows: pa.Table):
    """A PDF provider whose query rows are a file table's rows, in order.

    A `document` of `<path>` makes one row per file. A `document` of
    `<path>#page=<n>` makes one row per listed page. Either way the
    rows keep the benchmark ids, so Quail's answers name them, and
    their other columns, so ordinary join conditions can read them.

    Raises:
        ValueError: The references mix the two forms.
    """
    references = [parse_document_reference(reference)
                  for reference in rows.column("document").to_pylist()]
    paths = [path for path, _page in references]
    pages = [page for _path, page in references]
    carried = rows.drop_columns(
        [name for name in PDF_FORMED_COLUMNS if name in rows.schema.names])
    carried = carried.append_column("path", pa.array(paths, pa.string()))
    if all(page is None for page in pages):
        return quail.DocumentProvider.from_pdfs(
            carried, id_col="id", path_col="path", row_mode="pdf")
    if any(page is None for page in pages):
        raise ValueError("a file table names whole files or pages, not both")
    listed = carried.append_column("page", pa.array(pages, pa.int32()))
    return quail.DocumentProvider.from_pdf_pages(
        listed, id_col="id", path_col="path", page_col="page")


def register_relation(session, relation: Relation, table: pa.Table) -> pa.Table:
    """Register the rows one relation reads, once, and return them.

    The rows are the benchmark table inside the relation's bounds, in
    table order; that order is the row index Quail's answers use. A
    file table becomes a PDF provider, any other a text table.
    """
    rows = relation.rows(table)
    if relation.source not in session.catalog:
        if relation.table in FILE_TABLES:
            provider = pdf_provider(rows)
        else:
            provider = quail.DocumentProvider.from_table(rows, id_col="id")
        session.register(relation.source, provider)
    return rows


def register_plan(session, plan: QueryPlan, tables) -> dict[str, pa.Table]:
    """Register every relation of a plan; table name -> the rows it reads.

    Raises:
        ValueError: Two relations read the same table inside different
            bounds, so one set of rows cannot stand for the table.
    """
    rows = {}
    bounds = {}
    for relation in plan.relations:
        if bounds.setdefault(relation.table, relation.bounds) != relation.bounds:
            raise ValueError(
                f"{relation.table} is read under two different bounds")
        rows[relation.table] = register_relation(
            session, relation, tables[relation.table])
    return rows


def read_tables(data_dir, names) -> dict[str, pa.Table]:
    """Named tables from a directory of Parquet files.

    A file table's references are resolved against the directory, the
    way `quail_b.load_benchmark` hands them to the adapter.
    """
    tables = {}
    for name in names:
        table = pq.read_table(Path(data_dir) / f"{name}.parquet")
        if name in FILE_TABLES:
            table = resolve_documents(table, Path(data_dir))
        tables[name] = table
    return tables


def register_tables(session, data_dir) -> None:
    """Register the relations of every query whose tables the directory holds."""
    available = {path.stem for path in Path(data_dir).glob("*.parquet")}
    tables = read_tables(data_dir, sorted(available))
    for spec in query_specs(include_privacy=True).values():
        plan = read_plan(spec.plan)
        if all(relation.table in available for relation in plan.relations):
            register_plan(session, plan, tables)


def _build(session, plan: QueryPlan):
    return substrait.build_query(session, plan, SELECTIVITY, order="by_cost")


def build_query(session, spec: QuerySpec):
    """Build the Quail query of one benchmark query on a session."""
    return _build(session, read_plan(spec.plan))


def queries(session) -> dict:
    """Id -> (description, callable() -> Query), for every registered table.

    A query is listed when every relation it reads is registered. Fresh
    Query objects per call so each pass re-plans.
    """
    listed = {}
    for spec in query_specs(include_privacy=True).values():
        plan = read_plan(spec.plan)
        if all(relation.source in session.catalog for relation in plan.relations):
            listed[spec.id] = (
                spec.description, lambda plan=plan: _build(session, plan))
    return listed


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


def _ids(table: pa.Table) -> list[str]:
    return [str(row_id) for row_id in table.column("id").to_pylist()]


def answer_oracle(ground_truth, tables):
    """Return the `answer(prompt, assignment)` callable Quail's estimate takes."""
    templates = canonical_templates(ground_truth)
    ids_by_table = {name: _ids(table) for name, table in tables.items()}

    def answer(prompt, assignment):
        ids = tuple(
            ids_by_table[arg.provider][assignment[arg.alias]]
            for arg in prompt.args
        )
        return reference_answer(ground_truth, templates[prompt.template], ids)

    return answer


def run_output(result, plan: QueryPlan, tables) -> RunOutput:
    """Translate a Quail result's row indices into benchmark ids by operator.

    Args:
        result: The Quail result.
        plan: The query's plan.
        tables: Table name -> the rows registered for it, in the order
            the row indices count (`register_plan`).
    """
    ids = {relation.alias: pa.array(_ids(tables[relation.table]), pa.string())
           for relation in plan.relations}

    def id_column(alias, indices):
        return pc.take(ids[alias], indices)

    filter_answers = {}
    for (alias, position), table in result.answer_tables["filters"].items():
        filter_answers[plan.filter_id(alias, position)] = pa.table({
            alias: id_column(alias, table.column(alias)),
            "answer": table.column("answer"),
        })
    join_answers = {}
    for position, table in result.answer_tables["joins"].items():
        join = plan.joins[position]
        join_answers[join.id] = pa.table({
            **{alias: id_column(alias, table.column(alias))
               for alias in join.aliases},
            "answer": table.column("answer"),
        })
    aliases = []
    for name in plan.select:
        alias, column = name.split(".", 1)
        if column != "id":
            raise NotImplementedError(
                "QUAIL-B output accuracy needs id columns in the select list")
        aliases.append(alias)
    started = time.perf_counter()
    rows = result.collect()
    collection_s = time.perf_counter() - started
    return RunOutput(
        filter_answers, join_answers, rows.rename_columns(aliases),
        result.report["wall_s"], dict(result.report, collection_s=collection_s))


def join_anchors(result) -> dict:
    """Return written position -> the anchor alias the engine chose."""
    return {
        position: table.schema.metadata[b"quail.anchor"].decode("utf-8")
        for position, table in result.answer_tables["joins"].items()
    }


def image_lengths(session, plan: QueryPlan, tables) -> dict[str, dict[str, int]]:
    """Table name -> row id -> the prompt positions a PDF row's pages take.

    QUAIL-B cannot tokenize a rendered document, so the engine reports
    the length of each; the session priced every row when it prepared
    the PDF provider.
    """
    lengths = {}
    for relation in plan.relations:
        if relation.table not in FILE_TABLES or relation.table in lengths:
            continue
        prepared = session.prepare_pdf(relation.source)
        row_ids = _ids(tables[relation.table])
        if len(prepared.lengths) != len(row_ids):
            raise ValueError(
                f"{relation.source} has {len(prepared.lengths)} PDF rows for "
                f"{len(row_ids)} benchmark rows")
        lengths[relation.table] = dict(zip(row_ids, prepared.lengths))
    return lengths


def prompt_pieces(query, plan: QueryPlan, anchors, images=None) -> dict:
    """Return the prompt token ids around each document, for QUAIL-B.

    QUAIL-B sizes the prefix trie of the run's requests from these
    pieces and the saved answer tables, after the run; nothing is
    tracked while the query runs.

    Args:
        query: The built query, with bound prompts.
        plan: The query's plan, for the operator ids.
        anchors: Written join position -> the anchor alias.
        images: Table name -> row id -> prompt positions, for the
            PDF relations (`image_lengths`); omitted when empty.
    """
    operators = query.logical.operators()
    filters, joins = operators.filters, operators.joins
    preamble = next((list(prompt.preamble_token_ids)
                     for prompt in operators.prompts
                     if prompt.preamble_token_ids), [])
    pieces = {"tokenizer": query.session.model.hf_name, "preamble": preamble,
              "filters": [], "joins": []}
    if images:
        pieces["images"] = images
    for alias, predicates in filters.items():
        for position, predicate in enumerate(predicates):
            pieces["filters"].append({
                "id": plan.filter_id(alias, position),
                "tail": list(predicate.prompt.tail_token_ids)})
    for position, join in enumerate(joins):
        anchor = anchors[position]
        parts = {alias: (list(label), list(frame))
                 for alias, label, frame in join.prompt.label_token_ids}
        (partner,) = [alias for alias in parts if alias != anchor]
        pieces["joins"].append({
            "id": plan.join_id(position), "anchor": anchor,
            "frame": parts[anchor][1], "label": parts[partner][0],
            "tail": list(join.prompt.tail_token_ids)})
    return pieces


def run_query(session, spec: QuerySpec, tables) -> RunOutput:
    """Execute one query and return benchmark ids, answers, and measurements."""
    plan = read_plan(spec.plan)
    rows = register_plan(session, plan, tables)
    query = _build(session, plan)
    result = query.run()
    output = run_output(result, plan, rows)
    output.prompt_pieces = prompt_pieces(
        query, plan, join_anchors(result), image_lengths(session, plan, rows))
    return output


def refused_queries(session, query_ids, data_dir) -> dict[str, str]:
    """Return id -> reason for the queries the session's backend refuses to plan.

    A request backend refuses a query with an equality join or a user
    function. Planning happens on the CPU, before any engine boots.
    """
    register_tables(session, data_dir)
    refused = {}
    for query_id in query_ids:
        plan = build_query(session, benchmark.get_query(query_id)).plan()
        if isinstance(plan, Refusal):
            refused[query_id] = " ".join(plan.reasons)
            print(f"[quail-b] {query_id}: skipped on {session.config.backend}: "
                  f"{refused[query_id]}", flush=True)
    return refused


def run_suite(only=None, *, sf=0.1, config, data_dir=None,
              ground_truth_collection=None, output_dir,
              h100_usd_per_hour=H100_USD_PER_HOUR, root=None):
    """Run Quail queries through QUAIL-B and save the benchmark report.

    Queries the backend refuses to plan are left out of the run and
    listed under `skipped_queries` in the returned record.
    """
    with quail.Session(config) as session:
        skipped = {}
        if only and data_dir is not None:
            skipped = refused_queries(session, only, data_dir)
            only = [query_id for query_id in only if query_id not in skipped]
        record = benchmark.run(
            partial(run_query, session), queries=only, scale_factor=sf,
            output_dir=output_dir, data_dir=data_dir,
            collection_id=ground_truth_collection, root=root,
            gpu_count=config.gpus, gpu_hourly_rate_usd=h100_usd_per_hour,
            metadata={
                "engine": config.backend, "model": config.model,
                "prompt_format": MODELS[config.model].prompt_format,
                "configuration": asdict(config),
                "warmup": "engine startup and kernel warmup excluded",
                "cache_reuse": "one session per backend and query family",
                "planning": {
                    "collection_id": SELECTIVITY_ESTIMATE_COLLECTION,
                    "corpus_id": SELECTIVITY_ESTIMATE_CORPUS,
                    "scale_factor": SELECTIVITY_ESTIMATE_SCALE_FACTOR,
                },
            })
        record["skipped_queries"] = skipped
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sf", type=float, choices=(0.1, 0.5, 1.0), default=0.1)
    parser.add_argument("--only", help="comma-separated query IDs")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="quail")
    parser.add_argument("--device", required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--data-dir", help="directory containing input Parquet files")
    parser.add_argument("--ground-truth-collection")
    parser.add_argument("--output-dir", required=True, help="new run directory")
    args = parser.parse_args()
    run_suite(
        [value.strip() for value in args.only.split(",")] if args.only else None,
        sf=args.sf,
        config=quail.EngineConfig(
            gpus=args.gpus,
            model=args.model,
            backend=args.backend,
            device=args.device,
        ),
        data_dir=args.data_dir,
        ground_truth_collection=args.ground_truth_collection,
        output_dir=args.output_dir)
    print(f"saved {args.output_dir}")


if __name__ == "__main__":
    main()

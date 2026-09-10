"""Score saved benchmark answers and write a Markdown report."""

import argparse
import json
from pathlib import Path

from quail_b._files import download_cache
from quail_b.benchmark import load_benchmark
from quail_b.run import _query_hash, _read_output, _score, _write_json


def _number(value):
    return "unavailable" if value is None else f"{value:.6g}"


def _write_report(directory, record):
    lines = [
        "# QUAIL-B results", "",
        f"Status: {record['status']}", "",
        f"Scale factor: {record['scale_factor']}", "",
        f"Corpus: {record['corpus_id']}", "",
        f"Reference collection: {record['collection_id']}", "",
        f"Reference model: {record['reference_model']}", "",
        "Query duration excludes model startup, result collection, scoring, "
        "and result saving.", "",
        "Predicate accuracy is agreement on evaluated answers. Engines may "
        "evaluate different documents and pairs.", "",
        "## Query results", "",
        "| Query | Status | Seconds | $/query | Throughput | Unit |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in record["queries"]:
        metrics = item.get("metrics", {})
        joins = "document_pairs_per_second" in metrics
        throughput = metrics.get(
            "document_pairs_per_second" if joins else "documents_per_second")
        unit = "document pairs/second" if joins else "documents/second"
        lines.append(
            f"| {item['id']} | {item['status']} | "
            f"{_number(item.get('runtime_s'))} | {_number(metrics.get('cost_usd'))} | "
            f"{_number(throughput)} | {unit} |")
    lines.extend([
        "", "## Accuracy", "",
        "| Query | Predicate accuracy | Evaluated answers | Output precision | "
        "Output recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for item in record["queries"]:
        accuracy = item.get("metrics", {}).get("accuracy", {})
        answers = accuracy.get("answer_accuracy") or {}
        output = accuracy.get("output_accuracy", {})
        lines.append(
            f"| {item['id']} | {_number(answers.get('accuracy'))} | "
            f"{_number(answers.get('evaluated'))} | "
            f"{_number(output.get('precision'))} | {_number(output.get('recall'))} |")
    lines.extend([
        "", "## Input rows and token counts", "",
        "| Query | Input rows by alias | Fresh tokens | Recomputed KV tokens |",
        "| --- | --- | ---: | ---: |",
    ])
    for item in record["queries"]:
        inputs = item.get("metrics", {}).get("input_rows", {})
        counts = ", ".join(f"{alias}: {count}" for alias, count in inputs.items())
        measurements = item.get("measurements", {})
        lines.append(
            f"| {item['id']} | {counts or 'unavailable'} | "
            f"{_number(measurements.get('fresh_tokens'))} | "
            f"{_number(measurements.get('regret_tokens'))} |")
    lines.extend(["", "## Configuration", "", "```json",
                  json.dumps(record["metadata"], indent=2), "```", ""])
    failures = [item for item in record["queries"] if "error" in item]
    if failures:
        lines.extend(["## Errors", ""])
        lines.extend(f"- {item['id']}: {item['error']}" for item in failures)
        lines.append("")
    path = directory / "report.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(path)
    return path


def _query_directory(directory, item):
    path = (directory / item.get("directory", item["id"])).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("query directory is outside the run directory")
    return path


def report(run_dir, *, rescore=True, cache_dir=None, root=None):
    """Write a report, optionally reusing scores already saved in the run."""
    directory = Path(run_dir)
    path = directory / "run.json"
    record = json.loads(path.read_text())
    if record["schema_version"] != 1:
        raise ValueError("unsupported run format")
    if not rescore:
        return _write_report(directory, record)
    ids = [item["id"] for item in record["queries"]]
    with download_cache(cache_dir):
        suite = load_benchmark(
            ids, scale_factor=record["scale_factor"],
            collection_id=record["collection_id"], root=root)
    if suite.corpus_id != record["corpus_id"]:
        raise ValueError("saved corpus does not match the benchmark")
    for spec, item in zip(suite.queries, record["queries"]):
        if _query_hash(spec) != item["definition_hash"]:
            raise ValueError(f"query definition changed: {spec.id}")
    for spec, item in zip(suite.queries, record["queries"]):
        if "files" not in item:
            continue
        try:
            output = _read_output(_query_directory(directory, item), item)
            item["metrics"] = _score(
                spec, output, suite, record["gpu_count"],
                record["gpu_hourly_rate_usd"])
            item["status"] = "complete"
            item.pop("error", None)
        except Exception as error:
            item.pop("metrics", None)
            item.update(status="scoring_failed",
                        error=f"{type(error).__name__}: {error}")
            record["status"] = "failed"
            _write_json(path, record)
            _write_report(directory, record)
            raise
    record["status"] = (
        "complete" if all(item["status"] == "complete" for item in record["queries"])
        else "failed")
    _write_json(path, record)
    return _write_report(directory, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "report", help="score saved answers and write a report")
    command.add_argument("run_dir")
    command.add_argument("--cache-dir")
    command.add_argument("--root", help="optional offline mirror of published data")
    args = parser.parse_args()
    print(report(args.run_dir, cache_dir=args.cache_dir, root=args.root))

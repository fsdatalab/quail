r"""Generate the raw-prompt QUAIL-B report and all six comparison PDFs.

Pull the CPU-derived summary and regenerate the figures:

    W=/tmp/quailb-raw; mkdir -p "$W"
    uv run modal volume get quail-results \
      reports/quailb-raw-2026-09-19/comparison.json "$W/comparison.json"
    uv run modal volume get quail-results \
      benchmarks/quailb/family-runs/20260920T062701Z-bio4-4b/quail/run.json \
      "$W/bio4-sf0.1-quail.json"
    uv run modal volume get quail-results \
      benchmarks/quailb/family-runs/20260920T062701Z-bio4-4b/pipelined_vllm/run.json \
      "$W/bio4-sf0.1-vllm.json"
    uv run modal volume get quail-results \
      sol/2026-09-20-bio4-qwen3-4b-sf0.1.json "$W/bio4-sf0.1-sol.json"
    BIO4_SF1_RUN=benchmarks/quailb/family-runs/20260920T064415Z-bio4-4b-sf1.0
    uv run modal volume get quail-results "$BIO4_SF1_RUN/quail/run.json" \
      "$W/bio4-sf1-quail.json"
    uv run modal volume get quail-results \
      "$BIO4_SF1_RUN/pipelined_vllm/run.json" \
      "$W/bio4-sf1-vllm.json"
    uv run modal volume get quail-results \
      sol/2026-09-20-bio4-qwen3-4b-sf1.0.json "$W/bio4-sf1-sol.json"
    BENCH=git+https://github.com/fsdatalab/quail-bench.git
    REV=35d026dc2f5b5c1e787268173e81e512b749081a
    uv run --with matplotlib --with "quail-b@$BENCH@$REV" \
      python reports/make_quailb_comparison_plots.py "$W"

To rebuild the summary on a CPU with the results volume mounted, use Quail
a79de8e8 and its pinned QUAIL-B version. Recalculate the SoL estimates first:

    W=/results/reports/quailb-raw-2026-09-19
    uv run python reports/make_sol_quailb.py "$W" 0.1 --root /results \
      --collection gt_be81cb241d74555dc2da79b5b0662554
    uv run python reports/make_quailb_comparison_plots.py "$W" --prepare \
      --root /results --sol-file "$W/sol_quailb_sf0.1.json"

Preparation validates saved plans and prompt token pieces, then recalculates
scores, complete input token counts, and KV regret from saved answers.
It does not run inference or change the original measurements.
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from plot_colors import BLUE, DARK, ORANGE

from quail.specs import H100_USD_PER_HOUR

HERE = Path(__file__).resolve().parent
METHODS = [("quail", "Quail", BLUE),
           ("pipelined_vllm", "Pipelined stock vLLM", ORANGE)]
BASE_QUERY_ORDER = (
    [f"IMDB-{n}" for n in range(1, 11)] + [f"BIO-{n}" for n in range(1, 4)]
    + [f"FEV-{n}" for n in range(1, 11)] + [f"LEP-{n}" for n in range(1, 6)]
    + ["AGENT-1", "AGENT-2"])
QUERY_ORDER = BASE_QUERY_ORDER[:13] + ["BIO-4"] + BASE_QUERY_ORDER[13:]
SOURCE_QUERY_IDS = {q: "LEP-7" if q == "LEP-5" else q
                    for q in BASE_QUERY_ORDER}
CURRENT_QUERY_IDS = {old: new for new, old in SOURCE_QUERY_IDS.items()}
SOURCE_RUN = "benchmarks/quailb/20260914T070913Z-f7beefb6"
COLLECTION = "gt_be81cb241d74555dc2da79b5b0662554"
BIO4_COLLECTIONS = {
    0.1: "gt_cd3ebdb784f64b9e028e50ea73cdedd0",
    1.0: "gt_e87691add604b02c4e43f0ff5bf0cc4f",
}
BIO4_RUNS = {
    0.1: "benchmarks/quailb/family-runs/20260920T062701Z-bio4-4b",
    1.0: "benchmarks/quailb/family-runs/20260920T064415Z-bio4-4b-sf1.0",
}


def row_metrics(row):
    """Derive throughput, GPU cost, and accuracy from one measurement row."""
    matched = row["matching_rows"]
    predicted = row["predicted_rows"]
    expected = row["expected_rows"]
    return {
        "seconds": row["runtime_s"],
        "fresh_tokens": row["fresh_tokens"],
        "regret_tokens": row["regret_tokens"],
        "regret_percent": (100 * row["regret_tokens"] / row["fresh_tokens"]
                           if row["regret_tokens"] is not None
                           and row["fresh_tokens"] else None),
        "tokens_per_second": row["requested_tokens"] / row["runtime_s"],
        "cost_per_million": (row["runtime_s"] / 3600 * H100_USD_PER_HOUR
                             / row["requested_tokens"] * 1e6),
        "cost": row["runtime_s"] / 3600 * H100_USD_PER_HOUR,
        "agreement": 100 * row["answers_correct"] / row["answers_evaluated"],
        "precision": (100 * matched / predicted if predicted
                      else (0 if expected else 100)),
        "recall": 100 * matched / expected if expected else (0 if predicted else 100),
        "throughput": ((row["evaluated_document_pairs"] or row["input_rows"])
                       / row["runtime_s"]),
        "throughput_unit": ("document pairs/s"
                            if row["evaluated_document_pairs"] else "documents/s"),
    }


METRICS = (
    ("seconds", "Latency", "seconds"),
    ("fresh_tokens", "Fresh input tokens computed", "tokens"),
    ("regret_tokens", "Recomputed KV tokens", "tokens"),
    ("agreement", "Answer agreement", "percent"),
)


def series_value(rows, sol, method, query, metric):
    """Return a measured value or an explicitly modeled value."""
    if method == "sol":
        return {
            "seconds": sol[query]["sol_s"],
            "fresh_tokens": sol[query]["tokens"],
            "regret_tokens": 0,
            "agreement": None,
        }[metric]
    if query not in rows[method]:
        return None
    return row_metrics(rows[method][query])[metric]


def metric_bars(axis, queries, rows, sol, metric, overview):
    """Draw measured bars and a SoL line across each query group."""
    measured = METHODS
    unit = next(unit for key, _, unit in METRICS if key == metric)
    methods = measured + ([] if metric == "agreement"
                          else [("sol", "SoL estimate", DARK)])
    positive = [value for key, _, _ in methods for query in queries
                if (value := series_value(rows, sol, key, query, metric)) is not None
                and value > 0]
    maximum = max(positive, default=0)
    logarithmic = (metric != "agreement" and positive
                   and maximum / min(positive) > 10)
    if metric == "agreement":
        axis.set_ylim(0, min(105, max(1, maximum * 1.4)))
        axis.set_ylabel("percent")
    elif logarithmic and metric == "regret_tokens":
        axis.set_yscale("symlog", linthresh=max(1, min(positive) / 10))
        axis.set_ylim(0, maximum * 8)
        axis.set_ylabel(f"{unit} (symmetric log scale)")
    elif logarithmic:
        axis.set_yscale("log")
        axis.set_ylim(min(positive) / 2, maximum * 8)
        axis.set_ylabel(f"{unit} (log scale)")
    else:
        axis.set_ylim(0, maximum * 1.6 if maximum else 1)
        axis.set_ylabel(unit)
        if not maximum:
            axis.set_yticks([0])
    width = 0.82 / len(measured)
    floor = axis.get_ylim()[0]
    for method_index, (key, label, color) in enumerate(measured):
        xs, values = [], []
        for index, query in enumerate(queries):
            position = index - 0.41 + (method_index + 0.5) * width
            value = series_value(rows, sol, key, query, metric)
            if value is None or value == 0:
                axis.plot(position, 0.015, marker="x" if value is None else "_",
                          color=color, markersize=4, linestyle="none",
                          transform=axis.get_xaxis_transform(), clip_on=False)
                continue
            xs.append(position)
            values.append(value)
            if not overview:
                shown = f"{value / 1e6:.2f}M" if value >= 1e6 else (
                    f"{value / 1e3:.1f}k" if value >= 1000 else f"{value:.2f}")
                if metric == "agreement":
                    shown = f"{value:.1f}%"
                axis.annotate(shown, (position, value), xytext=(0, 3),
                              textcoords="offset points", rotation=90,
                              ha="center", va="bottom", fontsize=8)
        axis.bar(xs, [value - floor for value in values], bottom=floor,
                 width=width * 0.9, color=color, label=label, edgecolor="none")
    if metric == "seconds" and not overview:
        for index, query in enumerate(queries):
            if any(query not in rows[key] for key, _, _ in METHODS):
                continue
            quail_time = rows["quail"][query]["runtime_s"]
            baseline_time = rows["pipelined_vllm"][query]["runtime_s"]
            top = max(quail_time, baseline_time)
            ratio = baseline_time / quail_time
            axis.annotate(f"{ratio:.2f}x", (index, top), xytext=(0, 35),
                          textcoords="offset points", ha="center",
                          va="bottom", fontsize=8)
    if metric != "agreement":
        for index, query in enumerate(queries):
            axis.hlines(series_value(rows, sol, "sol", query, metric),
                        index - 0.41, index + 0.41, color=DARK, linewidth=1.7,
                        zorder=3, clip_on=False)
    axis.set_xlim(-0.7, len(queries) - 0.3)
    axis.set_xticks(range(len(queries)),
                   queries,
                   rotation=55 if overview else 35, ha="right")
    axis.tick_params(axis="x", labelsize=9)
    latency_title = "Latency" if overview else "Latency (ratios: vLLM / Quail)"
    title = next(title for key, title, _ in METRICS if key == metric)
    axis.set_title(latency_title if metric == "seconds" else title, fontsize=13)


def document_page(title, queries, relations):
    """Create a readable input-count page for the PDF."""
    figure = plt.figure(figsize=(14, 9))
    figure.suptitle(f"{title}: input documents before filtering", y=0.96,
                    fontsize=16)
    axis = figure.add_axes((0.045, 0.12, 0.91, 0.77))
    axis.axis("off")
    cells = [[query, "; ".join(f"{alias} ({provider}): {count:,}"
                               for alias, provider, count in relations[query])]
             for query in queries]
    table = axis.table(cellText=cells,
                       colLabels=["Query", "Relation alias (set): input documents"],
                       colWidths=[0.10, 0.90], cellLoc="left", colLoc="left",
                       loc="upper left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, _), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_height(min(0.065, 0.95 / (len(cells) + 1)))
        cell.PAD = 0.02
        if row == 0:
            cell.set_text_props(weight="bold")
    note_text = (
        "Each alias lists its full input before filters. Repeated aliases can "
        "refer to the same underlying set.\n"
        "Counts come from the saved corpus manifest. SoL uses the same inputs "
        "and the saved reference labels for survivors.")
    figure.text(0.055, 0.075, note_text, fontsize=11, linespacing=1.5)
    return figure


def plot_comparison(title, queries, rows, relations, sol, name, overview=False):
    """Export metric charts and input counts as a vector PDF."""
    destination = HERE / "plots" / name
    groups = [[metric] for metric, _, _ in METRICS] if overview else [
        [metric for metric, _, _ in METRICS]]
    with PdfPages(destination.with_suffix(".pdf")) as pdf:
        for metrics in groups:
            single = len(metrics) == 1
            figure, axes = (plt.subplots(1, 1, figsize=(14, 8.5)) if single
                            else plt.subplots(2, 2, figsize=(14, 10)))
            axes = [axes] if single else list(axes.flat)
            for axis, metric in zip(axes, metrics):
                metric_bars(axis, queries, rows, sol, metric, overview)
            figure.suptitle(f"{title}, Qwen3 4B FP8, sf=0.1, one H100", y=0.97,
                            fontsize=16)
            handles = [Patch(facecolor=color, label=label)
                       for _, label, color in METHODS]
            if any(metric != "agreement" for metric in metrics):
                handles.append(Line2D([0], [0], color=DARK, linewidth=1.7,
                                      label="SoL estimate"))
            figure.legend(handles=handles, loc="upper center",
                          bbox_to_anchor=(0.5, 0.925), ncol=5, fontsize=11,
                          frameon=False)
            figure.subplots_adjust(left=0.075, right=0.97, top=0.83,
                                   bottom=0.14 if overview else 0.10, hspace=0.60,
                                   wspace=0.28)
            note = "Raw prompts. References: Qwen3 32B FP8 and dataset annotations."
            empty = [q for q in queries if q in rows["quail"]
                     and not rows["quail"][q]["expected_rows"]]
            if empty:
                note += " Empty reference output: " + ", ".join(empty) + "."
            figure.text(0.075, 0.02, note + " x = not measured.", fontsize=9)
            pdf.savefig(figure, bbox_inches=None)
            plt.close(figure)
        figure = document_page(title, queries, relations)
        pdf.savefig(figure, bbox_inches=None)
        plt.close(figure)
    return destination.with_suffix(".pdf").name


def reference_requested_tokens(query, answer, estimate):
    """Replay the saved SoL stage order to count its complete input prompts."""
    from quail.planner.estimate import _Search
    from quail.planner.live_rows import PairRelation, exact_live_rows

    session = query.session
    search = _Search(query, answer, session.model, session.device,
                     estimate["chunk_tokens"], False)
    live = {alias: list(range(len(data.tokens)))
            for alias, data in search.aliases.items()}
    total = 0
    for stage in estimate["filter_stages"]:
        alias = stage["alias"]
        predicate = search.filters[alias][stage["written_pos"]]
        assert len(live[alias]) == stage["evaluated"]
        total += sum(search.pre + search.aliases[alias].tokens[row]
                     + predicate.prompt.tail_tokens for row in live[alias])
        live[alias] = [row for row in live[alias]
                       if answer(predicate.prompt, {alias: row})]
        assert len(live[alias]) == stage["passed"]
    base_rows = dict(live)
    edges = []
    for stage in estimate["join_stages"]:
        position = stage["written_pos"]
        prompt = search.joins[position].prompt
        anchor = stage["anchor"]
        (partner,) = stage["partners"]
        labels = {alias: (label, frame) for alias, label, frame in prompt.labels}
        fixed = search.pre + labels[anchor][1] + labels[partner][0]
        fixed += prompt.tail_tokens
        allowed = search.allowed_pairs.get(position, {}).get(anchor)
        passing = set()
        evaluated = 0
        for left in live[anchor]:
            for right in live[partner]:
                if allowed is not None and right not in allowed.get(left, ()):
                    continue
                total += (fixed + search.aliases[anchor].tokens[left]
                          + search.aliases[partner].tokens[right])
                evaluated += 1
                if answer(prompt, {anchor: left, partner: right}):
                    passing.add((left, right))
        assert evaluated == stage["evaluated_pairs"]
        assert len(passing) == stage["passing_pairs"]
        edges.append(PairRelation(anchor, partner, frozenset(passing)))
        live = exact_live_rows(base_rows, edges)
    return total


def saved_directory(root, method, item):
    """Locate the saved query files, including the earlier BioDEX rerun."""
    directory = root / SOURCE_RUN / method / item["directory"]
    if not (directory / item["files"]["plan"]).exists():
        candidates = [p.parent for p in (root / "benchmarks/quailb").rglob(
            item["files"]["plan"])
            if "20260914T070913Z-f7beefb6-biodex-rerun" in str(p)
            and method in p.parts and p.parent.name == item["id"]]
        assert len(candidates) == 1, (item["id"], candidates)
        directory = candidates[0]
    return directory


def prepare(workdir, root, sol_path):
    """Validate and rescore saved raw answers without inference."""
    from functools import cache

    import quail
    from quail.bench.quailb import answer_oracle, build_query, prompt_pieces
    from quail.bench.substrait import read_plan
    from quail.planner.plan import EngineConfig
    from quail_b.benchmark import load_benchmark
    from quail_b.minimum import load_tokenizer
    from quail_b.queries import QuerySpec, get_query
    from quail_b.run import _query_hash, _read_output, _score

    root = Path(root)
    benchmark = load_benchmark(
        scale_factor=0.1, root=str(root), collection_id=COLLECTION)
    tokenizer = load_tokenizer("Qwen/Qwen3-4B-FP8")

    @cache
    def encode(text):
        return tuple(tokenizer([text])[0])

    session = quail.Session(EngineConfig(
        model="qwen3-4b-fp8", device="h100-sxm"), tokenizer=encode)
    for name, table in benchmark.tables.items():
        session.register(name, quail.DocumentProvider.from_table(table, id_col="id"))
    result = {
        "prompt_format": "raw-v1", "source_run": "/results/" + SOURCE_RUN,
        "collection_id": COLLECTION, "corpus_id": benchmark.ground_truth.corpus_id,
        "source_hashes": {}, "rows": {}, "missing": {}, "settings": {},
        "query_hashes": {q: _query_hash(get_query(q)) for q in BASE_QUERY_ORDER},
        "relations": {
            q: [(r.alias, r.table, len(benchmark.tables[r.table]))
                for r in get_query(q)._info.relations] for q in BASE_QUERY_ORDER},
    }
    stores = {}
    for method, _, _ in METHODS:
        raw = (root / SOURCE_RUN / method / "run.json").read_bytes()
        suite = json.loads(raw)
        assert suite["corpus_id"] == result["corpus_id"]
        assert suite["scale_factor"] == 0.1
        assert suite["metadata"]["model"] == "qwen3-4b-fp8"
        result["source_hashes"][method] = hashlib.sha256(raw).hexdigest()
        result["rows"][method] = {}
        result["missing"][method] = {}
        result["settings"][method] = {}
        for item in suite["queries"]:
            if item["id"] not in CURRENT_QUERY_IDS:
                continue
            qid = CURRENT_QUERY_IDS[item["id"]]
            directory = saved_directory(root, method, item)
            spec = get_query(qid)
            saved = QuerySpec(qid, "Saved query", (
                directory / item["files"]["plan"]).read_bytes())
            assert _query_hash(saved) == item["definition_hash"], qid
            if _query_hash(saved) != _query_hash(spec):
                assert qid in ("BIO-1", "BIO-3"), qid
                result["missing"][method][qid] = "Saved query uses the old filter."
                continue
            assert item["status"] == "complete", qid
            output = _read_output(directory, item, rows=False)
            query = build_query(session, spec)
            positions = {j.id: n for n, j in enumerate(spec._info.joins)}
            anchors = {positions[p["id"]]: p["anchor"]
                       for p in output.prompt_pieces["joins"]}
            expected = prompt_pieces(query, read_plan(spec.plan), anchors)
            assert output.prompt_pieces == expected, (method, qid, "prompt pieces")
            metrics = _score(spec, output, benchmark, 1, H100_USD_PER_HOUR, stores)
            token_counter = None
            if method == "pipelined_vllm":
                token_counter = (output.measurements["fresh_tokens"]
                                 + output.measurements["cached_tokens"])
                # The saved runs used bpe-qwen for document tokenization.
                assert abs(metrics["input_tokens"] - token_counter) <= (
                    token_counter * 0.00001), qid
            accuracy = metrics["accuracy"]
            answers = accuracy["answer_accuracy"]
            final = accuracy["output_accuracy"]
            result["rows"][method][qid] = {
                "runtime_s": item["runtime_s"],
                "requested_tokens": metrics["input_tokens"],
                "recorded_input_tokens": token_counter,
                "fresh_tokens": metrics["fresh_tokens"],
                "regret_tokens": metrics["regret_tokens"],
                "matching_rows": final["matching_rows"],
                "predicted_rows": final["predicted_rows"],
                "expected_rows": final["expected_rows"],
                "answers_correct": answers["correct"],
                "answers_evaluated": answers["evaluated"],
                "evaluated_document_pairs": metrics["evaluated_document_pairs"],
                "input_rows": sum(metrics["input_rows"].values()),
                "source_directory": str(directory),
            }
            backend = item["measurements"].get("backend_metrics") or {}
            result["settings"][method][qid] = backend.get("capacity")
            print("Rescored saved answers:", method, qid, flush=True)
    source = json.loads(Path(sol_path).read_text())
    assert source["corpus_id"] == result["corpus_id"]
    assert source["collection_id"] == COLLECTION
    assert source["optimizer"]["persistent_kv_capacity"] == "unlimited"
    answer = answer_oracle(benchmark.ground_truth, benchmark.tables)
    result["sol"] = {}
    for qid in BASE_QUERY_ORDER:
        spec = get_query(qid)
        plan_hash = hashlib.sha256(spec.plan_bytes).hexdigest()
        matching = [r for r in source["queries"].values()
                    if r["plan_sha256"] == plan_hash]
        assert len(matching) == 1, qid
        record = matching[0]
        assert record["prompt_format"] == "raw-v1"
        assert record["plan_sha256"] == hashlib.sha256(spec.plan_bytes).hexdigest()
        estimate = record["models"]["qwen3-4b-fp8"]
        estimate["requested_tokens"] = reference_requested_tokens(
            build_query(session, spec), answer, estimate)
        result["sol"][qid] = estimate
        print("Counted SoL prompts:", qid, flush=True)
    result["sol_source"] = str(sol_path)
    result["sol_sha256"] = hashlib.sha256(Path(sol_path).read_bytes()).hexdigest()
    session.close()
    Path(workdir, "comparison.json").write_text(json.dumps(result, indent=2) + "\n")


def measurement_row(suite, source_directory):
    """Convert one saved BIO-4 suite into a comparison row."""
    assert suite["status"] == "complete"
    assert suite["metadata"]["model"] == "qwen3-4b-fp8"
    assert suite["metadata"]["prompt_format"] == "raw-v1"
    assert suite["gpu_count"] == 1
    assert len(suite["queries"]) == 1
    item = suite["queries"][0]
    assert item["id"] == "BIO-4"
    assert item["status"] == "complete"
    metrics = item["metrics"]
    assert metrics["runtime_s"] == item["runtime_s"]
    accuracy = metrics["accuracy"]
    answers = accuracy["answer_accuracy"]
    final = accuracy["output_accuracy"]
    measurements = item["measurements"]
    recorded = None
    if measurements.get("cached_tokens") is not None:
        recorded = measurements["fresh_tokens"] + measurements["cached_tokens"]
    return {
        "runtime_s": item["runtime_s"],
        "requested_tokens": metrics["input_tokens"],
        "recorded_input_tokens": recorded,
        "fresh_tokens": metrics["fresh_tokens"],
        "regret_tokens": metrics["regret_tokens"],
        "matching_rows": final["matching_rows"],
        "predicted_rows": final["predicted_rows"],
        "expected_rows": final["expected_rows"],
        "answers_correct": answers["correct"],
        "answers_evaluated": answers["evaluated"],
        "evaluated_document_pairs": metrics["evaluated_document_pairs"],
        "input_rows": sum(metrics["input_rows"].values()),
        "source_directory": source_directory,
    }


def load_bio4(workdir, query_hash, plan_hash):
    """Load and validate the saved BIO-4 measurements and SoL estimates."""
    workdir = Path(workdir)
    result = {}
    for scale, stem in ((0.1, "bio4-sf0.1"), (1.0, "bio4-sf1")):
        suites = {
            method: json.loads((workdir / f"{stem}-{suffix}.json").read_text())
            for method, suffix in (("quail", "quail"),
                                   ("pipelined_vllm", "vllm"))
        }
        sol_source = json.loads((workdir / f"{stem}-sol.json").read_text())
        for method, suite in suites.items():
            assert suite["scale_factor"] == scale
            assert suite["collection_id"] == BIO4_COLLECTIONS[scale]
            assert suite["reference_model"] == "qwen3-32b-fp8"
            assert suite["queries"][0]["definition_hash"] == query_hash
            assert suite["queries"][0]["metrics"]["accuracy"][
                "ground_truth_collection_id"] == BIO4_COLLECTIONS[scale]
            if method == "pipelined_vllm":
                assert suite["metadata"]["engine"] == "pipelined_vllm"
        assert suites["quail"]["corpus_id"] == suites["pipelined_vllm"][
            "corpus_id"]
        assert sol_source["query"] == "BIO-4"
        assert sol_source["scale_factor"] == scale
        assert sol_source["corpus_id"] == suites["quail"]["corpus_id"]
        assert sol_source["collection_id"] == BIO4_COLLECTIONS[scale]
        assert sol_source["reference_model"] == "qwen3-32b-fp8"
        assert sol_source["plan_sha256"] == plan_hash
        assert sol_source["estimate"]["model"] == "qwen3-4b-fp8"
        assert sol_source["estimate"]["assumptions"][
            "persistent_kv_capacity"] == "unlimited"
        rows = {
            method: measurement_row(
                suite, f"/results/{BIO4_RUNS[scale]}/{method}/biodex")
            for method, suite in suites.items()
        }
        settings = suites["pipelined_vllm"]["queries"][0]["measurements"][
            "backend_metrics"]["capacity"]
        estimate = dict(sol_source["estimate"])
        estimate["volume_path"] = sol_source["volume_path"]
        result[scale] = {
            "rows": rows,
            "sol": estimate,
            "relations": [
                (alias, column.split(".", 1)[0],
                 estimate["documents_by_alias"][alias])
                for alias, column in estimate["alias_columns"].items()
            ],
            "setting": settings,
        }
    return result


def sol_metrics(estimate):
    """Derive throughput and cost for one SoL estimate."""
    work = estimate.get("join_pair_evaluations") or estimate["input_document_rows"]
    return {
        "seconds": estimate["sol_s"],
        "fresh_tokens": estimate["tokens"],
        "regret_tokens": 0,
        "regret_percent": 0,
        "throughput": work / estimate["sol_s"],
        "throughput_unit": ("document pairs/s"
                            if estimate.get("join_pair_evaluations")
                            else "documents/s"),
        "cost": estimate.get(
            "usd_per_query", estimate["sol_s"] / 3600 * H100_USD_PER_HOUR),
    }


def main(workdir):
    """Regenerate all benchmark figures and the report from the saved summary."""
    data = json.loads(Path(workdir, "comparison.json").read_text())
    assert data["prompt_format"] == "raw-v1"
    assert data["collection_id"] == COLLECTION
    from quail_b.queries import get_query
    from quail_b.run import _query_hash

    bio4_spec = get_query("BIO-4")
    bio4 = load_bio4(
        workdir, _query_hash(bio4_spec),
        hashlib.sha256(bio4_spec.plan_bytes).hexdigest())
    by_hash = {value: key for key, value in data["query_hashes"].items()}
    sources = {q: by_hash[_query_hash(get_query(q))] for q in BASE_QUERY_ORDER}
    rows = {method: {q: data["rows"][method][source]
                     for q, source in sources.items()
                     if source in data["rows"][method]}
            for method, _, _ in METHODS}
    sol = {q: data["sol"][source] for q, source in sources.items()}
    relations = {q: data["relations"][source] for q, source in sources.items()}
    for method, _, _ in METHODS:
        rows[method]["BIO-4"] = bio4[0.1]["rows"][method]
    sol["BIO-4"] = bio4[0.1]["sol"]
    relations["BIO-4"] = bio4[0.1]["relations"]
    for method, _, _ in METHODS:
        assert set(rows[method]) == set(QUERY_ORDER) - {"BIO-1", "BIO-3"}
    plt.style.use(HERE / "quail.mplstyle")
    plt.rcParams.update({"pdf.fonttype": 42, "figure.autolayout": False,
                         "savefig.bbox": None})
    plot_comparison("QUAIL-B", QUERY_ORDER, rows, relations, sol,
                    "quailb_main.pdf", overview=True)
    paired = [q for q in QUERY_ORDER if all(q in rows[m] for m, _, _ in METHODS)]
    speedups = {q: rows["pipelined_vllm"][q]["runtime_s"]
                / rows["quail"][q]["runtime_s"] for q in paired}
    fastest = max(speedups, key=speedups.get)
    settings = [data["settings"]["pipelined_vllm"][sources[q]]
                for q in paired if q != "BIO-4"] + [bio4[0.1]["setting"]]
    batch = sorted({s["max_num_batched_tokens"] for s in settings})
    capacity = [s["kv_cache_size_tokens"] for s in settings]
    capacity_text = (f"{min(capacity):,}" if min(capacity) == max(capacity)
                     else f"{min(capacity):,} to {max(capacity):,}")
    sequences = sorted({s["max_num_seqs"] for s in settings})
    bio4_sf01_speedup = speedups["BIO-4"]
    bio4_sf1_speedup = (
        bio4[1.0]["rows"]["pipelined_vllm"]["runtime_s"]
        / bio4[1.0]["rows"]["quail"]["runtime_s"])
    report_text = [
        "# QUAIL-B comparison", "",
        "- All 31 queries use raw document/question prompts ending in `ANSWER:`.",
        "  LePaRD has five queries. Original LEP-7 is now LEP-5.",
        "  Both methods use Qwen3 4B FP8, sf=0.1, and one H100.",
        "- Measurements are reused from saved raw runs. No inference was repeated.",
        f"  Base source on `quail-results`: `{data['source_run']}`.",
        f"  BIO-4 source: `/results/{BIO4_RUNS[0.1]}`.",
        "  Base saved plans and prompt token pieces match the restored definitions.",
        "  Base scores and token counts were recalculated from saved answers.",
        "  BIO-4 uses the saved current query hash, plan hash, and runner scores.",
        "- BIO-1 and BIO-3 are not measured for the serious-adverse-event filter.",
        "  The saved demographic-filter results do not match those queries.",
        "  All plots include both queries, with missing measurements marked x.",
        f"  Comparisons below use the {len(paired)} queries with both measurements.",
        "- Original LEP-5, LEP-6, and LEP-8 are excluded because their reference",
        "  outputs are empty at sf=0.1 with raw prompts too. Historical IDs are",
        "  matched by query-definition hash before measurements are reused.",
        "- References use Qwen3 32B FP8 and the benchmark's dataset annotations.",
        f"  Base reference collection: `{COLLECTION}`.",
        f"  BIO-4 sf=0.1 reference collection: `{BIO4_COLLECTIONS[0.1]}`.",
        "  Answer agreement counts matching predicate answers. Output precision",
        "  and recall compare final rows with the reference output.",
        "- Quail uses pipelining, token-based admission, and KV rewind.",
        "  Stock vLLM uses pipelining and prefix caching. Filter stages advance",
        "  independently; joins begin after filtering finishes.",
        f"  vLLM batched-token limits: {', '.join(f'{n:,}' for n in batch)}.",
        f"  Sequence limits: {', '.join(f'{n:,}' for n in sequences)}.",
        f"  Measured vLLM KV capacity: {capacity_text} tokens.",
        f"- Quail is faster on {sum(s > 1 for s in speedups.values())} of "
        f"{len(paired)} queries.",
        f"  Mean speedup: {mean(speedups.values()):.2f}x; median: "
        f"{median(speedups.values()):.2f}x; maximum: "
        f"{speedups[fastest]:.2f}x ({fastest}).",
        "  Each query has equal weight. Speedup is vLLM time divided by Quail time.",
        "- Latency excludes startup and result collection. GPU cost is query",
        f"  seconds / 3,600 * ${H100_USD_PER_HOUR:.4f}.",
        "  Throughput is input documents/second for filter-only queries. For queries",
        "  with joins, it is evaluated document pairs across all join stages divided",
        "  by query seconds. Different answers can change the work each method does.",
        "- KV regret is recomputed tokens / fresh computed tokens * 100%.",
        "  The minimum computes each distinct input prefix once with unlimited KV.",
        "  A pair's partner suffix is counted after its anchor. Regret is",
        "  recalculated with the current benchmark rule from saved prompt pieces.",
        "- SoL estimates ideal compute and memory time with unlimited retained KV",
        "  and exact raw-reference survivors. Matching prefixes are reused across",
        "  requests, documents, and aliases. The join search uses left-deep plans.",
        "  It excludes software overhead. Different answers change the work, so",
        "  the gap from SoL is not solely execution overhead. It has no accuracy.",
        "  These estimates were recalculated on CPU for the restored queries.",
        f"  Base estimates: `{data['sol_source']}`.",
        f"  BIO-4 estimate: `{bio4[0.1]['sol']['volume_path']}`.",
        "- PDFs show latency, fresh input tokens, recomputed KV tokens, and answer",
        "  agreement. Each PDF lists input counts separately for every alias.",
        "  SoL uses lines for latency and token totals. Measurements use bars.",
        "  A dash marks zero. An x marks a missing measurement.", "",
        "[Main comparison PDF](plots/quailb_main.pdf)", "",
        "CPU-derived summary on `quail-results`:",
        "`/results/reports/quailb-raw-2026-09-19/comparison.json`.", "",
        "Rebuild commands: `reports/make_quailb_comparison_plots.py`.", "",
    ]
    for family in ("IMDB", "BIO", "FEV", "LEP", "AGENT"):
        selected = [q for q in QUERY_ORDER if q.startswith(family + "-")]
        name = plot_comparison(f"QUAIL-B {family}", selected, rows, relations,
                               sol, f"quailb_{family.lower()}.pdf")
        report_text.extend([f"## {family}", "",
                            f"[{family} comparison PDF](plots/{name})", "",
                            "| Query | Input documents by alias and set |",
                            "|---|---|"])
        for q in selected:
            counts = ", ".join(f"{a} ({t}) = {n:,}" for a, t, n in relations[q])
            report_text.append(f"| {q} | {counts} |")
        report_text.extend([
            "", "| Query | Method | Seconds | Throughput | Unit | $/query "
            "| Fresh tokens | Recomputed KV tokens | KV regret (%) |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|"])
        for q in selected:
            for method, label, _ in METHODS:
                if q not in rows[method]:
                    report_text.append(
                        f"| {q} | {label} | Not measured | | | | | | |")
                    continue
                m = row_metrics(rows[method][q])
                report_text.append(
                    f"| {q} | {label} | {m['seconds']:.2f} "
                    f"| {m['throughput']:,.2f} | {m['throughput_unit']} "
                    f"| {m['cost']:.5f} | {m['fresh_tokens']:,.0f} "
                    f"| {m['regret_tokens']:,.0f} | {m['regret_percent']:.2f} |")
            v = sol_metrics(sol[q])
            report_text.append(
                f"| {q} | SoL estimate | {v['seconds']:.3f} "
                f"| {v['throughput']:,.2f} | {v['throughput_unit']} "
                f"| {v['cost']:.5f} | {v['fresh_tokens']:,.0f} "
                "| 0 (assumed) | 0 (assumed) |")
        report_text.extend([
            "", "| Query | Method | Reference rows | Returned rows "
            "| Answer agreement (%) "
            "| Output precision (%) | Output recall (%) |",
            "|---|---|---:|---:|---:|---:|---:|"])
        for q in selected:
            for method, label, _ in METHODS:
                if q not in rows[method]:
                    report_text.append(f"| {q} | {label} | | Not measured | | | |")
                    continue
                m = row_metrics(rows[method][q])
                row = rows[method][q]
                recall = (f"{m['recall']:.5g}" if row["expected_rows"]
                          else "Not defined")
                report_text.append(
                    f"| {q} | {label} | {row['expected_rows']:,} "
                    f"| {row['predicted_rows']:,} | {m['agreement']:.2f} "
                    f"| {m['precision']:.5g} | {recall} |")
        report_text.append("")
        if family == "BIO":
            sf1 = bio4[1.0]
            sf1_counts = {alias: count for alias, _, count in sf1["relations"]}
            quail_row = sf1["rows"]["quail"]
            vllm_row = sf1["rows"]["pipelined_vllm"]
            report_text.extend([
                "### BIO-4 at sf=1.0", "",
                f"The prediction was that Quail would beat its "
                f"{bio4_sf01_speedup:.2f}x "
                "speedup at sf=0.1. The working target was 10x. At sf=1.0, Quail "
                f"was {bio4_sf1_speedup:.2f}x faster.", "",
                f"Quail took {quail_row['runtime_s'] / 60:.2f} minutes. Pipelined "
                f"stock vLLM took {vllm_row['runtime_s'] / 3600:.2f} hours. Quail "
                f"recomputed {quail_row['regret_tokens']:,} KV tokens, compared with "
                f"{vllm_row['regret_tokens']:,} for pipelined stock vLLM.", "",
                "The methods evaluated different numbers of document pairs because",
                "their answers changed which rows reached the joins. The throughput",
                "for each method uses its own evaluated pair count.", "",
                "Both methods returned many incorrect final rows. Quail's output",
                "precision was 1.57%, compared with 1.41% for pipelined stock vLLM.",
                "Output recall was 22.02% for Quail and 22.57% for stock vLLM.", "",
                "| Method | Seconds | Document pairs/s | $/query | Fresh tokens "
                "| Recomputed KV tokens | KV regret (%) | Answer agreement (%) "
                "| Output precision (%) | Output recall (%) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for method, label, _ in METHODS:
                row = sf1["rows"][method]
                m = row_metrics(row)
                report_text.append(
                    f"| {label} | {m['seconds']:.2f} | {m['throughput']:,.2f} "
                    f"| {m['cost']:.5f} | {m['fresh_tokens']:,.0f} "
                    f"| {m['regret_tokens']:,.0f} | {m['regret_percent']:.2f} "
                    f"| {m['agreement']:.2f} | {m['precision']:.5g} "
                    f"| {m['recall']:.5g} |")
            v = sol_metrics(sf1["sol"])
            report_text.extend([
                f"| SoL estimate | {v['seconds']:.3f} | {v['throughput']:,.2f} "
                f"| {v['cost']:.5f} | {v['fresh_tokens']:,.0f} "
                "| 0 (assumed) | 0 (assumed) | Not measured | Not measured "
                "| Not measured |", "",
                f"Input documents: r (reports) = {sf1_counts['r']:,}, "
                f"n (terms) = {sf1_counts['n']:,}, and "
                f"c (terms) = {sf1_counts['c']:,}.", "",
                f"Measured source on `quail-results`: `/results/{BIO4_RUNS[1.0]}`.",
                f"SoL source: `{sf1['sol']['volume_path']}`.",
                f"Reference collection: `{BIO4_COLLECTIONS[1.0]}`.", "",
                "Quail result function call: `fc-01M2YRXV1TAVKDDPPBKNHM90XH`.",
                "Stock vLLM result function call: `fc-01M2YXPZA8E6EJYJMSGDTR0X69`.", "",
            ])
    (HERE / "quailb-comparison.md").write_text("\n".join(report_text))
    print("Updated the report and all six PDFs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--root")
    parser.add_argument("--sol-file")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.workdir, args.root, args.sol_file)
    else:
        main(args.workdir)

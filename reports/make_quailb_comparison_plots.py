"""Generate the QUAIL-B overview and dataset plots without inference.

Pull the saved measurements, corpus manifest, and SoL estimates:

    W=/tmp/quail-comparison; mkdir -p "$W/run"
    RUN=benchmarks/quailb/family-runs/20260918T071546Z-all-chat
    uv run modal volume get quail-results "$RUN/manifest.json" "$W/run/manifest.json"
    uv run modal volume get quail-results "$RUN/measurements.parquet" \
      "$W/run/measurements.parquet"
    for METHOD in quail pipelined_vllm; do
      mkdir -p "$W/run/$METHOD"
      uv run modal volume get quail-results "$RUN/$METHOD/run.json" \
        "$W/run/$METHOD/run.json"
    done
    uv run modal volume get quail-results "$RUN/requested_tokens.json" \
      "$W/requested_tokens.json"
    CORPUS=ground_truth/quailb/schema_v1/corpora/c_1aa2c4f0d0b6c816fd37aa5748c33341
    uv run modal volume get quail-results "$CORPUS/manifest.json" "$W/corpus.json"
    uv run modal volume get quail-results \
      sol/2026-09-18-all-chat/sol_quailb_sf0.1.json "$W/sol.json"
    B="$W/bio-comparison"; mkdir -p "$B"
    OLD=benchmarks/quailb/family-runs/20260912T225100Z-902686c5
    NEW=benchmarks/quailb/family-runs/20260918T060700Z-biodex-chat
    uv run modal volume get quail-results "$OLD/quail/biodex/BIO-2/joins-0.parquet" \
      "$B/raw.parquet"
    uv run modal volume get quail-results "$NEW/quail/biodex/BIO-2/joins-0.parquet" \
      "$B/chat.parquet"
    for TABLE in reports terms; do
      uv run modal volume get quail-results "quailb_data/sf0.1/$TABLE.parquet" \
        "$B/$TABLE.parquet"
    done
    LABELS=ground_truth/quailb/schema_v1/label_sets/biodex/report_experienced_reaction
    uv run modal volume get quail-results \
      "$LABELS/ls_558442b4193e9a48bfe1aea9bc87a66a/labels.parquet" \
      "$B/raw-reference.parquet"
    uv run modal volume get quail-results \
      "$LABELS/ls_4c0b69af26ad590e1b64ac5ffebf2484/labels.parquet" \
      "$B/chat-reference.parquet"
    uv run modal volume get quail-results \
      ablations/bio2-prompt-layout-20260918T155056Z/result.json "$B/layout.json"
    BENCH=git+https://github.com/fsdatalab/quail-bench.git
    REV=d93a53583a0de61978916f349f21c9ecd6ce1ba3
    uv run --with matplotlib --with "quail-b@$BENCH@$REV" \
      python reports/make_quailb_comparison_plots.py "$W"

To recount complete input prompts on a mounted results volume with Quail
revision 307e4b2 and the benchmark revision above, run on CPU:

    python reports/make_quailb_comparison_plots.py /results/$RUN \
      --count-requested-tokens --root /results \
      --sol-file /results/sol/2026-09-18-all-chat/sol_quailb_sf0.1.json

All queries use non-thinking chat prompts. The saved run reuses the completed
September 18 BioDEX measurements after checking its reference label identities.
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from plot_colors import BLUE, DARK, ORANGE

from quail.specs import H100_USD_PER_HOUR

HERE = Path(__file__).resolve().parent
METHODS = [
    ("quail", "Quail", BLUE),
    ("pipelined_vllm", "Pipelined vLLM", ORANGE),
]
QUERY_ORDER = (
    [f"IMDB-{n}" for n in range(1, 11)] + [f"BIO-{n}" for n in range(1, 4)]
    + [f"FEV-{n}" for n in range(1, 11)] + [f"LEP-{n}" for n in range(1, 6)]
    + ["AGENT-1", "AGENT-2"])

SOURCE_QUERY_IDS = {q: "LEP-7" if q == "LEP-5" else q for q in QUERY_ORDER}
CURRENT_QUERY_IDS = {source: q for q, source in SOURCE_QUERY_IDS.items()}


def row_metrics(row):
    """Derive throughput, GPU cost, and accuracy from one measurement row."""
    matched = row["matching_rows"]
    predicted = row["predicted_rows"]
    expected = row["expected_rows"]
    return {
        "seconds": row["runtime_s"],
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
    }


METRICS = (
    ("seconds", "Latency", "seconds"),
    ("tokens_per_second", "Total requested input tokens per second", "tokens/second"),
    ("cost", "GPU cost per query", "dollars/query"),
    ("cost_per_million", "GPU cost per million input tokens", "dollars/million tokens"),
    ("regret_percent", "Recomputed share of computed tokens", "percent"),
)


def input_relations(queries, corpus):
    """Resolve each query alias to its saved input table count."""
    from quail.bench.substrait import read_plan
    from quail_b.queries import get_query

    return {
        query: [
            (relation.alias, relation.table,
             corpus["tables"][relation.table]["rows"])
            for relation in read_plan(get_query(query).plan).relations]
        for query in queries
    }


def load_sol(root, queries, rows, corpus, manifest):
    """Load estimates matching the measured query and reference identities."""
    from quail_b.queries import get_query
    from quail_b.rendering import PROMPT_FORMAT

    source = json.loads((root / "sol.json").read_text())
    assert source["corpus_id"] == corpus["corpus_id"]
    assert source["collection_id"] == manifest["collection_id"]
    assert source["scale_factor"] == 0.1
    assert source["optimizer"]["persistent_kv_capacity"] == "unlimited"
    assert set(SOURCE_QUERY_IDS.values()) <= set(source["queries"])
    estimates = {}
    for query in queries:
        record = source["queries"][SOURCE_QUERY_IDS[query]]
        assert record["prompt_format"] == PROMPT_FORMAT
        assert record["plan_sha256"] == hashlib.sha256(
            get_query(query).plan_bytes).hexdigest()
        estimate = record["models"]["qwen3-4b-fp8"]
        assert estimate["input_document_rows"] == rows["quail"][query]["input_rows"]
        assert estimate["tokens"] <= estimate["per_document"]["tokens"], query
        estimates[query] = estimate
    counts = json.loads((root / "requested_tokens.json").read_text())
    sol_hash = hashlib.sha256((root / "sol.json").read_bytes()).hexdigest()
    assert counts["sol_sha256"] == sol_hash
    for query, estimate in estimates.items():
        estimate["requested_tokens"] = counts["sol"][SOURCE_QUERY_IDS[query]]
    return estimates


def series_value(rows, sol, method, query, metric):
    """Return a measured value or an explicitly modeled value."""
    if method == "sol":
        seconds = sol[query]["sol_s"]
        cost = seconds / 3600 * H100_USD_PER_HOUR
        return {
            "seconds": seconds,
            "tokens_per_second": sol[query]["requested_tokens"] / seconds,
            "cost": cost,
            "cost_per_million": cost / sol[query]["requested_tokens"] * 1e6,
            "regret_percent": 0,
        }[metric]
    if query not in rows[method]:
        return None
    return row_metrics(rows[method][query])[metric]


def metric_bars(axis, queries, rows, sol, metric, overview):
    """Draw measured bars and a SoL line across each query group."""
    measured = METHODS
    unit = next(unit for key, _, unit in METRICS if key == metric)
    methods = measured + [("sol", "SoL estimate", DARK)]
    positive = [value for key, _, _ in methods for query in queries
                if (value := series_value(rows, sol, key, query, metric)) is not None
                and value > 0]
    maximum = max(positive, default=0)
    logarithmic = (metric != "regret_percent" and positive
                   and maximum / min(positive) > 10)
    if metric == "regret_percent":
        axis.set_ylim(0, min(105, max(1, maximum * 1.4)))
        axis.set_ylabel("percent")
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
                if metric in ("cost", "cost_per_million"):
                    shown = f"{value:.2g}"
                axis.annotate(shown, (position, value), xytext=(0, 3),
                              textcoords="offset points", rotation=90,
                              ha="center", va="bottom", fontsize=8)
        axis.bar(xs, [value - floor for value in values], bottom=floor,
                 width=width * 0.9, color=color, label=label, edgecolor="none")
    if metric == "seconds" and not overview:
        for index, query in enumerate(queries):
            quail_time = rows["quail"][query]["runtime_s"]
            baseline_time = rows["pipelined_vllm"][query]["runtime_s"]
            top = max(quail_time, baseline_time)
            ratio = baseline_time / quail_time
            axis.annotate(f"{ratio:.2f}x", (index, top), xytext=(0, 35),
                          textcoords="offset points", ha="center",
                          va="bottom", fontsize=8)
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


def plot_comparison(title, queries, rows, relations, sol, name, overview=False,
                    model_label="Qwen3 4B FP8",
                    reference_note="Reference labels: Qwen3 32B FP8 and "
                                   "dataset annotations."):
    """Export metric charts and input counts as a vector PDF."""
    destination = HERE / "plots" / name
    groups = [[metric] for metric, _, _ in METRICS] if overview else [
        ["seconds", "cost", "cost_per_million", "regret_percent"],
        ["tokens_per_second"]]
    with PdfPages(destination.with_suffix(".pdf")) as pdf:
        for metrics in groups:
            single = len(metrics) == 1
            figure, axes = (plt.subplots(1, 1, figsize=(14, 8.5)) if single
                            else plt.subplots(2, 2, figsize=(14, 10)))
            axes = [axes] if single else list(axes.flat)
            for axis, metric in zip(axes, metrics):
                metric_bars(axis, queries, rows, sol, metric, overview)
            figure.suptitle(f"{title}, {model_label}, sf=0.1, one H100", y=0.97,
                            fontsize=16)
            handles = [Patch(facecolor=color, label=label)
                       for _, label, color in METHODS]
            handles.append(Line2D([0], [0], color=DARK, linewidth=1.7,
                                  label="SoL estimate"))
            figure.legend(handles=handles, loc="upper center",
                          bbox_to_anchor=(0.5, 0.925), ncol=5, fontsize=11,
                          frameon=False)
            figure.subplots_adjust(left=0.075, right=0.97, top=0.83,
                                   bottom=0.14 if overview else 0.10, hspace=0.60,
                                   wspace=0.28)
            figure.text(0.075, 0.02, reference_note, fontsize=9)
            pdf.savefig(figure, bbox_inches=None)
            plt.close(figure)
        figure = document_page(title, queries, relations)
        pdf.savefig(figure, bbox_inches=None)
        plt.close(figure)
    return destination.with_suffix(".pdf").name


def measurement_rows(path):
    """Read a run's measurements.parquet as {method: {query: row}}."""
    rows = {}
    for row in pq.read_table(path).to_pylist():
        rows.setdefault(row["method"], {})[row["query"]] = row
    return rows


def load_rows(root, corpus):
    """Load complete measurements for both methods and all 30 retained queries."""
    from quail_b.queries import get_query
    from quail_b.rendering import PROMPT_FORMAT
    from quail_b.run import _query_hash

    manifest = json.loads((root / "run" / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["model"] == "qwen3-4b-fp8"
    assert manifest["sf"] == 0.1
    assert set(SOURCE_QUERY_IDS.values()) <= set(manifest["query_ids"])
    assert set(manifest["methods"]) == {key for key, _, _ in METHODS}
    rows = measurement_rows(root / "run" / "measurements.parquet")
    counts = json.loads((root / "requested_tokens.json").read_text())
    for key in ("run_id", "corpus_id", "collection_id"):
        assert counts[key] == manifest[key]
    suites = {}
    for method, _, _ in METHODS:
        raw = (root / "run" / method / "run.json").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == counts["run_sha256"][method]
        suite = json.loads(raw)
        assert suite["corpus_id"] == corpus["corpus_id"]
        assert suite["collection_id"] == manifest["collection_id"]
        assert suite["metadata"]["prompt_format"] == PROMPT_FORMAT
        assert set(rows[method]) == set(manifest["query_ids"])
        assert {item["id"] for item in suite["queries"]} == set(manifest["query_ids"])
        for item in suite["queries"]:
            if item["id"] not in CURRENT_QUERY_IDS:
                continue
            query = CURRENT_QUERY_IDS[item["id"]]
            assert item["status"] == "complete"
            assert item["definition_hash"] == _query_hash(get_query(query))
            assert rows[method][item["id"]]["regret_tokens"] is not None
            rows[method][item["id"]]["requested_tokens"] = (
                counts["methods"][method][item["id"]])
        rows[method] = {
            query: rows[method][source]
            for query, source in SOURCE_QUERY_IDS.items()}
        suites[method] = suite
    return rows, manifest, suites


def requested_input_tokens(spec, output, documents):
    """Sum complete prompt lengths for every evaluated predicate answer."""
    import pyarrow as pa
    import pyarrow.compute as pc

    from quail_b.minimum import validate_prompt_pieces

    pieces = validate_prompt_pieces(spec, output.prompt_pieces)
    relations = {r.alias: (r.table, r.text_column) for r in spec._info.relations}

    def document_sum(table, alias):
        counts = pc.value_counts(pc.cast(table.column(alias), pa.string()))
        ids = counts.field("values").to_pylist()
        keys = [(*relations[alias], row_id) for row_id in ids]
        documents.fetch(keys)
        return sum(len(documents[key]) * count for key, count in zip(
            keys, counts.field("counts").to_pylist()))

    total = 0
    tails = {p["id"]: p["tail"] for p in pieces["filters"]}
    for operator in spec._info.filters:
        table = output.filter_answers[operator.id]
        total += document_sum(table, operator.relation)
        total += len(table) * (len(pieces["preamble"]) + len(tails[operator.id]))
    joins = {p["id"]: p for p in pieces["joins"]}
    for operator in spec._info.joins:
        table = output.join_answers[operator.id]
        piece = joins[operator.id]
        total += len(table) * sum(len(piece[key]) for key in (
            "frame", "label", "tail")) + len(table) * len(pieces["preamble"])
        total += sum(document_sum(table, alias) for alias in operator.relations)
    return total


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


def count_requested_tokens(run_directory, root, sol_path):
    """Save input counts from existing answers and the saved SoL plan on CPU."""
    from functools import cache

    import quail
    from quail.bench.quailb import answer_oracle, build_query
    from quail.planner.plan import EngineConfig
    from quail_b.benchmark import load_benchmark
    from quail_b.minimum import DocumentTokens, load_tokenizer
    from quail_b.queries import get_query
    from quail_b.run import _query_hash, _read_output

    run_directory = Path(run_directory)
    manifest = json.loads((run_directory / "manifest.json").read_text())
    benchmark = load_benchmark(
        scale_factor=0.1, root=root, collection_id=manifest["collection_id"])
    sol_bytes = Path(sol_path).read_bytes()
    sol = json.loads(sol_bytes)
    assert sol["corpus_id"] == manifest["corpus_id"]
    assert sol["collection_id"] == manifest["collection_id"]
    tokenizer = load_tokenizer("Qwen/Qwen3-4B-FP8")
    documents = DocumentTokens(benchmark.tables, tokenizer)
    result = {
        "run_id": manifest["run_id"], "corpus_id": manifest["corpus_id"],
        "collection_id": manifest["collection_id"],
        "sol_sha256": hashlib.sha256(sol_bytes).hexdigest(),
        "methods": {}, "sol": {}, "run_sha256": {},
    }
    for method, _, _ in METHODS:
        raw = (run_directory / method / "run.json").read_bytes()
        result["run_sha256"][method] = hashlib.sha256(raw).hexdigest()
        result["methods"][method] = {}
        for item in json.loads(raw)["queries"]:
            if item["id"] not in CURRENT_QUERY_IDS:
                continue
            spec = get_query(CURRENT_QUERY_IDS[item["id"]])
            assert item["definition_hash"] == _query_hash(spec)
            directory = run_directory / method / item["directory"]
            output = _read_output(directory, item, rows=False)
            total = requested_input_tokens(spec, output, documents)
            if method == "pipelined_vllm":
                measured = item["measurements"]
                assert total == measured["fresh_tokens"] + measured["cached_tokens"]
            result["methods"][method][item["id"]] = total
        print("Counted complete prompts:", method, flush=True)

    @cache
    def encode(text):
        return tuple(tokenizer([text])[0])

    session = quail.Session(EngineConfig(
        model="qwen3-4b-fp8", device="h100-sxm"), tokenizer=encode)
    try:
        for name, table in benchmark.tables.items():
            session.register(name, quail.DocumentProvider.from_table(
                table, id_col="id"))
        answer = answer_oracle(benchmark.ground_truth, benchmark.tables)
        for query in QUERY_ORDER:
            spec = get_query(query)
            saved = sol["queries"][SOURCE_QUERY_IDS[query]]
            assert saved["plan_sha256"] == hashlib.sha256(spec.plan_bytes).hexdigest()
            result["sol"][SOURCE_QUERY_IDS[query]] = reference_requested_tokens(
                build_query(session, spec), answer, saved["models"]["qwen3-4b-fp8"])
            print("Counted SoL prompts:", query, flush=True)
    finally:
        session.close()
    destination = run_directory / "requested_tokens.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(destination, flush=True)


def biodex_prompt_comparison(root):
    """Compare saved prompt runs and score BIO-2 against dataset annotations."""
    directory = root / "bio-comparison"
    reports = pq.read_table(directory / "reports.parquet", columns=["id", "reactions"])
    terms = pq.read_table(directory / "terms.parquet").to_pylist()
    term_ids = {row["term"]: row["id"] for row in terms}
    expected = {(row["id"], term_ids[reaction]) for row in reports.to_pylist()
                for reaction in row["reactions"] if reaction in term_ids}
    pairs = len(reports) * len(terms)
    layout = json.loads((directory / "layout.json").read_text())
    assert layout["query"] == "BIO-2" and layout["scale_factor"] == 0.1
    lines = [
        "## BIO-2 raw and chat comparison", "",
        "The chat closing adds nine tokens after each pair's text. Both",
        "benchmark runs evaluated 563,500 pairs. Quail's fresh input tokens rose",
        "49%, and its query time rose from 127.75 to 187.01 seconds. Its computed",
        "token rate stayed close to 82,000 per second.", "",
        "A separate pipelined stock vLLM experiment compared raw and chat prompts",
        "in one container. It used a fresh engine for each join and ran BIO-1",
        "first. The prediction was that times would differ by less than 5%,",
        "with chat no faster than raw.", "",
        "| Round | Format | Seconds | Milliseconds/pair | Fresh tokens |",
        "|---|---|---:|---:|---:|",
    ]
    times = {"raw": [], "chat": []}
    for run in layout["joins"]:
        assert run["pairs"] == pairs
        times[run["layout"]].append(run["wall_s"])
        lines.append(
            f"| {run['round'] + 1} | {run['layout']} | {run['wall_s']:.1f} "
            f"| {run['wall_s'] / pairs * 1000:.3f} | {run['fresh_tokens']:,} |")
    difference = (mean(times["chat"]) / mean(times["raw"]) - 1) * 100
    lines.extend([
        "", f"Chat time differed from raw by {difference:.2f}% on average.",
        "The result supports the 5% prediction, but chat was slightly faster.",
        "It did not reproduce the 13% vLLM time reduction between benchmark runs.",
        "Those runs used different machines. Host variation is a plausible",
        "explanation; this experiment does not isolate every host difference.", "",
        "Result on `quail-results`:",
        f"`{layout['result_volume_path']}`.",
        "Modal call: `fc-01M2TKCMJQVM0HV104NVCHTKY5`.",
        "The experiment is `experiments/bio2_prompt_layout.py` at commit",
        "`1f35316` on `claude/focused-carson-huqcvm`.", "",
        "Earlier attempts used a cold engine or encountered throttling. They",
        "are excluded from the controlled comparison above. Their records are",
        "`/results/ablations/bio2-prompt-layout-20260918T142630Z/joins.json`",
        "(`fc-01M2TEJJY4Z2G2DGG25NE8F28B`) and the cancelled call",
        "`fc-01M2TG6KG63GSZTN2X3BRZ43AD`.", "",
        "The September 12 recomputed KV values used an older minimum rule.",
        "That rule counted a pair's partner label once per anchor and shared",
        "partner document prefixes across pairs. It understated BIO-2's minimum",
        "by 4,146,500 tokens. Under the current rule, vLLM recomputed 154,747",
        "tokens in that run and 157,341 in the chat run. Those values are close.", "",
        "### Accuracy against the same dataset annotations", "",
        "The main correctness tables compare 4B answers with 32B reference",
        "answers. Those references changed when the prompt format changed.",
        "Here both formats are scored against the same reactions recorded in",
        f"BioDEX: {len(expected):,} positive pairs out of {pairs:,}.",
        "Matching is exact; a synonym absent from the recorded list counts as",
        "wrong. These scores therefore measure agreement with dataset annotations.",
        "F1 combines precision and recall and is shown as a percentage.", "",
        "| Model and format | Predicted matches | Precision (%) "
        "| Recall (%) | F1 (%) |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, title, columns in [
        ("raw", "4B raw", ["r", "m", "answer"]),
        ("chat", "4B chat", ["r", "m", "answer"]),
        ("raw-reference", "32B raw", ["left_id", "right_id", "answer"]),
        ("chat-reference", "32B chat", ["left_id", "right_id", "answer"]),
    ]:
        table = pq.read_table(directory / f"{name}.parquet", columns=columns)
        values = zip(*(table[column].to_pylist() for column in columns))
        answers = {(left, right): answer for left, right, answer in values}
        assert len(answers) == len(table) == pairs
        assert {left for left, _ in answers} == set(reports["id"].to_pylist())
        assert {right for _, right in answers} == set(term_ids.values())
        predicted = {pair for pair, answer in answers.items() if answer}
        matches = len(predicted & expected)
        precision = matches / len(predicted) if predicted else 0
        recall = matches / len(expected)
        f1 = 2 * matches / (len(predicted) + len(expected))
        lines.append(
            f"| {title} | {len(predicted):,} | {100 * precision:.1f} "
            f"| {100 * recall:.1f} | {100 * f1:.1f} |")
    lines.extend([
        "", "Chat improves precision and F1 on this annotation comparison, while",
        "missing more recorded reactions. It does not establish an accuracy",
        "improvement across the whole benchmark.", "",
        "The 4B answers come from `quail/biodex/BIO-2/joins-0.parquet` under",
        "`/results/benchmarks/quailb/family-runs/20260912T225100Z-902686c5/` and",
        "`/results/benchmarks/quailb/family-runs/20260918T060700Z-biodex-chat/`.",
        "The 32B label sets are `ls_558442b4193e9a48bfe1aea9bc87a66a` and",
        "`ls_4c0b69af26ad590e1b64ac5ffebf2484`. Annotation tables are under",
        "`/results/quailb_data/sf0.1/`. Download commands are in the generator.", "",
        "Possible follow-ups are to score different TRUE/FALSE thresholds and",
        "test shorter pair prompts. Neither was measured here. Moving prompt",
        "text requires checking answer quality and updating token accounting.", "",
    ])
    return lines


def main(workdir):
    """Regenerate the report and figures from the downloaded measurements."""
    root = Path(workdir)
    corpus = json.loads((root / "corpus.json").read_text())
    rows, manifest, suites = load_rows(root, corpus)
    queries = list(QUERY_ORDER)
    plt.style.use(HERE / "quail.mplstyle")
    plt.rcParams.update({"pdf.fonttype": 42, "figure.autolayout": False,
                         "savefig.bbox": None})
    relations = input_relations(queries, corpus)
    for method in rows.values():
        for query, row in method.items():
            assert sum(count for _, _, count in relations[query]) == row["input_rows"]
    sol = load_sol(root, queries, rows, corpus, manifest)
    overview = plot_comparison("QUAIL-B", queries, rows, relations, sol,
                               "quailb_main.pdf", overview=True)
    faster = sum(rows["quail"][q]["runtime_s"]
                 < rows["pipelined_vllm"][q]["runtime_s"] for q in queries)
    speedups = {q: rows["pipelined_vllm"][q]["runtime_s"]
                / rows["quail"][q]["runtime_s"] for q in queries}
    fastest = max(speedups, key=speedups.get)
    settings = [item["measurements"]["backend_metrics"]["capacity"]
                for item in suites["pipelined_vllm"]["queries"]]
    batch_tokens = sorted({item["max_num_batched_tokens"] for item in settings})
    sequences = sorted({item["max_num_seqs"] for item in settings})
    kv_capacity = [item["kv_cache_size_tokens"] for item in settings]
    bio2 = row_metrics(rows["quail"]["BIO-2"])
    bio3 = row_metrics(rows["quail"]["BIO-3"])
    checks = manifest["reference_checks"]
    kv_range = (f"{min(kv_capacity):,} tokens"
                if min(kv_capacity) == max(kv_capacity)
                else f"{min(kv_capacity):,} to {max(kv_capacity):,} tokens")
    imdb9 = row_metrics(rows["quail"]["IMDB-9"])
    fev8 = row_metrics(rows["quail"]["FEV-8"])
    lines = [
        "# QUAIL-B comparison", "",
        "- All 30 retained queries use Qwen3's chat format with thinking disabled.",
        "  Both methods use Qwen3 4B FP8, sf=0.1, lf=1, and one H100.",
        "  Quail and pipelined stock vLLM share a physical GPU within each family.",
        "  Pipelined stock vLLM advances documents through filter stages",
        "  independently, then starts joins after filtering finishes.",
        "- Previous LEP-5, LEP-6, and LEP-8 were removed because their reference",
        "  filters leave no rows at sf=0.1. Previous LEP-7 is now LEP-5.",
        "  This report selects 60 measurements from the saved 66-measurement run.",
        "  It checks query definitions before mapping historical IDs.",
        "- The measured run is on `quail-results`:",
        f"  `/results/benchmarks/quailb/family-runs/{manifest['run_id']}/`.",
        "  It reuses the completed BioDEX run after checking that its corpus,",
        "  query definitions, prompt format, and reference answers match.",
        f"  BioDEX source: `{manifest['reused_biodex']}`.",
        "  All other measurements are new. Earlier prompt formats are omitted.",
        "- Model references use Qwen3 32B FP8 with thinking disabled.",
        f"  The collection is `{manifest['collection_id']}`.",
        "  Reference join prompts put the first argument first. FEVER joins",
        "  were regenerated after fixing automatic prompt reordering. Saved",
        "  benchmark answers were rescored without changing timings.",
        "  The existing FEVER annotation and LePaRD citation rules still apply.",
        f"  Saved-answer checks repeated {checks['compared']} answers",
        f"  with {checks['answer_differences']} differences.",
        "- Quail's planned limits are 110,376 tokens per chunk and 362,250",
        "  resident KV tokens. Pipelined stock vLLM uses prefix caching,",
        f"  {', '.join(f'{n:,}' for n in batch_tokens)} batched tokens, and",
        f"  {', '.join(f'{n:,}' for n in sequences)} sequences.",
        f"  Its measured KV capacity is {kv_range}.",
        "  Both methods use the same planner's filter and join ordering rules.",
        f"- Quail is faster on {faster} of {len(queries)} queries.",
        f"  The arithmetic mean speedup is {mean(speedups.values()):.2f}x,",
        f"  the median is {median(speedups.values()):.2f}x, and the maximum is",
        f"  {speedups[fastest]:.2f}x on {fastest}. Each query has equal weight.",
        "  Speedup is pipelined stock vLLM time divided by Quail time.",
        "  Query time excludes startup and result collection.",
        f"  GPU cost is query seconds / 3,600 * ${H100_USD_PER_HOUR:.4f}.",
        "- SoL means speed of light. It estimates ideal GPU time by dividing",
        "  arithmetic and memory traffic by the hardware's peak rates. For each",
        "  model component, it takes the larger time, then adds component times.",
        "  It assumes ideal batching and unlimited retained KV. Matching token",
        "  prefixes are computed once across requests, documents, and aliases.",
        "  It excludes startup and software scheduling overhead and uses exact",
        "  reference-label survivors. The supported join search uses left-deep",
        "  plans. Different measured answers change the work, so the gap from",
        "  SoL is not purely execution overhead. SoL has no measured accuracy.",
        "- Fresh input tokens count every input position processed by a model",
        "  forward pass. Repeated computation counts again. Recomputed KV tokens",
        "  are fresh tokens minus the minimum for the run's actual requests with",
        "  unlimited KV: each document once, each question and join frame once",
        "  per document, and each pair's partner label, partner document, and",
        "  answer cue once per pair. They are included in fresh tokens, not added",
        "  to them. The benchmark computes this minimum from saved answers after",
        "  the run. KV regret is recomputed tokens / fresh tokens * 100%.",
        "  A missing minimum or zero computed tokens leaves regret unreported.",
        "- Token throughput is total requested input tokens divided by query seconds.",
        "  Count each complete prompt once per evaluated filter or join pair,",
        "  including tokens served from KV. Exclude generated answer tokens.",
        "  Counts come from saved answers, prompt pieces, and document tokens.",
        "  All retained vLLM totals match its recorded fresh plus cached token counts.",
        "  Cost per million input tokens is query dollars / requested tokens * 1e6.",
        "  It uses the same complete-prompt count as tokens per second.",
        "  Different survivors can change which prompts a method evaluates.",
        "  The SoL line counts complete prompts under its reference survivors.",
        "- Answer agreement counts matching evaluated predicate answers. Output",
        "  precision is the fraction of returned rows matching the reference.",
        "  Output recall is the fraction of reference rows returned. Most join",
        "  pairs can be negative, so high agreement can coexist with low recall.",
        f"  Quail's BIO-2 and BIO-3 recall is {bio2['recall']:.2f}% and",
        f"  {bio3['recall']:.2f}%, respectively.",
        f"  Its IMDB-9 output recall is {imdb9['recall']:.4f}%, and its FEV-8",
        f"  output precision is {fev8['precision']:.4f}%.",
        "- The main PDF shows all queries with one metric per page. Each dataset",
        "  PDF includes every query in that dataset. Input counts list each alias",
        "  separately, before filtering. SoL is a horizontal line, not a measured",
        "  bar. Latency labels show vLLM time divided by Quail time.",
        "  Log scales are labeled; a dash marks zero.", "",
        f"[Open the main vector PDF](plots/{overview})", "",
        f"Figure: plots/{overview}", "",
        "SoL estimates on `quail-results`:",
        "`/results/sol/2026-09-18-all-chat/sol_quailb_sf0.1.json`.", "",
        "The download commands are in `reports/make_quailb_comparison_plots.py`.", "",
    ]
    for family in ("IMDB", "BIO", "FEV", "LEP", "AGENT"):
        selected = [q for q in queries if q.startswith(family + "-")]
        name = plot_comparison(f"QUAIL-B {family}", selected, rows, relations,
                               sol, f"quailb_{family.lower()}.pdf")
        lines.extend([f"## {family}", "",
                      f"[Open the {family} vector PDF](plots/{name})", "",
                      f"Figure: plots/{name}", "",
                      "| Query | Input documents by alias and set |", "|---|---|"])
        for query in selected:
            counts = ", ".join(f"{alias} ({provider}) = {count:,}"
                               for alias, provider, count in relations[query])
            lines.append(f"| {query} | {counts} |")
        if family == "BIO":
            terms = corpus["tables"]["terms"]["rows"]
            survivors = {}
            for method, _, _ in METHODS:
                pairs = rows[method]["BIO-3"]["evaluated_document_pairs"]
                assert pairs % terms == 0
                survivors[method] = pairs // terms
            lines.extend([
                "", "BIO-3 filter survivors: "
                f"{sol['BIO-3']['documents_after_filters']:,} in the reference, "
                f"{survivors['quail']:,} in Quail, and "
                f"{survivors['pipelined_vllm']:,} in pipelined stock vLLM.",
                "",
                "The BIO-2 prompt comparison below explains the change in time",
                "and separately scores both formats against dataset annotations."])
        table_header_text = (
            "| Query | Method | Seconds | Tokens/second | $/query "
            "| $/million input tokens | KV regret (%) |")
        lines.extend(["", table_header_text,
                      "|---|---|---:|---:|---:|---:|---:|"])
        for query in selected:
            for key, label, _ in METHODS:
                m = row_metrics(rows[key][query])
                regret = ("Not measured" if m["regret_percent"] is None
                          else f"{m['regret_percent']:.2f}")
                lines.append(
                    f"| {query} | {label} | {m['seconds']:.2f} "
                    f"| {m['tokens_per_second']:,.2f} | {m['cost']:.5f} "
                    f"| {m['cost_per_million']:.6f} | {regret} |")
            values = {metric: series_value(rows, sol, "sol", query, metric)
                      for metric, _, _ in METRICS}
            lines.append(
                f"| {query} | SoL estimate | {values['seconds']:.3f} "
                f"| {values['tokens_per_second']:,.2f} | {values['cost']:.5f} "
                f"| {values['cost_per_million']:.6f} | 0 (assumed) |")
        lines.extend(["", "Correctness against saved reference labels:", "",
                      "| Query | Method | Answer agreement (%) "
                      "| Output precision (%) | Output recall (%) |",
                      "|---|---|---:|---:|---:|"])
        for query in selected:
            for key, label, _ in METHODS:
                m = row_metrics(rows[key][query])
                lines.append(
                    f"| {query} | {label} | {m['agreement']:.2f} "
                    f"| {m['precision']:.5g} | {m['recall']:.5g} |")
        lines.append("")
    lines.extend(biodex_prompt_comparison(root))
    report = HERE / "quailb-comparison.md"
    report.write_text("\n".join(lines))
    print(f"Updated {report}, the main figure, and five dataset figures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir")
    parser.add_argument("--count-requested-tokens", action="store_true")
    parser.add_argument("--root")
    parser.add_argument("--sol-file")
    args = parser.parse_args()
    if args.count_requested_tokens:
        count_requested_tokens(args.workdir, args.root, args.sol_file)
    else:
        main(args.workdir)

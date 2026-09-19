"""QUAIL-B figures and tables for DiffusionGemma: Quail, pipelined vLLM, SoL.

Pull the run, the corpus manifest, and the SoL estimate:

    W=/tmp/gemma-comparison; mkdir -p "$W/run"
    RUN=benchmarks/quailb/20260919T054043Z-8c1fa18b
    for F in manifest.json measurements.parquet quail/run.json \
        pipelined_vllm/run.json; do
      mkdir -p "$W/run/$(dirname $F)"
      uv run modal volume get quail-results "$RUN/$F" "$W/run/$F"
    done
    CORPUS=ground_truth/quailb/schema_v1/corpora/c_1aa2c4f0d0b6c816fd37aa5748c33341
    uv run modal volume get quail-results "$CORPUS/manifest.json" "$W/corpus.json"
    uv run modal volume get quail-results \
      sol/2026-09-19-diffusion-gemma/sol_quailb_sf0.1_diffusion-gemma-26b-a4b-fp8.json \
      "$W/sol.json"

Counting the requested input tokens needs the corpora and labels
under a root laid out like the volume (quailb_data/sf0.1 and
ground_truth/quailb/schema_v1); pass it once with
--count-requested-tokens --root <root>, which writes
run/requested_tokens.json. Then:

    uv run --with matplotlib python reports/make_diffusion_gemma_quailb_plots.py $W

Writes reports/plots/quailb_diffusion_gemma_main.pdf, one PDF per
dataset, and reports/2026-09-19-diffusion-gemma-quailb.md.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from make_quailb_comparison_plots import (  # noqa: E402
    METHODS,
    METRICS,
    QUERY_ORDER,
    input_relations,
    measurement_rows,
    plot_comparison,
    reference_requested_tokens,
    requested_input_tokens,
    row_metrics,
    series_value,
)

MODEL = "diffusion-gemma-26b-a4b-fp8"
MODEL_LABEL = "DiffusionGemma 26B-A4B FP8"
SLUG = "diffusion_gemma"
REPORT = HERE / "2026-09-19-diffusion-gemma-quailb.md"
FAMILIES = ("IMDB", "BIO", "FEV", "LEP", "AGENT")


def load_rows(root):
    """Measurements of both methods for the run's queries, in benchmark order."""
    from quail_b.queries import get_query
    from quail_b.run import _query_hash

    manifest = json.loads((root / "run" / "manifest.json").read_text())
    assert manifest["status"] == "complete", manifest["status"]
    assert manifest["model"] == MODEL
    assert manifest["sf"] == 0.1
    assert set(manifest["summaries"]) == {key for key, _, _ in METHODS}
    queries = [q for q in QUERY_ORDER if q in manifest["query_ids"]]
    rows = measurement_rows(root / "run" / "measurements.parquet")
    counts = json.loads((root / "run" / "requested_tokens.json").read_text())
    assert counts["run_id"] == manifest["run_id"]
    suites = {}
    for method, _, _ in METHODS:
        raw = (root / "run" / method / "run.json").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == counts["run_sha256"][method]
        suite = json.loads(raw)
        assert suite["collection_id"] == manifest["collection_id"]
        for item in suite["queries"]:
            if item["id"] not in queries:
                continue
            assert item["status"] == "complete", (method, item["id"])
            assert item["definition_hash"] == _query_hash(get_query(item["id"]))
            rows[method][item["id"]]["requested_tokens"] = (
                counts["methods"][method][item["id"]])
        rows[method] = {query: rows[method][query] for query in queries}
        suites[method] = suite
    return rows, manifest, suites, queries


def load_sol(root, queries, rows, corpus, manifest):
    """The model's SoL estimate per query, with its requested token count."""
    from quail_b.queries import get_query

    source = json.loads((root / "sol.json").read_text())
    assert source["corpus_id"] == corpus["corpus_id"]
    assert source["collection_id"] == manifest["collection_id"]
    assert source["scale_factor"] == 0.1
    counts = json.loads((root / "run" / "requested_tokens.json").read_text())
    assert counts["sol_sha256"] == hashlib.sha256(
        (root / "sol.json").read_bytes()).hexdigest()
    estimates = {}
    for query in queries:
        record = source["queries"][query]
        assert record["plan_sha256"] == hashlib.sha256(
            get_query(query).plan_bytes).hexdigest()
        estimate = record["models"][MODEL]
        assert estimate["input_document_rows"] == rows["quail"][query]["input_rows"]
        estimate["requested_tokens"] = counts["sol"][query]
        estimates[query] = estimate
    return estimates


def count_requested_tokens(root, data_root):
    """Count every complete prompt the answers imply, on the CPU."""
    from functools import cache

    import quail
    from quail.bench.quailb import answer_oracle, build_query
    from quail.planner.plan import EngineConfig
    from quail.specs import MODELS
    from quail_b.benchmark import load_benchmark
    from quail_b.minimum import DocumentTokens, load_tokenizer
    from quail_b.queries import get_query
    from quail_b.run import _query_hash, _read_output

    run_directory = root / "run"
    manifest = json.loads((run_directory / "manifest.json").read_text())
    benchmark = load_benchmark(
        scale_factor=0.1, root=data_root, collection_id=manifest["collection_id"])
    sol_bytes = (root / "sol.json").read_bytes()
    sol = json.loads(sol_bytes)
    assert sol["collection_id"] == manifest["collection_id"]
    tokenizer = load_tokenizer(MODELS[MODEL].hf_name)
    documents = DocumentTokens(benchmark.tables, tokenizer)
    queries = [q for q in QUERY_ORDER if q in manifest["query_ids"]]
    result = {
        "run_id": manifest["run_id"], "collection_id": manifest["collection_id"],
        "sol_sha256": hashlib.sha256(sol_bytes).hexdigest(),
        "methods": {}, "sol": {}, "run_sha256": {},
    }
    for method, _, _ in METHODS:
        raw = (run_directory / method / "run.json").read_bytes()
        result["run_sha256"][method] = hashlib.sha256(raw).hexdigest()
        result["methods"][method] = {}
        for item in json.loads(raw)["queries"]:
            if item["id"] not in queries:
                continue
            spec = get_query(item["id"])
            assert item["definition_hash"] == _query_hash(spec)
            output = _read_output(run_directory / method / item["directory"],
                                  item, rows=False)
            result["methods"][method][item["id"]] = requested_input_tokens(
                spec, output, documents)
        print("Counted complete prompts:", method, flush=True)

    @cache
    def encode(text):
        return tuple(tokenizer([text])[0])

    session = quail.Session(EngineConfig(model=MODEL, device="h100-sxm"),
                            tokenizer=encode)
    try:
        for name, table in benchmark.tables.items():
            session.register(name, quail.DocumentProvider.from_table(
                table, id_col="id"))
        answer = answer_oracle(benchmark.ground_truth, benchmark.tables)
        for query in queries:
            spec = get_query(query)
            saved = sol["queries"][query]
            assert saved["plan_sha256"] == hashlib.sha256(spec.plan_bytes).hexdigest()
            result["sol"][query] = reference_requested_tokens(
                build_query(session, spec), answer, saved["models"][MODEL])
            print("Counted SoL prompts:", query, flush=True)
    finally:
        session.close()
    destination = run_directory / "requested_tokens.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(destination, flush=True)


def report_lines(rows, queries, relations, sol, manifest, suites, figures):
    """The report's markdown, from the measurements alone."""
    speedups = {q: rows["pipelined_vllm"][q]["runtime_s"]
                / rows["quail"][q]["runtime_s"] for q in queries}
    faster = sum(ratio > 1 for ratio in speedups.values())
    fastest = max(speedups, key=speedups.get)
    settings = [item["measurements"]["backend_metrics"]["capacity"]
                for item in suites["pipelined_vllm"]["queries"]]
    batch_tokens = sorted({item["max_num_batched_tokens"] for item in settings})
    sequences = sorted({item["max_num_seqs"] for item in settings})
    lines = [
        "# QUAIL-B on DiffusionGemma 26B-A4B", "",
        f"- Both methods run `{MODEL}` at sf=0.1 on one H100, on the same",
        "  prompts: the Gemma 4 chat turn with the empty thinking channel",
        "  prefilled, and one canvas row after the answer cue on Quail's side.",
        "  Quail and pipelined stock vLLM share a physical GPU within each",
        "  query family.",
        f"- The measured run is `/results/benchmarks/quailb/{manifest['run_id']}/`",
        "  on `quail-results`; reference labels are collection",
        f"  `{manifest['collection_id']}` (Qwen3 32B FP8 answering).",
        "- Quail reads the TRUE and FALSE logits at its one canvas row and",
        "  compares them. Pipelined stock vLLM denoises the checkpoint's",
        "  256-row canvas per request, since its diffusion sampler takes no",
        "  temperature, min_tokens, or allowed_token_ids; its answer is the",
        "  first TRUE or FALSE word in the generated text, and a request that",
        "  ends its turn without an answer word counts as FALSE. That readout",
        "  is what stock vLLM gives, and it costs the baseline recall.",
        f"- Pipelined stock vLLM batches {', '.join(f'{b:,}' for b in batch_tokens)}"
        f" tokens and {', '.join(f'{s:,}' for s in sequences)} sequences, with",
        "  prefix caching. vLLM caps this model at 8 sequences per step because",
        "  its diffusion sampler holds a [sequences, canvas rows, vocabulary]",
        "  float32 tensor. Quail's chunk budget is 65,536 tokens.",
        f"- Quail is faster on {faster} of {len(queries)} queries.",
        f"  The median speedup is {median(speedups.values()):.2f}x and the maximum",
        f"  is {speedups[fastest]:.2f}x on {fastest}.",
        "  Speedup is pipelined stock vLLM time divided by Quail time.",
        "  Query time excludes startup and result collection.",
        "  GPU cost is query seconds / 3,600 times $3.9492.",
        "- SoL means speed of light: ideal GPU time from arithmetic and memory",
        "  traffic at the hardware's peak rates, with ideal batching, unlimited",
        "  retained KV, every distinct prompt prefix computed once, one canvas",
        "  row per evaluation, and exact reference-label survivors. It is an",
        "  estimate, not a measured backend, and has no accuracy.",
        "- Fresh input tokens count every input position a forward pass",
        "  processes, repeated computation included. KV regret is recomputed",
        "  tokens as a share of fresh tokens. Token throughput is total",
        "  requested input tokens divided by query seconds.", "",
        "[Open the main vector PDF](plots/quailb_diffusion_gemma_main.pdf)", "",
        "Figure: plots/quailb_diffusion_gemma_main.pdf", "",
    ]
    for family in FAMILIES:
        selected = [q for q in queries if q.startswith(family + "-")]
        if not selected:
            continue
        name = figures[family]
        lines.extend([f"## {family}", "",
                      f"[Open the {family} vector PDF](plots/{name})", "",
                      f"Figure: plots/{name}", "",
                      "| Query | Input documents by alias and set |", "|---|---|"])
        for query in selected:
            counts = ", ".join(f"{alias} ({provider}) = {count:,}"
                               for alias, provider, count in relations[query])
            lines.append(f"| {query} | {counts} |")
        lines.extend(["", "| Query | Method | Seconds | Tokens/second | $/query "
                      "| $/million input tokens | KV regret (%) |",
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
    return lines


def main(workdir):
    """Regenerate the figures and the report from the pulled measurements."""
    root = Path(workdir)
    corpus = json.loads((root / "corpus.json").read_text())
    rows, manifest, suites, queries = load_rows(root)
    plt.style.use(HERE / "quail.mplstyle")
    plt.rcParams.update({"pdf.fonttype": 42, "figure.autolayout": False,
                         "savefig.bbox": None})
    relations = input_relations(queries, corpus)
    for method in rows.values():
        for query, row in method.items():
            assert sum(count for _, _, count in relations[query]) == row["input_rows"]
    sol = load_sol(root, queries, rows, corpus, manifest)
    note = "Reference labels: Qwen3 32B FP8 and dataset annotations."
    plot_comparison("QUAIL-B", queries, rows, relations, sol,
                    f"quailb_{SLUG}_main.pdf", overview=True,
                    model_label=MODEL_LABEL, reference_note=note)
    figures = {}
    for family in FAMILIES:
        selected = [q for q in queries if q.startswith(family + "-")]
        if selected:
            figures[family] = plot_comparison(
                family, selected, rows, relations, sol,
                f"quailb_{SLUG}_{family.lower()}.pdf",
                model_label=MODEL_LABEL, reference_note=note)
    REPORT.write_text("\n".join(report_lines(
        rows, queries, relations, sol, manifest, suites, figures)))
    print(f"Updated {REPORT}, the main figure, and {len(figures)} dataset figures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir")
    parser.add_argument("--count-requested-tokens", action="store_true")
    parser.add_argument("--root", help="corpora and labels laid out like the volume")
    args = parser.parse_args()
    if args.count_requested_tokens:
        count_requested_tokens(Path(args.workdir), args.root)
    main(args.workdir)

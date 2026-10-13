"""Sessions for the end-to-end tests: a fake tokenizer and a planted executor."""

from quail.planner.plan import EngineConfig

CONFIG = EngineConfig(model="qwen3-4b-fp8", device="h100-sxm")

TRUTH = {"r": {"q1:": [1, 1, 0, 1, 1, 0], "q2:": [1, 0, 1, 1, 0, 1]}}


def fake_tok(text):
    return text.split()


def _run(query, execute):
    from quail.api.execute import execute_query

    return execute_query(query, physical_executor=execute)


def make_executor(filter_truth, join_truth=None):
    """Build a fake executor from filter and join truth tables."""
    def _match_key(alias, q):
        for key in filter_truth[alias]:
            if any(t.startswith(key) for t in q):
                return key
        raise KeyError(f"no filter_truth key for alias {alias!r} "
                       f"matches tokens {q[:5]}")

    def _exec(request):
        from helpers.fakes import graph_state
        from quail.backends.quail.graph import execute_single_graph
        from quail.builtins import built_in_registry
        from quail.execution.metrics import NodeMetrics, NodeResult
        from quail.execution.types import PhysicalResponse
        from quail.physical import AiFilter, Scan, decode_graph

        graph = decode_graph(request.plan["graph"], built_in_registry().codecs)
        docs = {
            node.alias: request.inputs[node.input_id].documents
            for node in graph.nodes if isinstance(node, Scan)
        }

        class FixedAnswers:
            def execute(self, node, inputs):
                if isinstance(node, AiFilter):
                    answers = filter_truth[node.alias][
                        _match_key(node.alias, node.question_token_ids)]
                    rows = {d: [answers[d]] for d in inputs["document_ids"]}
                    survivors = [d for d, row in rows.items() if row[0]]
                    # the real loop reports a chunk's finished documents
                    # by position; the hook streams them
                    if inputs.get("document_done") is not None:
                        inputs["document_done"]([
                            (position, len(row) - 1, bool(row[-1]))
                            for position, row in enumerate(rows.values())])
                    return NodeResult({
                        f"ids:{node.alias}": survivors,
                        f"filter_answers:{node.alias}": rows,
                    }, NodeMetrics(input_rows=len(rows),
                                   output_rows=len(survivors),
                                   evaluated_documents=len(rows),
                                   fresh_tokens=10 * len(rows)))
                anchors = list(inputs["anchor_ids"])
                outputs = {}
                partners = inputs["partner_indices"][node.written_pos]
                rule = join_truth[(node.anchor, *node.partners)]
                rows = {i: [rule(a, *pair) for pair in partners]
                        for i, a in enumerate(anchors)}
                outputs[f"join_answers:{node.written_pos}"] = {
                    "rows": rows, "anchor_index": anchors,
                    "partner_index": partners, "anchor": node.anchor,
                    "partners": list(node.partners),
                    "semantics": node.semantics,
                    "selectivity": node.selectivity,
                    "written_pos": node.written_pos,
                }
                passing = {anchors[i] for i, row in rows.items() if any(row)}
                outputs[f"ids:{node.anchor}"] = [a for a in anchors
                    if (a not in passing if node.semantics == "anti"
                        else a in passing)]
                # the real loop reports each anchor's last-stage row as
                # it finishes; the hook frees its KV and streams answers
                if inputs.get("anchor_done") is not None:
                    for i, row in rows.items():
                        inputs["anchor_done"](i, row)
                return NodeResult(outputs, NodeMetrics(
                    input_rows=len(anchors),
                    output_rows=len(outputs[f"ids:{node.anchor}"]),
                    evaluated_document_pairs=sum(
                        len(row) for row in rows.values())))

        state = graph_state(FixedAnswers(), docs)
        if getattr(request, "relations", None):
            state["columns"] = request.column_tables()
        report = execute_single_graph(state, {
            "filter_limit": None, "pre_ids": [],
        }, graph)
        outputs = report.pop("_outputs")
        return PhysicalResponse(outputs, {
            **report, "wall_s": 1.0, "boot_s": 0.5, "fresh_tokens": 1234,
        })

    return _exec

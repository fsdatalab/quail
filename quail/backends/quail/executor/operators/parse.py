"""AI.PARSE on the executor: parse rows and hand their text to the reader.

The Parse node's rows are parsed by the session's parser (context
state ``parser``, a callable from the node to a parser), and each
parsed row is tokenized with the reading model's tokenizer (state
``tokenize``). The table's documents in state ``docs`` become those
token lists, so every operator after the parse reads the parsed text.

In a pipeline that starts with the parse, the operators after it run
in rounds: once the parsed rows hold a forward pass's worth of tokens,
they run through the rest of the pipeline while the parser waits, and
then parsing goes on. Parsed rows reach the reader before the last row
is parsed.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import pyarrow as pa

from quail.execution.runner import NodeMetrics, NodeResult


class ParsedDocuments(Sequence):
    """A table's rows as token lists, filled in as rows are parsed.

    Args:
        rows: The table's row count.
    """

    def __init__(self, rows: int):
        self._tokens = [None] * rows

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, row):
        tokens = self._tokens[row]
        if tokens is None:
            raise KeyError(f"row {row} has not been parsed")
        return tokens

    def set(self, row: int, tokens) -> None:
        self._tokens[row] = tokens


def _rows(pages, ids):
    for row in ids:
        path_file, numbers = pages[row]
        yield row, path_file, numbers


def parsed_table(node, texts: dict) -> pa.Table:
    """The parsed column of the rows in ``texts``, by row id."""
    ids = sorted(texts)
    schema = pa.schema(
        [pa.field(node.alias, pa.int32(), nullable=False),
         pa.field(node.name, pa.string(), nullable=False)],
        metadata={b"quail.kind": b"parsed_text",
                  b"quail.alias": node.alias.encode("utf-8")})
    return pa.Table.from_arrays(
        [pa.array(ids, pa.int32()), pa.array([texts[i] for i in ids], pa.string())],
        schema=schema)


def parse_result(node, texts: dict, stats: dict, wall_s: float,
                 parse_tokens: int) -> NodeResult:
    """The Parse node's outputs: the parsed row ids and the parsed column."""
    return NodeResult(
        outputs={f"ids:{node.alias}": sorted(texts),
                 "parsed": parsed_table(node, texts)},
        metrics=NodeMetrics(
            wall_s=wall_s, input_rows=len(texts), output_rows=len(texts),
            extension={**stats, "parser": node.parser,
                       "parsed_reader_tokens": parse_tokens}))


class _Parsing:
    """Parse one Parse node's rows into the table's documents."""

    def __init__(self, node, ids, context):
        state = context.state
        self.node = node
        self.parser = state["parser"](node)
        self.tokenize = state["tokenize"]
        pages = state["docs"][node.alias]
        self.documents = ParsedDocuments(len(pages))
        state["docs"][node.alias] = self.documents
        self.rows = _rows(pages, ids)
        self.texts = {}
        self.tokens = 0
        self.started = time.perf_counter()

    def __iter__(self):
        """Yield (row, token count) as each row is parsed and tokenized."""
        for row, text in self.parser.parse(self.rows):
            tokens = list(self.tokenize(text))
            self.documents.set(row, tokens)
            self.texts[row] = text
            self.tokens += len(tokens)
            yield row, len(tokens)

    def result(self) -> NodeResult:
        return parse_result(self.node, self.texts, dict(self.parser.stats),
                            time.perf_counter() - self.started, self.tokens)


class ParseRuntime:
    """Run a Parse node on its own: parse every input row, then pass them on."""

    def execute(self, node, inputs, context) -> NodeResult:
        ids = list(next(iter(inputs.values())))
        limit = context.state.get("parse_limit")
        if limit is not None:
            ids = ids[:limit]
        parsing = _Parsing(node, ids, context)
        for _ in parsing:
            pass
        return parsing.result()


def _merge(results: list) -> NodeResult:
    """Combine one member's results from several rounds."""
    outputs = {}
    for result in results:
        for port, value in result.outputs.items():
            if port not in outputs:
                outputs[port] = value
            elif isinstance(value, pa.Table):
                outputs[port] = pa.concat_tables([outputs[port], value])
            elif isinstance(value, dict):
                outputs[port] = {**outputs[port], **value}
            else:
                outputs[port] = sorted([*outputs[port], *value])
    metrics = NodeMetrics()
    for result in results:
        metrics = metrics + result.metrics
    return NodeResult(outputs, metrics)


def execute_parsed_pipeline(run_rest, pipeline, inputs, context,
                            round_tokens: int):
    """Parse a pipeline's rows and run the rest of it in rounds.

    Args:
        run_rest: Callable(pipeline, inputs, context) that runs a
            pipeline without the parse and returns node id ->
            NodeResult.
        pipeline: The Pipeline; its first member is the Parse node.
        inputs: Member node id -> its inputs from outside the pipeline.
        context: The graph's execution context.
        round_tokens: Parsed tokens that start a round; the last round
            takes what is left.

    Returns:
        Node id -> NodeResult for every member.
    """
    from quail.execution.pipelines import Pipeline

    parse, *members = pipeline.members
    rest = Pipeline(pipeline.alias, tuple(members))
    ids = list(next(iter(inputs[parse.node_id].values())))
    # the first member's ids input, which the parse supplies
    port = next(port.name for port in members[0].inputs
                if port.source.node_id == parse.node_id)
    parsing = _Parsing(parse, ids, context)
    rounds = {member.node_id: [] for member in members}
    waiting, tokens = [], 0
    reader_s = 0.0

    def run(rows):
        nonlocal reader_s
        started = time.perf_counter()
        round_inputs = {member.node_id: dict(inputs[member.node_id])
                        for member in members}
        round_inputs[members[0].node_id][port] = sorted(rows)
        for node_id, result in run_rest(rest, round_inputs, context).items():
            rounds[node_id].append(result)
        reader_s += time.perf_counter() - started

    limit = context.state.get("filter_limit")
    sink = members[-1]

    def enough() -> bool:
        kept = sum(len(result.outputs.get(f"ids:{pipeline.alias}", ()))
                   for result in rounds[sink.node_id])
        return limit is not None and kept >= limit

    rows = iter(parsing)
    for row, count in rows:
        waiting.append(row)
        tokens += count
        if tokens >= round_tokens:
            run(waiting)
            waiting, tokens = [], 0
            if enough():
                # the rows already kept fill the LIMIT
                rows.close()
                break
    if waiting or not any(rounds.values()):
        run(waiting)
    result = parsing.result()
    extension = {**result.metrics.extension,
                 "parse_rounds": len(rounds[members[0].node_id]),
                 "reader_s": reader_s}
    results = {parse.node_id: NodeResult(result.outputs, NodeMetrics(
        wall_s=result.metrics.wall_s - reader_s,
        input_rows=result.metrics.input_rows,
        output_rows=result.metrics.output_rows, extension=extension))}
    for member in members:
        results[member.node_id] = _merge(rounds[member.node_id])
    return results

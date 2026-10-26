"""What operators that decode an answer in rounds share.

An extraction and a map put each of their questions to a document the
same way: the question's tail is the frame of its stages, written
right after the document in place of the tokens the question before it
appended, so a document is read once however many questions it takes.
Each question is the tail stage, whose row chooses the answer's first
token, then up to the configured answer limit in rounds, each feeding
the chosen token and a draft and verifying the draft against every
row. The operators differ in how a token is chosen and drafted, which
their decoders (extract/decoder.py, map/decoder.py) define.

AnswerStages is the shared stage shape, and AnswerPart the shared
execution part: it builds the stages over a pipeline's documents, runs
them as one group, and turns each question's settled answers into its
node's rows and metrics.
"""

import time
from dataclasses import replace
from functools import partial

import numpy as np

from quail.backends.quail.executor.chunk import Suffixes
from quail.backends.quail.executor.model_loading import full_output_head
from quail.backends.quail.executor.parts import (
    DocumentAnswerPart,
    execute_parts,
    gpu_seconds,
    shared_prefix_tree,
    stage_spans,
)
from quail.backends.quail.executor.readout import AsyncTopLogprobs, readout_rows
from quail.backends.quail.executor.stages import Stage, StageRequests
from quail.backends.quail.executor.vocabulary import load_vocabulary
from quail.execution.metrics import NodeMetrics, NodeResult


class AnswerStages:
    """The stages of one pipeline's decoded answers.

    A subclass prepares its documents and decoder, then calls
    _build(). It supplies _draft(document, answer, q), the tokens a
    round feeds, and a decoder with verify(document, q, rows, fed) and
    settle(document, q).

    Attributes set by the subclass before _build(): specs, documents,
    document_ids (sorted), document_order, decoder, readout, tails,
    recurrent. A subclass whose rounds run ahead of their answers sets
    ahead and supplies _stage_launched.
    """

    operator = ""   # the operator's name in error messages
    ahead = False   # whether the stages run ahead of their answers

    def _parse_specs(self, state, specs) -> None:
        """Check the specs share a table, head, and decoding settings."""
        self.specs = list(specs)
        first = self.specs[0]
        self.spec = first
        self.recurrent = bool(getattr(state.loaded_model.arena, "num_state_slots", 0))
        if any(not first.same_decoding(spec) for spec in self.specs):
            raise ValueError(
                f"a pipeline's {self.operator}s share decoding settings")
        if any(spec.alias != first.alias for spec in self.specs):
            raise ValueError(f"a pipeline's {self.operator}s share a table")
        parts = [spec.prompt_token_parts for spec in self.specs]
        self.head = list(parts[0][0])
        if any(list(head) != self.head for head, _ in parts):
            raise ValueError(f"a pipeline's {self.operator}s share a prompt head")
        self.tails = [list(tail) for _, tail in parts]
        if any(len(tail) < 2 for tail in self.tails):
            raise ValueError(
                f"a {self.operator}'s tail ends in its cue after its question")

    def _order_documents(self, document_ids) -> None:
        """Index the documents by id for the suffix callbacks."""
        document_ids = np.asarray(document_ids, dtype=np.int64)
        self.document_order = np.argsort(document_ids, kind="stable")
        self.document_ids = document_ids[self.document_order]
        self.prefixes = [document.prefix for document in self.documents]
        self.keys = [document.key for document in self.documents]
        self.documents_by_key = {}

    def _build(self) -> None:
        """Build each question's stages, in spec order."""
        self.stages = []
        self.question_starts = []
        for q, (spec, tail) in enumerate(zip(self.specs, self.tails)):
            self.question_starts.append(len(self.stages))
            self.stages.extend(self._question_stages(q, spec.name, tail))
        self.question_starts.append(len(self.stages))

    def _question_stages(self, q: int, name: str, tail: list) -> list:
        """Build one question's stages: its tail, then the answer's rounds.

        The tail but its last token is the frame of both stages, so
        the scheduler writes it once after the document; the first
        stage rewinds past what the question before appended. The
        last token of the tail, the cue, is the stage's suffix, whose
        row chooses the answer's first token. The second stage repeats
        a round, feeding the chosen token and a draft, until the answer
        is done and its callback passes the document on.
        """
        frame, last = tail[:-1], tail[-1]
        return [
            Stage(suffixes=[[last]], frame=frame, rewind=True,
                  readout=self.readout, decide=partial(self._tail_decide, q),
                  prepare=self._stage_prepare(q, tail=True),
                  launched=self._stage_launched(q, tail=True), ahead=self.ahead,
                  single=True, append=True, label=f"{name} tail"),
            Stage(suffixes=partial(self._step_suffix, q),
                  suffix_tokens=(1 + self.spec.draft_tokens)
                  * (2 if self.recurrent else 1), frame=frame,
                  readout=self.readout, decide=partial(self._step_decide, q),
                  prepare=self._stage_prepare(q, tail=False),
                  launched=self._stage_launched(q, tail=False), ahead=self.ahead,
                  costs=self._stage_costs(q), rounds=self._stage_rounds(q),
                  read_all_rows=True, single=True, append=True, repeats=True,
                  label=f"{name} rounds"),
        ]

    def _stage_prepare(self, q: int, tail: bool):
        """Return a stage's prepare callback, or None to decide per document."""
        return None

    def _stage_launched(self, q: int, tail: bool):
        """Return a stage's launched callback, or None when none is needed."""
        return None

    def _stage_costs(self, q: int):
        """Return the rounds stage's costs callback, or None to ask per document."""
        return None

    def _stage_rounds(self, q: int):
        """Return the rounds stage's rounds callback, or None to ask per document."""
        return None

    @property
    def appended_tokens(self) -> int:
        """The most tokens a question leaves in KV after its frame.

        The cue, and the answer's tokens with the last round's draft
        past them.
        """
        return 1 + self.spec.max_answer_tokens + self.spec.draft_tokens

    def _readout(self, state):
        """The readout of every row: the stop, then the top tokens."""
        torch = state.torch
        head = full_output_head(state.loaded_model.model)
        return AsyncTopLogprobs(torch, torch.nn.functional, head,
                                self.spec.top_tokens, [self.vocabulary.stop_ids],
                                ragged=True)

    # ---- the tail, then the answer, a round of tokens at a time ------------

    def _tail_decide(self, q, anchor, row):
        """Choose the answer's first token from the row after the cue."""
        document = self.documents[anchor]
        answer = document.answers[q]
        answer.asked = True
        # a tail equal to the question's before it is in KV already:
        # only its cue token is fed
        same = (not self.recurrent and q > 0
                and self.tails[q] == self.tails[q - 1])
        answer.fed_tokens += 1 if same else len(self.tails[q])
        self.decoder.verify(document, q, self._rows(row), [])
        return True

    def _rows(self, row):
        """Return a readout's rows for one answer."""
        # the admission hands over the document's one answer in a list
        # or an object array
        if isinstance(row, list) and len(row) == 1:
            row = row[0]
        elif isinstance(row, np.ndarray) and row.dtype == object and row.size == 1:
            row = row.reshape(-1)[0]
        return readout_rows(row, self.readout.width)

    def _step_suffix(self, q, key):
        """Feed the chosen token, then a draft of the answer's next tokens.

        A recurrent model replays the accepted tokens to advance its
        checkpoint, so they are fed again and written as the suffix.
        """
        document = self._document(key)
        answer = document.answers[q]
        if answer.done:
            return Stage.SkipTo(self.question_starts[q + 1])
        accepted = answer.fed[:len(answer.fed) - answer.discard]
        tokens = self._draft(document, answer, q)
        if self.recurrent:
            answer.fed_tokens += len(accepted)
            return StageRequests([accepted + tokens],
                                 write_suffix_tokens=len(accepted),
                                 read_rows=[len(tokens)])
        return StageRequests(
            Suffixes(np.array(tokens, dtype=np.int64), [len(tokens)]),
            discard=answer.discard)

    def _draft(self, document, answer, q) -> list:
        """Return the tokens the next round feeds: the chosen token, then a draft."""
        raise NotImplementedError

    def _document(self, key):
        document = self.documents_by_key.get(key)
        if document is None:
            position = np.searchsorted(self.document_ids, key[1],
                                       side="right") - 1
            document = self.documents[self.document_order[position]]
            self.documents_by_key[key] = document
        return document

    def _step_decide(self, q, anchor, row):
        """Verify the round's draft by the row after each of its tokens."""
        document = self.documents[anchor]
        answer = document.answers[q]
        answer.fed_tokens += len(answer.fed)
        self.decoder.verify(document, q, self._rows(row), answer.fed)
        return True

    def finish(self) -> None:
        """Answer every question the steps left open."""
        for document in self.documents:
            for q, answer in enumerate(document.answers):
                if not answer.done:
                    self.decoder.settle(document, q)

    def asked(self) -> list:
        """Return the indices of the documents the questions were asked of."""
        return [index for index, document in enumerate(self.documents)
                if document.answers[0].asked]


class AnswerPart(DocumentAnswerPart):
    """One pipeline of decoded-answer nodes, run as one stage group.

    A subclass names its node_type, its stages_type (an AnswerStages),
    and its operator for messages and progress labels, and builds each
    node's rows from the settled answers (_answer_rows).
    """

    combine = True
    node_type = None
    stages_type = None
    operator = ""
    # whether the stages put a pipeline's questions to a document at
    # once, as branches, when the pipeline runs on its own
    branches = False

    @classmethod
    def execute_group(cls, state, nodes, inputs) -> dict:
        """Run the pipeline's questions over its documents as one stage run.

        Each document is read once and asked every node's question
        while its KV is resident. The first node is charged the
        documents' tokens and the wall time. Each question is charged
        its own suffix tokens and GPU time.

        Args:
            state: The loaded model and current query state.
            nodes: The pipeline's nodes, in order.
            inputs: The documents' ids, the tokenized documents and
                texts by alias, the query's shared preamble, and the
                gpu_timing setting.

        Returns:
            Node id -> NodeResult with the node's answer table.
        """
        state.gpu_timing = bool(inputs.get("gpu_timing", False))
        started = time.perf_counter()
        # a limit counts documents, so their questions are asked in
        # turn, not as branches whose anchors it would count
        part = cls(state, nodes, inputs["document_ids"], inputs,
                   standalone=inputs.get("limit") is None)
        stages = part.answer_stages
        branches = getattr(stages, "branches", False)
        tree = getattr(stages, "prefix_tree", None)
        if tree is None and nodes[0].spec.share_prefixes:
            tree = shared_prefix_tree(nodes[0].spec.name, stages.prefixes,
                                      state.loaded_model.arena)
        results = execute_parts(
            state, [part], stages.prefixes, stages.keys,
            # a branch's settled pages are still shared with its
            # siblings, so branches retain no KV
            keep_kv=nodes[-1].keep_kv and not branches, prefix_tree=tree,
            limit=inputs.get("limit"),
            label=f"{stages.operator} "
                  f"{', '.join(node.spec.name for node in nodes)}")
        first = nodes[0].node_id
        results[first] = replace(results[first], metrics=replace(
            results[first].metrics, wall_s=time.perf_counter() - started))
        return results

    @classmethod
    def prepare(cls, state, nodes, ids, parts, inputs, context):
        """Prepare consecutive questions over the preceding operator's documents.

        Raises:
            ValueError: A document uses a different prompt or token sequence.
        """
        first = parts[0].node if parts else nodes[0]
        part = cls(state, nodes, ids, {
            "documents": context.state["documents"],
            "texts": context.state.get("texts", {}),
            "pipeline_head": first.document_head})
        alias = part.node.spec.alias
        for document, document_id in zip(part.answer_stages.documents, part.ids):
            if document.key != (alias, document_id):
                raise ValueError(
                    f"{cls.operator} {part.node.spec.name!r} shows document "
                    f"{document_id} of {alias!r} unlike the pipeline before it")
        return part

    def __init__(self, state, nodes, ids, inputs, standalone=False):
        self.nodes = list(nodes)
        self.node = self.nodes[0]
        self.ids = np.asarray(ids, dtype=np.int64)
        if any(not isinstance(node, self.node_type) for node in nodes):
            raise TypeError(
                f"a {self.operator} pipeline holds "
                f"{self.node_type.__name__} nodes only")
        alias = self.node.spec.alias
        # branches need anchors of their own, so only a pipeline run on
        # its own puts its questions at once
        options = {"branches": True} if self.branches and standalone else {}
        self.answer_stages = self.stages_type(
            state, [node.spec for node in nodes], load_vocabulary(state),
            self.ids, inputs["documents"][alias],
            inputs.get("texts", {}).get(alias),
            pipeline_head=inputs.get("pipeline_head", self.node.document_head),
            document_positions=inputs.get("document_positions"), **options)
        self.stages = self.answer_stages.stages
        self._results = {}

    @property
    def appended_tokens(self) -> int:
        return self.answer_stages.appended_tokens

    def _answer_rows(self, node, q: int, kept, document_ids):
        """Return one node's rows from its settled answers."""
        raise NotImplementedError

    def finish(self, every) -> int:
        """Settle every answer; return the tokens the questions fed."""
        self.answer_stages.finish()
        self.counts = [self.answer_stages.counts(q)
                       for q in range(len(self.nodes))]
        return sum(count["suffix_tokens"] for count in self.counts)

    def complete(self, fresh, spans, stats, offset):
        """Build each question's answer table and attribute its GPU work.

        The first node is charged the run's fresh tokens less the
        later questions' suffix tokens, the packing time, and the
        borrowed prefix tokens; each later node its own suffix tokens.
        Questions put to a document at once share their stages, so the
        first node is charged all of their GPU time.
        """
        branches = getattr(self.answer_stages, "branches", False)
        per_question = (len(self.stages) if branches
                        else len(self.stages) // len(self.nodes))
        kept = self.answer_stages.asked()
        document_ids = self.ids[kept]
        counts = self.counts
        later = sum(count["suffix_tokens"] for count in counts[1:])
        results = {}
        for q, node in enumerate(self.nodes):
            start = offset + (0 if branches else q * per_question)
            selected = (stage_spans(spans, start, start + per_question)
                        if q == 0 or not branches else [])
            first = q == 0
            results[node.node_id] = NodeResult(
                {node.rows_port: self._answer_rows(node, q, kept, document_ids)},
                NodeMetrics(
                    input_rows=len(document_ids),
                    output_rows=len(document_ids),
                    evaluated_documents=len(document_ids),
                    fresh_tokens=(fresh - later if first
                                  else counts[q]["suffix_tokens"]),
                    gpu_s=gpu_seconds(selected), chunks=len(selected),
                    extension={
                        "output": node.spec.name,
                        "aliases": list(node.spec.aliases),
                        "input_rows": len(document_ids), **counts[q],
                        "borrowed_prefix_tokens": (
                            stats.get("borrowed_tokens", 0) if first else 0),
                        "pack_s": (round(stats.get("pack_s", 0.0), 3)
                                   if first else 0.0),
                        "host_s": stats.get("host_s") if first else None,
                        "run_s": stats.get("run_s") if first else None,
                        "graph_replays": (stats.get("graph_replays")
                                          if first else None),
                        "graph_fallbacks": (stats.get("graph_fallbacks")
                                            if first else None),
                        "readout_gpu_s": (stats.get("readout_gpu_s")
                                          if first else None),
                        "idle_gpu_s": (stats.get("idle_gpu_s")
                                       if first else None),
                        "kv_growth": stats.get("kv_growth") if first else None,
                        "branches": bool(branches)}))
        self._results = results
        self.result_value = self._results[self.node.node_id]

    def results(self) -> dict:
        """Return the answer table and metrics for each question."""
        return self._results

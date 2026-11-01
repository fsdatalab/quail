"""Prepare join requests, retain matching pairs, and collect predicate results."""

import itertools
from functools import partial

import numpy as np

from quail.backends.quail.executor.parts import (
    ExecutionPart,
    execute_parts,
    execution_prefix_tree,
    gpu_seconds,
    input_staging,
    stage_spans,
)
from quail.backends.quail.executor.readout import AsyncSlotMargins
from quail.backends.quail.executor.set_passes import split_passes, state_tails
from quail.backends.quail.executor.stages import Stage, run_stages
from quail.backends.quail.prompts import join_request
from quail.backends.quail.retention import apply_retention, join_settlement
from quail.catalog.pairs import (
    RowGroups,
    allowed_members,
    members_by_partner,
    pair_partner,
    partner_map,
)
from quail.catalog.tokens import (
    DocumentKeys,
    DocumentPrefixes,
    concat_tokens,
    select_documents,
)
from quail.execution.ids import id_positions
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.rows import Rows
from quail.execution.runner import ExecutionContext
from quail.progress import answer_sink


class JoinPart(ExecutionPart):
    """Join predicates and their anchor KV within one scheduler run."""

    combine = True

    @classmethod
    def execute_group(cls, state, nodes, inputs):
        """Run prepared join predicates through the shared document scheduler."""
        state.gpu_timing = bool(inputs.get("gpu_timing", False))
        part = cls(state, nodes, inputs, inputs["anchor_ids"])
        if state.loaded_model.pipeline.whole_sequences:
            return execute_set_joins(state, part)
        prefixes = inputs["prefixes"]
        return execute_parts(
            state, [part], prefixes, inputs["anchor_keys"],
            keep_kv=nodes[-1].keep_kv,
            prefix_tree=execution_prefix_tree(
                nodes[0], prefixes, state.loaded_model.arena),
            attention=nodes[0].attention, label=f"join ({len(nodes)} predicates)")

    @staticmethod
    def model_inputs(node, inputs, context, upstream=None, nodes=None):
        return prepare_join_inputs(node, inputs, context, upstream, nodes)

    @classmethod
    def prepare(cls, state, nodes, ids, parts, inputs, context):
        """Bind partner inputs to the documents of the preceding operators."""
        first = parts[0].node if parts else nodes[0]
        prefixes = DocumentPrefixes(
            first.document_head, context.state["documents"][nodes[0].anchor], ids)
        upstream = {"documents": prefixes, "document_ids": ids, "parts": tuple(parts)}
        prepared = context.model_inputs(
            nodes[0], inputs, context, upstream, nodes=nodes)
        return cls(state, nodes, prepared, ids, context=context)

    def __init__(self, state, nodes, inputs, ids, *, context=None):
        self.nodes = nodes
        self.node = nodes[0]
        self.inputs = inputs
        self.ids = ids
        self.context = context
        self.canvas_tokens = len(state.loaded_model.pipeline.canvas_ids)
        self.attention = self.node.attention or None
        self.settled = np.zeros(len(ids), dtype=bool)
        self.final_rows = [None] * len(ids)
        self.readers = []
        lists_for = inputs.get("anchor_partners")
        self.stages = [
            Stage(suffixes=suffixes, readout=state.async_answers, frame=frame,
                  requests=(None if lists_for is None
                            else lambda key, j=j: lists_for(key[1], j)),
                  label=f"join predicate {j}")
            for j, (suffixes, frame) in enumerate(
                zip(inputs["stage_suffixes"], inputs["stage_frames"]))]
        self._results = {}

    def add_reader(self, kept):
        """Expose surviving pairs to a following classification."""
        self.readers.append(kept)
        for index in (kept.stage, len(self.stages) - 1):
            self.stages[index].decide = partial(self._record, index)

    def _record(self, index, anchor, row):
        lists_for = self.inputs.get("anchor_partners")
        key = self.inputs["anchor_keys"][anchor]
        indices = None if lists_for is None else lists_for(key[1], index)
        for reader in self.readers:
            if reader.stage == index:
                reader.record(anchor, row, indices)
        if index == len(self.stages) - 1:
            self.final_rows[anchor] = row
        return bool(any(row))

    def on_settled(self, anchor, survived, row):
        """Settle anchor KV using join answers after all readers finish."""
        self.settled[anchor] = True
        self.inputs["anchor_done"](
            anchor, row if self.final_rows[anchor] is None else self.final_rows[anchor])

    def finish(self, every) -> int:
        """Map predicate answers to reached anchors and count suffix work."""
        self.own_tokens = _predicate_tokens(
            self.stages, every, self.inputs.get("anchor_partners"),
            self.inputs["anchor_keys"], self.canvas_tokens)
        upstream = self.inputs.get("upstream")
        if upstream is not None:
            if every:
                self.settled[np.fromiter(every[0], dtype=np.int64)] = True
            reached = np.flatnonzero(self.settled)
            local_of = np.full(len(self.ids), -1, dtype=np.int32)
            local_of[reached] = np.arange(len(reached), dtype=np.int32)
            self.answers = [
                {local_of[position]: row for position, row in answers.items()}
                for answers in every]
            self.anchor_ids = [self.ids[position] for position in reached]
            self.kv_round = ({"hits": len(reached), "misses": 0} if upstream["parts"]
                             else {"hits": 0, "misses": len(reached)})
        else:
            self.answers = every
            self.anchor_ids = list(self.ids)
            self.kv_round = self.inputs.get("kv_round") or {}
        return sum(self.own_tokens)

    def complete(self, tokens, spans, stats, offset):
        """Build each join predicate's answers and measured costs."""
        nodes, inputs = self.nodes, self.inputs
        node = self.node
        anchor_ids, answers = self.anchor_ids, self.answers
        lists_for = inputs.get("anchor_partners")
        own_tokens = list(self.own_tokens)
        kv_round = self.kv_round
        group = inputs["group"]
        partner_lists = stage_partner_lists(group, lists_for, anchor_ids)
        results = {}
        own_tokens[0] = tokens - sum(own_tokens[1:])
        for j, (member, rows, members) in enumerate(zip(nodes, answers, partner_lists)):
            reached_ids = [anchor_ids[int(local)] for local in sorted(rows)]
            matched = [anchor_ids[int(local)] for local in sorted(rows)
                       if any(rows[local])]
            survivors = matched
            # Anchors with no candidates never launch a request.
            if member.semantics == "anti":
                candidates = np.asarray(
                    anchor_ids if j == 0 else id_positions(results[
                        nodes[j - 1].node_id].outputs[f"rows:{node.anchor}"]),
                    dtype=np.int64)
                survivors = candidates[member.keeps(np.isin(candidates, matched))]
            answer = {
                "rows": rows,
                "anchor_index": anchor_ids,
                "partner_index": inputs["partner_indices"][member.written_position],
                "anchor_partners": members,
                "anchor": member.anchor,
                "partners": list(member.partners),
                "semantics": member.semantics,
                "selectivity": member.selectivity,
                "written_position": member.written_position,
            }
            own_spans = stage_spans(spans, offset + j, offset + j + 1)
            results[member.node_id] = NodeResult(
                outputs={f"rows:{member.anchor}": Rows({member.anchor: survivors}),
                         f"join_answers:{member.written_position}": answer},
                metrics=NodeMetrics(
                    input_rows=len(anchor_ids) if j == 0 else len(reached_ids),
                    output_rows=len(survivors),
                    evaluated_document_pairs=sum(len(row) for row in rows.values()),
                    kv_hits=kv_round.get("hits", 0) if j == 0 else len(rows),
                    kv_misses=kv_round.get("misses", 0) if j == 0 else 0,
                    fresh_tokens=own_tokens[j],
                    gpu_s=gpu_seconds(own_spans),
                    chunks=len(own_spans),
                    extension={
                        "answers": [rows],
                        **({"borrowed_prefix_tokens": stats["borrowed_tokens"]}
                           if j == 0 and stats.get("borrowed_tokens") else {}),
                        **({"pack_s": round(stats["pack_s"], 3)}
                           if j == 0 and stats.get("pack_s") else {}),
                    }))
        self._results = results
        self.result_value = results[node.node_id]
        if self.context is not None:
            # the first predicate's hits count the anchors that reached
            # the group on resident KV
            first = nodes[0]
            self.context.model_result(first, results[first.node_id], self.context)

    def results(self) -> dict:
        """Return results for every join predicate in the group."""
        return self._results


def set_join_questions(node, documents, tails) -> list:
    """Each partner tuple's question tokens, in member order.

    A question is every partner's label and document, each document
    followed by the blank line that opens the tail and joined to it as
    the model's tokenizer joins them, then the rest of the tail.

    Args:
        node: The AiJoin.
        documents: alias -> the partner documents' token ids, by local
            position.
        tails: The model's StateTails.
    """
    labels = dict(node.label_token_ids)
    tail = list(node.tail_token_ids)
    separator, rest = tail[:node.separator_tokens], tail[node.separator_tokens:]
    blocks = {alias: [list(labels[alias]) + tails.prefix(document, separator)
                      for document in documents[alias]]
              for alias in node.partners}
    questions = []
    for positions in itertools.product(
            *(range(len(blocks[alias])) for alias in node.partners)):
        tokens = []
        for alias, position in zip(node.partners, positions):
            tokens += blocks[alias][position]
        questions.append(tokens + rest)
    return questions


def join_pass_stage(node, prefixes, anchors, members_of, documents, tails,
                    readout, set_size: int, budget: int):
    """One join's passes over the anchors, as one unpaged stage.

    Args:
        node: The AiJoin.
        prefixes: Per anchor, its head and document token ids.
        anchors: The anchors to ask, by index into prefixes.
        members_of: Callable(anchor) -> the member indices it asks, or
            None for every member.
        documents: alias -> the partner documents' token ids.
        tails: The model's StateTails.
        readout: The readout of each question's last row.
        set_size: The most questions in one pass.
        budget: The chunk token budget a pass fits.

    Returns:
        The Stage, each pass's state tokens, and each pass's anchor;
        the Stage is None when no anchor asks a member.
    """
    questions = set_join_questions(node, documents, tails)
    lengths = [len(question) for question in questions]
    header = list(node.question_header_token_ids)
    frame = list(node.frame_token_ids)
    states, suffixes, shapes, owners = [], [], [], []
    for anchor in anchors:
        members = members_of(anchor)
        members = range(len(questions)) if members is None else members
        if not len(members):
            continue
        prefix = tails.prefix(prefixes[anchor], frame)
        for block in split_passes(len(prefix), members, lengths, len(header),
                                  set_size, budget):
            suffix, shape = [], []
            for index, member in enumerate(block):
                question = (header if index else []) + questions[member]
                suffix += question
                shape.append(len(question))
            states.append(prefix)
            suffixes.append(suffix)
            shapes.append(tuple(shape))
            owners.append(anchor)
    if not states:
        return None, states, owners
    return Stage(suffixes=suffixes, readout=readout, requests=lambda key: [key],
                 questions=tuple(shapes), decide=lambda _, row: True,
                 label=f"join ({len(states)} passes)"), states, owners


def execute_set_joins(state, part) -> dict:
    """Run a group's joins under set attention.

    Each anchor is the state of its passes, and each of its partner
    tuples is one question; a pass asks up to the model's set size.
    A later join of the group asks only the anchors with a match in
    the join before it.

    Args:
        state: The query's execution state.
        part: The group's JoinPart, which turns the answers into results.
    """
    nodes, inputs = part.nodes, part.inputs
    tails = state_tails(state)
    readout = AsyncSlotMargins(state.torch, state.answer_rows)
    keys = inputs["anchor_keys"]
    lists_for = inputs.get("anchor_partners")
    every, own_tokens, all_spans = [], [], []
    reached = range(len(part.ids))
    for j, node in enumerate(nodes):
        stage, states, owners = join_pass_stage(
            node, inputs["prefixes"], reached,
            lambda anchor, j=j: (None if lists_for is None
                                 else lists_for(keys[anchor][1], j)),
            inputs["partner_documents"], tails, readout,
            state.loaded_model.model_spec.set_size, state.chunk_tokens)
        rows, tokens = {}, 0
        if stage is not None:
            answers, spans, tokens = run_stages(
                state.torch, state.loaded_model.arena, state.loaded_model.pipeline,
                [stage], states, state.chunk_tokens, staging=input_staging(state),
                gpu_timing=state.gpu_timing, paged=False,
                label=f"join predicate {j}")
            for position, anchor in enumerate(owners):
                margins = np.asarray(answers[0][position][0], dtype=float)
                rows.setdefault(anchor, []).extend((margins > 0).tolist())
            # one run per join: its stage 0 is the group's stage j
            all_spans += [({j: by_stage.get(0, 0)}, chunk, start, end)
                          for by_stage, chunk, start, end in spans]
        every.append(rows)
        own_tokens.append(tokens)
        reached = [anchor for anchor in sorted(rows) if any(rows[anchor])]
    last = {anchor: row for rows in every for anchor, row in rows.items()}
    for anchor in sorted(last):
        part.on_settled(anchor, any(last[anchor]), last[anchor])
    part.finish(every)
    part.own_tokens = own_tokens
    part.complete(sum(own_tokens), all_spans, {}, 0)
    return part.results()


def _predicate_tokens(join_stages, answers, lists_for, anchor_keys,
                      canvas_tokens) -> list[int]:
    """Count each predicate's fresh frame and partner suffix tokens."""
    totals = []
    previous = None
    for j, (stage, rows) in enumerate(zip(join_stages, answers)):
        frame = list(stage.frame)
        writes = bool(frame) and frame != previous
        previous = frame
        lengths = np.fromiter(
            (len(suffix) + canvas_tokens for suffix in stage.suffixes),
            dtype=np.int64, count=len(stage.suffixes))
        suffix_tokens = int(lengths.sum())
        total = len(rows) * (len(frame) + canvas_tokens if writes else 0)
        if lists_for is None:
            total += len(rows) * suffix_tokens
        else:
            for anchor in rows:
                indices = lists_for(anchor_keys[anchor][1], j)
                total += (suffix_tokens if indices is None
                          else int(lengths[indices].sum()))
        totals.append(total)
    return totals


def _join_round_kv(anchor_keys, arena) -> dict:
    hits = sum(arena.is_resident(key) for key in anchor_keys)
    return {"hits": hits, "misses": len(anchor_keys) - hits}


def _tuple_suffix(join, documents, member):

    parts = []
    for alias, document in zip(join["partners"], member):
        parts.extend((join["labels"][alias], documents[alias][document]))
    parts.append(join["tail"])
    return concat_tokens(*parts)


def join_suffixes(group, documents, indices):
    """Build partner suffixes and their global document IDs in the same order.

    Args:
        group: Join request descriptions in execution order.
        documents: Selected partner documents indexed by local position.
        indices: Global document IDs for those local positions, by alias.
    """
    suffixes, tuples = [], {}
    for join in group:
        partners = join["partners"]
        members, tokens = [], []
        for positions in itertools.product(*(range(len(indices[p])) for p in partners)):
            members.append([indices[p][i] for p, i in zip(partners, positions)])
            tokens.append(_tuple_suffix(join, documents, positions))
        tuples[join["written_position"]] = members
        suffixes.append(tokens)
    return suffixes, tuples


def runs_over_pairs(join: dict) -> bool:
    """Whether a stage streams each anchor's own pairs."""
    return bool(join.get("pairs_from"))


def partner_maps(group, pair_tables) -> dict:
    """Written position -> anchor row -> partner rows, per pair stage."""
    maps = {}
    for join in group:
        if not runs_over_pairs(join):
            continue
        table = pair_tables.get(join["written_position"])
        if table is None:
            raise ValueError(
                f"join {join['written_position']} runs over pairs but no pair "
                f"table reached it")
        maps[join["written_position"]] = partner_map(
            table, join["anchor"],
            pair_partner(join["anchor"], join["partners"]))
    return maps


def partner_list_builder(group, tuples_by_stage, maps):
    """Per-anchor partner member lists for the stages that run over pairs.

    Args:
        group: The stages' runtime specs.
        tuples_by_stage: Per stage, its partner member tuples of global
            document ids, in partner order.
        maps: written position -> anchor row -> partner rows.

    Returns:
        callable(anchor global id, stage) -> the sorted member
        indices the anchor streams, or None for every member; or None
        when no stage runs over pairs.
    """
    stage_maps = []
    for join, tuples in zip(group, tuples_by_stage):
        if not runs_over_pairs(join):
            stage_maps.append(None)
            continue
        position = join["partners"].index(
            pair_partner(join["anchor"], join["partners"]))
        rows = maps[join["written_position"]]
        cache = (np.full(len(rows.keys), None, dtype=object)
                 if isinstance(rows, RowGroups) else None)
        stage_maps.append((rows, members_by_partner(tuples, position), cache))
    if all(entry is None for entry in stage_maps):
        return None

    def lists_for(anchor, stage):
        entry = stage_maps[stage]
        if entry is None:
            return None
        rows, members, cache = entry
        if cache is None:
            # Per-batch apply results can arrive after the lookup is constructed.
            return allowed_members(rows, members, anchor)
        position = np.searchsorted(rows.keys, anchor)
        if position == len(rows.keys) or rows.keys[position] != anchor:
            return np.empty(0, dtype=np.int64)
        if cache[position] is None:
            cache[position] = allowed_members(rows, members, anchor)
        return cache[position]

    return lists_for


def stage_partner_lists(group, lists_for, anchor_ids) -> list:
    """Per stage, anchor local index -> member indices, or None."""
    out = []
    for index, join in enumerate(group):
        if lists_for is None or not runs_over_pairs(join):
            out.append(None)
            continue
        out.append({local: lists_for(anchor, index)
                    for local, anchor in enumerate(anchor_ids)})
    return out


def prepare_join_inputs(node, inputs, context: ExecutionContext, upstream=None,
                        nodes=None) -> dict:
    """Prepare join requests from graph ports and preceding operator state."""
    state = context.state
    grouped = nodes is not None
    nodes = (node,) if nodes is None else nodes
    by_alias = {}
    port_pairs = {}        # written position -> pair table from a port
    for member, input_port in ((member, port) for member in nodes
                               for port in member.inputs):
        values = inputs[member.node_id] if grouped else inputs
        port = input_port.source.port
        if input_port.name not in values:
            continue           # fed by the upstream itself
        value = values[input_port.name]
        if port.startswith("pairs:"):
            port_pairs[int(port.split(":", 1)[1])] = value
            continue
        if not port.startswith("rows:"):
            continue
        # the suffix and partner builders index these as Python ints
        by_alias[port.split(":", 1)[1]] = id_positions(value).tolist()
    if not state["joins_started"]:
        state["joins_started"] = True
        arena = state["gpu"].arena
        retained = arena.retained_keys()
        state["kv_stats"].update(
            retained_after_filters=len(retained),
            retained_pages_after_filters=arena.retained_pages,
            retained_prefix_tokens_after_filters=arena.retained_prefix_tokens,
            retained_by_alias_after_filters={
                alias: sum(key[0] == alias for key in retained)
                for alias in state["documents"]
            },
        )
    config = state["retention"]
    apply_retention(state["gpu"].arena, config,
                    config.get("before", {}).get(node.node_id, {}), by_alias)
    documents = {alias: select_documents(state["documents"][alias], by_alias[alias])
                 for member in nodes for alias in member.partners}
    prefixes = (DocumentPrefixes(node.document_head, state["documents"][node.anchor],
                                 by_alias[node.anchor])
                if upstream is None else upstream["documents"])
    prepared = join_inputs(nodes, documents, by_alias, prefixes, state["gpu"].arena,
                           config, port_pairs, upstream=upstream, report_answers=True)
    if upstream is None:
        counts = prepared["kv_round"]
        state["kv_stats"]["join_anchor_hits"] += counts["hits"]
        state["kv_stats"]["join_anchor_misses"] += counts["misses"]
    state["join_streamed"] = upstream is not None
    return prepared


def join_inputs(nodes, documents, indices, prefixes, arena, retention, pair_tables,
                *, upstream=None, report_answers=False) -> dict:
    """Build join requests and KV settlement for local or worker execution."""
    node = nodes[0]
    group = [join_request(member) for member in nodes]
    stage_suffixes, tuple_indices = join_suffixes(group, documents, indices)
    maps = partner_maps(group, pair_tables)
    lists_for = partner_list_builder(
        group, [tuple_indices[stage.written_position] for stage in nodes], maps)
    anchor_ids = indices[node.anchor] if upstream is None else None
    anchor_keys = DocumentKeys(node.anchor, anchor_ids if upstream is None
                               else upstream["document_ids"])
    round_kv = _join_round_kv(anchor_keys, arena) if upstream is None else None
    last = group[-1]
    last_tuples = tuple_indices[last["written_position"]]

    def finished_answers(document, row) -> dict:
        """The anchor's last-stage matches: partner tuples that answered true.

        Every pair the anchor was asked about and is not listed answered
        false; ``asked`` says how many pairs that covers.
        """
        members = None
        if lists_for is not None and runs_over_pairs(last):
            members = lists_for(document, len(group) - 1)
        matches = []
        for position, answer in enumerate(row):
            if answer:
                member = position if members is None else int(members[position])
                matches.append(list(last_tuples[member]))
        return {"kind": "join", "node": node.node_id, "anchor": node.anchor,
                "partners": list(last["partners"]),
                "semantics": last["semantics"], "document": int(document),
                "asked": len(row), "matches": matches}

    settle = join_settlement(arena, anchor_keys, prefixes, nodes[-1], retention)

    def anchor_done(local_index, row):
        key = anchor_keys[local_index]
        settle(local_index, row)
        sink = answer_sink() if report_answers else None
        if sink is not None:
            sink(finished_answers(key[1], row))

    return {
        "prefixes": prefixes,
        "partner_documents": documents,
        "stage_suffixes": stage_suffixes,
        "stage_frames": [join.get("frame") or [] for join in group],
        "anchor_keys": anchor_keys,
        "anchor_done": anchor_done,
        "upstream": upstream,
        "kv_round": round_kv,
        "anchor_ids": anchor_ids,
        "partner_indices": tuple_indices,
        "anchor_partners": lists_for,
        "group": group,
    }

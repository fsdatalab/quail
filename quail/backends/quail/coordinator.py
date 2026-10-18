"""Coordinator: split plan payloads across GPU workers and merge answers.

Called by the worker's parent process between child GPUs.
"""

import pyarrow as pa
import pyarrow.compute as pc

from quail.backends.quail.executor.operators.filter import stop_groups
from quail.backends.quail.executor.operators.join import runs_over_pairs
from quail.catalog.pairs import pair_partner
from quail.catalog.tokens import select_documents
from quail.planner import balanced_shards


def _common_payload(payload: dict) -> dict:
    return {
        "model": payload["model"],
        "chunk_tokens": payload["chunk_tokens"],
        "arena_pages": payload.get("arena_pages"),
        "true_ids": payload["true_ids"],
        "false_ids": payload["false_ids"],
        "filter_limit": payload.get("filter_limit"),
        "retention": payload.get("retention", {}),
        "gpu_timing": payload.get("gpu_timing", False),
    }


def begin_query_payloads(payload: dict, gpus: int) -> list[dict]:
    """Build one empty query start payload per GPU executor."""
    outputs = []
    for worker in range(gpus):
        sub = _common_payload(payload)
        sub["physical_plan"] = payload["physical_plan"]
        sub.update(
            documents={},
            document_index={},
            node_id=None,
            worker=worker,
            gpus=gpus,
        )
        outputs.append(sub)
    return outputs


def filter_node_payloads(payload: dict, node, shards: dict,
                         gpus: int, *, final: bool) -> list[dict]:
    """Build one typed filter node payload per GPU executor.

    The result limit ends the filters' run only when final: the
    filters are the query's last model operators, so no later
    operator can drop a survivor.
    """
    outputs = []
    tokens = payload["documents"][node.alias]
    for worker in range(gpus):
        shard = shards[node.alias][worker]
        indices = shard if isinstance(shard, range) else list(shard)
        sub = _common_payload(payload)
        sub["physical_plan"] = payload["physical_plan"]
        sub["filter_limit"] = payload.get("filter_limit") if final else None
        sub.update(
            documents={node.alias: select_documents(tokens, indices)},
            document_index={node.alias: indices},
            stop_groups=stop_groups(
                node, payload.get("columns", {}).get(node.alias), indices),
            node_id=node.node_id,
            worker=worker,
            gpus=gpus,
        )
        outputs.append(sub)
    return outputs


def keyed_shards(groups, document_tokens, gpus: int) -> tuple:
    """Shard documents by stop key group, balanced by token count.

    Every document of a group lands on one worker, so the worker's
    per-key stop sees the whole group.
    """
    tokens = {}
    for group, length in zip(groups, document_tokens):
        tokens[group] = tokens.get(group, 0) + length
    ordered = list(tokens)
    shards, _ = balanced_shards([tokens[group] for group in ordered], gpus)
    placed = {ordered[index]: worker
              for worker, shard in enumerate(shards) for index in shard}
    out = [[] for _ in range(gpus)]
    for document, group in enumerate(groups):
        out[placed[group]].append(document)
    return tuple(tuple(shard) for shard in out)


def join_group_payloads(payload: dict, gpus: int, survivors: dict,
                        joins: list, prior_shards: dict | None = None,
                        *, filtered_aliases: set[str] | None = None,
                        shards: dict | None = None,
                        pair_tables: dict | None = None) -> list:
    """Build per-worker sub-payloads for one anchor joins's join round.

    Anchors follow the shards their KV already sits on - the shards of
    an earlier joins that kept the anchor's KV (prior_shards), else
    their filter shards - otherwise they are re-sharded over the live
    set. Partners are replicated to all workers.
    """
    if not joins:
        return []

    anchor_alias = joins[0]["anchor"]

    def surv(alias):
        # an alias with no filters survives whole
        if alias in survivors:
            return list(survivors[alias])
        return list(range(len(payload["documents"][alias])))

    expected_rows = surv(anchor_alias)
    shards = shards if shards is not None else payload.get("shards") or {}
    filtered_aliases = (filtered_aliases if filtered_aliases is not None
                        else set(payload.get("filters", ())))
    alive = set(expected_rows)
    if prior_shards and anchor_alias in prior_shards:
        anchor_shards = [[g for g in shard if g in alive]
                         for shard in prior_shards[anchor_alias]]
        placed = {g for shard in anchor_shards for g in shard}
        missing = [g for g in expected_rows if g not in placed]
        if missing:
            tokens = payload["documents"][anchor_alias]
            index_shards, _ = balanced_shards(
                [len(tokens[g]) for g in missing], gpus)
            for worker, shard in enumerate(index_shards):
                anchor_shards[worker].extend(missing[i] for i in shard)
    elif anchor_alias in filtered_aliases and anchor_alias in shards:
        anchor_shards = [[g for g in shard if g in alive]
                         for shard in shards[anchor_alias]]
    else:
        tokens = payload["documents"][anchor_alias]
        index_shards, _ = balanced_shards([len(tokens[g]) for g in expected_rows],
                                        gpus)
        anchor_shards = [[expected_rows[i] for i in s] for s in index_shards]
    partner_aliases = sorted({p for j in joins for p in j["partners"]})
    partners = {}
    for alias in partner_aliases:
        indices = surv(alias)
        partners[alias] = {
            "index": indices,
            "documents": select_documents(payload["documents"][alias], indices),
        }
    # a pair stage ships each worker its anchors' live partner rows
    pair_rows = {}
    for j in joins:
        if not runs_over_pairs(j):
            continue
        partner = pair_partner(anchor_alias, j["partners"])
        table = (pair_tables or {}).get(j["written_position"])
        if table is None:
            raise ValueError(f"join {j['written_position']} received no pair table")
        pair_rows[j["written_position"]] = table.filter(pc.is_in(
            table[partner], value_set=pa.array(partners[partner]["index"],
                                               type=table[partner].type)))
    subs = []
    for w in range(gpus):
        sub = _common_payload(payload)
        sub["physical_plan"] = payload["physical_plan"]
        sub.update(joins=joins,
                   anchor_alias=anchor_alias,
                   anchor_index=list(anchor_shards[w]),
                   anchor_documents=select_documents(
                       payload["documents"][anchor_alias], anchor_shards[w]
                   ),
                   partners=partners,
                   pairs={position: table.filter(pc.is_in(
                       table[anchor_alias], value_set=pa.array(
                           anchor_shards[w], type=table[anchor_alias].type)))
                          for position, table in pair_rows.items()},
                   worker=w, gpus=gpus)
        subs.append(sub)
    return subs


def merge_join_answers(parts: list) -> dict:
    """Merge answer rows while translating each worker's local anchor indices."""
    rows, anchor_index = {}, []
    members = None
    for part in parts:
        base = len(anchor_index)
        anchor_index.extend(part["anchor_index"])
        rows.update((base + int(local), row) for local, row in part["rows"].items())
        if part.get("anchor_partners") is not None:
            if members is None:
                members = {}
            members.update((base + int(local), streamed)
                           for local, streamed in part["anchor_partners"].items())
    return dict(parts[0], rows=rows, anchor_index=anchor_index,
                anchor_partners=members)

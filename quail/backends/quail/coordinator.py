"""Coordinator: split plan payloads across GPU workers and merge answers.

Pure dict-and-list logic (no torch). Called by the worker's parent
process between child GPUs.
"""

from quail.backends.quail.graph import partner_maps, runs_over_pairs
from quail.execution.pairs import pair_partner
from quail.execution.tokens import select_documents
from quail.execution.types import join_answer_cells
from quail.planner import balanced_shards


def _common_payload(payload: dict) -> dict:
    return {
        "model": payload["model"],
        "chunk_tokens": payload["chunk_tokens"],
        "arena_pages": payload.get("arena_pages"),
        "true_ids": payload["true_ids"],
        "false_ids": payload["false_ids"],
        "pre_ids": payload.get("pre_ids", []),
        "filter_limit": payload.get("filter_limit"),
        "retention": payload.get("retention", {}),
        "gpu_timing": payload.get("gpu_timing", False),
    }


def begin_query_payloads(payload: dict, k: int) -> list[dict]:
    """Build one empty query start payload per GPU executor."""
    outputs = []
    for worker in range(k):
        sub = _common_payload(payload)
        sub["physical_plan"] = payload["physical_plan"]
        sub.update(
            docs={},
            doc_index={},
            node_id=None,
            worker=worker,
            workers=k,
        )
        outputs.append(sub)
    return outputs


def filter_node_payloads(payload: dict, node, shards: dict,
                         k: int, *, has_joins: bool) -> list[dict]:
    """Build one typed filter node payload per GPU executor."""
    outputs = []
    tokens = payload["docs"][node.alias]
    for worker in range(k):
        shard = shards[node.alias][worker]
        indices = shard if isinstance(shard, range) else list(shard)
        sub = _common_payload(payload)
        sub["physical_plan"] = payload["physical_plan"]
        sub["filter_limit"] = (
            None if has_joins else payload.get("filter_limit")
        )
        sub.update(
            docs={node.alias: select_documents(tokens, indices)},
            doc_index={node.alias: indices},
            node_id=node.node_id,
            worker=worker,
            workers=k,
        )
        outputs.append(sub)
    return outputs


def merge_filter_round(outs: list, limit: int | None = None) -> dict:
    """Merge workers' filter answers, survivors, and token counts."""
    filters, survivors = {}, {}
    tokens = 0
    for out in outs:
        tokens += out["fresh_tokens"]
        for alias, rows in out["filters"].items():
            filters.setdefault(alias, {}).update(rows)
        for alias, surv in out["survivors"].items():
            survivors.setdefault(alias, []).extend(surv)
    merged = {a: sorted(v) for a, v in survivors.items()}
    if limit is not None:
        merged = {a: v[:limit] for a, v in merged.items()}
    return dict(filters=filters, survivors=merged,
                fresh_tokens=tokens)


def gate_group(stage_out: dict, semantics: str) -> list:
    """Return surviving anchor indices after one group's last stage.

    Full/exists keep anchors with any TRUE; anti keeps those with none.
    """
    evaluated = list(stage_out["anchor_index"])
    kept = {evaluated[a] for a, row in stage_out["rows"].items()
            if any(row)}
    if semantics == "anti":
        return [g for g in evaluated if g not in kept]
    return sorted(kept)


def thin_survivors(full_stage_outs: list, survivors: dict) -> dict:
    """Thin survivor lists at a barrier.

    A document stays live only if every finished full stage touching
    its table has it in at least one surviving pair. Affects what later
    stages evaluate, never the final query result.
    """
    for out in full_stage_outs:
        anchor, partners = out["anchor"], out["partners"]
        alive = set(survivors.get(anchor, out["anchor_index"]))
        seen = {al: set() for al in (anchor, *partners)}
        for la, ti, bit in join_answer_cells(out):
            ga = out["anchor_index"][la]
            if ga not in alive or not bit:
                continue
            seen[anchor].add(ga)
            for al, gp in zip(partners, out["partner_index"][ti]):
                seen[al].add(gp)
        for al, ids in seen.items():
            if al in survivors:
                survivors[al] = [g for g in survivors[al] if g in ids]
            else:
                survivors[al] = sorted(ids)
    return survivors


def join_group_payloads(payload: dict, k: int, survivors: dict,
                        group: list, prior_shards: dict | None = None,
                        *, filtered_aliases: set[str] | None = None,
                        shards: dict | None = None,
                        pair_tables: dict | None = None) -> list:
    """Build per-worker sub-payloads for one anchor group's join round.

    Anchors follow the shards their KV already sits on - the shards of
    an earlier group that kept the anchor's KV (prior_shards), else
    their filter shards - otherwise they are re-sharded over the live
    set. Partners are replicated to all workers.
    """
    if not group:
        return []

    anchor_alias = group[0]["anchor"]

    def surv(alias):
        # an alias with no filters survives whole
        if alias in survivors:
            return list(survivors[alias])
        return list(range(len(payload["docs"][alias])))

    live = surv(anchor_alias)
    shards = shards if shards is not None else payload.get("shards") or {}
    filtered_aliases = (filtered_aliases if filtered_aliases is not None
                        else set(payload.get("filters", ())))
    alive = set(live)
    if prior_shards and anchor_alias in prior_shards:
        anchor_shards = [[g for g in shard if g in alive]
                         for shard in prior_shards[anchor_alias]]
        placed = {g for shard in anchor_shards for g in shard}
        missing = [g for g in live if g not in placed]
        if missing:
            toks = payload["docs"][anchor_alias]
            idx_shards, _ = balanced_shards(
                [len(toks[g]) for g in missing], k)
            for worker, shard in enumerate(idx_shards):
                anchor_shards[worker].extend(missing[i] for i in shard)
    elif anchor_alias in filtered_aliases and anchor_alias in shards:
        anchor_shards = [[g for g in shard if g in alive]
                         for shard in shards[anchor_alias]]
    else:
        toks = payload["docs"][anchor_alias]
        idx_shards, _ = balanced_shards([len(toks[g]) for g in live],
                                        k)
        anchor_shards = [[live[i] for i in s] for s in idx_shards]
    partner_aliases = sorted({p for j in group for p in j["partners"]})
    partners = {}
    for alias in partner_aliases:
        indices = surv(alias)
        partners[alias] = {
            "index": indices,
            "docs": select_documents(payload["docs"][alias], indices),
        }
    # a pair stage ships each worker its anchors' live partner rows
    maps = partner_maps(group, pair_tables or {})
    pair_rows = {}
    for j in group:
        if not runs_over_pairs(j):
            continue
        live_partners = set(partners[pair_partner(
            anchor_alias, j["partners"])]["index"])
        pair_rows[j["written_pos"]] = {
            anchor: [p for p in matched if p in live_partners]
            for anchor, matched in maps[j["written_pos"]].items()}
    subs = []
    for w in range(k):
        sub = _common_payload(payload)
        sub["physical_plan"] = payload["physical_plan"]
        sub.update(joins=group,
                   anchor_alias=anchor_alias,
                   anchor_index=list(anchor_shards[w]),
                   anchor_docs=select_documents(
                       payload["docs"][anchor_alias], anchor_shards[w]
                   ),
                   partners=partners,
                   pairs={position: {anchor: rows.get(anchor, [])
                                     for anchor in anchor_shards[w]}
                          for position, rows in pair_rows.items()},
                   worker=w, workers=k)
        subs.append(sub)
    return subs


def merge_join_round(outs: list) -> list:
    """Merge workers' per-stage join answer rows."""
    if not outs:
        return []
    n_stages = len(outs[0]["joins"])
    merged = []
    for s in range(n_stages):
        rows, anchor_index = {}, []
        members = None
        partner_index = outs[0]["joins"][s]["partner_index"]
        for out in outs:
            stage = out["joins"][s]
            base = len(anchor_index)
            anchor_index.extend(stage["anchor_index"])
            for local, row in stage["rows"].items():
                rows[base + int(local)] = row
            if stage.get("anchor_partners") is not None:
                members = members or {}
                for local, streamed in stage["anchor_partners"].items():
                    members[base + int(local)] = streamed
        merged.append(dict(rows=rows, anchor_index=anchor_index,
                           partner_index=partner_index,
                           anchor_partners=members))
    return merged

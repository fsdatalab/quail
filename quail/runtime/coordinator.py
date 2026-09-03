"""Coordinator: split plan payloads across GPU workers and merge answers.

Pure dict-and-list logic (no torch). Called by the worker's parent
process between child GPUs.
"""

def _common_payload(payload: dict) -> dict:
    return {
        "model": payload["model"],
        "chunk_tokens": payload["chunk_tokens"],
        "true_ids": payload["true_ids"],
        "false_ids": payload["false_ids"],
        "pre_ids": payload["pre_ids"],
        "filter_limit": payload["filter_limit"],
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
    from quail.runtime.tokens import select_documents

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


def search_specs(joins: list) -> list:
    """The payload's join specs in the shared search's count form."""
    out = []
    for i, j in enumerate(joins):
        out.append(dict(
            written_pos=j.get("written_pos", i),
            aliases=list(j["aliases"]),
            anchor=j.get("anchor"),
            anchor_free=bool(j.get("anchor_free")),
            semantics=j["semantics"],
            selectivity=j.get("selectivity"),
            frame_tokens={a: len(t) for a, t in j["frames"].items()},
            label_tokens={a: len(t) for a, t in j["labels"].items()},
            tail_tokens=len(j["tail"])))
    return out


def runtime_join_steps(sequence, joins: list) -> tuple:
    """Build typed anchored join and exchange steps for a search result."""
    from quail.physical import AnchoredJoin, Exchange

    pos_to_idx = {j.get("written_pos", i): i
                  for i, j in enumerate(joins)}
    aliases = tuple(sorted({
        alias for join in joins for alias in join.get("aliases", ())
    }))
    groups = []
    for wp, anchor in sequence:
        idx = pos_to_idx[wp]
        full = joins[idx]["semantics"] == "full"
        if groups and full and groups[-1]["full"] \
                and groups[-1]["anchor"] == anchor:
            groups[-1]["idxs"].append(idx)
        else:
            groups.append(dict(anchor=anchor, full=full, idxs=[idx]))
    nodes = []
    barriers = 0
    prev = None
    for i, g in enumerate(groups):
        if prev is not None and prev != g["anchor"]:
            nodes.append(Exchange(
                node_id=f"runtime-exchange:{barriers}",
                next_anchor=g["anchor"], aliases=aliases,
            ))
            barriers += 1
        nodes.append(AnchoredJoin(
            node_id=f"runtime-join:{i}",
            anchor=g["anchor"],
            stage_idxs=tuple(g["idxs"]),
        ))
        prev = g["anchor"]
    return tuple(nodes)


def report_join_plan(sequence, joins: list) -> list:
    """Return typed join steps for a query report."""
    pos_to_join = {
        join.get("written_pos", index): join
        for index, join in enumerate(joins)
    }
    groups = []
    for written_pos, anchor in sequence:
        join = pos_to_join[written_pos]
        full = join["semantics"] == "full"
        if groups and full and groups[-1]["full"] \
                and groups[-1]["anchor"] == anchor:
            groups[-1]["written_positions"].append(written_pos)
        else:
            groups.append({
                "anchor": anchor,
                "full": full,
                "written_positions": [written_pos],
            })

    records = []
    previous_anchor = None
    for index, group in enumerate(groups):
        if previous_anchor is not None \
                and previous_anchor != group["anchor"]:
            records.append({
                "type": "quail.exchange",
                "id": f"runtime-exchange:{len(records)}",
                "next_anchor": group["anchor"],
            })
        records.append({
            "type": "quail.anchored_join",
            "id": f"runtime-join:{index}",
            "anchor": group["anchor"],
            "written_positions": group["written_positions"],
        })
        previous_anchor = group["anchor"]
    return records


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


def stage_for_anchor(spec: dict, anchor: str) -> dict:
    """Return a child-facing copy of one join stage spec with the given anchor."""
    partners = [a for a in spec["aliases"] if a != anchor]
    out = dict(spec)
    out.update(anchor=anchor, partners=partners,
               frame=spec["frames"][anchor],
               labels={p: spec["labels"][p] for p in partners})
    return out


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
        for la, row in out["rows"].items():
            ga = out["anchor_index"][la]
            if ga not in alive:
                continue
            hit = False
            for ti, bit in enumerate(row):
                if bit:
                    hit = True
                    for al, gp in zip(partners,
                                      out["partner_index"][ti]):
                        seen[al].add(gp)
            if hit:
                seen[anchor].add(ga)
        for al, ids in seen.items():
            if al in survivors:
                survivors[al] = [g for g in survivors[al] if g in ids]
            else:
                survivors[al] = sorted(ids)
    return survivors


def join_group_payloads(payload: dict, k: int, survivors: dict,
                        group: list, prior_shards: dict | None = None,
                        *, filtered_aliases: set[str] | None = None,
                        shards: dict | None = None) -> list:
    """Build per-worker sub-payloads for one anchor group's join round.

    Anchors follow the shards their KV already sits on - the shards of
    an earlier group that kept the anchor's KV (prior_shards), else
    their filter shards - otherwise they are re-sharded over the live
    set. Partners are replicated to all workers.
    """
    if not group:
        return []
    from quail.runtime.tokens import select_documents

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
            from quail.planner.decide import balanced_shards
            toks = payload["docs"][anchor_alias]
            idx_shards, _ = balanced_shards(
                [len(toks[g]) for g in missing], k)
            for worker, shard in enumerate(idx_shards):
                anchor_shards[worker].extend(missing[i] for i in shard)
    elif anchor_alias in filtered_aliases and anchor_alias in shards:
        anchor_shards = [[g for g in shard if g in alive]
                         for shard in shards[anchor_alias]]
    else:
        from quail.planner.decide import balanced_shards
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
        partner_index = outs[0]["joins"][s]["partner_index"]
        for out in outs:
            stage = out["joins"][s]
            base = len(anchor_index)
            anchor_index.extend(stage["anchor_index"])
            for local, row in stage["rows"].items():
                rows[base + int(local)] = row
        merged.append(dict(rows=rows, anchor_index=anchor_index,
                           partner_index=partner_index))
    return merged

"""Coordinator: split plan payloads across GPU workers and merge answers.

Pure dict-and-list logic (no torch). Called by the worker's parent
process between child GPUs.
"""

COMMON_KEYS = ("model", "kv_dtype", "chunk_tokens", "true_ids",
               "false_ids", "pre_ids", "limit")


def filter_round_limit(payload: dict):
    """Return the filter round's limit: the payload's limit for filter-only
    queries, None when joins are present.
    """
    if payload.get("joins"):
        return None
    return payload.get("limit")


def filter_round_payloads(payload: dict, shards: dict, k: int) -> list:
    """Build per-worker sub-payloads for the filter round.

    Args:
        shards: alias -> tuple of global document indices per worker.
        k: Number of workers.
    """
    subs = []
    for w in range(k):
        docs, index = {}, {}
        for alias in payload["filters"]:
            toks = payload["docs"][alias]
            idx = list(shards[alias][w]) if alias in shards \
                else list(range(len(toks)))
            docs[alias] = [toks[i] for i in idx]
            index[alias] = idx
        sub = {key: payload[key] for key in COMMON_KEYS}
        # the child never sees the joins, so the join-vs-filter limit
        # rule (filter_round_limit) must be applied at split time
        sub["limit"] = filter_round_limit(payload)
        sub.update(docs=docs, doc_index=index,
                   filters=payload["filters"],
                   filter_arena_writes=payload["filter_arena_writes"],
                   store=payload.get("store"),
                   store_flush=payload.get("store_flush", False),
                   worker=w, workers=k)
        subs.append(sub)
    return subs


def merge_filter_round(outs: list, limit: int | None = None) -> dict:
    """Merge workers' filter answers, survivors, token counts, and store stats."""
    filters, survivors, store = {}, {}, {}
    tokens = 0
    for out in outs:
        tokens += out["fresh_tokens"]
        for alias, rows in out["filters"].items():
            filters.setdefault(alias, {}).update(rows)
        for alias, surv in out["survivors"].items():
            survivors.setdefault(alias, []).extend(surv)
        for alias, st in (out.get("store") or {}).items():
            agg = store.setdefault(alias, {})
            for key, v in st.items():
                agg[key] = agg.get(key, 0) + v
    merged = {a: sorted(v) for a, v in survivors.items()}
    if limit is not None:
        merged = {a: v[:limit] for a, v in merged.items()}
    return dict(filters=filters, survivors=merged,
                fresh_tokens=tokens, store=store)


def stage_for_anchor(spec: dict, anchor: str) -> dict:
    """Return a child-facing copy of one join stage spec with the given anchor."""
    if "frames" not in spec:
        return spec
    partners = [a for a in spec["aliases"] if a != anchor]
    out = dict(spec)
    out.update(anchor=anchor, partners=partners,
               frame=spec["frames"][anchor],
               labels={p: spec["labels"][p] for p in partners})
    return out


def derive_plan_nodes(joins: list) -> list:
    """Reconstruct JoinGroup/Barrier nodes from bare stage specs.

    Used for payloads built without a planner.
    """
    nodes = []
    cur = None
    n_groups = n_barriers = 0

    def flush():
        nonlocal cur, n_groups
        if cur is None:
            return
        nodes.append(dict(id=f"group:{n_groups}", op="JoinGroup",
                          inputs=(), anchor=cur["anchor"],
                          stage_idxs=tuple(cur["idxs"]), stages=()))
        n_groups += 1
        cur = None

    for i, j in enumerate(joins):
        if (cur is not None and j["semantics"] == "full" and cur["full"]
                and cur["anchor"] == j["anchor"]):
            cur["idxs"].append(i)
            continue
        prev_anchor = cur["anchor"] if cur is not None else None
        flush()
        if prev_anchor is not None and prev_anchor != j["anchor"]:
            nodes.append(dict(id=f"barrier:{n_barriers}", op="Barrier",
                              inputs=(), next_anchor=j["anchor"],
                              thins=()))
            n_barriers += 1
        cur = dict(anchor=j["anchor"], full=(j["semantics"] == "full"),
                   idxs=[i])
    flush()
    return nodes


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
                        group: list) -> list:
    """Build per-worker sub-payloads for one anchor group's join round.

    Anchors follow their filter shards when available, otherwise are
    re-sharded over the live set. Partners are replicated to all workers.
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
    shards = payload.get("shards") or {}
    if anchor_alias in payload["filters"] and anchor_alias in shards:
        alive = set(live)
        anchor_shards = [[g for g in shard if g in alive]
                         for shard in shards[anchor_alias]]
    else:
        from quail.planner.decide import balanced_shards
        toks = payload["docs"][anchor_alias]
        idx_shards, _ = balanced_shards([len(toks[g]) for g in live],
                                        k)
        anchor_shards = [[live[i] for i in s] for s in idx_shards]
    partner_aliases = sorted({p for j in group for p in j["partners"]})
    partners = {alias: dict(index=surv(alias),
                            docs=[payload["docs"][alias][g]
                                  for g in surv(alias)])
                for alias in partner_aliases}
    subs = []
    for w in range(k):
        sub = {key: payload[key] for key in COMMON_KEYS}
        sub.update(joins=group,
                   anchor_alias=anchor_alias,
                   anchor_index=list(anchor_shards[w]),
                   anchor_docs=[payload["docs"][anchor_alias][g]
                                for g in anchor_shards[w]],
                   partners=partners,
                   store=payload.get("store"),
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

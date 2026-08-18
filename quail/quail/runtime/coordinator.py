"""The coordinator's arithmetic: splitting a plan payload across GPU
workers and merging their answers back. Pure dict-and-list logic, no
torch, CPU-tested - the worker's parent process calls it between its
children, so the split and the merge never cross a network hop.

The sharding contract, from the design: filters split documents by
token count; joins split by anchor document, so gating, dedup, and
the next stage's pair list stay local to the GPU holding the anchor.
Shards are deterministic for a given corpus, which is what lets each
GPU's store slice serve the same documents query after query.

Two rounds per query:

  round 1 (filters): every worker filters its shard of every alias.
  round 2 (joins): anchors follow their filter shard (their KV is on
  that GPU); every worker sees every surviving partner. Stages that
  anchor on different aliases would need a re-shard between stages;
  that is refused plainly until the benchmark needs it.
"""

COMMON_KEYS = ("model", "kv_dtype", "chunk_tokens", "yes_ids",
               "no_ids")


def filter_round_payloads(payload: dict, shards: dict, k: int) -> list:
    """Per-worker sub-payloads for the filter round. shards:
    alias -> one tuple of global document indices per worker.

    Only aliases WITH filters ship documents in this round: an
    unfiltered partner's documents would otherwise cross the parent-
    child pipe twice (sharded here, replicated in the join round) for
    no work at all. Its survivors default to everything at the join
    round."""
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
        sub.update(docs=docs, doc_index=index,
                   filters=payload["filters"],
                   store=payload.get("store"),
                   store_flush=payload.get("store_flush", False),
                   worker=w, workers=k)
        subs.append(sub)
    return subs


def merge_filter_round(outs: list) -> dict:
    """Merge the workers' filter answers (global-keyed), survivors,
    token counts, and store stats."""
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
    return dict(filters=filters,
                survivors={a: sorted(v) for a, v in survivors.items()},
                fresh_tokens=tokens, store=store)


def join_round_payloads(payload: dict, shards: dict, k: int,
                        survivors: dict) -> list:
    """Per-worker sub-payloads for the join round.

    Every join stage must share one anchor alias (the same-anchor
    chain and star shapes; a stage anchored elsewhere needs the
    deferred re-shard). Anchors follow their filter shard; partners
    are the full surviving lists, identical on every worker so the
    partner index space matches at the merge."""
    joins = payload["joins"]
    if not joins:
        return []
    anchor_alias = joins[0]["anchor"]
    if any(j["anchor"] != anchor_alias for j in joins):
        raise NotImplementedError(
            "join stages anchored on different aliases need the "
            "between-stage re-shard, which is not built yet")

    def surv(alias):
        # an alias with no filters survives whole
        if alias in survivors:
            return list(survivors[alias])
        return list(range(len(payload["docs"][alias])))

    anchor_shards = shards.get(anchor_alias)
    partner_aliases = sorted({j["partner"] for j in joins})
    partners = {alias: dict(index=surv(alias),
                            docs=[payload["docs"][alias][g]
                                  for g in surv(alias)])
                for alias in partner_aliases}
    subs = []
    anchor_live = set(surv(anchor_alias))
    for w in range(k):
        shard = anchor_shards[w] if anchor_shards \
            else range(len(payload["docs"][anchor_alias]))
        anchors = [g for g in shard if g in anchor_live]
        sub = {key: payload[key] for key in COMMON_KEYS}
        sub.update(joins=joins,
                   anchor_alias=anchor_alias,
                   anchor_index=anchors,
                   anchor_docs=[payload["docs"][anchor_alias][g]
                                for g in anchors],
                   partners=partners, worker=w, workers=k)
        subs.append(sub)
    return subs


def merge_join_round(outs: list) -> list:
    """Concatenate the workers' per-stage answer rows. Anchors are
    disjoint across workers; partner index lists are identical, so
    the merged rows read exactly like a single worker's."""
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

"""Apply saved KV retention decisions to a GPU arena."""

from quail.cost.retention import RetentionPolicy


def policy(config, uses):
    """Build the priority rule for one planned execution boundary."""
    return RetentionPolicy(
        config["linear_seconds"], config["pair_seconds"], uses,
        sliding_pair_seconds=config.get("sliding_pair_seconds", 0.0),
        sliding_window=config.get("sliding_window", 0))


def apply_retention(arena, config, uses, survivors=None):
    """Release expired prefixes and update priorities of retained KV."""
    if not config:
        return
    alive = {alias: set(documents) for alias, documents in (survivors or {}).items()}
    for key in arena.retained_keys():
        alias, document = key
        if alias not in uses or (alias in alive and document not in alive[alias]):
            arena.free_key(key)
    arena.configure_retention(
        policy(config, uses), min(config["cap_pages"], arena.num_pages))
    arena.evict_retained(max(0, arena.retained_pages - arena.retention_cap_pages))


def join_settlement(arena, keys, prefixes, node, config):
    """Return the callback that retains or frees a completed join anchor's KV."""
    uses = config.get("after", {}).get(node.node_id, {})
    priority = policy(config, uses) if config else None

    def settle(anchor, row):
        key = keys[anchor]
        if not arena.is_resident(key):
            return
        matched = any(row)
        alive = not matched if node.semantics == "anti" else matched
        if alive and node.keep_kv:
            tokens = len(prefixes[anchor])
            value = (priority.priority(key, tokens, arena.pages_needed(tokens))
                     if priority is not None else None)
            arena.retain(key, tokens, priority=value)
        else:
            arena.free_key(key)

    return settle

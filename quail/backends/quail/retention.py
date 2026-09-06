"""Apply saved KV retention decisions to a GPU arena."""

from quail.executor.retention import RetentionPolicy


def policy(config, uses):
    """Build the priority rule for one planned execution boundary."""
    return RetentionPolicy(config["linear_seconds"], config["pair_seconds"], uses)


def apply_retention(arena, config, uses, survivors=None):
    """Release expired prefixes and update priorities of retained KV."""
    if not config:
        return
    alive = {alias: set(documents) for alias, documents in (survivors or {}).items()}
    for key in list(arena.accounting.retained):
        alias, document = key
        if alias not in uses or (alias in alive and document not in alive[alias]):
            arena.free_key(key)
    arena.accounting.configure_retention(
        policy(config, uses), min(config["cap_pages"], arena.accounting.n_pages))
    arena.evict_retained(max(
        0, arena.accounting.retained_pages - arena.accounting.retention_cap_pages))


def retain_after_join(arena, key, tokens, config, uses):
    """Offer a completed anchor using its next planned reuse priority."""
    priority = None
    if config:
        priority = policy(config, uses).priority(
            key, tokens, arena.accounting.pages_needed(tokens))
    return arena.retain(key, tokens, priority=priority)

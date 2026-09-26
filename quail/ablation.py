"""Switches that turn off one engine feature each, to measure its effect.

`EngineConfig.disabled_features` names the features a session runs
without. Each off setting recreates, in the current code, what the
engine did before the change that added the feature. The settings
exist for the BIO-4 history ablation and are not tuned for use.

The switches are process-wide: a Session sets them when it starts, and
the planner, worker, and vLLM boot read them. Only one-GPU sessions can
turn a feature off, because GPU child processes do not inherit them.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    """One engine change that a switch can turn off.

    Args:
        name: The switch name used in `disabled_features`.
        merged: When the change merged, as a UTC ISO 8601 timestamp.
        pull_request: The pull request number that merged it.
        startup: Whether the change affects startup rather than query
            time.
        summary: What the engine does with the feature on.
        off: What the engine does with the feature off.
    """

    name: str
    merged: str
    pull_request: int
    startup: bool
    summary: str
    off: str


FEATURES = {feature.name: feature for feature in (
    Feature(
        "triton_kernels", "2026-08-16T05:21:37Z", 3, False,
        "fused Triton kernels for norm and quantize, q and k norm with "
        "rotary, and SiLU with quantize",
        "vLLM's unfused kernels for those three steps"),
    Feature(
        "pinned_staging", "2026-08-19T03:07:56Z", 12, False,
        "chunk inputs copied to the GPU through pinned memory without "
        "blocking",
        "pageable, blocking copies"),
    Feature(
        "attention_paths", "2026-08-23T05:57:28Z", 35, False,
        "filters use the unified attention path and joins use merge_quant",
        "filters use the join's two-call attention path"),
    Feature(
        "skip_arena_writes", "2026-08-23T18:17:22Z", 36, False,
        "a one-stage filter whose KV nothing reads skips the KV arena",
        "every filter writes its KV into the arena"),
    Feature(
        "join_search", "2026-08-24T19:29:11Z", 42, False,
        "one search over join order and anchor choice",
        "joins in written order; each join anchors on the input with "
        "the most tokens"),
    Feature(
        "compile_once", "2026-08-25T03:14:02Z", 47, True,
        "the kernel compile pass runs once; later boots only touch kernels",
        "every boot runs the compile pass"),
    Feature(
        "shared_join_prompts", "2026-08-26T07:46:14Z", 52, False,
        "the join question is written once into each anchor's KV",
        "the join question follows every partner document"),
    Feature(
        "filter_kv_reuse", "2026-08-29T22:52:51Z", 70, False,
        "filter survivors keep their KV for the joins; filters ordered "
        "by cost",
        "joins recompute every anchor prefix; filters in written order"),
    Feature(
        "scan_ring", "2026-08-30T21:25:12Z", 72, False,
        "retained KV is capped to leave two chunks of pages for admission",
        "retained KV may fill the arena"),
    Feature(
        "boot_cache", "2026-09-01T03:09:36Z", 78, True,
        "vLLM's cache on the kernel volume and pinned model revisions",
        "a fresh vLLM cache per boot and the model's main branch"),
    Feature(
        "shared_retention", "2026-09-07T06:55:15Z", 79, False,
        "one KV retention pool shared by every planned anchor input",
        "only the first join anchor's filter survivors are retained"),
    Feature(
        "join_continuous_batching", "2026-09-07T18:15:20Z", 81, False,
        "join anchors admitted continuously, stages mixed in one chunk",
        "each arena-sized group of anchors runs one stage at a time and "
        "waits for every answer"),
    Feature(
        "projection_pushdown", "2026-09-07T20:49:14Z", 82, False,
        "scans keep only the columns the query reads",
        "scans keep every source column"),
    Feature(
        "plan_on_estimates", "2026-09-08T01:16:41Z", 85, False,
        "planning uses estimated token counts while tokenization runs",
        "planning waits for exact token counts"),
    Feature(
        "filter_join_streaming", "2026-09-13T01:28:03Z", 92, False,
        "filter survivors stream into the join with their KV pinned",
        "the join starts after its filter finishes"),
    Feature(
        "gigatoken", "2026-09-16T21:20:17Z", 103, False,
        "Gigatoken tokenizes documents and prompts",
        "bpe-qwen tokenizes documents and prompts"),
    Feature(
        "vllm_gigatoken", "2026-09-22T20:51:30Z", 167, False,
        "vLLM receives prompt text and tokenizes it with Gigatoken",
        "vLLM receives prompt token ids built from each document's tokens, "
        "and loads the Hugging Face tokenizer"),
)}

_disabled: frozenset = frozenset()


def configure(disabled) -> None:
    """Set the features this process runs without.

    Raises:
        ValueError: A name is not a known feature.
    """
    global _disabled
    names = frozenset(disabled)
    unknown = sorted(names - set(FEATURES))
    if unknown:
        raise ValueError(
            f"unknown features {unknown}; known: {sorted(FEATURES)}")
    _disabled = names


def enabled(name: str) -> bool:
    """Return whether the named feature is on in this process."""
    if name not in FEATURES:
        raise KeyError(name)
    return name not in _disabled


def disabled_features() -> frozenset:
    """Return the features this process runs without."""
    return _disabled


def merged_after(timestamp: str) -> frozenset:
    """Return the features merged after a UTC ISO 8601 timestamp."""
    return frozenset(
        name for name, feature in FEATURES.items()
        if feature.merged > timestamp)

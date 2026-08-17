"""Join scheduling decisions - no GPU, no vLLM, unit-tested.

A 2-way join evaluates one yes/no question per (anchor, partner)
pair. The anchor document sits first in the prompt, right after the
fixed preamble, so its KV depends on nothing pair-specific and can be
computed once; the partner streams as the per-pair suffix and is
recomputed every pair in every design. The functions here decide the
schedule: which side anchors, how an anchor's suffix stream splits
into chunks, how chunks pack across anchors, and what survives
between the stages of an n-way join.

Length units are tokens everywhere. A "suffix" is one partner
document plus the question tail; suffixes are atomic - a suffix's
tokens attend to each other, so one suffix never splits across two
chunks.
"""


def orient(mean_left_tokens, mean_right_tokens):
    """Which side anchors: the longer one. Anchor tokens are paid
    once per document plus a cheap shared read; partner tokens are
    paid once per pair, so the short side must stream."""
    return "left" if mean_left_tokens >= mean_right_tokens else "right"


def plan_groups(prefix_tokens, suffix_tokens, budget):
    """Split one anchor's suffix stream into consecutive chunk groups.

    Each group is a (start, end) slice of the suffix list whose
    tokens fit under `budget` together with one copy of the prefix
    (the prefix is recomputed - or read from a kept tensor - at the
    head of every group). len(result) is the m of the size rule.
    """
    room = budget - prefix_tokens
    if room <= 0:
        raise ValueError(
            f"prefix {prefix_tokens} tokens leaves no room in a "
            f"{budget}-token chunk")
    groups, start, used = [], 0, 0
    for i, s in enumerate(suffix_tokens):
        if s > room:
            raise ValueError(
                f"suffix {i} has {s} tokens; at most {room} fit beside "
                f"a {prefix_tokens}-token prefix (suffixes are atomic)")
        if used + s > room:
            groups.append((start, i))
            start, used = i, 0
        used += s
    if used or not suffix_tokens:
        groups.append((start, len(suffix_tokens)))
    return groups


def pack_stream(anchors, budget, keep=(), already_kept=()):
    """Brim-pack a whole pair list into chunks, keeping cut prefixes.

    anchors: list of (prefix_tokens, [suffix_tokens]) in run order.
    keep: anchor indices whose prefix K/V a later stage needs.
    already_kept: anchor indices whose prefix K/V is in the store
    from an earlier stage; their groups never pack prefix tokens.

    Returns (chunks, capture). Each chunk is a list of groups
    (anchor_index, start, end, carried); carried means the group
    packs a fresh copy of the anchor's prefix tokens ahead of
    suffixes start..end. An anchor's prefix is packed at most once,
    ever: when the budget cuts its stream, the anchor continues in
    the next chunk with carried False, reading the captured prefix
    K/V instead of re-emitting tokens. capture is the set of anchors
    whose fresh prefix K/V must outlive its chunk - cut mid-stream,
    or named in keep. Cuts happen only at suffix boundaries; when an
    anchor's stream ends mid-chunk, the next anchor starts in the
    same chunk.
    """
    keep, already = set(keep), set(already_kept)
    chunks, chunk, used = [], [], 0
    capture = set()
    for a, (prefix, suffixes) in enumerate(anchors):
        n = len(suffixes)
        placed = a in already
        if n == 0:
            if placed or a not in keep:
                continue    # nothing streams against it and no later
                            # stage needs it: computing it serves no one
            # capture-only placement: a prefix a later stage needs,
            # with nothing streamed against it in this one
            if prefix > budget:
                raise ValueError(
                    f"anchor {a}: prefix {prefix} tokens exceeds the "
                    f"{budget}-token chunk budget")
            if used + prefix > budget:
                chunks.append(chunk)
                chunk, used = [], 0
            chunk.append((a, 0, 0, True))
            used += prefix
            if a in keep:
                capture.add(a)
            continue
        i = 0
        while i < n:
            carried = 0 if placed else prefix
            if suffixes[i] + carried > budget:
                raise ValueError(
                    f"anchor {a} suffix {i} has {suffixes[i]} tokens; "
                    f"at most {budget - carried} fit beside what its "
                    f"group must carry (suffixes are atomic)")
            room = budget - used - carried
            if room < suffixes[i]:
                chunks.append(chunk)
                chunk, used = [], 0
                continue
            j, group_tokens = i, 0
            while j < n and group_tokens + suffixes[j] <= room:
                group_tokens += suffixes[j]
                j += 1
            chunk.append((a, i, j, not placed))
            used += carried + group_tokens
            placed = True
            i = j
            if i < n and a not in already:
                capture.add(a)      # the stream continues elsewhere
        if a in keep and a not in already:
            capture.add(a)
    if chunk:
        chunks.append(chunk)
    return chunks, capture


def gate(answer_rows):
    """Anchors that survive a conjunctive stage: any YES in the row.
    answer_rows: dict anchor_index -> iterable of 0/1 answers.
    Returns the sorted surviving anchor indices."""
    return sorted(a for a, row in answer_rows.items() if any(row))


def matches(answer_rows):
    """anchor_index -> sorted list of partner indices answered YES."""
    return {a: sorted(i for i, v in enumerate(row) if v)
            for a, row in answer_rows.items()}


def assemble(ans1_rows, ans2_rows):
    """Output triples of a chain 3-way from recorded answers.

    ans1_rows: b -> row of 0/1 over A (stage 1, anchored on b).
    ans2_rows: b -> row of 0/1 over C, present only for gated
    survivors. Triples fan out from each surviving b's matched a's
    crossed with its matched c's - no model calls.
    """
    m1, m2 = matches(ans1_rows), matches(ans2_rows)
    out = []
    for b in sorted(ans2_rows):
        for a in m1.get(b, ()):
            for c in m2[b]:
                out.append((a, b, c))
    return sorted(out)


def brute_force_triples(ans1_rows, ans2_rows):
    """The nested-loop reference over the same recorded answers: no
    gating, no dedup. Short-circuit order means a gated b - whose
    stage-2 row was never recorded - is never looked up, because its
    stage-1 row has no YES."""
    out = []
    for b, row1 in sorted(ans1_rows.items()):
        for a, v1 in enumerate(row1):
            if v1:
                for c, v2 in enumerate(ans2_rows[b]):
                    if v2:
                        out.append((a, b, c))
    return sorted(out)

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


def pack_stream(anchors, budget):
    """Brim-pack a whole pair list into chunks.

    anchors: list of (prefix_tokens, [suffix_tokens]) in run order.
    Returns a list of chunks; each chunk is a list of groups
    (anchor_index, start, end) whose token total - every group
    costing one prefix copy plus its suffixes - fits under budget.
    Cuts happen only at suffix boundaries. When an anchor's stream
    ends mid-chunk, the next anchor's prefix starts in the same
    chunk; when the budget lands mid-stream, the anchor continues in
    the next chunk with its prefix re-emitted there.
    """
    chunks, chunk, used = [], [], 0
    for a, (prefix, suffixes) in enumerate(anchors):
        i, n = 0, len(suffixes)
        while i < n or (n == 0 and i == 0):
            room = budget - used - prefix
            if room <= 0:
                if not chunk:
                    raise ValueError(
                        f"anchor {a}: prefix {prefix} tokens exceeds the "
                        f"{budget}-token chunk budget")
                chunks.append(chunk)
                chunk, used = [], 0
                continue
            j, group_tokens = i, 0
            while j < n and group_tokens + suffixes[j] <= room:
                group_tokens += suffixes[j]
                j += 1
            if j == i and n:
                if suffixes[i] > budget - prefix:
                    raise ValueError(
                        f"anchor {a} suffix {i} has {suffixes[i]} tokens; "
                        f"at most {budget - prefix} fit beside its prefix "
                        f"(suffixes are atomic)")
                chunks.append(chunk)
                chunk, used = [], 0
                continue
            chunk.append((a, i, j))
            used += prefix + group_tokens
            i = j
            if n == 0:
                break
    if chunk:
        chunks.append(chunk)
    return chunks


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

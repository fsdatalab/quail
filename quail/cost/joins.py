"""Model work for a join with given row counts and KV residency."""

from quail.cost.work import Work, triangle


def stage_work(spec: dict, anchor: str, n: float, tuples: float, lengths: dict,
               pre: int, *, resident: bool = False, window: int = 0) -> Work:
    """Expected Work of one stage at the current live counts.

    The length sums make the calculation constant time in the number
    of documents without a window. With a window, it also visits the
    distinct lengths below that window. A resident prefix pays its
    frame only. A missing prefix scans the preamble, document, and frame;
    the frame includes the first partner's label. Every tuple then
    carries the other partner labels, partner documents, and the answer
    cue.
    """
    stats = lengths[anchor]
    if window != stats.window:
        raise ValueError("length summaries must use the model attention window")
    if stats.count == 0 or n <= 0:
        return Work()
    partners = [a for a in spec["aliases"] if a != anchor]
    # matches JoinStage.runtime_spec: the first label is in the frame
    u = spec["tail_tokens"] + sum(
        spec["label_tokens"][p] + lengths[p].mean for p in partners[1:]
    ) + (lengths[partners[0]].mean if partners else 0.0)
    frame = spec["frame_tokens"][anchor] + (
        spec["label_tokens"][partners[0]] if partners else 0)
    per_anchor = tuples / n
    frac = n / stats.count

    count = stats.count
    prefix_sum = stats.total + pre * count
    prefix_squared = (
        stats.squared + 2 * pre * stats.total + pre * pre * count)
    if resident:
        start = Work(
            tokens=count * frame,
            pairs=frame * prefix_sum + count * triangle(frame),
            kv_written=count * frame,
            kv_read=prefix_sum,
            sliding_pairs=(stats.window_pairs(pre + frame) - stats.window_pairs(pre)
                           if window else 0.0),
            sliding_kv_read=stats.window_reads(pre) if window else 0.0,
        )
    else:
        scan_sum = prefix_sum + count * frame
        scan_squared = (
            prefix_squared + 2 * frame * prefix_sum + count * frame * frame)
        start = Work(
            tokens=scan_sum,
            pairs=(scan_squared + scan_sum) / 2,
            kv_written=scan_sum,
            sliding_pairs=stats.window_pairs(pre + frame) if window else 0.0,
        )
    stream = Work(
        tokens=count * per_anchor * u,
        pairs=per_anchor * (
            u * (prefix_sum + count * frame)
            + count * triangle(u)),
        kv_written=count * per_anchor * u,
        kv_read=prefix_sum + count * frame,
        sliding_pairs=(per_anchor * (stats.window_pairs(pre + frame + u)
                                    - stats.window_pairs(pre + frame))
                       if window else 0.0),
        sliding_kv_read=stats.window_reads(pre + frame) if window else 0.0,
    )
    return (start + stream) * frac

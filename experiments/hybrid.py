"""Reference calculations and shared runners for hybrid model validation."""

L2_EPS = 1e-6


def l2_normalize(x):
    """Scale the last dimension of x to unit length."""
    import torch

    x = x.float()
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + L2_EPS)


def causal_conv_reference(x, weight, lengths, history=None, bias=None,
                          activate=True):
    """Depthwise causal convolution, then SiLU, per sequence of packed rows.

    Args:
        x: (rows, channels) inputs of the sequences back to back.
        weight: (channels, width) filter taps; the last tap is the
            current row.
        lengths: Rows per sequence.
        history: (sequences, width - 1, channels) input columns before
            each sequence, oldest first; None means zeros.
        bias: Optional (channels,) bias.
        activate: Whether SiLU follows the convolution; a short
            convolution layer has none.

    Returns:
        (output with x's shape, tails), where tails is
        (sequences, width - 1, channels): each sequence's last width - 1
        input columns, which the next row would read as its history.
    """
    import torch
    import torch.nn.functional as F

    rows, channels = x.shape
    width = weight.shape[1]
    lengths_t = torch.as_tensor(list(lengths), device=x.device)
    starts = torch.cumsum(lengths_t, 0) - lengths_t
    sequence = torch.repeat_interleave(
        torch.arange(len(lengths), device=x.device), lengths_t)
    offset = torch.arange(rows, device=x.device) - starts[sequence]
    out = torch.zeros((rows, channels), dtype=torch.float32, device=x.device)
    for back in range(width):
        tap = weight[:, width - 1 - back].float()
        source = (torch.arange(rows, device=x.device) - back).clamp(min=0)
        column = x[source].float()
        before = offset < back
        if history is None:
            column = torch.where(before[:, None], 0.0, column)
        elif before.any():
            past = history[sequence[before], width - 1 - back + offset[before]]
            column[before] = past.float()
        out += column * tap
    if bias is not None:
        out += bias.float()
    tails = _conv_tails(x, lengths_t, starts, history, width)
    return (F.silu(out) if activate else out).to(x.dtype), tails


def _conv_tails(x, lengths, starts, history, width):
    """Each sequence's last width - 1 input columns, oldest first."""
    import torch

    count = lengths.shape[0]
    index = lengths[:, None] + torch.arange(width - 1, device=x.device)
    in_history = index < width - 1
    row = (starts[:, None] + index - (width - 1)).clamp(min=0)
    tails = x[row.clamp(max=x.shape[0] - 1)]
    pad = (history if history is not None
           else torch.zeros((count, width - 1, x.shape[1]), dtype=x.dtype,
                            device=x.device))
    old = pad.gather(1, index.clamp(max=width - 2)[:, :, None].expand(
        -1, -1, x.shape[1]))
    return torch.where(in_history[:, :, None], old.to(x.dtype), tails)


def gated_delta_reference(q, k, v, g, beta, state, lengths, scale):
    """Gated delta rule over packed sequences, one token at a time.

    Per token: S <- exp(g) * S (I - beta k k^T) + beta v k^T, and the
    output is S q. Queries and keys are L2-normalized and the query is
    multiplied by scale first.

    Args:
        q: (rows, key heads, key dim).
        k: (rows, key heads, key dim).
        v: (rows, value heads, value dim).
        g: (rows, value heads) log of the decay.
        beta: (rows, value heads) write strength.
        state: (sequences, value heads, value dim, key dim), float32;
            updated in place to each sequence's final state.
        lengths: Rows per sequence.
        scale: Query multiplier after normalization.

    Returns:
        (rows, value heads, value dim) outputs, in v's dtype.
    """
    import torch

    value_heads = v.shape[1]
    repeat = value_heads // q.shape[1]
    q = l2_normalize(q).repeat_interleave(repeat, dim=1) * scale
    k = l2_normalize(k).repeat_interleave(repeat, dim=1)
    out = torch.empty(v.shape, dtype=torch.float32, device=v.device)
    decay = g.float().exp()
    row = 0
    for index, length in enumerate(lengths):
        s = state[index]
        for t in range(row, row + length):
            s = s * decay[t][:, None, None]
            predicted = (s * k[t][:, None, :]).sum(-1)
            delta = (v[t].float() - predicted) * beta[t].float()[:, None]
            s = s + delta[:, :, None] * k[t][:, None, :]
            out[t] = (s * q[t][:, None, :]).sum(-1)
        state[index] = s
        row += length
    return out.to(v.dtype)


def run_filter(torch, arena, pipeline, answers, documents, questions, budget,
                document_done=None):
    """Run Boolean filter stages over the documents; return bits per document."""
    from quail.backends.quail.executor.stages import filter_stages, run_stages

    stages = filter_stages(questions, answers)
    last = len(stages) - 1
    arena.retention_cap_pages = max(
        0, arena.num_pages - arena.pages_needed(2 * budget))

    def on_settled(document, survived, row):
        if arena.is_resident(document):
            arena.free_key(document)

    def on_chunk(transitions):
        finished = [(document, stage, passed)
                    for document, stage, passed in transitions
                    if not passed or stage == last]
        if finished:
            document_done(finished)

    stage_answers, _, _ = run_stages(
        torch, arena, pipeline, stages, documents, budget,
        anchor_keys=list(range(len(documents))), on_settled=on_settled,
        attention_mode="unified", default_attention="unified",
        on_chunk=on_chunk if document_done is not None else None)
    bits = {}
    for stage in stage_answers:
        for document, row in stage.items():
            bits.setdefault(document, []).append(int(row[0]))
    return bits

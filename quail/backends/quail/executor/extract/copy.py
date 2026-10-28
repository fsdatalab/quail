"""Document continuations and the GPU constraint for copy decoding."""

import numpy as np

from quail.backends.quail.executor.extract.candidates import (
    CandidateIndex,
    SqueezedText,
    Stream,
    ensure_tokens,
)
from quail.cost.draft_policy import LIVE_BUCKETS, PRIOR_WEIGHT


def _padded(values, dtype=np.int64):
    width = max(1, max(map(len, values), default=0))
    out = np.full((len(values), width), -1, dtype=dtype)
    for i, row in enumerate(values):
        out[i, : len(row)] = row
    return out


class CopyConstraint:
    """Tokenized document positions shared by a pipeline's extraction fields.

    Tokenization happens before decoding. Each spelling shares the continuation
    after its document token with every other spelling of that token.
    """

    def __init__(self, torch, device, documents, spec):
        self.torch, self.device, self.spec = torch, device, spec
        self.documents = documents
        vocabulary = self.vocabulary = documents.vocabulary
        options, fillers, positions, streams, keys = [], [], [], [], []
        self.num_keys = 1
        for document in documents.documents:
            scan = document.scan
            index = scan.candidates = CandidateIndex.of(scan, vocabulary)
            normalized = scan.normalized = SqueezedText(document.text)
            option0, position0 = len(options), len(positions)
            options.extend(index.token.tolist())
            fillers.extend(
                vocabulary.filler(int(scan.tokens[index.anchor[k]]), int(n))
                for k, n in zip(index.segment, index.length)
            )
            positions.extend(
                Stream(
                    int(a),
                    [int(t)],
                    [int(b)],
                    document.text,
                    normalized=normalized,
                    first_window_chars=spec.first_window_chars,
                    window_chars=spec.window_chars,
                )
                for a, b, t in zip(scan.starts, scan.ends, scan.tokens)
            )
            local = []
            for at, k in enumerate(index.inverse):
                start = index.offsets[k]
                end = (
                    index.offsets[k + 1]
                    if k + 1 < len(index.kids)
                    else len(index.token)
                )
                local.extend(
                    (position0 + at, option0 + j, int(k)) for j in range(start, end)
                )
            streams.append(local)
            keys.append(len(index.kids))
        num_tokens = spec.max_answer_tokens + spec.draft_tokens
        ensure_tokens(positions, num_tokens - 1, vocabulary)
        for stream in positions:
            del stream.stream[num_tokens:]
            del stream.ends[num_tokens:]
        self.num_keys = max(1, max(keys, default=0))
        self.characters = np.array([s.char for s in positions], dtype=np.int64)
        self.ends = _padded([s.ends[: spec.max_answer_tokens] for s in positions])
        self.filler_lengths = np.array(list(map(len, fillers)), dtype=np.int64)
        self.position_lengths = np.array([len(s.stream) - 1 for s in positions])
        bounds = np.r_[0, np.cumsum(list(map(len, streams)))]
        self.streams = _padded(
            [list(range(start, end)) for start, end in zip(bounds[:-1], bounds[1:])]
        )
        flat = np.array([s for row in streams for s in row], dtype=np.int64)
        # Index zero is also the harmless gather target for padded positions.
        if not len(flat):
            flat = np.zeros((1, 3), dtype=np.int64)
            options, fillers, positions = [0], [[]], [Stream(0, [0], [0], "")]
            self.characters = np.array([0])
            self.ends = np.array([[0]])
            self.filler_lengths = np.array([0])
            self.position_lengths = np.array([0])
        self.position, self.option, self.key = flat.T
        self.stream_ids = self.tensor(self.streams)
        self.position_ids = self.tensor(self.position)
        self.option_ids = self.tensor(self.option)
        self.key_ids = self.tensor(self.key)
        self.first_tokens = self.tensor(options)
        self.fillers = self.tensor(_padded(fillers))
        self.num_fillers = self.tensor(self.filler_lengths)
        self.continuations = self.tensor(
            _padded([s.stream[1:] for s in positions], dtype=np.int32)
        )
        self.characters_device = self.tensor(self.characters)
        self.is_stop = self.tensor(vocabulary.is_stop)
        self.stop_ids = self.tensor(vocabulary.stop_ids)

    def tensor(self, value):
        return self.torch.as_tensor(value).to(self.device, non_blocking=True)

    def tokens_at(self, streams, at):
        """Gather a stream's token at each answer index, or -1 past its end."""
        torch = self.torch
        valid = streams >= 0
        streams = streams.clamp_min(0)
        position, option = self.position_ids[streams], self.option_ids[streams]
        size = self.num_fillers[option]
        following = at - size - 1
        continuation = self.continuations[
            position, following.clamp(0, self.continuations.shape[1] - 1)
        ]
        continuation = torch.where(
            following < self.continuations.shape[1], continuation, -1
        )
        filler = self.fillers[option, (at - 1).clamp(0, self.fillers.shape[1] - 1)]
        return torch.where(
            valid,
            torch.where(
                at == 0,
                self.first_tokens[option],
                torch.where(at <= size, filler, continuation),
            ),
            -1,
        )

    def choose(self, logits, streams, alive, at):
        """Choose allowed tokens, stopping before a token leaves the document."""
        torch = self.torch
        logits = logits.float()
        logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        top_values, top_ids = torch.topk(logits, self.spec.top_tokens, dim=1)
        scores = torch.full_like(logits, -float("inf"))
        scores.scatter_(1, top_ids, top_values)
        stop = torch.logsumexp(logits.index_select(1, self.stop_ids), dim=1)
        top = top_ids[:, 0]
        tokens = self.tokens_at(streams, at[:, None])
        values = scores.gather(1, tokens.clamp_min(0))
        values = torch.where(alive & (tokens >= 0), values, -float("inf"))
        key = self.key_ids[streams.clamp_min(0)]
        maxima = torch.full((len(at), self.num_keys), -float("inf"), device=self.device)
        maxima.scatter_reduce_(1, key, values, reduce="amax", include_self=True)
        option = self.option_ids[streams.clamp_min(0)]
        ranked = torch.where(
            values == maxima.gather(1, key), option, len(self.first_tokens)
        )
        best_option = torch.full(
            (len(at), self.num_keys),
            len(self.first_tokens),
            dtype=torch.int64,
            device=self.device,
        )
        best_option.scatter_reduce_(1, key, ranked, reduce="amin", include_self=True)
        first_key = maxima.argmax(dim=1)
        first_option = best_option.gather(1, first_key[:, None]).squeeze(1)
        first = self.first_tokens[first_option.clamp_max(len(self.first_tokens) - 1)]
        best = values.max(dim=1).values
        following = (
            torch.where(values == best[:, None], tokens, logits.shape[1])
            .min(dim=1)
            .values
        )
        chosen = torch.where(at == 0, first, following)
        preferred = (option == best_option.gather(1, key)) & values.isfinite()
        allows = torch.where(at[:, None] == 0, preferred, alive) & (
            tokens == top[:, None]
        )
        leaves = (
            (top_values[:, 0] > torch.maximum(best, stop))
            & ~allows.any(dim=1)
            & ~self.is_stop[top]
        )
        ended = leaves | torch.where(at == 0, stop > best, stop >= best)
        ended |= (at > 0) & ~(alive & (tokens >= 0)).any(dim=1)
        keep = alive & (tokens == chosen[:, None])
        keep &= (at[:, None] != 0) | preferred
        return chosen, ended, keep

    def source(self, streams, alive):
        """Return the first surviving position in document order."""
        chars = self.characters_device[self.position_ids[streams.clamp_min(0)]]
        chars = self.torch.where(alive, chars, np.iinfo(np.int64).max)
        return streams.gather(1, chars.argmin(dim=1)[:, None]).squeeze(1)

    def output_answers(self, documents, sources, lengths):
        """Map chosen streams and answer lengths back to original text spans."""
        values, spans = [], []
        for document, source, length in zip(documents, sources, lengths):
            if length <= 0 or source < 0:
                values.append(None)
                spans.append(None)
                continue
            position, option = self.position[source], self.option[source]
            start = int(self.characters[position])
            index = max(0, length - 1 - self.filler_lengths[option])
            end = int(self.ends[position, index])
            text = self.documents.documents[document].text
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            values.append(text[start:end])
            spans.append((start, end))
        return values, spans


def draft_lengths(spec, lengths, written, num_streams, rho, trials, accepts):
    """Choose all running answers' adaptive draft lengths with array operations."""
    cap = spec.draft_tokens
    if spec.draft_policy != "adaptive":
        return np.full(len(written), cap, dtype=np.int64)
    j = np.arange(spec.max_answer_tokens + 1)
    observed = np.bincount(lengths[lengths > 0], minlength=len(j))
    q = 1 - 1 / max(1, spec.expected_answer_tokens)
    curve = PRIOR_WEIGHT * q**j + observed.sum() - np.cumsum(observed)
    columns = written[:, None] + np.arange(cap + 1)
    survival = curve[np.minimum(columns, spec.max_answer_tokens)]
    base = survival[:, :1]
    survival = np.divide(survival, base, out=np.ones_like(survival), where=base > 0)
    buckets = np.abs(np.log1p(num_streams[:, None]) - np.log1p(LIVE_BUCKETS)).argmin(1)
    odds = np.where(
        num_streams <= 1,
        1,
        (PRIOR_WEIGHT / np.array(LIVE_BUCKETS)[buckets] + accepts[buckets])
        / (PRIOR_WEIGHT + trials[buckets]),
    )
    writes = 1 + odds[:, None] * np.cumsum(
        np.c_[np.zeros(len(written)), survival[:, 1:]], axis=1
    )
    costs = (1 + np.arange(cap + 1) + rho[:, None]) / writes
    costs[columns >= spec.max_answer_tokens] = np.inf
    return costs.argmin(axis=1)

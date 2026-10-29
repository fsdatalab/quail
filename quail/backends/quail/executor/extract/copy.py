"""Document continuations and the GPU constraint for copy decoding."""

import numpy as np

from quail.backends.quail.executor.extract.candidates import (
    CandidateIndex,
    SqueezedText,
    Stream,
    ensure_tokens,
)
from quail.backends.quail.executor.extract.trace import TRACE_WIDTH, trace_path
from quail.cost.draft_policy import LIVE_BUCKETS, PRIOR_WEIGHT


def _padded(values, dtype=np.int64):
    width = max(1, max(map(len, values), default=0))
    out = np.full((len(values), width), -1, dtype=dtype)
    for i, row in enumerate(values):
        out[i, : len(row)] = row
    return out


class CopyConstraint:
    """Tokenized document positions shared by a pipeline's extraction fields.

    Continuations are prepared after the first token is chosen, only for starts
    it can reach. Fields and spellings share each prepared continuation.
    """

    def __init__(self, torch, device, documents, spec):
        self.torch, self.device, self.spec = torch, device, spec
        self.documents = documents
        self.trace = trace_path()
        self.trace_width = TRACE_WIDTH if self.trace is not None else 0
        vocabulary = self.vocabulary = documents.vocabulary
        options, filler_tokens, filler_chars = [], [], []
        position_keys, option_keys = [], []
        position_local_keys, option_local_keys = [], []
        key0 = 0
        characters, ends, position_tokens, position_documents = [], [], [], []
        streams, keys, option_rows = [], [], []
        option0 = position0 = 0
        for d, document in enumerate(documents.documents):
            scan = document.scan
            index = scan.candidates = CandidateIndex.of(scan, vocabulary)
            options.append(index.token)
            position_keys.append(index.inverse + key0)
            option_keys.append(index.segment + key0)
            position_local_keys.append(index.inverse)
            option_local_keys.append(index.segment)
            key0 += len(index.kids)
            filler_tokens.append(scan.tokens[index.anchor[index.segment]])
            filler_chars.append(index.length)
            characters.append(scan.starts)
            ends.append(scan.ends)
            position_tokens.append(scan.tokens)
            position_documents.append(np.full(len(scan.tokens), d, dtype=np.int64))
            streams.append(np.arange(position0, position0 + len(scan.tokens)))
            keys.append(len(index.kids))
            option_rows.append(np.arange(option0, option0 + len(index.token)))
            option0 += len(index.token)
            position0 += len(scan.tokens)
        self.num_keys = max(1, max(keys, default=0))
        self.characters = np.concatenate(characters) if position0 else np.array([0])
        self.position_ends = np.concatenate(ends) if position0 else np.array([0])
        self.position_tokens = (
            np.concatenate(position_tokens) if position0 else np.array([0])
        )
        self.position_documents = (
            np.concatenate(position_documents) if position0 else np.array([0])
        )
        self.option_tokens = np.concatenate(options) if option0 else np.array([0])
        self.filler_tokens = np.concatenate(filler_tokens) if option0 else np.array([0])
        self.filler_chars = np.concatenate(filler_chars) if option0 else np.array([0])
        self.option_rows = _padded(option_rows)
        self.position_keys = (
            np.concatenate(position_keys) if position0 else np.array([0])
        )
        lookup = (np.concatenate(option_keys) if option0 else np.array([0])) * len(
            vocabulary.vocab
        ) + self.option_tokens
        self.option_order = np.argsort(lookup)
        self.option_lookup = lookup[self.option_order]
        self.filler_lengths = np.full(len(self.option_tokens), -1, dtype=np.int64)
        self.position_lengths = np.zeros(len(self.characters), dtype=np.int64)
        self.position_rows = np.full(len(self.characters), -1, dtype=np.int64)
        counts = np.array(list(map(len, position_tokens)), dtype=np.int64)
        self.position_offsets = np.r_[0, np.cumsum(counts)]
        self.max_positions = max(1, int(counts.max(initial=0)))
        self.mask_words = -(-self.max_positions // 64)
        self.position_offsets_device = self.tensor(self.position_offsets)
        self.mask_bits = self.tensor(np.arange(64))
        self.positions = []
        self.exhausted = np.zeros(len(self.characters), dtype=bool)
        self.prepared = np.empty(0, dtype=np.int64)
        self.num_positions = 0
        self.num_options = len(self.option_tokens)
        positions = _padded(streams)
        self.streams = np.where(positions >= 0, positions * self.num_options, -1)
        self.stream_ids = self.tensor(self.streams)
        self.option_rows_device = self.tensor(self.option_rows)
        self.key_ids = self.tensor(
            np.concatenate(position_local_keys) if position0 else np.array([0])
        )
        self.option_key_ids = self.tensor(
            np.concatenate(option_local_keys) if option0 else np.array([0])
        )
        self.first_tokens = self.tensor(self.option_tokens)
        self.filler_rows = np.full(self.num_options, -1, dtype=np.int64)
        self.filler_rows_device = self.tensor(self.filler_rows.copy())
        self.num_filler_rows = 0
        self.fillers = torch.full((1, 1), -1, dtype=torch.int64, device=device)
        self.num_fillers = torch.zeros(1, dtype=torch.int64, device=device)
        self.continuations = torch.full(
            (1, spec.max_answer_tokens + spec.draft_tokens),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.ends = np.full((1, spec.max_answer_tokens), -1, dtype=np.int64)
        self.continuations_host = np.full(self.continuations.shape, -1, dtype=np.int32)
        self.fillers_host = np.full(self.fillers.shape, -1, dtype=np.int64)
        self.changed = np.zeros(len(self.characters), dtype=bool)
        self.position_rows_device = self.tensor(self.position_rows.copy())
        self.characters_device = self.tensor(self.characters)
        self.is_stop = self.tensor(vocabulary.is_stop)
        self.stop_ids = self.tensor(vocabulary.stop_ids)

    def _prepare_fillers(self, documents, tokens):
        width = len(self.vocabulary.vocab)
        pairs = np.setdiff1d(np.unique(documents * width + tokens), self.prepared)
        if not len(pairs):
            return
        self.prepared = np.union1d(self.prepared, pairs)
        documents, tokens = np.divmod(pairs, width)
        options = self.option_rows[documents]
        matches = (options >= 0) & (
            self.option_tokens[np.maximum(options, 0)] == tokens[:, None]
        )
        options = np.unique(options[matches])
        missing = options[self.filler_lengths[options] < 0]
        fillers = [
            self.vocabulary.filler(
                int(self.filler_tokens[option]), int(self.filler_chars[option])
            )
            for option in missing
        ]
        if len(missing):
            lengths = np.array(list(map(len, fillers)), dtype=np.int64)
            self.filler_lengths[missing] = lengths
            rows = np.arange(self.num_filler_rows, self.num_filler_rows + len(missing))
            self.num_filler_rows += len(missing)
            size = max(self.fillers.shape[1], int(lengths.max()))
            capacity = (
                len(self.fillers)
                if self.num_filler_rows <= len(self.fillers)
                else max(self.num_filler_rows, 2 * len(self.fillers))
            )
            if capacity > len(self.fillers) or size > self.fillers.shape[1]:
                wider = self.torch.full(
                    (capacity, size), -1, dtype=self.torch.int64, device=self.device
                )
                wider[: len(self.fillers), : self.fillers.shape[1]] = self.fillers
                self.fillers = wider
                wider_host = np.full((capacity, size), -1, dtype=np.int64)
                wider_host[: len(self.fillers_host), : self.fillers_host.shape[1]] = (
                    self.fillers_host
                )
                self.fillers_host = wider_host
                counts = self.torch.zeros(
                    capacity, dtype=self.torch.int64, device=self.device
                )
                counts[: len(self.num_fillers)] = self.num_fillers
                self.num_fillers = counts
            values = _padded(fillers)
            self.filler_rows[missing] = rows
            self.fillers_host[rows, : values.shape[1]] = values
            selected = self.tensor(rows)
            self.filler_rows_device[self.tensor(missing)] = selected
            self.fillers[selected, : values.shape[1]] = self.tensor(values)
            self.num_fillers[selected] = self.tensor(lengths)

    def packed_positions(self, streams, alive, documents):
        """Pack surviving document positions into 64-bit words for the host copy."""
        torch = self.torch
        positions = (
            (streams.clamp_min(0) // self.num_options)
            - self.position_offsets_device[documents, None]
        ).clamp_min(0)
        mask = torch.zeros(
            (len(alive), self.mask_words * 64), dtype=torch.int64, device=self.device
        )
        mask.scatter_reduce_(1, positions, alive.to(torch.int64), reduce="amax")
        return (mask.reshape(len(alive), self.mask_words, 64) << self.mask_bits).sum(
            dim=2
        )

    def prepare(self, documents, tokens, masks, written, drafts, sources):
        """Prepare only continuations that can verify the next copy draft.

        A row after a rejected draft is never accepted. Later verification rows
        therefore need only positions matching the draft's earlier tokens.
        """
        if not len(documents):
            return
        self._prepare_fillers(documents, tokens)
        source_positions, source_options = np.divmod(sources, self.num_options)
        self._extend(
            source_positions,
            np.maximum(0, written - 1 + drafts - self.filler_lengths[source_options]),
        )
        remaining = (
            1
            + self.filler_lengths[source_options]
            + self.position_lengths[source_positions]
            - written
        )
        drafts = np.minimum(drafts, np.maximum(0, remaining))
        mask = np.unpackbits(
            np.ascontiguousarray(masks).view(np.uint8), axis=1, bitorder="little"
        )[:, : self.max_positions]
        answers, columns = np.nonzero(mask)
        positions = self.position_offsets[documents[answers]] + columns
        lookup = (
            self.position_keys[positions] * len(self.vocabulary.vocab) + tokens[answers]
        )
        options = self.option_order[np.searchsorted(self.option_lookup, lookup)]
        for step in range(int(drafts.max(initial=0)) + 1):
            at = written[answers] + step
            valid = (step <= drafts[answers]) & (at < self.spec.max_answer_tokens)
            positions, options, answers, at = (
                positions[valid],
                options[valid],
                answers[valid],
                at[valid],
            )
            if not len(positions):
                break
            self._extend(positions, np.maximum(0, at - self.filler_lengths[options]))
            expected = self._tokens(
                source_positions[answers], source_options[answers], at
            )
            following = self._tokens(positions, options, at)
            keep = (following >= 0) & (following == expected)
            positions, options, answers = positions[keep], options[keep], answers[keep]
        self._upload()

    def _tokens(self, positions, options, at):
        sizes = self.filler_lengths[options]
        following = at - sizes - 1
        rows = np.maximum(self.position_rows[positions], 0)
        continuation = self.continuations_host[
            rows, np.clip(following, 0, self.continuations_host.shape[1] - 1)
        ]
        filler = self.fillers_host[
            np.maximum(self.filler_rows[options], 0),
            np.clip(at - 1, 0, self.fillers_host.shape[1] - 1),
        ]
        return np.where(
            at == 0,
            self.option_tokens[options],
            np.where(at <= sizes, filler, continuation),
        )

    def _extend(self, positions, at):
        positions, inverse = np.unique(positions, return_inverse=True)
        needed = np.zeros(len(positions), dtype=np.int64)
        np.maximum.at(needed, inverse, at)
        pending = (self.position_rows[positions] < 0) | (
            (self.position_lengths[positions] < needed) & ~self.exhausted[positions]
        )
        positions, needed = positions[pending], needed[pending]
        if not len(positions):
            return
        streams = []
        for position in positions:
            row = self.position_rows[position]
            if row < 0:
                document = self.documents.documents[self.position_documents[position]]
                scan = document.scan
                if scan.normalized is None:
                    scan.normalized = SqueezedText(document.text)
                row = len(self.positions)
                self.position_rows[position] = row
                self.positions.append(
                    Stream(
                        int(self.characters[position]),
                        [int(self.position_tokens[position])],
                        [int(self.position_ends[position])],
                        document.text,
                        normalized=scan.normalized,
                        first_window_chars=self.spec.first_window_chars,
                        window_chars=self.spec.window_chars,
                    )
                )
            streams.append(self.positions[row])
        ensure_tokens(streams, needed, self.vocabulary)
        rows = self.position_rows[positions]
        self.num_positions = len(self.positions)
        num_tokens = self.spec.max_answer_tokens + self.spec.draft_tokens
        if self.num_positions > len(self.ends):
            capacity = max(self.num_positions, 2 * len(self.ends))
            ends = np.full((capacity, self.ends.shape[1]), -1, dtype=np.int64)
            ends[: len(self.ends)] = self.ends
            self.ends = ends
            wider = np.full((capacity, num_tokens), -1, dtype=np.int32)
            wider[: len(self.continuations_host)] = self.continuations_host
            self.continuations_host = wider
        tokens = _padded([stream.stream[1:num_tokens] for stream in streams], np.int32)
        ends = _padded(
            [stream.ends[: self.spec.max_answer_tokens] for stream in streams]
        )
        self.position_lengths[positions] = np.array(
            [min(num_tokens, len(s.stream)) - 1 for s in streams]
        )
        self.exhausted[positions] = [stream.exhausted for stream in streams]
        self.ends[rows, : ends.shape[1]] = ends
        self.continuations_host[rows, : tokens.shape[1]] = tokens
        self.changed[positions] = True

    def _upload(self):
        positions = np.flatnonzero(self.changed)
        if not len(positions):
            return
        rows = self.position_rows[positions]
        if len(self.continuations_host) > len(self.continuations):
            wider = self.torch.full(
                self.continuations_host.shape,
                -1,
                dtype=self.torch.int32,
                device=self.device,
            )
            wider[: len(self.continuations)] = self.continuations
            self.continuations = wider
        self.position_rows_device[self.tensor(positions)] = self.tensor(rows)
        self.continuations[self.tensor(rows)] = self.tensor(
            self.continuations_host[rows]
        )
        self.changed[positions] = False

    def tensor(self, value):
        return self.torch.as_tensor(value).to(self.device, non_blocking=True)

    def tokens_at(self, streams, at):
        """Gather a stream's token at each answer index, or -1 past its end."""
        torch = self.torch
        valid = streams >= 0
        streams = streams.clamp_min(0)
        position, option = streams // self.num_options, streams % self.num_options
        filler_row = self.filler_rows_device[option].clamp_min(0)
        size = self.num_fillers[filler_row]
        following = at - size - 1
        continuation = self.continuations[
            self.position_rows_device[position].clamp_min(0),
            following.clamp(0, self.continuations.shape[1] - 1),
        ]
        continuation = torch.where(
            following < self.continuations.shape[1], continuation, -1
        )
        filler = self.fillers[filler_row, (at - 1).clamp(0, self.fillers.shape[1] - 1)]
        return torch.where(
            valid,
            torch.where(
                at == 0,
                self.first_tokens[option],
                torch.where(at <= size, filler, continuation),
            ),
            -1,
        )

    def choose(self, logits, streams, alive, at, documents, first=True):
        """Choose allowed tokens, stopping before a token leaves the document."""
        torch = self.torch
        logits = logits.float()
        logits = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        top_values, top_ids = torch.topk(logits, self.spec.top_tokens, dim=1)
        stop = torch.logsumexp(logits.index_select(1, self.stop_ids), dim=1)
        return self._choose(
            top_values, top_ids, stop, streams, alive, at, documents, first)

    def choose_readout(self, values, streams, alive, at, documents, first=True):
        """Apply the token rule to an already projected top-k readout."""
        k = self.spec.top_tokens
        return self._choose(values[:, 1 + k:], values[:, 1:1 + k].to(self.torch.int64),
                            values[:, 0], streams, alive, at, documents, first)

    def _choose(self, top_values, top_ids, stop, streams, alive, at, documents, first):
        torch = self.torch
        sorted_ids, order = top_ids.sort(dim=1)
        sorted_values = top_values.gather(1, order)

        def scores(tokens):
            columns = torch.searchsorted(sorted_ids, tokens.contiguous())
            columns = columns.clamp_max(sorted_ids.shape[1] - 1)
            return torch.where(sorted_ids.gather(1, columns) == tokens,
                               sorted_values.gather(1, columns), -float("inf"))

        top = top_ids[:, 0]
        possible = alive
        if first:
            options = self.option_rows_device[documents]
            option = options.clamp_min(0)
            key = self.option_key_ids[option]
            values = scores(self.first_tokens[option])
            values = torch.where(options >= 0, values, -float("inf"))
            if self.trace is not None:
                first_choices = self.first_tokens[option], values, options >= 0
            maxima = torch.full(
                (len(at), self.num_keys), -float("inf"), device=self.device
            )
            maxima.scatter_reduce_(1, key, values, reduce="amax", include_self=True)
            ranked = torch.where(
                values == maxima.gather(1, key), option, self.num_options
            )
            best_option = torch.full(
                (len(at), self.num_keys),
                self.num_options,
                dtype=torch.int64,
                device=self.device,
            )
            best_option.scatter_reduce_(
                1, key, ranked, reduce="amin", include_self=True
            )
            best_option.clamp_max_(self.num_options - 1)
            first_key = maxima.argmax(dim=1)
            first_token = self.first_tokens[
                best_option.gather(1, first_key[:, None])
            ].squeeze(1)
            position = streams.clamp_min(0) // self.num_options
            keys = self.key_ids[position]
            initial = position * self.num_options + best_option.gather(1, keys)
            initial = torch.where(streams >= 0, initial, -1)
            streams.copy_(torch.where(at[:, None] == 0, initial, streams))
            possible = alive & ((at[:, None] > 0) | maxima.gather(1, keys).isfinite())
        tokens = self.tokens_at(streams, at[:, None])
        values = scores(tokens.clamp_min(0))
        allowed = possible & (tokens >= 0)
        values = torch.where(allowed, values, -float("inf"))
        best = values.max(dim=1).values
        following = (
            torch.where(values == best[:, None], tokens, len(self.vocabulary.vocab))
            .min(dim=1)
            .values
        )
        chosen = torch.where(at == 0, first_token, following) if first else following
        allows = (allowed & (tokens == top[:, None])).any(dim=1)
        leaves = (
            (top_values[:, 0] > torch.maximum(best, stop))
            & ~allows
            & ~self.is_stop[top]
        )
        ended = leaves | torch.where(at == 0, stop > best, stop >= best)
        ended |= (at > 0) & ~allowed.any(dim=1)
        keep = allowed & (tokens == chosen[:, None])
        if self.trace is not None:
            runner, runner_lp = self._runner(tokens, values, allowed, chosen)
            if first:
                initial, initial_lp = self._runner(*first_choices, chosen)
                runner = torch.where(at == 0, initial, runner)
                runner_lp = torch.where(at == 0, initial_lp, runner_lp)
            self.trace_values = torch.stack(
                (
                    torch.where(best.isfinite(), chosen, -1).float(),
                    torch.where(runner_lp.isfinite(), runner, -1).float(),
                    best - runner_lp,
                    best,
                    stop,
                    top.float(),
                    top_values[:, 0],
                ),
                dim=1,
            )
        return chosen, ended, keep

    def _runner(self, tokens, values, allowed, chosen):
        torch = self.torch
        others = allowed & (tokens != chosen[:, None])
        score = torch.where(others, values, -float("inf")).max(dim=1).values
        runner = (
            torch.where(
                others & (values == score[:, None]), tokens, len(self.vocabulary.vocab)
            )
            .min(dim=1)
            .values
        )
        return runner, score

    def source(self, streams, alive):
        """Return the first surviving position in document order."""
        chars = self.characters_device[(streams.clamp_min(0) // self.num_options)]
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
            position, option = divmod(source, self.num_options)
            start = int(self.characters[position])
            index = max(0, length - 1 - max(0, self.filler_lengths[option]))
            end = int(
                self.position_ends[position]
                if index == 0
                else self.ends[self.position_rows[position], index]
            )
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

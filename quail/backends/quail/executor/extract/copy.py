"""Document continuations and the GPU constraint for copy decoding."""

from types import SimpleNamespace

import numpy as np

from quail.backends.quail.executor.extract.candidates import (
    TextWindows,
)
from quail.backends.quail.executor.extract.trace import TRACE_WIDTH, trace_path


def verification_blocks(counts, stream_width, num_streams, num_options, num_keys,
                        top_tokens, mask_words, workspace_bytes):
    """Yield row indices, position block sizes, and workspace byte bounds.

    The bound excludes resident inputs and the caller's final output. It includes
    compaction, first-choice options, block outputs, and simultaneous token,
    index, score and trace intermediates. Rows too wide for one position fail
    before allocating device temporaries.
    """
    widths = np.broadcast_to(stream_width, counts.shape)
    num_streams = np.maximum(1, np.broadcast_to(num_streams, counts.shape))
    order = np.lexsort((counts, num_streams, widths))
    # Lookup's live int64 gathers dominate; score/trace passes retain its results.
    options = 64 * num_options + 32 * num_keys
    per_position = 512 + 96 * top_tokens + 16 * mask_words
    # CUDA rounds each allocation to 512 bytes; fewer than 128 are live here.
    padding = 128 * 512
    start = 0
    while start < len(order):
        row = order[start]
        num_steps = int(counts[row])
        width, num_columns = int(widths[row]), int(num_streams[row])
        base = padding + 64 * width + options + num_steps * per_position
        steps = min(num_steps, (workspace_bytes - base) // (192 * num_columns))
        if steps < 1:
            raise ValueError("one copy verification position exceeds its workspace")
        end = start + 1
        peak = base + steps * 192 * num_columns
        if steps == num_steps:
            while end < len(order):
                row = order[end]
                num_next_steps = max(num_steps, int(counts[row]))
                next_width = max(width, int(widths[row]))
                num_next_columns = max(num_columns, int(num_streams[row]))
                per_row = (64 * next_width + options
                           + num_next_steps * (per_position + 192 * num_next_columns))
                if padding + (end + 1 - start) * per_row > workspace_bytes:
                    break
                end += 1
                steps = num_steps = num_next_steps
                width, num_columns = next_width, num_next_columns
                peak = padding + (end - start) * per_row
        yield order[start:end], steps, peak
        start = end


def _padded(values, dtype=np.int64):
    width = max(1, max(map(len, values), default=0))
    out = np.full((len(values), width), -1, dtype=dtype)
    for i, row in enumerate(values):
        out[i, : len(row)] = row
    return out


# A stream stores position * STREAM_STRIDE + option.
STREAM_STRIDE = 1 << 24


def _flat_ranges(starts, counts):
    """Return the ranges start to start + count, concatenated."""
    return (np.repeat(starts - (np.cumsum(counts) - counts), counts)
            + np.arange(int(counts.sum())))


def _concatenated(arrays):
    return np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.int64)


def copy_tables(documents) -> SimpleNamespace:
    """Return the host arrays a CopyConstraint reads, for all documents at once.

    A key is one distinct word key id within one document; keys are
    numbered by document, then by key id. A key's anchor is the
    document's token at the key's first position. Options are stored once
    per distinct anchor token: the vocabulary's spellings of the anchor's
    word, each followed by the rest of the anchor. A key reads its
    anchor's options.

    Args:
        documents: The extraction inputs, with each document's token scan.

    Returns:
        The document positions and their keys, each document's range of
        keys, each key's range of options, the options, and the lookup
        from (anchor, token) to option, as numpy arrays.
    """
    vocabulary = documents.vocabulary
    scans = [document.scan for document in documents.documents]
    lengths = np.array([len(scan.tokens) for scan in scans], dtype=np.int64)
    num_documents, num_positions = len(scans), int(lengths.sum())
    position_starts = np.r_[0, np.cumsum(lengths)]
    zero = np.array([0])
    if not num_positions:
        empty = np.zeros(num_documents, dtype=np.int64)
        return SimpleNamespace(
            num_keys=1, max_options=1, characters=zero, position_ends=zero,
            position_documents=zero, position_offsets=position_starts,
            max_positions=1, position_keys=zero, position_local_keys=zero,
            document_key_starts=empty, document_key_ends=empty, key_anchors=zero,
            key_option_starts=zero, key_option_counts=zero, option_tokens=zero,
            filler_tokens=zero, filler_chars=zero, option_order=zero,
            option_lookup=zero)
    kids = _concatenated([scan.kids for scan in scans]).astype(np.int64)
    tokens = _concatenated([scan.tokens for scan in scans]).astype(np.int64)
    position_documents = np.repeat(np.arange(num_documents), lengths)
    width = int(kids.max()) + 1
    keys, first, position_keys = np.unique(
        position_documents * width + kids, return_index=True, return_inverse=True)
    key_documents = keys // width
    key_starts = np.searchsorted(key_documents, np.arange(num_documents + 1))
    anchors, key_anchors = np.unique(tokens[first], return_inverse=True)
    anchor_kids = vocabulary.key_of[anchors]
    offsets, table_tokens, table_lengths = vocabulary.option_table(anchor_kids)
    anchor_counts = offsets[anchor_kids + 1] - offsets[anchor_kids]
    indices = _flat_ranges(offsets[anchor_kids], anchor_counts)
    option_tokens = table_tokens[indices]
    lookup = (np.repeat(np.arange(len(anchors)), anchor_counts)
              * len(vocabulary.vocab) + option_tokens)
    order = np.argsort(lookup)
    key_option_counts = anchor_counts[key_anchors]
    document_options = np.bincount(
        key_documents, weights=key_option_counts, minlength=num_documents)
    return SimpleNamespace(
        num_keys=max(1, int(np.diff(key_starts).max(initial=0))),
        max_options=max(1, int(document_options.max(initial=0))),
        characters=_concatenated([scan.starts for scan in scans]),
        position_ends=_concatenated([scan.ends for scan in scans]),
        position_documents=position_documents,
        position_offsets=position_starts,
        max_positions=max(1, int(lengths.max(initial=0))),
        position_keys=position_keys,
        position_local_keys=position_keys - key_starts[position_documents],
        document_key_starts=key_starts[:-1],
        document_key_ends=key_starts[1:],
        key_anchors=key_anchors,
        key_option_starts=(np.cumsum(anchor_counts) - anchor_counts)[key_anchors],
        key_option_counts=key_option_counts,
        option_tokens=option_tokens,
        filler_tokens=np.repeat(anchors, anchor_counts),
        filler_chars=table_lengths[indices],
        option_order=order,
        option_lookup=lookup[order],
    )


class CopyConstraint:
    """Tokenized document positions shared by a pipeline's extraction fields.

    Continuations are prepared after the first token is chosen, only for starts
    it can reach. Fields and spellings share each prepared continuation.
    """

    def __init__(self, torch, device, documents, spec):
        self.torch, self.device, self.spec = torch, torch.device(device), spec
        self.documents = documents
        self.trace = trace_path()
        self.trace_width = TRACE_WIDTH if self.trace is not None else 0
        vocabulary = self.vocabulary = documents.vocabulary
        tables = copy_tables(documents)
        self.num_keys = tables.num_keys
        self.max_options = tables.max_options
        self.characters = tables.characters
        self.position_ends = tables.position_ends
        self.position_documents = tables.position_documents
        self.option_tokens = tables.option_tokens
        self.filler_tokens = tables.filler_tokens
        self.filler_chars = tables.filler_chars
        self.position_keys = tables.position_keys
        self.document_key_starts = tables.document_key_starts
        self.document_key_ends = tables.document_key_ends
        self.key_anchors = tables.key_anchors
        self.option_order = tables.option_order
        self.option_lookup = tables.option_lookup
        self.num_options = len(self.option_tokens)
        if self.num_options >= STREAM_STRIDE:
            raise ValueError("too many copy options for one stream stride")
        self.filler_lengths = np.full(self.num_options, -1, dtype=np.int64)
        self.position_lengths = np.zeros(len(self.characters), dtype=np.int64)
        self.position_rows = np.full(len(self.characters), -1, dtype=np.int64)
        self.position_offsets = tables.position_offsets
        self.max_positions = tables.max_positions
        self.mask_words = -(-self.max_positions // 64)
        self.position_offsets_device = self.tensor(self.position_offsets)
        self.windows = None
        self.window_ends = self.position_ends.copy()
        self.window_started = np.zeros(len(self.characters), dtype=bool)
        self.exhausted = np.zeros(len(self.characters), dtype=bool)
        self.prepared = np.empty(0, dtype=np.int64)
        self.num_positions = 0
        starts = self.position_offsets_device[:-1, None]
        columns = torch.arange(self.max_positions, device=self.device)
        self.stream_ids = torch.where(
            columns < self.position_offsets_device[1:, None] - starts,
            (starts + columns) * STREAM_STRIDE, -1)
        self.document_key_starts_device = self.tensor(self.document_key_starts)
        self.document_key_ends_device = self.tensor(self.document_key_ends)
        self.key_option_starts = self.tensor(tables.key_option_starts)
        self.key_option_counts = self.tensor(tables.key_option_counts)
        self.key_ids = self.tensor(tables.position_local_keys)
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
        self.stop_ids = self.tensor(vocabulary.tokenizer.stop_ids)

    def _prepare_fillers(self, documents, tokens):
        width = len(self.vocabulary.vocab)
        pairs = np.setdiff1d(np.unique(documents * width + tokens), self.prepared)
        if not len(pairs):
            return
        self.prepared = np.union1d(self.prepared, pairs)
        documents, tokens = np.divmod(pairs, width)
        counts = (self.document_key_ends[documents]
                  - self.document_key_starts[documents])
        keys = _flat_ranges(self.document_key_starts[documents], counts)
        lookup = (self.key_anchors[keys] * width
                  + np.repeat(tokens, counts))
        found = np.minimum(np.searchsorted(self.option_lookup, lookup),
                           len(self.option_lookup) - 1)
        options = np.unique(
            self.option_order[found[self.option_lookup[found] == lookup]])
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
            (streams.clamp_min(0) // STREAM_STRIDE)
            - self.position_offsets_device[documents, None]
        ).clamp_min(0)
        mask = torch.zeros(
            (len(alive), self.mask_words), dtype=torch.int64, device=self.device
        )
        # Each surviving stream has a distinct document position.
        mask.scatter_add_(1, positions // 64, alive.to(torch.int64) << (positions % 64))
        return mask

    def prepare(self, documents, tokens, masks, written, drafts, sources):
        """Prepare only continuations that can verify the next copy draft.

        A row after a rejected draft is never accepted. Later verification rows
        therefore need only positions matching the draft's earlier tokens.
        """
        if not len(documents):
            return
        self._prepare_fillers(documents, tokens)
        source_positions, source_options = np.divmod(sources, STREAM_STRIDE)
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
        answers, positions = self.live_positions(documents, masks)
        lookup = (self.key_anchors[self.position_keys[positions]]
                  * len(self.vocabulary.vocab) + tokens[answers])
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

    def live_positions(self, documents, masks):
        """Expand only nonempty mask words into answer and document positions."""
        answers, words = np.nonzero(masks)
        bits = np.arange(64, dtype=np.uint64)
        member, columns = np.nonzero(
            (masks[answers, words].astype(np.uint64)[:, None] >> bits) & 1)
        positions = words[member] * 64 + columns
        answers = answers[member]
        return answers, self.position_offsets[documents[answers]] + positions

    def prepare_ahead(self, documents, masks, written):
        """Extend live starts far enough for the next pass before its result lands."""
        answers, positions = self.live_positions(documents, masks)
        limits = np.minimum(
            written + 2 * self.spec.draft_tokens + 1,
            self.spec.max_answer_tokens + self.spec.draft_tokens - 1)
        self._extend(positions, limits[answers])
        self._upload()
        return limits

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
        missing = positions[self.position_rows[positions] < 0]
        self.position_rows[missing] = np.arange(
            self.num_positions, self.num_positions + len(missing))
        self.num_positions += len(missing)
        rows = self.position_rows[positions]
        num_tokens = self.spec.max_answer_tokens + self.spec.draft_tokens
        if self.num_positions > len(self.ends):
            capacity = max(self.num_positions, 2 * len(self.ends))
            ends = np.full((capacity, self.ends.shape[1]), -1, dtype=np.int64)
            ends[: len(self.ends)] = self.ends
            self.ends = ends
            wider = np.full((capacity, num_tokens), -1, dtype=np.int32)
            wider[: len(self.continuations_host)] = self.continuations_host
            self.continuations_host = wider
        self.ends[rows, 0] = self.position_ends[positions]
        self.changed[positions] = True
        while True:
            pending = ((self.position_lengths[positions] < needed)
                       & ~self.exhausted[positions])
            positions, needed = positions[pending], needed[pending]
            if not len(positions):
                break
            self._extend_windows(positions)

    def prepare_documents(self):
        """Index normalized text while the documents' prefill is in flight."""
        if self.windows is None:
            self.windows = TextWindows([
                document.text for document in self.documents.documents])

    def _extend_windows(self, positions):
        self.prepare_documents()
        documents = self.position_documents[positions]
        sizes = np.where(self.window_started[positions], self.spec.window_chars,
                         self.spec.first_window_chars)
        text, left, cuts = self.windows.windows(
            documents, self.window_ends[positions], sizes)
        self.window_started[positions] = True
        self.window_ends[positions] = cuts
        tokens, offsets, boundaries = self.vocabulary.encode_packed(text)
        counts = np.diff(boundaries)
        owner = np.repeat(np.arange(len(positions)), counts)
        columns = (np.arange(len(tokens)) - np.repeat(boundaries[:-1], counts)
                   + self.position_lengths[positions[owner]])
        rows = self.position_rows[positions[owner]]
        valid = columns < self.continuations_host.shape[1] - 1
        self.continuations_host[rows[valid], columns[valid]] = tokens[valid]
        valid = columns + 1 < self.ends.shape[1]
        end = np.minimum(self.windows.ends[left[owner] + offsets[:, 1] - 1]
                         - self.windows.offsets[documents[owner]], cuts[owner])
        self.ends[rows[valid], columns[valid] + 1] = end[valid]
        self.position_lengths[positions] = np.minimum(
            self.position_lengths[positions] + counts,
            self.continuations_host.shape[1] - 1)
        self.exhausted[positions] = (
            (cuts + self.windows.offsets[documents] == self.windows.limits[documents])
            | (counts == 0)
            | (self.position_lengths[positions]
               == self.continuations_host.shape[1] - 1))

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
        host = self.torch.as_tensor(value)
        if self.device.type == "cuda":
            host = host.pin_memory()
        return host.to(self.device, non_blocking=True)

    def tokens_at(self, streams, at):
        """Gather a stream's token at each answer index, or -1 past its end."""
        torch = self.torch
        valid = streams >= 0
        streams = streams.clamp_min(0)
        position, option = streams // STREAM_STRIDE, streams % STREAM_STRIDE
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

    def _document_options(self, documents):
        """Return each row's document options, padded with -1, and their keys.

        A key is the option's key index within its document.
        """
        torch = self.torch
        keys = (self.document_key_starts_device[documents, None]
                + torch.arange(self.num_keys, device=self.device))
        inside = keys < self.document_key_ends_device[documents, None]
        keys = torch.where(inside, keys, 0)
        counts = torch.where(inside, self.key_option_counts[keys], 0)
        ends = counts.cumsum(dim=1)
        columns = torch.arange(self.max_options, device=self.device).expand(
            len(documents), -1).contiguous()
        key = torch.searchsorted(ends, columns, right=True).clamp_max(
            self.num_keys - 1)
        options = (self.key_option_starts[keys.gather(1, key)] + columns
                   - (ends - counts).gather(1, key))
        return torch.where(columns < ends[:, -1:], options, -1), key

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
            options, key = self._document_options(documents)
            option = options.clamp_min(0)
            values = scores(self.first_tokens[option])
            values = torch.where(options >= 0, values, -float("inf"))
            if self.trace is not None:
                first_choices = self.first_tokens[option], values, options >= 0
            maxima = torch.full(
                (len(at), self.num_keys), -float("inf"), device=self.device
            )
            maxima.scatter_reduce_(1, key, values, reduce="amax", include_self=True)
            ranked = torch.where(
                (values == maxima.gather(1, key)) & (options >= 0),
                option, self.num_options)
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
            position = streams.clamp_min(0) // STREAM_STRIDE
            keys = self.key_ids[position]
            initial = position * STREAM_STRIDE + best_option.gather(1, keys)
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
        # No finite candidate ends the answer; dead padding cannot choose its token.
        following = torch.where(best.isfinite(), following, -1)
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

    def compact(self, streams, alive, num_streams):
        """Keep live columns first, preserving their original order."""
        torch = self.torch
        columns = torch.arange(streams.shape[1], device=self.device)
        ranks = alive.to(torch.int64).cumsum(dim=1)
        ranks = torch.where(alive, ranks - 1, ranks[:, -1:] + columns - ranks)
        order = torch.empty_like(ranks)
        order.scatter_(1, ranks, columns.expand_as(ranks))
        columns = order[:, :num_streams]
        return streams.gather(1, columns), alive.gather(1, columns)

    def verify(self, projected, streams, alive, source, at, documents, limits,
               valid, first, active):
        """Verify every draft position, retaining the first rejected choice."""
        torch = self.torch
        chosen, ended, keep = self.choose_readout(
            projected[:, 0], streams, alive, at[:, 0], documents, first=first)
        use = active & valid[:, 0]
        alive = torch.where((use & ~ended)[:, None], keep, alive)
        following = self.source(streams, alive)
        hit = chosen == self.tokens_at(source, at[:, 0])
        values = torch.stack((torch.where(ended, -1, chosen), following,
                              alive.sum(dim=1), hit.to(torch.int64)), dim=1)[:, None]
        values = torch.where(use[:, None, None], values, -2)
        traces = (self.trace_values[:, None].view(torch.int32).to(torch.int64)
                  if self.trace is not None else None)
        active = use & ~ended & hit & (at[:, 0] + 1 < limits)
        if at.shape[1] == 1:
            return values, alive, following, traces, active
        # The first choice can change a tied source after stream compaction.
        tail, alive, source, tail_traces, active = self._verify_tail(
            projected[:, 1:], streams, alive, following, at[:, 1:], documents,
            limits, valid[:, 1:], active)
        if traces is not None:
            traces = torch.cat((traces, tail_traces), dim=1)
        return torch.cat((values, tail), dim=1), alive, source, traces, active

    def _verify_tail(self, projected, streams, alive, source, at, documents,
                     limits, valid, active):
        torch = self.torch
        num_rows, num_steps = at.shape
        expanded = streams[:, None, :].expand(-1, num_steps, -1)
        tokens = self.tokens_at(expanded, at[:, :, None])
        drafted = self.tokens_at(source[:, None], at)
        # A hit retains the earliest source, so it stays fixed until rejection.
        matches = tokens == drafted[:, :, None]
        prefixes = matches.to(torch.int32).cumprod(dim=1).bool()
        before = torch.cat((alive[:, None], alive[:, None] & prefixes[:, :-1]), dim=1)
        flat_streams = expanded.reshape(num_rows * num_steps, -1)
        flat_at = at.flatten()
        flat_documents = documents[:, None].expand(-1, num_steps).flatten()
        chosen, ended, keep = self.choose_readout(
            projected.flatten(0, 1), flat_streams, before.flatten(0, 1), flat_at,
            flat_documents, first=False)
        chosen, ended = chosen.view_as(at), ended.view_as(at)
        keep = keep.view_as(before)
        hit = chosen == drafted
        continues = valid & ~ended & hit & (at + 1 < limits[:, None])
        prefixes = continues[:, :-1].to(torch.int32).cumprod(dim=1).bool()
        use = torch.cat((active[:, None], active[:, None] & prefixes), dim=1)
        use &= valid
        after = torch.where(ended[:, :, None], before, keep)
        # Rejected and terminal choices update the state once; later rows do not.
        last = (use.sum(dim=1) - 1).clamp_min(0)
        rows = torch.arange(num_rows, device=self.device)
        final = torch.where(use[:, :1], after[rows, last], alive)
        following = self.source(flat_streams, after.flatten(0, 1)).view_as(at)
        values = torch.stack((torch.where(ended, -1, chosen), following,
                              after.sum(dim=2), hit.to(torch.int64)), dim=2)
        values = torch.where(use[:, :, None], values, -2)
        traces = None
        if self.trace is not None:
            # The sequential trace also records unused rows against frozen state.
            traced = torch.where(use[:, :, None], before, final[:, None])
            self.choose_readout(projected.flatten(0, 1), flat_streams,
                                traced.flatten(0, 1), flat_at, flat_documents,
                                first=False)
            traces = self.trace_values.view(num_rows, num_steps, -1)
            traces = traces.view(torch.int32).to(torch.int64)
        active = use[:, -1] & continues[:, -1]
        return values, final, self.source(streams, final), traces, active

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
        chars = self.characters_device[(streams.clamp_min(0) // STREAM_STRIDE)]
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
            position, option = divmod(source, STREAM_STRIDE)
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

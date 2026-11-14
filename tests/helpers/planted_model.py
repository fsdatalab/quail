"""The fake model of the end-to-end tests: a forward pass that answers from truth.

``load`` binds a production QuailModelExecution to a CPU arena and to a
pipeline with no weights; ``PlantedModel.forward_chunk`` is the
pipeline's forward pass for one region. The production operators, GPU
loop, chunk packer, decoder, and label readout run as they do on a
GPU.

The forward pass reads each packed chunk as a model does: it follows
every read row's tokens back through the KV the row reads, finds the
operator whose prompt those tokens are and the documents in it, and
returns the row's next-token scores from the oracle's truth
(helpers/oracle.py). The head is the identity, so a row's hidden state
is its scores over the vocabulary.
"""

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from quail.backends.quail import QuailModelExecution
from quail.backends.quail.executor.arena import KVArena
from quail.backends.quail.executor.chunk import InputStaging
from quail.backends.quail.executor.extract.vocabulary import same_word
from quail.backends.quail.executor.sae import SaeEncoder
from quail.backends.quail.prompts import join_request
from quail.physical import (
    AiClassify,
    AiExtract,
    AiFilter,
    AiJoin,
    AiMap,
    AiSaeFeatures,
    AiScore,
)

# the token that ends an answer: the tests' tokenizer has none of its own
STOP = 0
# a token the model does not choose
UNLIKELY = -100.0
# the share of probability a classification gives the planted label
LABEL_SHARE = 0.8
# the arena's pages and tokens per page, and the features a row keeps
PAGES, PAGE_TOKENS, KEPT_FEATURES = 1 << 14, 16, 8


def _alias(name):
    """The scan alias in a document's arena key name.

    A map's branch is keyed (name, map index) and an extraction's own
    documents ("extract", alias).
    """
    while isinstance(name, tuple):
        name = name[1] if name[0] == "extract" else name[0]
    return name


class PlantedModel:
    """A forward pass that answers each read row from an oracle's truth.

    Args:
        oracle: The Oracle.
        graph: The region's physical graph.
        gpu: The worker's loaded resources: its arena and vocabulary.
        settings: The plan's settings, with its true and false token ids.
    """

    def __init__(self, oracle, graph, gpu, settings):
        self.oracle = oracle
        self.arena = gpu.arena
        self.vocabulary = gpu.execution.loaded_model.vocabulary
        self.yes = min(settings["true_ids"])
        self.no = min(settings["false_ids"])
        self.pages = {}         # arena page -> the prefix tokens it holds
        self.kv = {}            # arena key -> the tokens its KV holds
        self.prefix_tokens = {}  # arena key -> the tokens before its questions
        self.memo = {}          # (node id, ids) -> the planted answer
        self.called = {}        # node id -> the rows the oracle scored or labeled
        self.writes = []        # (node, document, planted text), as asked
        self.by_tokens = {}     # alias -> document tokens -> document
        # node id -> each node the model answers for
        self.asking = {node.node_id: node for node in graph.nodes if isinstance(
            node, (AiFilter, AiJoin, AiClassify, AiMap, AiExtract, AiSaeFeatures,
                   AiScore))}

    # ---- reading a chunk ------------------------------------------

    def forward_chunk(self, chunk):
        """Return each read row's scores over the vocabulary."""
        ids = chunk.input_ids.tolist()
        positions = chunk.positions.tolist()
        page = self.arena.page_tokens
        # a document may borrow pages a document later in the chunk writes
        for key, row0, suffix0, _ in chunk.row_spans:
            if suffix0 > row0 and self.arena.is_resident(key):
                owned = self.arena.owned_pages(key)
                for row in range(row0, suffix0):
                    held = self.pages.setdefault(owned[positions[row] // page],
                                                 [None] * page)
                    held[positions[row] % page] = ids[row]
        prompts = {}    # read row -> (its arena key, the tokens up to it)
        for span in chunk.row_spans:
            self._follow(ids, positions, *span, prompts)
        decode = chunk.decode_rows
        if decode is not None:
            # an answer's rows continue the tokens its KV holds
            first = len(ids) - len(decode)
            bounds = decode.sequence_offsets.tolist()
            for begin, end in zip(bounds, bounds[1:]):
                key = self.arena.answer_keys[int(decode.slots[begin])]
                held = self.kv[key][:int(decode.positions[begin])]
                for row in range(first + begin, first + end):
                    held = [*held, ids[row]]
                    prompts[row] = (key, held)
                self.kv[key] = held
        finals = chunk.final_indices.tolist()
        out = torch.full((len(finals), len(self.vocabulary.vocab)), UNLIKELY)
        for index, row in enumerate(finals):
            self._answer(out[index], *prompts[row])
        return out

    def _follow(self, ids, positions, key, row0, suffix0, end, prompts):
        """Follow one group's rows through the KV they read.

        Args:
            ids: The chunk's token ids.
            positions: Each row's position in its sequence.
            key: The group's arena key.
            row0: The group's first row, of its prefix.
            suffix0: Its first row after the prefix.
            end: The row after its last.
            prompts: Row -> (key, the tokens up to the row), filled here.
        """
        if suffix0 > row0:
            # a prefix follows the pages before it: its own, or borrowed
            start, page = positions[row0], self.arena.page_tokens
            owned = self.arena.owned_pages(key) if start else ()
            borrowed = [self.pages[owned[at // page]][at % page]
                        for at in range(start)]
            self.kv[key] = borrowed + ids[row0:suffix0]
            self.prefix_tokens[key] = len(self.kv[key])
        # a row follows the latest row before it at the position before
        # its own: the row before it in its suffix, or its trie parent
        latest = {}
        first = True
        for row in range(suffix0, end):
            at = positions[row]
            before = latest.get(at - 1)
            held = [*(self.kv[key][:at] if before is None else before), ids[row]]
            latest[at] = held
            prompts[row] = (key, held)
            last = row + 1 == end or positions[row + 1] != at + 1
            if first and last:
                # the first suffix's tokens are the ones a group may keep
                self.kv[key] = held
            first = first and not last

    # ---- answering a row ------------------------------------------

    def _answer(self, row, key, tokens) -> None:
        """Write the scores of the row whose prompt so far is tokens.

        Raises:
            KeyError: No operator of the region asks the prompt.
        """
        if key[0] == "score":
            node = next(node for node in self.asking.values()
                        if isinstance(node, AiScore) and node.spec.name == key[1])
            return self._score(row, node, tokens)
        alias, document = _alias(key[0]), int(key[1])
        after = tuple(tokens[self.prefix_tokens[key]:])
        frame = False
        for node in self.asking.values():
            if isinstance(node, AiScore) or alias != node.document_aliases[0]:
                continue
            head = self._head(node)
            if after[:len(head)] == head and self._asked(
                    row, node, document, after[len(head):]):
                return None
            frame = frame or head[:len(after)] == after
        if not frame:
            raise KeyError(f"no operator of {alias!r} asks {self._text(after)!r}")
        # a frame's own row: nothing reads it
        return None

    @staticmethod
    def _head(node) -> tuple:
        """The tokens a node's prompt has straight after its document."""
        if isinstance(node, AiFilter):
            return tuple(node.question_token_ids)
        if isinstance(node, AiJoin):
            return tuple(join_request(node)["frame"])
        if isinstance(node, AiClassify) and node.spec.partner is not None:
            note, partner_label = node.spec.join_layout
            return (*note, *partner_label)
        return tuple(node.spec.prompt_token_parts[1])

    def _asked(self, row, node, document, rest) -> bool:
        """Answer a node's prompt about a document; False if rest is not its.

        Args:
            row: The row's scores, written here.
            node: The node whose prompt starts the tokens after the document.
            document: The document.
            rest: The prompt's tokens after ``_head(node)``.
        """
        oracle, spec = self.oracle, getattr(node, "spec", None)
        if isinstance(node, AiFilter):
            if rest:
                return False
            self._bit(row, self._planted(
                node, (document,), lambda: oracle._filter_answer(node, document)))
        elif isinstance(node, AiJoin):
            request = join_request(node)
            partners = self._documents(
                rest, [(alias, request["labels"][alias]) for alias in node.partners],
                tuple(request["tail"]))
            if partners is None:
                return False
            self._bit(row, self._planted(
                node, (document, *partners),
                lambda: oracle._join_answer(node, document, partners)))
        elif isinstance(node, AiClassify) and spec.partner is not None:
            partners = self._documents(rest, [(spec.partner, ())],
                                       tuple(spec.prompt_token_parts[1]))
            if partners is None:
                return False
            self._label(row, node, (document, *partners), ())
        elif isinstance(node, AiClassify):
            self._label(row, node, (document,), rest)
        elif isinstance(node, AiSaeFeatures):
            if rest:
                return False
            # the SAE reads the row itself: each feature's strength
            row[:] = 0.0
            for found in self._planted(node, (document,), lambda:
                                       oracle.feature_list(spec, document)):
                row[int(found["feature"])] = float(found["strength"])
        elif isinstance(node, AiMap):
            text = self._planted(node, (document,), lambda:
                                 oracle.written(spec, document))
            self._next(row, self.vocabulary.tokenizer(text), rest)
        else:
            self._next(row, self._copied(node, document), rest)
        return True

    def _planted(self, node, ids, ask):
        """The oracle's answer about a node's row, asked once."""
        key = (node.node_id, ids)
        if key not in self.memo:
            if isinstance(node, (AiScore, AiClassify)):
                if node.node_id not in self.called:
                    self.called[node.node_id] = []
                    self.oracle.calls.append(
                        (node.spec.name, self.called[node.node_id]))
                self.called[node.node_id].append(list(ids))
            self.memo[key] = ask()
            if isinstance(node, (AiMap, AiExtract)):
                self.writes.append((node, ids[0], self.memo[key]))
        return self.memo[key]

    def _documents(self, tokens, slots, tail):
        """The documents a prompt names after its anchor, or None.

        Args:
            tokens: The prompt's tokens after its frame.
            slots: Per document, its alias and the label before it.
            tail: The tokens after the last document.
        """
        if not slots:
            return () if tuple(tokens) == tail else None
        (alias, label), rest = slots[0], slots[1:]
        if tuple(tokens[:len(label)]) != tuple(label):
            return None
        tokens = tokens[len(label):]
        by_tokens = self.by_tokens.get(alias)
        if by_tokens is None:
            by_tokens = self.by_tokens[alias] = {}
            for document, found in enumerate(self.oracle.documents[alias]):
                by_tokens.setdefault(tuple(int(token) for token in found), document)
        for length in sorted({len(found) for found in by_tokens}, reverse=True):
            document = by_tokens.get(tuple(tokens[:length]))
            if document is None or length > len(tokens):
                continue
            later = self._documents(tokens[length:], rest, tail)
            if later is not None:
                return (document, *later)
        return None

    def _bit(self, row, answer) -> None:
        row[self.yes if answer else self.no] = 0.0

    def _score(self, row, node, tokens) -> None:
        """Write a score's value as the yes token's score.

        Raises:
            KeyError: The score does not ask the prompt.
        """
        spec = node.spec
        parts = [tuple(part) for part in spec.prompt_token_parts]
        ids = None
        if tuple(tokens[:len(parts[0])]) == parts[0]:
            ids = self._documents(
                tokens[len(parts[0]):],
                list(zip(spec.aliases, [(), *parts[1:-1]])), parts[-1])
        if ids is None:
            raise KeyError(f"{spec.name!r} does not ask {self._text(tokens)!r}")
        row[self.yes] = self._planted(node, ids, lambda:
                                      self.oracle.score(spec, ids))

    def _label(self, row, node, ids, path) -> None:
        """Write the planted label's next token after the path so far.

        The label's token has LABEL_SHARE of the probability and the
        other labels' tokens share the rest.

        Raises:
            ValueError: The planted label is not one of the labels.
        """
        spec = node.spec
        label = self._planted(node, ids, lambda: self.oracle.label(spec, ids))
        if label not in spec.labels:
            # the executor's readout always chooses one of the labels
            raise ValueError(f"{spec.name!r} has no label {label!r}; the model "
                             f"answers one of {list(spec.labels)}")
        label_ids = [tuple(found) for found in spec.label_token_ids]
        path = tuple(path)
        wanted = label_ids[spec.labels.index(label)]
        children = {found[len(path)] for found in label_ids
                    if len(found) > len(path) and found[:len(path)] == path}
        if len(wanted) > len(path) and wanted[:len(path)] == path:
            others = children - {wanted[len(path)]}
            row[wanted[len(path)]] = math.log(LABEL_SHARE if others else 1.0)
            for token in others:
                row[token] = math.log((1 - LABEL_SHARE) / len(others))
        elif wanted != path:
            # a path the label does not take: any of its tokens
            for token in children:
                row[token] = -math.log(len(children))

    def _next(self, row, answer, written) -> None:
        """Write the answer's next token, or the stop once it is written."""
        words, fed = self._words(answer), self._words(written)
        on_answer = fed == words[:len(fed)] and len(fed) < len(words)
        row[answer[len(fed)] if on_answer else STOP] = 0.0

    def _text(self, tokens) -> str:
        """The text of token ids, for an error message."""
        return self.vocabulary.tokenizer.decode([int(token) for token in tokens])

    def _words(self, tokens) -> list:
        """Each token's word, as the decoder matches a copied token."""
        vocab = self.vocabulary.vocab
        return [same_word(vocab[int(token)]) or int(token) for token in tokens]

    def _copied(self, node, document) -> list:
        """The tokens of the text the model answers an extraction with.

        The model says the planted text, as its document spells it when
        the document holds it; the executor's decoder decides what of
        it is an answer.
        """
        spec = node.spec
        text = self._planted(node, (document,), lambda:
                             self.oracle.extracted(spec, document))
        if not text:
            return []
        source = self.oracle._texts([spec.alias], [document])[0]
        start = source.find(text)
        # a word after a space is one token with its space
        spaced = start > 0 and source[start - 1] == " "
        return self.vocabulary.tokenizer((" " if spaced else "") + text)


class _HostTensor(torch.Tensor):
    """A host tensor that is its own pinned copy and its own copy on the GPU."""

    def pin_memory(self):
        return self

    def to(self, *args, **options):
        options.pop("non_blocking", None)
        args = tuple(found for found in args if found != "cuda")
        return super().to(*args, **options) if args or options else self


def host_torch():
    """CPU torch for the executor: the host stands for the GPU too."""
    def event(**options):
        return SimpleNamespace(record=lambda: None, query=lambda: True,
                               synchronize=lambda: None,
                               elapsed_time=lambda other: 2.0)

    def on_host(make):
        def made(*args, pin_memory=False, device=None, **options):
            return make(*args, **options).as_subclass(_HostTensor)
        return made

    host = SimpleNamespace(**{name: getattr(torch, name) for name in dir(torch)
                              if not name.startswith("_")})
    host.cuda = SimpleNamespace(
        Event=event, synchronize=lambda: None, max_memory_allocated=lambda: 0,
        reset_peak_memory_stats=lambda: None)
    for name in ("empty", "tensor", "as_tensor", "from_numpy"):
        setattr(host, name, on_host(getattr(torch, name)))
    return host


class HostStaging(InputStaging):
    """Chunk input buffers that stay on the host."""

    def host(self, name, count, dtype):
        self.buffers[name] = self.torch.empty(count, dtype=dtype)
        return self.buffers[name]

    def upload(self, name, count):
        return self.buffers[name][:count]


class StoppingTokenizer:
    """The tests' tokenizer with a token that ends an answer."""

    stop_ids = [STOP]
    eos_token_id = STOP

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __len__(self):
        return len(self.tokenizer)

    def __call__(self, text):
        return self.tokenizer(text)


# loaded resources no worker is using, by what they were loaded for
_IDLE = {}


def load(spec, device, gpu_index, gpus):
    """Bind a production model execution to a CPU arena and an identity head.

    A worker that ends puts what it loaded in ``idle`` for the next one.

    Args:
        spec: The model's spec.
        device: The GPUs' device spec.
        gpu_index: Which of the run's GPUs the worker stands for.
        gpus: How many GPUs the run has.

    Returns:
        The worker's torch, arena, pipeline, and execution.
    """
    from helpers.fakes import fake_pipeline
    from helpers.sessions import encoder

    idle = _IDLE.setdefault((spec.name, device.name, gpu_index, gpus), [])
    try:
        return idle.pop()
    except IndexError:
        pass
    tokenizer = StoppingTokenizer(encoder())
    arena = KVArena(num_layers=1, num_pages=PAGES, page_tokens=PAGE_TOKENS,
                    num_kv_heads=1, d_head=1, dtype=torch.float32, device="cpu")
    pipeline = fake_pipeline()
    execution = QuailModelExecution(SimpleNamespace(
        model=spec, device=device, gpu_index=gpu_index, gpus=gpus))
    head = torch.eye(len(tokenizer))
    execution.bind_loaded_model(
        model=SimpleNamespace(quail_full_head=head), arena=arena,
        pipeline=pipeline, tokenizer=tokenizer)
    loaded = execution.loaded_model
    host = host_torch()
    loaded.input_staging = HostStaging(host)
    loaded.sae = SaeEncoder(host, F, head.to(torch.bfloat16),
                            torch.zeros(len(tokenizer)), KEPT_FEATURES)
    return SimpleNamespace(torch=host, arena=arena, pipeline=pipeline,
                           execution=execution, spec=spec, idle=idle)

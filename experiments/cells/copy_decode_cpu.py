"""Compare batched copy selection with the old host rule on CPU.

Run from the repository root:
    PYTHONPATH=tests:tests/engine:. uv run --no-sync python \
        experiments/cells/copy_decode_cpu.py

The synthetic workload has 41 fields, 45 documents and 7,500 tokens per
contract. Each answer has one matching start and accepts its drafts. It
reports first-token setup and admission, continuing passes, and row packing
with shared contract pages. It does not measure model time, attention, real
CUAD answer lengths, or GPU performance. New-path timings include CPU tensor
execution and the small identity output head. Old-path timings exclude head
projection and use precomputed readouts. The named phases are cumulative
cProfile times; nested phases must not be added together.
"""

import argparse
import cProfile
import pstats
import statistics
import time

import numpy as np
import torch
from test_copy_batching import Dispatches, continuation, copy_batch

from quail.backends.quail.executor.extract.decode import Decoder
from quail.backends.quail.executor.readout import AsyncTopLogprobs


def compare(num_fields, num_documents, document_tokens, num_passes):
    """Time continuing draft passes with the same accepted tokens."""
    torch.set_num_threads(1)
    with cProfile.Profile() as setup:
        decoder, vocabulary = copy_batch(
            num_fields, num_documents, document_tokens, prepare=False)
        assert decoder.rows() is None
    phase_times(setup, "first-token", [
        ("map_decoder.py", name) for name in
        ("configure_copy", "_configure_copy", "start", "submit", "harvest", "rows")
    ] + [("copy.py", "prepare_documents")])
    with cProfile.Profile() as deferred:
        decoder.prepare()
        decoder.rows()
    phase_times(deferred, "deferred", [("copy.py", "prepare"),
                                       ("map_decoder.py", "rows")])
    group = decoder.copy_groups[0]
    spec = group.constraint.spec
    documents = group.constraint.documents.documents
    reference = Decoder(vocabulary, spec.top_tokens, 1 + 2 * spec.top_tokens,
                        num_fields, spec)
    hidden = torch.full((4, len(vocabulary.vocab)), -10.0)
    hidden[0, vocabulary.encode_ids("five")[0]] = 1
    hidden[1:, vocabulary.encode_ids(" years")[0]] = 1
    readout = AsyncTopLogprobs(torch, torch.nn.functional, decoder.head,
                              spec.top_tokens, [vocabulary.stop_ids])
    values = readout.values(hidden).numpy()
    for document in documents:
        for q, answer in enumerate(document.answers):
            reference.choose(document, q, values[0])
            answer.fed = reference.draft(answer, document.rho, q)

    def old_pass():
        for document in documents:
            for q, answer in enumerate(document.answers):
                reference.verify(document, q, values[1:], answer.fed)
                answer.fed = reference.draft(answer, document.rho, q)

    continuation(decoder, vocabulary)
    old_pass()
    current, old = [], []
    for _ in range(num_passes):
        start = time.perf_counter()
        continuation(decoder, vocabulary)
        current.append(time.perf_counter() - start)
        start = time.perf_counter()
        old_pass()
        old.append(time.perf_counter() - start)
    for q in range(num_fields):
        for d, document in enumerate(documents):
            slot = q * num_documents + d
            written = decoder.written[q][slot]
            assert np.array_equal(decoder.tokens[q][slot, :written],
                                  document.answers[q].written)
    with Dispatches() as dispatches:
        num_rows = continuation(decoder, vocabulary)
    print(f"fields={num_fields}, documents={num_documents}, "
          f"tokens/document={document_tokens}, rows/pass={num_rows}")
    print("median CPU milliseconds/pass: "
          f"batched={statistics.median(current)*1000:.2f}, "
          f"old={statistics.median(old)*1000:.2f}")
    print(f"tensor dispatches/pass={sum(dispatches.counts.values())}, "
          f"head GEMMs/pass={dispatches.counts['aten.mm.default']}")
    with cProfile.Profile() as continuing:
        continuation(decoder, vocabulary)
    phase_times(continuing, "continuing", [
        ("map_decoder.py", name) for name in ("rows", "harvest", "_constrain")
    ] + [("copy.py", name) for name in ("prepare", "prepare_ahead", "_extend")])
    shared_layout(decoder, num_documents, document_tokens)


def phase_times(profile, label, names):
    """Print cumulative phase times without merging same-named methods."""
    stats = pstats.Stats(profile)
    for filename, name in names:
        seconds = sum(value[3] for key, value in stats.stats.items()
                      if key[2] == name and key[0].endswith("/" + filename))
        print(f"{label} CPU milliseconds {filename}:{name}={seconds * 1000:.2f}")


def shared_layout(decoder, num_documents, document_tokens):
    """Time row packing with full contract pages shared across fields."""
    decoder.share = True
    width = -(-(document_tokens + 32) // decoder.page_tokens)
    decoder.pages = np.tile(np.arange(width, dtype=np.int32), (len(decoder.map), 1))
    decoder.num_pages[:] = width
    decoder.origin[:] = document_tokens
    decoder.parent[:] = np.arange(len(decoder.map)) % num_documents
    shared = document_tokens // decoder.page_tokens * decoder.page_tokens
    decoder.borrowed[num_documents:] = shared
    decoder.lead[:num_documents] = shared
    elapsed = []
    for _ in range(20):
        start = time.perf_counter()
        decoder.rows()
        elapsed.append(time.perf_counter() - start)
    print("shared contract rows CPU milliseconds="
          f"{statistics.median(elapsed)*1000:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=int, default=41)
    parser.add_argument("--documents", type=int, default=45)
    parser.add_argument("--document-tokens", type=int, default=7500)
    parser.add_argument("--passes", type=int, default=5)
    args = parser.parse_args()
    compare(args.fields, args.documents, args.document_tokens, args.passes)

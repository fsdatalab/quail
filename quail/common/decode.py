"""Spec-only pass time for answer admission."""

from dataclasses import dataclass

import numpy as np

from quail.common.budgets import budgets_of
from quail.common.retention import RetentionPolicy
from quail.common.work import Work, triangle


@dataclass(frozen=True)
class DecodeCost:
    """Price decode passes and prefix recomputation in seconds."""

    recompute_cost: RetentionPolicy
    weight_seconds: float
    kv_seconds: float
    sliding_kv_seconds: float
    state_seconds: float = 0.0

    @classmethod
    def from_specs(cls, model, device):
        """Build coefficients from model shapes and hardware peak rates."""
        budgets = budgets_of(model, device)
        return cls(RetentionPolicy(**budgets.coefficients(True)),
                   *budgets.decode_seconds())

    def row_work(self, contexts, groups=None, shared=None, covered=None):
        """Count per-row work, assigning each shared read to one group member.

        Args:
            contexts: Each row's context tokens.
            groups: Each row's shared-read group, or None.
            shared: Each row's leading tokens its group reads once.
            covered: Each row's leading tokens a row outside contexts
                already reads, which it does not read again; or None.
        """
        contexts = np.asarray(contexts, dtype=float)
        window = self.recompute_cost.sliding_window
        sliding = np.minimum(contexts, window)
        reads, sliding_reads = contexts.copy(), sliding.copy()
        if covered is not None:
            covered = np.minimum(contexts, covered)
            reads -= covered
            sliding_reads -= np.maximum(0, covered - np.maximum(0, contexts - window))
        if groups is not None and len(contexts):
            _, firsts, inverse = np.unique(groups, return_index=True,
                                           return_inverse=True)
            shared = np.minimum(contexts, shared)
            shared_sliding = np.maximum(0, shared - np.maximum(0, contexts - window))
            reads -= shared
            sliding_reads -= shared_sliding
            full = np.zeros(len(firsts))
            capped = np.zeros(len(firsts))
            np.maximum.at(full, inverse, shared)
            np.maximum.at(capped, inverse, shared_sliding)
            reads[firsts] += full
            sliding_reads[firsts] += capped
        # Each answer advances its own state, even when its KV reads are shared.
        state = np.ones(len(contexts)) if self.state_seconds else 0.0
        return Work(tokens=np.ones(len(contexts)), pairs=contexts,
                    kv_read=reads, sliding_pairs=sliding,
                    sliding_kv_read=sliding_reads,
                    state_read=state, state_written=state)

    def seconds(self, work):
        """Return the larger of pass memory time and pass compute time."""
        policy = self.recompute_cost
        memory = (self.weight_seconds + self.kv_seconds * work.kv_read
                  + self.sliding_kv_seconds * work.sliding_kv_read
                  + self.state_seconds * (work.state_read + work.state_written))
        compute = (policy.linear_seconds * work.tokens
                   + policy.pair_seconds * work.pairs
                   + policy.sliding_pair_seconds * work.sliding_pairs)
        return max(memory, compute) if work.tokens else 0.0

    def replay_seconds(self, work, tokens, chunk_tokens=0):
        """Return the seconds rebuilding prefixes of these lengths adds to a pass.

        Each prefix is rebuilt beside the pass's work: its tokens, causal
        attention pairs, and KV writes are added to the pass, and the
        difference in pass time is its cost. A prefix longer than
        chunk_tokens takes more passes, and each extra pass reads the
        weights again. chunk_tokens of 0 packs any prefix in one pass.

        Args:
            work: The pass's work without the rebuild, as one Work total.
            tokens: Prefix lengths to rebuild, an array.
            chunk_tokens: The most tokens one pass packs, or 0.

        Returns:
            Seconds per prefix length, an array.
        """
        policy = self.recompute_cost
        tokens = np.asarray(tokens, dtype=float)
        memory = (self.weight_seconds + self.kv_seconds * work.kv_read
                  + self.sliding_kv_seconds * work.sliding_kv_read
                  + self.state_seconds * (work.state_read + work.state_written))
        compute = (policy.linear_seconds * work.tokens
                   + policy.pair_seconds * work.pairs
                   + policy.sliding_pair_seconds * work.sliding_pairs)
        added = (policy.linear_seconds * tokens
                 + policy.pair_seconds * triangle(tokens)
                 + policy.sliding_pair_seconds
                 * triangle(tokens, policy.sliding_window))
        before = max(memory, compute) if work.tokens else 0.0
        extra = (np.maximum(0.0, np.ceil(tokens / chunk_tokens) - 1)
                 if chunk_tokens else 0.0)
        return (np.maximum(memory, compute + added) - before
                + extra * self.weight_seconds)

    def gain(self, work, added, expected_tokens):
        """Return saved seconds from admitting a group of answer rows now."""
        if not added.tokens:
            return 0.0
        # An empty pass compares with the candidate's standalone pass.
        before = self.seconds(work) if work.tokens else self.seconds(added)
        average = before / (work.tokens or added.tokens)
        marginal = ((self.seconds(work + added) - self.seconds(work))
                    / added.tokens)
        return expected_tokens * max(0.0, average - marginal)

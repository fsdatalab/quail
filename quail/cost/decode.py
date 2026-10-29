"""Spec-only pass time for answer admission."""

from dataclasses import dataclass

import numpy as np

from quail.cost.dense_decoder_cost import dense_decoder_components
from quail.cost.retention import RetentionPolicy, coefficients
from quail.cost.work import Work


@dataclass(frozen=True)
class DecodeCost:
    """Price decode passes and prefix recomputation in seconds."""

    recompute_cost: RetentionPolicy
    weight_seconds: float
    kv_seconds: float
    sliding_kv_seconds: float

    @classmethod
    def from_specs(cls, model, device):
        """Build coefficients from model shapes and hardware peak rates."""
        weight_bytes = sum(component.bytes_moved for component in
                           dense_decoder_components(Work(), model, passes=1))
        return cls(RetentionPolicy(**coefficients(model, device, peak=True)),
                   weight_bytes / device.hbm_bw,
                   model.kappa_full / device.hbm_bw,
                   model.kappa_sliding / device.hbm_bw)

    def row_work(self, contexts, groups=None, shared=None):
        """Count per-row work, assigning each shared read to one group member."""
        contexts = np.asarray(contexts, dtype=float)
        window = self.recompute_cost.sliding_window
        sliding = np.minimum(contexts, window)
        reads, sliding_reads = contexts.copy(), sliding.copy()
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
        return Work(tokens=np.ones(len(contexts)), pairs=contexts,
                    kv_read=reads, sliding_pairs=sliding,
                    sliding_kv_read=sliding_reads)

    def seconds(self, work):
        """Return the larger of pass memory time and pass compute time."""
        policy = self.recompute_cost
        memory = (self.weight_seconds + self.kv_seconds * work.kv_read
                  + self.sliding_kv_seconds * work.sliding_kv_read)
        compute = (policy.linear_seconds * work.tokens
                   + policy.pair_seconds * work.pairs
                   + policy.sliding_pair_seconds * work.sliding_pairs)
        return max(memory, compute) if work.tokens else 0.0

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

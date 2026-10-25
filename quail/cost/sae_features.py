"""Work and time estimates for SAE feature calls.

A feature call runs prefill over each document and its question tail,
then encodes the pooling's tokens with the model's sparse autoencoder.
The encode is priced by the dense decoder's sae_encode component from
Work.sae_encoded. The pooling 'last' encodes one token per row; 'max'
encodes every prompt token, and each document's maxima are kept, so a
later prompt over the same rows encodes only its question tokens.
"""

from dataclasses import dataclass, replace

from quail.cost.budgets import parallel_copies
from quail.cost.score import score_work
from quail.cost.sol import speed_of_light
from quail.cost.work import Work
from quail.specs import DeviceSpec, ModelSpec


@dataclass(frozen=True)
class SaeFeaturesCost:
    """Prepared numeric inputs for one feature call's cost at any count.

    Attributes:
        token_lengths: Token counts of the head and the question tail.
        summary: The document length summary of the read alias.
        pooling: 'last' or 'max'.
        model: The model spec, with the SAE fields set.
        device: The device spec.
        chunk_tokens: The chunk budget.
        gpus: GPU copies the documents are split over.
        max_pooling: Whether another call in the pipeline needs all token rows.
    """

    token_lengths: tuple[int, ...]
    summary: object
    pooling: str
    model: ModelSpec
    device: DeviceSpec
    chunk_tokens: int
    gpus: int
    max_pooling: bool = False

    def estimate(self, expected_rows: float,
                 resident: bool = False) -> tuple[Work, float]:
        """Return model work and seconds for the expected documents.

        Args:
            expected_rows: Documents the call reads.
            resident: Whether the documents' KV is already read, by a
                step of the same pipeline before this one. The call
                then pays only its question tails and, with the
                pooling 'max', encodes only its question tokens.
        """
        fixed = sum(self.token_lengths)
        mean = self.summary.mean
        shared = self.token_lengths[0]
        prefix = float(shared) + (mean if resident else 0.0)
        # resident, every row's prefix is read from KV; fresh, the
        # shared head is computed once per GPU copy
        groups = 0.0 if resident else min(float(self.gpus), expected_rows)
        work = score_work(
            expected_rows, mean, fixed,
            variance=self.summary.variance if not resident else 0.0,
            prefix_tokens=prefix, groups=groups,
            shared_tokens=shared, copies=self.gpus)
        encoded = expected_rows if self.pooling == "last" else 0.0
        if self.pooling == "max" or self.max_pooling:
            encoded += expected_rows * (
                self.token_lengths[1] if resident else fixed + mean)
        work = replace(work, sae_encoded=encoded)
        copies = parallel_copies(self.gpus, min(expected_rows,
                                                self.summary.count))
        seconds = speed_of_light(
            work * (1 / copies), self.model, self.device,
            self.chunk_tokens).seconds
        return work, seconds

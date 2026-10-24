"""The SAE encoder and the two poolings of AI.SAE_FEATURES.

The encode follows the published SAE: per token with hidden state h,
pre = h @ W_enc.T + b_enc, keep the SaeSpec.keep largest values, and set
the rest and any negatives to 0. The pooling 'last' encodes each
answer row through a readout (encoder.py); 'max' encodes every chunk
row through the pipeline's forward hook and keeps each document's
per-feature maxima across the pipeline's feature calls
(max_pooling.py).
"""

from quail.backends.quail.executor.sae.encoder import (
    SaeEncoder,
    SaeFeaturesReadout,
    feature_lists_of_rows,
    sae_encoder,
)
from quail.backends.quail.executor.sae.max_pooling import SaeMaxAccumulator

__all__ = [
    "SaeEncoder",
    "SaeFeaturesReadout",
    "SaeMaxAccumulator",
    "feature_lists_of_rows",
    "sae_encoder",
]

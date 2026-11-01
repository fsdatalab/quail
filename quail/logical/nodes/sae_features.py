"""The SemanticSaeFeatures node: one AI.SAE_FEATURES call over one table."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from quail.logical.nodes.column import SemanticColumn, column_nodes

# the Arrow type of one feature list
FEATURE_LIST = pa.list_(
    pa.struct([("feature", pa.int32()), ("strength", pa.float32())]))


@dataclass(frozen=True)
class SemanticSaeFeatures(SemanticColumn):
    """One AI.SAE_FEATURES call over its input rows, adding the list column.

    Every input row passes through with its sparse autoencoder (SAE)
    features in the column named ``name``, a FEATURE_LIST holding the
    nonzero features. The call reads one table and generates no text.
    The root Project returns the column through an Alias of the same
    call.
    """

    type_name = "quail.semantic_sae_features"
    call_kind = "features"
    function = "AI.SAE_FEATURES"

    @property
    def pooling(self) -> str:
        """Which tokens become a row's list: 'last' or 'max'."""
        return self.call.pooling

    def explain_fields(self) -> dict:
        return {**super().explain_fields(), "pooling": self.pooling}


def feature_calls(node) -> tuple:
    """Return the SemanticSaeFeatures nodes at or below a node, children first."""
    return column_nodes(node, SemanticSaeFeatures)

"""A loaded model's tokenizer: encoding, decoding, and the tokens that end an answer."""

import numpy as np
import pyarrow as pa

from quail.common.tokens import NativeEncoder, TokenOffsets, TokenView


class ModelTokenizer:
    """One model's tokenizer, as AI.MAP and AI.EXTRACT read it on the GPU worker.

    Args:
        encoder: The model's NativeEncoder, or an object with its methods.
        turn_suffix: The model's chat-turn text after a prompt; its first
            token closes a turn. Empty for a model without chat turns.
        eos_token_id: The model's end-of-text token, or None.

    Raises:
        ValueError: The model has neither a turn close nor an
            end-of-text token, so no answer could end.
    """

    def __init__(self, encoder, turn_suffix: str = "",
                 eos_token_id: int | None = None):
        self.encoder = encoder
        self.offsets = TokenOffsets(encoder.token_bytes())
        # the answer ends where the model ends its turn: the turn close
        # or the end of text, which no document holds
        stops = {self.encode_ids(turn_suffix)[0]} if turn_suffix else set()
        if eos_token_id is not None:
            stops.add(int(eos_token_id))
        if not stops:
            raise ValueError("an answer needs a turn close or an end-of-text token")
        self.stop_ids = tuple(sorted(stops))
        self.is_stop = np.zeros(len(self), dtype=bool)
        self.is_stop[list(self.stop_ids)] = True

    @classmethod
    def load(cls, spec, model_path: str = "") -> "ModelTokenizer":
        """Load a model's tokenizer.

        Args:
            spec: The model spec.
            model_path: A local checkpoint that replaces the spec's
                Hugging Face repository at its pinned revision.

        Returns:
            The model's tokenizer.
        """
        from quail.specs.files import resolve_model_path

        path = resolve_model_path(model_path or spec.hf_name,
                                  None if model_path else spec.revision,
                                  include_weights=False)
        encoder = NativeEncoder(path)
        eos = eos_token(path)
        return cls(encoder, spec.turn_suffix,
                   None if eos is None else encoder.added_token_id(eos))

    def __len__(self) -> int:
        return len(self.encoder)

    def token_texts(self) -> list:
        """Return each token id's token text, in the tokenizer's own spelling."""
        return self.encoder.token_texts()

    def decode_batch(self, tokens: list) -> list[str]:
        """Return the text of each token id list, without special tokens."""
        return self.encoder.decode_batch(tokens)

    def encode_ids(self, text: str) -> list:
        """Return the token ids of a text, without special tokens."""
        return list(self.encoder(text))

    def encode(self, text: str) -> tuple:
        """Return the token ids of a text and each token's character range."""
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list) -> list:
        """Return (token ids, character ranges) of each text, encoded together."""
        if not texts:
            return []
        ids, offsets, boundaries = self.encode_packed(texts)
        return [(ids[start:end], offsets[start:end])
                for start, end in zip(boundaries[:-1], boundaries[1:])]

    def encode_packed(self, texts):
        """Return flat tokens, character ranges and row boundaries.

        Raises:
            ValueError: The token bytes do not reproduce the source text.
        """
        source = pa.array(texts, type=pa.large_string())
        if not len(source):
            return (np.empty(0, dtype=np.int64), np.empty((0, 2), dtype=np.int64),
                    np.zeros(1, dtype=np.int64))
        ids, boundaries = self.encoder.batch_numpy(source)
        tokens = pa.LargeListArray.from_arrays(boundaries, pa.array(ids))
        offsets, exact = self.offsets.numpy(tokens, source)
        if not exact.to_numpy(zero_copy_only=False).all():
            raise ValueError(
                "AI.EXTRACT requires token bytes that preserve the source text")
        return ids, offsets, boundaries

    def document_batch(self, texts, stored) -> list[TokenView]:
        """Return document token views with source character ranges.

        Stored views with offsets are reused. Other inputs are encoded
        together through the same tokenizer used for continuations.
        """
        missing = [index for index, tokens in enumerate(stored)
                   if not isinstance(tokens, TokenView)
                   or tokens.character_offsets is None]
        result = list(stored)
        for index, (ids, offsets) in zip(
                missing, self.encode_batch([texts[index] for index in missing])):
            bounds = np.asarray(offsets, dtype=np.int64).reshape(-1)
            result[index] = TokenView(
                pa.array(ids, type=pa.int32()),
                pa.FixedSizeListArray.from_arrays(pa.array(bounds), 2))
        return result


def eos_token(path) -> str | None:
    """Return the end-of-text token's text a model's tokenizer config names."""
    import json
    from pathlib import Path

    config = Path(path) / "tokenizer_config.json"
    if not config.exists():
        return None
    token = json.loads(config.read_text()).get("eos_token")
    return token.get("content") if isinstance(token, dict) else token

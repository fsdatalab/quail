"""A loaded model's tokenizer: encoding, decoding, and the tokens that end an answer."""

import numpy as np
import pyarrow as pa

from quail.common.tokens import GigatokenEncoder, TokenOffsets, TokenView


class ModelTokenizer:
    """One model's tokenizer, as AI.MAP and AI.EXTRACT read it on the GPU worker.

    Args:
        tokenizer: A Gigatoken encoder, or a Hugging Face style tokenizer
            supplied by the caller.
        turn_suffix: The model's chat-turn text after a prompt; its first
            token closes a turn. Empty for a model without chat turns.

    Raises:
        ValueError: The model has neither a turn close nor an
            end-of-text token, so no answer could end.
    """

    def __init__(self, tokenizer, turn_suffix: str = ""):
        self.encoder = tokenizer if isinstance(tokenizer, GigatokenEncoder) else None
        self.hf = (self.encoder.tokenizer.as_hf() if self.encoder is not None
                   else tokenizer)
        self.offsets = (TokenOffsets(self.encoder.token_bytes())
                        if self.encoder is not None else None)
        # the answer ends where the model ends its turn: the turn close
        # or the end of text, which no document holds
        stops = {self.encode_ids(turn_suffix)[0]} if turn_suffix else set()
        if self.hf.eos_token_id is not None:
            stops.add(int(self.hf.eos_token_id))
        if not stops:
            raise ValueError("an answer needs a turn close or an end-of-text token")
        self.stop_ids = tuple(sorted(stops))
        self.is_stop = np.zeros(len(self.hf), dtype=bool)
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
        return cls(GigatokenEncoder(path), spec.turn_suffix)

    def __len__(self) -> int:
        return len(self.hf)

    def token_texts(self) -> list:
        """Return each token id's token text, in the tokenizer's own spelling."""
        return self.hf.convert_ids_to_tokens(list(range(len(self.hf))))

    def decode_batch(self, tokens: list) -> list[str]:
        """Return the text of each token id list, without special tokens."""
        return self.hf.batch_decode(tokens, skip_special_tokens=True)

    def encode_ids(self, text: str) -> list:
        """Return the token ids of a text, without special tokens."""
        return list(self.hf.encode(text, add_special_tokens=False))

    def encode(self, text: str) -> tuple:
        """Return the token ids of a text and each token's character range."""
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list) -> list:
        """Return (token ids, character ranges) of each text, encoded together."""
        if not texts:
            return []
        if self.encoder is not None:
            ids, offsets, boundaries = self.encode_packed(texts)
            return [(ids[start:end], offsets[start:end])
                    for start, end in zip(boundaries[:-1], boundaries[1:])]
        backend = getattr(self.hf, "backend_tokenizer", None)
        if backend is not None:
            return [(enc.ids, enc.offsets) for enc in backend.encode_batch(
                list(texts), add_special_tokens=False)]
        enc = self.hf(list(texts), return_offsets_mapping=True,
                      add_special_tokens=False)
        return [(list(ids), list(offsets))
                for ids, offsets in zip(enc["input_ids"], enc["offset_mapping"])]

    def encode_packed(self, texts):
        """Return flat tokens, character ranges and row boundaries.

        Raises:
            ValueError: The token bytes do not reproduce the source text.
        """
        source = pa.array(texts, type=pa.large_string())
        if not len(source):
            return (np.empty(0, dtype=np.int64), np.empty((0, 2), dtype=np.int64),
                    np.zeros(1, dtype=np.int64))
        if self.encoder is not None:
            ids, boundaries = self.encoder.batch_numpy(source)
            tokens = pa.LargeListArray.from_arrays(boundaries, pa.array(ids))
            offsets, exact = self.offsets.numpy(tokens, source)
            if not exact.to_numpy(zero_copy_only=False).all():
                raise ValueError(
                    "AI.EXTRACT requires token bytes that preserve the source text")
            return ids, offsets, boundaries
        encoded = self.encode_batch(source.to_pylist())
        boundaries = np.r_[0, np.cumsum([len(ids) for ids, _ in encoded])]
        ids = np.concatenate([ids for ids, _ in encoded]).astype(np.int64)
        offsets = np.concatenate([
            np.asarray(offsets, dtype=np.int64).reshape(-1, 2)
            for _, offsets in encoded])
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

"""SAE weights, the per-token top-k encode, and the 'last' readout."""

import numpy as np

from quail.backends.quail.executor.readout import _HostReadout

# bytes of fp32 pre-activation one encode block materializes: it lives
# in the device memory the arena leaves unclaimed (POOL_FRACTION in
# quail.cost.budgets), and fewer, larger blocks mean fewer launches
ENCODE_BLOCK_BYTES = 256 * 2**20


class SaeEncoder:
    """W_enc and b_enc on the GPU, with the per-token top-k encode.

    The GEMM runs in bf16 and the bias adds in fp32; the agreement
    check against the fp32 reference covers the cast.
    """

    def __init__(self, torch, F, weight, bias, keep: int):
        self.torch = torch
        self.F = F
        self.weight = weight    # (sae_num_features, hidden) bf16
        self.bias = bias        # (sae_num_features,) fp32
        self.keep = keep
        self.num_features = weight.shape[0]

    @classmethod
    def load(cls, torch, F, spec):
        """Load the spec's SAE tensors onto the GPU.

        Raises:
            ValueError: The spec names no SAE, or the tensors do not
                match its width.
        """
        if not spec.sae_hf_name or not spec.sae_file:
            raise ValueError(f"{spec.name!r} names no SAE checkpoint")
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(spec.sae_hf_name, spec.sae_file,
                               revision=spec.sae_revision or None)
        tensors = torch.load(path, map_location="cpu", weights_only=True)
        weight = tensors["W_enc"]
        bias = tensors["b_enc"]
        if tuple(weight.shape) != (spec.sae_num_features, spec.hidden):
            raise ValueError(
                f"{spec.sae_file} W_enc is {tuple(weight.shape)}, expected "
                f"({spec.sae_num_features}, {spec.hidden})")
        return cls(torch, F, weight.to(device="cuda", dtype=torch.bfloat16),
                   bias.to(device="cuda", dtype=torch.float32), spec.sae_keep)

    def top_features(self, rows):
        """Return (indices int32, strengths fp32), the keep largest per row."""
        torch = self.torch
        indices, strengths = [], []
        block = max(1, ENCODE_BLOCK_BYTES // (4 * self.num_features))
        for start in range(0, rows.shape[0], block):
            # the fp32 bias promotes the bf16 GEMM output in one pass; a
            # negative value is selected only when a row has fewer than
            # keep positives, and the clamp on the selection zeroes it
            pre = self.F.linear(rows[start:start + block].to(torch.bfloat16),
                                self.weight) + self.bias
            values, index = torch.topk(pre, self.keep, dim=1)
            indices.append(index.to(torch.int32))
            strengths.append(values.clamp_(min=0.0))
        return torch.cat(indices), torch.cat(strengths)

    def packed_top_features(self, rows):
        """One fp32 tensor per block of rows: the indices, then the strengths.

        Feature indices are below 2**24, so fp32 carries them exactly.
        """
        indices, strengths = self.top_features(rows)
        return self.torch.cat([indices.float(), strengths], dim=1)


class SaeFeaturesReadout(_HostReadout):
    """Encode the answer rows' residual stream, for the pooling 'last'.

    result() returns a (rows, 2 * keep) fp32 array: each row's feature
    indices, then their strengths. The dtype carries the row width, so
    the scheduler's answer array holds one vector per answer.
    """

    trailing_rows = 1

    def __init__(self, torch, encoder: SaeEncoder):
        super().__init__(torch)
        self.encoder = encoder
        self.dtype = np.dtype((np.float32, 2 * encoder.keep))

    def submit(self, stream):
        return self._copy(self.encoder.packed_top_features(stream))


def feature_lists_of_rows(packed, keep: int) -> tuple:
    """Return (lengths, indices, strengths) of packed readout rows.

    Each row is a readout's packed vector: keep feature indices, then
    their strengths, zero past the kept ones. Every row unpacks in one
    pass, with no per-feature or per-row Python work.
    """
    packed = np.asarray(packed, dtype=np.float32).reshape(-1, 2 * keep)
    indices = packed[:, :keep]
    strengths = packed[:, keep:]
    row_ids, columns = np.nonzero(strengths > 0.0)
    lengths = np.bincount(row_ids, minlength=packed.shape[0])
    return (lengths, indices[row_ids, columns].astype(np.int32),
            strengths[row_ids, columns])


def sae_encoder(state) -> SaeEncoder:
    """Return the loaded model's SAE encoder, loading it on first use."""
    loaded = state.loaded_model
    if getattr(loaded, "sae", None) is None:
        import torch.nn.functional as F

        loaded.sae = SaeEncoder.load(state.torch, F, loaded.model_spec)
    return loaded.sae

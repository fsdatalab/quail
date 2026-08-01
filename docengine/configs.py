"""Model and device configurations (paper Tables 2-4, conventions C1/C5/C6/C7).

All byte quantities are decimal unless suffixed _kib. Rates are FLOP/s and
bytes/s. W_mem / W_run are nominal placeholders until measured from safetensors
(convention C5); both are recorded per run.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelConfig:
    name: str
    P: float            # dense non-embedding params repeatedly used per token
    L: int              # transformer layers
    h: int              # hidden width
    n_q: int            # query heads
    n_kv: int           # KV heads
    d_h: int            # head dim
    L_ctx: int          # max tokens on any root-to-leaf causal path
    q_kv: int           # bytes per stored KV element (1=fp8, 2=bf16)
    W_mem: float        # resident weight footprint, bytes
    W_run: float        # compulsory transformer-weight traffic per nonempty batch, bytes

    @property
    def kappa(self) -> int:
        """KV bytes per cached token (eq. 18)."""
        return 2 * self.L * self.n_kv * self.d_h * self.q_kv

    @property
    def attn_width(self) -> int:
        """Attention width n_q * d_h (convention C1; paper eq. 20 uses h,
        which undercounts for Qwen3 where n_q*d_h != h)."""
        return self.n_q * self.d_h

    def with_kv_dtype(self, q_kv: int) -> "ModelConfig":
        return replace(self, q_kv=q_kv)


@dataclass(frozen=True)
class DeviceConfig:
    name: str
    M: float            # physical memory, bytes
    BW: float           # memory bandwidth, bytes/s
    R_D: float          # dense FP8 throughput ceiling, FLOP/s
    R_A: float          # attention throughput ceiling, FLOP/s (C7: = R_D primary)
    S: float = 2e9      # non-weight non-KV reserve, bytes (C6)


QWEN3_4B_FP8 = ModelConfig(
    name="Qwen3-4B-FP8", P=3.6e9, L=36, h=2560, n_q=32, n_kv=8, d_h=128,
    L_ctx=40_960, q_kv=1, W_mem=4.5e9, W_run=3.6e9,
)

QWEN3_32B_FP8 = ModelConfig(
    name="Qwen3-32B-FP8", P=31.2e9, L=64, h=5120, n_q=64, n_kv=8, d_h=128,
    L_ctx=40_960, q_kv=1, W_mem=33.5e9, W_run=31.2e9,
)

H100_SXM = DeviceConfig(name="H100-SXM-80GB", M=80e9, BW=3.35e12,
                        R_D=1.979e15, R_A=1.979e15)

L40S = DeviceConfig(name="L40S-48GB", M=48e9, BW=864e9,
                    R_D=0.733e15, R_A=0.733e15)

MODELS = {m.name: m for m in (QWEN3_4B_FP8, QWEN3_32B_FP8)}
DEVICES = {d.name: d for d in (H100_SXM, L40S)}

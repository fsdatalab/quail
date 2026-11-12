from .qwen3 import qwen3_spec

# config.json of Qwen/Qwen3-0.6B at the pinned revision, the backbone
# Decision-2.0-Kai-0.6B was fine tuned from
_CONFIG = {
    "model_type": "qwen3",
    "torch_dtype": "bfloat16",
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151_936,
    "tie_word_embeddings": True,
    "use_sliding_window": False,
}

QWEN3_0_6B_BF16 = qwen3_spec(
    "qwen3-0.6b-bf16", "Qwen/Qwen3-0.6B",
    "c1899de289a04d12100db370d81485cdf75e47ca", config=_CONFIG,
    # the same shapes as Decision-2.0-Kai-0.6B
    chunk_cap_tokens=65_536,
)

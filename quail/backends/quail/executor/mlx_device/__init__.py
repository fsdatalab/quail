"""Quail's model execution on Apple silicon with MLX.

The pieces mirror the CUDA executor's: KV pools in pages, a paged KV
write and paged attention from vllm-metal's Metal kernels, a Qwen3
forward pass over packed rows, a weight loader, and the Decision 2.0
readouts. mlx and vllm-metal are imported on first use, so this package
imports on any platform.
"""

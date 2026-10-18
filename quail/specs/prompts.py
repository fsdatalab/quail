"""Shared instructions and model chat turns."""

TEXT_ANALYSIS_SYSTEM_PROMPT = "You are an expert at text analysis"

QWEN3_TURN_PREFIX = (
    f"<|im_start|>system\n{TEXT_ANALYSIS_SYSTEM_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n"
)
QWEN3_TURN_SUFFIX = (
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
QWEN3_PROMPT_FORMAT = "qwen3-chat-text-analysis-v1"
QWEN3_RERANKER_PROMPT_FORMAT = "qwen3-reranker-chat-text-analysis-v1"

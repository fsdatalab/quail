"""Shared Qwen3 reranker input format."""

QWEN3_RERANKER_INSTRUCTION = (
    "Judge whether the document meets the requirements in the query."
)
QWEN3_RERANKER_SYSTEM_TEXT = (
    "You are performing a data processing task. "
    "Judge whether the Document meets the requirements based on the Query "
    "and the Instruct provided. Note that the answer can only be \"yes\" "
    "or \"no\"."
)

def render_qwen3_reranker_input(query: str, document: str) -> str:
    """Render a complete Qwen3 reranker prompt."""
    return (
        f"<|im_start|>system\n{QWEN3_RERANKER_SYSTEM_TEXT}<|im_end|>\n"
        f'<|im_start|>user\n<Instruct>: {QWEN3_RERANKER_INSTRUCTION}\n'
        f'<Query>: {query}\n<Document>: {document}<|im_end|>\n'
        '<|im_start|>assistant\n<think>\n\n</think>\n\n'
    )

"""Shared Qwen3 reranker input format."""

QWEN3_RERANKER_INSTRUCTION = (
    "Judge whether the document meets the requirements in the query."
)
QWEN3_RERANKER_SYSTEM_TEXT = (
    "Judge whether the Document meets the requirements based on the Query "
    "and the\nInstruct provided. Note that the answer can only be \"yes\" "
    "or \"no\"."
)

QWEN3_RERANKER_TEMPLATE = """{%- set query_text = messages
    | selectattr("role", "eq", "query")
    | map(attribute="content") | first -%}
{%- set document_text = messages
    | selectattr("role", "eq", "document")
    | map(attribute="content") | first -%}
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the
Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Judge whether the document meets the requirements in the query.
<Query>: {{ query_text }}
<Document>: {{ document_text }}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""


def render_qwen3_reranker_input(query: str, document: str) -> str:
    """Render the text that vLLM sends to the Qwen3 reranker."""
    return (
        f"<|im_start|>system\n{QWEN3_RERANKER_SYSTEM_TEXT}<|im_end|>\n"
        f'<|im_start|>user\n<Instruct>: {QWEN3_RERANKER_INSTRUCTION}\n'
        f'<Query>: {query}\n<Document>: {document}<|im_end|>\n'
        '<|im_start|>assistant\n<think>\n\n</think>\n\n'
    )

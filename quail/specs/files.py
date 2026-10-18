"""Local files for one model revision."""

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=16)
def resolve_model_path(model_name: str, revision: str | None = None, *,
                       include_weights: bool = True) -> str:
    """Return the local directory holding one model revision.

    Args:
        model_name: A local directory or a Hugging Face repository id.
        revision: The repository revision; None uses its default branch.
        include_weights: Whether to download the weights beside the
            tokenizer and config files.
    """
    path = Path(model_name).expanduser()
    if path.is_dir():
        return str(path.resolve())
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "*.model", "*.txt", "*.tiktoken"]
    if include_weights:
        patterns.append("*.safetensors")
    return snapshot_download(repo_id=model_name, revision=revision or None,
                             allow_patterns=patterns)

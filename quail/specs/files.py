"""Local files for one model revision."""

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=16)
def resolve_model_path(model_name: str, revision: str | None = None, *,
                       include_weights: bool = True) -> str:
    """Return a local checkpoint directory at the requested revision.

    Args:
        model_name: A local directory or a Hugging Face repository ID.
        revision: The repository revision; None or empty uses its default branch.
        include_weights: Download weights as well as tokenizer and config files.
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

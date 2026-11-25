"""Image documents in prompts for stock vLLM.

Quail's token stores hold an image as one token id per image token
(executor/images.py). vLLM takes an image as a picture beside the
prompt and one ``<|image_pad|>`` token in it, which its processor
expands to the image's tokens. ``ImagePrompts.prompt`` rewrites a
prompt of token ids that way, so both engines read the same pixels.
"""

from __future__ import annotations

import io

from _quail_native import IMAGE_TOKEN_BASE

from quail.backends.quail.executor.images import RegionImages


class ImagePrompts:
    """Rewrites prompts that hold image tokens for vLLM's processor.

    Args:
        images: The region's images.
        pad_id: The id of the model's ``<|image_pad|>`` token.
    """

    def __init__(self, images: RegionImages, pad_id: int):
        self.images = images
        self.pad_id = pad_id

    def prompt(self, prompt):
        """The prompt with each image as one pad token and its picture."""
        if not isinstance(prompt, dict) or "prompt_token_ids" not in prompt:
            return prompt
        ids, pictures = [], []
        for token in prompt["prompt_token_ids"]:
            if token < IMAGE_TOKEN_BASE:
                ids.append(token)
                continue
            image = self.images.image_of(token)
            if token == image.first:
                ids.append(self.pad_id)
                pictures.append(_picture(image.encoded))
        if not pictures:
            return prompt
        return {**prompt, "prompt_token_ids": ids,
                "multi_modal_data": {"image": pictures}}


def _picture(encoded: bytes):
    from PIL import Image

    with Image.open(io.BytesIO(encoded)) as opened:
        return opened.convert("RGB")

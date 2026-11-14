"""Rotary positions of rows that read image tokens (executor/images.py).

Qwen's vision models place image token (i, j) of an image that starts
at position s at (s, s + i, s + j), and the token after the image at
s + max(rows, columns).
"""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import torch
from _quail_native import IMAGE_TOKEN_BASE
from helpers.images import png

from quail.backends.quail.executor.images import RegionImages

FIRST = IMAGE_TOKEN_BASE
START, END = 3060, 3061
# 2 columns by 3 rows of image tokens: 6 tokens, positions advance 3
IMAGE = [START, *range(FIRST, FIRST + 6), END]
KEY = ("pages", 0)


def _images():
    columns = {"pages": pa.table({"image": pa.array([png(64, 96)], pa.binary())})}
    return RegionImages({"pages": [IMAGE]}, columns)


def _chunk(ids, positions, row_spans, decode=None):
    return SimpleNamespace(input_ids=torch.tensor(ids), row_spans=row_spans,
                           positions=torch.tensor(positions), decode_rows=decode)


def test_an_image_token_takes_its_grid_position_and_later_rows_follow_the_grid():
    images = _images()
    ids = [7, 8, *IMAGE, 9, 10]
    chunk = _chunk(ids, list(range(len(ids))), [(KEY, 0, len(ids) - 2, len(ids))])
    rotary = images.rotary_positions(chunk, ()).numpy()
    image = [3 + offset for offset in range(6)]
    assert rotary[:, :3].tolist() == [[0, 1, 2]] * 3
    assert rotary[0, image].tolist() == [3] * 6
    assert rotary[1, image].tolist() == [3, 3, 4, 4, 5, 5]
    assert rotary[2, image].tolist() == [3, 4, 3, 4, 3, 4]
    # the vision end token and the question after it
    assert rotary[:, 9:].tolist() == [[6, 7, 8]] * 3


class _Decode:
    """Decode rows: each row's answer slot."""

    def __init__(self, slots):
        self.slots = np.asarray(slots)

    def __len__(self):
        return len(self.slots)


def test_a_question_and_an_answer_after_an_earlier_image_shift_by_its_grid():
    images = _images()
    ids = [7, 8, *IMAGE]
    images.rotary_positions(_chunk(ids, list(range(len(ids))),
                                   [(KEY, 0, len(ids), len(ids))]), ())
    # a question of two rows, then one answer row whose slot is the key's
    chunk = _chunk([9, 10, 11], [10, 11, 12], [(KEY, 0, 0, 2)], _Decode([0]))
    assert images.rotary_positions(chunk, [KEY]).tolist() == [7, 8, 9]


def test_a_chunk_with_no_image_document_keeps_its_positions():
    chunk = _chunk([7, 8], [0, 1], [(("notes", 0), 0, 0, 2)])
    assert _images().rotary_positions(chunk, ()) is None


class _Visual(torch.nn.Module):
    """A vision encoder whose token embeddings count its calls' tokens."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, pixel_values, grid_thw):
        self.calls += 1
        tokens = int(grid_thw.prod(-1).sum()) // 4
        assert pixel_values.shape == (tokens * 4, 1536)
        return torch.arange(tokens, dtype=torch.float32)[:, None].expand(tokens, 4)


def test_each_image_token_takes_its_own_embedding_and_text_takes_the_vocabulary():
    images, visual = _images(), _Visual()
    text = torch.nn.Embedding(4096, 4)
    with torch.no_grad():
        text.weight.copy_(-torch.ones(4096, 4))
        # the image's last three tokens, then text, then its first three
        ids = torch.tensor([FIRST + 3, FIRST + 4, FIRST + 5, 7, FIRST, FIRST + 1,
                            FIRST + 2])
        hidden = images.embed(torch, visual, ids, text)
    assert hidden[:, 0].tolist() == [3, 4, 5, -1, 0, 1, 2]
    assert visual.calls == 1

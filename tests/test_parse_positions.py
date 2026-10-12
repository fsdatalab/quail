import numpy as np
import pytest

from quail.backends.quail.executor.parse import mrope_positions

IMAGE = 7


def test_image_rows_and_columns_follow_the_text_before_them():
    # two text tokens, a 2 x 3 image (4 x 6 patches merged by 2), one text token
    ids = [1, 2] + [IMAGE] * 6 + [3]
    positions, offset = mrope_positions(ids, (1, 4, 6), IMAGE, 2)
    assert positions[:, :2].tolist() == [[0, 1]] * 3
    assert positions[0, 2:8].tolist() == [2] * 6
    assert positions[1, 2:8].tolist() == [2, 2, 2, 3, 3, 3]
    assert positions[2, 2:8].tolist() == [2, 3, 4, 2, 3, 4]
    # text after the image resumes one past its largest position, 4
    assert positions[:, 8].tolist() == [5, 5, 5]
    # the next token, at sequence index 9, takes position 6
    assert 9 + offset == 6


def test_a_prompt_whose_image_does_not_match_the_grid_is_refused():
    with pytest.raises(ValueError):
        mrope_positions([1] + [IMAGE] * 5, (1, 4, 6), IMAGE, 2)


def test_positions_are_int64():
    positions, _ = mrope_positions(np.array([IMAGE] * 4), (1, 4, 4), IMAGE, 2)
    assert positions.dtype == np.int64

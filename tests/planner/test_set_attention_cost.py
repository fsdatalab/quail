"""Set-attention costs: one pass, a join's passes, and pass groups."""

from helpers.rust_cost import cost_model

from quail.specs import D1_2B_BF16, H100_SXM


def _groups(selectivities, questions=50, state=300.0, longest=300,
            chunk=8192, header=0):
    count = len(selectivities)
    return cost_model().choose_pass_groups(
        [questions] * count, selectivities, 5000.0, state, longest,
        D1_2B_BF16.name, H100_SXM.name, chunk, 16, header)


def test_a_filter_that_keeps_few_rows_gets_its_own_pass_group():
    assert _groups([0.02, 0.5, 0.5]) == [0, 1, 1]


def test_a_pass_group_fits_one_chunk_with_the_longest_document_and_headers():
    # 300 + 50 + 3 * (50 + 5) = 515 tokens fit 520; a fifth question does not
    assert _groups([1.0] * 5, chunk=520, header=5) == [0, 0, 0, 0, 1]

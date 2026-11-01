"""CPU checks for the classification labeling cell."""

import math

from experiments.cells import classify_labels as cell

# billing refund request, billing refund status, billing address change,
# technical login problem, one word per token
BILLING = [(1, 2, 3), (1, 2, 4), (1, 5, 6), (7, 8, 9)]


def test_scores_sum_prefix_log_probabilities_and_ties_go_first():
    log = math.log
    logprobs = {(): {1: log(0.8), 7: log(0.1)}, (1,): {2: log(0.7), 5: log(0.2)},
                (1, 2): {3: log(0.6), 4: log(0.3)}, (1, 5): {6: log(0.9)},
                (7,): {8: log(0.8)}, (7, 8): {9: log(0.9)}}
    winner, scores = cell.score_labels(BILLING, logprobs)
    assert winner == 0
    assert [round(math.exp(s), 3) for s in scores] == [0.336, 0.168, 0.144, 0.072]
    winner, _ = cell.score_labels([(1,), (2,)], {(): {1: -1.0, 2: -1.0}})
    assert winner == 0

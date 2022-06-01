# coding = utf-8

import numpy as np


def Fbeta(predictions: np.ndarray, targets: np.ndarray, beta=1) -> float:
    assert len(predictions.shape) == len(targets.shape) == 1
    assert len(predictions) == len(targets)

    num_positive_predictions = np.sum(predictions)
    num_positive_targets = np.sum(targets)

    assert num_positive_targets > 0

    num_true_positive = np.sum(predictions[predictions == targets])

    if num_positive_predictions == 0:
        precision = 1
        recall = 0
        f = 0
    elif num_true_positive == 0:
        precision = 0
        recall = 0
        f = 0
    else:
        precision = num_true_positive / num_positive_predictions
        recall = num_true_positive / num_positive_targets

        if beta == 1:
            f = 2 * precision * recall / (precision + recall)
        else:
            beta2 = beta ** 2
            f = (1 + beta2) * precision * recall / (beta2 * precision + recall)

    return f, precision, recall

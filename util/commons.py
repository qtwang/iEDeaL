# coding = utf-8

import numpy as np


def discretize(values, threshold: np.float32 = 0.5):
    positive_indices = values > threshold

    binaries = np.zeros_like(values, dtype=int)
    binaries[positive_indices] = 1

    return binaries

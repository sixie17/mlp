"""
Loss functions. All take the softmax output `probabilities` of shape (n, 2)
and the integer `labels` of shape (n,).

With a 2-unit softmax output, binary and categorical cross-entropy give the same value:
    p0 = 1 - p1, so  -[y log p1 + (1 - y) log(1 - p1)] == -sum_k y_k log p_k
Their gradient with respect to the output weighted sums is therefore the same too:
probabilities - one_hot(labels).
"""
import numpy as np

from .preprocessing import one_hot

EPSILON = 1e-15


def binary_cross_entropy(probabilities, labels):
    """E = -1/N * sum[y log p + (1 - y) log(1 - p)], p = predicted probability of class 1 (malignant)."""
    malignant_probabilities = np.clip(probabilities[:, 1], EPSILON, 1.0 - EPSILON)
    return float(-np.mean(labels * np.log(malignant_probabilities)
                          + (1 - labels) * np.log(1 - malignant_probabilities)))


def categorical_cross_entropy(probabilities, labels):
    """E = -1/N * sum_n sum_k y_nk log p_nk"""
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    targets = one_hot(labels, probabilities.shape[1])
    return float(-np.mean(np.sum(targets * np.log(clipped), axis=1)))


LOSSES = {
    "binaryCrossentropy": binary_cross_entropy,
    "categoricalCrossentropy": categorical_cross_entropy,
}


def get_loss(name):
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {', '.join(LOSSES)}")
    return LOSSES[name]

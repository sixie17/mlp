"""
Activation functions and their derivatives.

Each derivative takes the pre-activation Z (Z = X @ W + b) and returns
d(activation)/dZ evaluated element-wise.
"""
import numpy as np


def identity(z):
    return z


def identity_derivative(z):
    return np.ones_like(z)


def relu(z):
    return np.maximum(0.0, z)


def relu_derivative(z):
    return (z > 0).astype(z.dtype)


def sigmoid(z):
    # Clipping avoids overflow in np.exp for large negative values.
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_derivative(z):
    sigmoid_output = sigmoid(z)
    return sigmoid_output * (1.0 - sigmoid_output)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    return 1.0 - np.tanh(z) ** 2


def softmax(z):
    """
    Row-wise softmax: turns each row of scores into a probability distribution.
    The row maximum is subtracted first for numerical stability (it does not change the result).
    """
    shifted = z - np.max(z, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


# name -> (function, derivative)
# softmax has no element-wise derivative: it is only allowed on the output layer,
# where it is combined with cross-entropy (see Network.backward).
ACTIVATIONS = {
    "identity": (identity, identity_derivative),
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "softmax": (softmax, None),
}


def get_activation(name):
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {', '.join(ACTIVATIONS)}")
    return ACTIVATIONS[name]

"""
Fully connected (dense) layer.

The single capital letters are the math symbols used in the formulas (see docs/08-backpropagation.md):
    X  = layer input          W = weights          b = biases          Z = weighted sums
    dZ = dLoss/dZ             dW = dLoss/dW        db = dLoss/db       dX = dLoss/dX

Forward:   Z = X @ W + b        output = activation(Z)
Backward:  dW = X.T @ dZ        db = sum of dZ over the batch        dX = dZ @ W.T
"""
import numpy as np

from .activations import get_activation
from .initializers import get_initializer


class DenseLayer:
    def __init__(self, units, activation="sigmoid", weights_initializer="heUniform"):
        if units < 1:
            raise ValueError("A layer needs at least one unit")
        self.units = int(units)
        self.activation = activation
        self.weights_initializer = weights_initializer
        self.activation_function, self.activation_derivative = get_activation(activation)
        self.initializer = get_initializer(weights_initializer)

        self.W = None
        self.b = None
        # Cache filled during forward, needed by backward.
        self.X = None
        self.Z = None

    def build(self, input_size, random_generator):
        self.W = self.initializer(input_size, self.units, random_generator)
        self.b = np.zeros((1, self.units))

    def forward(self, X):
        self.X = X
        self.Z = X @ self.W + self.b
        return self.activation_function(self.Z)

    def activation_backward(self, d_output):
        """Chain rule through the activation: dZ = d_output * activation'(Z)."""
        if self.activation_derivative is None:
            raise ValueError(f"'{self.activation}' can only be used on the output layer")
        return d_output * self.activation_derivative(self.Z)

    def backward(self, dZ):
        """
        dZ is dLoss/dZ for this layer, already averaged over the batch.
        Returns the gradients of W and b, and dLoss/dX to pass to the previous layer.
        """
        dW = self.X.T @ dZ
        db = np.sum(dZ, axis=0, keepdims=True)
        dX = dZ @ self.W.T
        return dW, db, dX

    def get_config(self):
        return {
            "units": self.units,
            "activation": self.activation,
            "weights_initializer": self.weights_initializer,
        }

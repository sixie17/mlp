"""
Optimizers. `apply(parameters, gradients)` updates each parameter array IN PLACE.
parameters and gradients are lists in the same order: [W0, b0, W1, b1, ...].

- SGD:       mini-batch gradient descent (mandatory part)
- Momentum / Nesterov, RMSprop, Adam: bonus part
"""
import numpy as np


class Optimizer:
    def __init__(self, learning_rate):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        self.learning_rate = learning_rate

    def apply(self, parameters, gradients):
        raise NotImplementedError


class SGD(Optimizer):
    """Gradient descent: w <- w - learning_rate * dLoss/dw"""

    def apply(self, parameters, gradients):
        for parameter, gradient in zip(parameters, gradients):
            parameter -= self.learning_rate * gradient


class Momentum(Optimizer):
    """
    Momentum keeps a velocity: velocity <- momentum * velocity - learning_rate * gradient ; w <- w + velocity
    Nesterov evaluates the step at the look-ahead point:   w <- w + momentum * velocity - learning_rate * gradient
    """

    def __init__(self, learning_rate, momentum=0.9, nesterov=False):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = {}

    def apply(self, parameters, gradients):
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            velocity = self.velocities.setdefault(index, np.zeros_like(parameter))
            velocity *= self.momentum
            velocity -= self.learning_rate * gradient
            if self.nesterov:
                parameter += self.momentum * velocity - self.learning_rate * gradient
            else:
                parameter += velocity


class RMSprop(Optimizer):
    """
    Divides the step by a running average of squared gradients:
    average <- rho * average + (1 - rho) * gradient^2 ; w <- w - learning_rate * gradient / sqrt(average)
    """

    def __init__(self, learning_rate, rho=0.9, epsilon=1e-7):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.squared_gradient_averages = {}

    def apply(self, parameters, gradients):
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            average = self.squared_gradient_averages.setdefault(index, np.zeros_like(parameter))
            average *= self.rho
            average += (1 - self.rho) * gradient ** 2
            parameter -= self.learning_rate * gradient / (np.sqrt(average) + self.epsilon)


class Adam(Optimizer):
    """Momentum (first moment) + RMSprop (second moment), with bias correction for the first steps."""

    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.first_moments = {}
        self.second_moments = {}
        self.step = 0

    def apply(self, parameters, gradients):
        self.step += 1
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            first_moment = self.first_moments.setdefault(index, np.zeros_like(parameter))
            second_moment = self.second_moments.setdefault(index, np.zeros_like(parameter))
            first_moment *= self.beta1
            first_moment += (1 - self.beta1) * gradient
            second_moment *= self.beta2
            second_moment += (1 - self.beta2) * gradient ** 2
            corrected_first_moment = first_moment / (1 - self.beta1 ** self.step)
            corrected_second_moment = second_moment / (1 - self.beta2 ** self.step)
            parameter -= (self.learning_rate * corrected_first_moment
                          / (np.sqrt(corrected_second_moment) + self.epsilon))


OPTIMIZERS = {
    "sgd": lambda learning_rate: SGD(learning_rate),
    "momentum": lambda learning_rate: Momentum(learning_rate),
    "nesterov": lambda learning_rate: Momentum(learning_rate, nesterov=True),
    "rmsprop": lambda learning_rate: RMSprop(learning_rate),
    "adam": lambda learning_rate: Adam(learning_rate),
}


def get_optimizer(name, learning_rate):
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from: {', '.join(OPTIMIZERS)}")
    return OPTIMIZERS[name](learning_rate)

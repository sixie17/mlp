"""
Weight initializers. Each takes (input_size, output_size, random_generator)
and returns a (input_size, output_size) matrix.

- Xavier / Glorot: keeps the variance of activations stable for sigmoid / tanh.
- He: scaled for ReLU, which zeroes half of its inputs on average.
"""
import numpy as np


def xavier_uniform(input_size, output_size, random_generator):
    limit = np.sqrt(6.0 / (input_size + output_size))
    return random_generator.uniform(-limit, limit, size=(input_size, output_size))


def xavier_normal(input_size, output_size, random_generator):
    standard_deviation = np.sqrt(2.0 / (input_size + output_size))
    return random_generator.normal(0.0, standard_deviation, size=(input_size, output_size))


def he_uniform(input_size, output_size, random_generator):
    limit = np.sqrt(6.0 / input_size)
    return random_generator.uniform(-limit, limit, size=(input_size, output_size))


def he_normal(input_size, output_size, random_generator):
    standard_deviation = np.sqrt(2.0 / input_size)
    return random_generator.normal(0.0, standard_deviation, size=(input_size, output_size))


INITIALIZERS = {
    "xavierUniform": xavier_uniform,
    "glorotUniform": xavier_uniform,
    "xavierNormal": xavier_normal,
    "glorotNormal": xavier_normal,
    "heUniform": he_uniform,
    "heNormal": he_normal,
}


def get_initializer(name):
    if name not in INITIALIZERS:
        raise ValueError(f"Unknown initializer '{name}'. Choose from: {', '.join(INITIALIZERS)}")
    return INITIALIZERS[name]

"""
The multilayer perceptron: a stack of DenseLayers.

Feedforward:     each layer's output is the next layer's input; the last layer is a softmax.
Backpropagation: the chain rule applied from the output back to the first layer.
                 For softmax + cross-entropy the output gradient simplifies to
                     dLoss/dZ_output = (probabilities - one_hot(labels)) / batch_size
"""
import json

import numpy as np

from .layers import DenseLayer
from .preprocessing import one_hot


class Network:
    def __init__(self, layers, input_size, seed=None):
        if len(layers) < 2:
            raise ValueError("The network needs at least one hidden layer and an output layer")
        if layers[-1].activation != "softmax":
            raise ValueError("The output layer must use the softmax activation")
        if any(layer.activation == "softmax" for layer in layers[:-1]):
            raise ValueError("softmax is only allowed on the output layer")

        self.layers = layers
        self.input_size = int(input_size)
        random_generator = np.random.default_rng(seed)
        layer_input_size = self.input_size
        for layer in self.layers:
            layer.build(layer_input_size, random_generator)
            layer_input_size = layer.units

    # ---------- feedforward ----------
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def predict_probabilities(self, X):
        return self.forward(X)

    # ---------- backpropagation ----------
    def backward(self, probabilities, labels):
        """
        probabilities: softmax output of the last forward pass, shape (batch_size, class_count)
        labels:        integer labels, shape (batch_size,)
        Returns gradients in the same order as parameters(): [dW0, db0, dW1, db1, ...]
        """
        batch_size = probabilities.shape[0]
        dZ = (probabilities - one_hot(labels, probabilities.shape[1])) / batch_size
        gradients = []
        for index in reversed(range(len(self.layers))):
            dW, db, dX = self.layers[index].backward(dZ)
            gradients.append(db)
            gradients.append(dW)
            if index > 0:
                dZ = self.layers[index - 1].activation_backward(dX)
        gradients.reverse()
        return gradients

    def parameters(self):
        parameters = []
        for layer in self.layers:
            parameters.extend([layer.W, layer.b])
        return parameters

    # ---------- weights ----------
    def get_weights(self):
        return [parameter.copy() for parameter in self.parameters()]

    def set_weights(self, weights):
        for layer, (W, b) in zip(self.layers, zip(weights[0::2], weights[1::2])):
            if W.shape != layer.W.shape or b.shape != layer.b.shape:
                raise ValueError("Saved weights do not match the network topology")
            layer.W = W.copy()
            layer.b = b.copy()

    def get_config(self):
        return {"input_size": self.input_size, "layers": [layer.get_config() for layer in self.layers]}

    def summary(self):
        lines = [f"input: {self.input_size} features"]
        for index, layer in enumerate(self.layers):
            kind = "output" if index == len(self.layers) - 1 else f"hidden {index + 1}"
            lines.append(f"{kind}: {layer.units} units, {layer.activation}, {layer.weights_initializer}")
        return "\n".join(lines)


def create_network(input_size, hidden_layers, output_size=2, activation="sigmoid",
                   weights_initializer="heUniform", seed=None):
    """Builds hidden DenseLayers of the given sizes followed by a softmax output layer."""
    layers = [DenseLayer(units, activation, weights_initializer) for units in hidden_layers]
    layers.append(DenseLayer(output_size, "softmax", weights_initializer))
    return Network(layers, input_size, seed=seed)


def save_model(path, network, scaler):
    """
    Saves the topology, the weights and the scaler used during training into one .npz file.
    The scaler is saved so prediction applies exactly the same preprocessing.
    """
    arrays = {f"parameter_{index}": parameter for index, parameter in enumerate(network.parameters())}
    config = {"network": network.get_config(), "scaler": scaler.to_dict()}
    np.savez(path, config=np.array(json.dumps(config)), **arrays)


def load_model(path):
    """Returns (network, scaler_values)."""
    with np.load(path, allow_pickle=False) as archive:
        config = json.loads(str(archive["config"]))
        parameter_count = len([key for key in archive.files if key.startswith("parameter_")])
        weights = [archive[f"parameter_{index}"] for index in range(parameter_count)]
    network_config = config["network"]
    layers = [DenseLayer(**layer_config) for layer_config in network_config["layers"]]
    network = Network(layers, network_config["input_size"])
    network.set_weights(weights)
    return network, config["scaler"]

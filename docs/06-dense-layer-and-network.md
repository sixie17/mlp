# 06 — Dense layer and network

Code: `src/utils/layers.py`, `src/utils/network.py`

## `DenseLayer` (layers.py)

One fully connected layer.

```python
DenseLayer(units, activation="sigmoid", weights_initializer="heUniform")
```

### Attributes

| Attribute | Content |
|-----------|---------|
| `units` | number of neurons |
| `activation`, `weights_initializer` | names (strings), saved in the model file |
| `activation_function`, `activation_derivative` | the functions, looked up in `ACTIVATIONS` |
| `initializer` | the initializer, looked up in `INITIALIZERS` |
| `W`, `b` | parameters, shapes `(input_size, units)` and `(1, units)` |
| `x`, `z` | **cache** filled by `forward`, read by `backward` |

### Methods

**`build(input_size, random_generator)`** — creates `W` with the initializer and `b` with zeros. Called by
`Network`, which knows the size of the previous layer.

**`forward(x)`** — the feedforward of one layer:

```python
self.X = x                        # remember the input
self.Z = x @ self.W + self.b      # weighted sums
return self.activation_function(self.Z) # output
```

The input `x` and `z` are stored because backpropagation needs them:
`dW` depends on `x`, and the activation derivative depends on `z`.

**`backward(dZ)`** — given `dZ = ∂Loss/∂Z` for this layer (already divided by the batch size):

```python
dW = self.X.T @ dZ                       # (input_size, batch) @ (batch, units) = (input_size, units)
db = np.sum(dZ, axis=0, keepdims=True)   # (1, units)
dX = dZ @ self.W.T                       # (batch, units) @ (units, input_size) = (batch, input_size)
return dW, db, dX
```

`dX` is the gradient for the previous layer's output. Derivation in [08](08-backpropagation.md).

**`activation_backward(d_output)`** — chain rule through the activation: `dZ = d_output * f'(z)`. Raises an
error for softmax, which has no element-wise derivative.

**`get_config()`** — `{"units", "activation", "weights_initializer"}`: everything needed to rebuild
the layer when loading a model.

## `Network` (network.py)

A list of `DenseLayer`s.

```python
Network(layers, input_size, seed=None)
```

### Validation at creation

- at least 2 layers (one hidden + the output);
- the last layer must be softmax;
- softmax is forbidden on hidden layers.

Then it builds every layer, chaining the sizes:

```python
input_size = input_size                 # 30
for layer in layers:
    layer.build(input_size, random_generator)       # 30→24, then 24→24, then 24→2
    input_size = layer.units
```

### Feedforward: `forward(x)`

```python
for layer in self.layers:
    x = layer.forward(x)           # each output becomes the next input
return x                           # (batch, 2) probabilities
```

`predict_probabilities(x)` is the same thing, named for readability where we only want predictions.
The predicted class is `np.argmax(probabilities, axis=1)`: the class with the higher probability
(equivalent to "P(M) > 0.5").

### Backpropagation: `backward(probabilities, y)`

```python
m = probabilities.shape[0]
dZ = (probabilities - one_hot(y, probabilities.shape[1])) / m        # gradient at the output
gradients = []
for i in reversed(range(len(self.layers))):          # last layer first
    dW, db, dX = self.layers[i].backward(dZ)
    gradients.append(db)
    gradients.append(dW)
    if i > 0:
        dZ = self.layers[i - 1].activation_backward(dX)
gradients.reverse()                                       # [dW0, db0, dW1, db1, dW2, db2]
return gradients
```

Explained in detail in [08](08-backpropagation.md). Note that `backward` only **computes**
gradients; it does not change any weight. The optimizer does that afterwards.

### Parameters and weights

- `parameters()` → `[W0, b0, W1, b1, W2, b2]`: the **actual arrays** inside the layers (not
  copies). The optimizer modifies them in place, which updates the network.
- `get_weights()` → copies of those arrays (used by early stopping to remember the best epoch).
- `set_weights(weights)` → puts copies back, checking that every shape matches.

### Description

- `get_config()` → `{"input_size": 30, "layers": [layer configs]}`.
- `summary()` → the text printed at the start of training:

```
input: 30 features
hidden 1: 24 units, sigmoid, heUniform
hidden 2: 24 units, sigmoid, heUniform
output: 2 units, softmax, heUniform
```

## `create_network(...)`

```python
create_network(input_size, hidden_layers, output_dim=2, activation="sigmoid",
               weights_initializer="heUniform", seed=None)
```

Shortcut used by training: one `DenseLayer` per number in `hidden_layers` (that is the `--layer`
option), then a 2-unit softmax output layer.

## Saving and loading the model

The subject requires saving the **topology and the weights**. We also save the **scaler** (training
mean and standard deviation), otherwise prediction could not standardize new data the same way.

### `save_model(path, network, scaler)`

Writes one NumPy `.npz` archive (a zip of arrays):

| Key | Content |
|-----|---------|
| `config` | a JSON string: `{"network": {input_size, layers}, "scaler": {mean, standard deviation}}` |
| `parameter_0 … parameter_5` | `W0, b0, W1, b1, W2, b2` |

### `load_model(path)`

1. Opens the file with `allow_pickle=False` — a model file can only contain plain arrays and text,
   never executable Python objects, so loading a file is safe.
2. Reads the JSON config and the `param_*` arrays in order.
3. Rebuilds the layers with `DenseLayer(**layer_config)` and a `Network`.
4. `set_weights` replaces the random initial weights with the saved ones.
5. Returns `(network, scaler_dict)`.

We verified the round trip: `predict` on the validation set gives exactly the last epoch's
`val_loss` (0.0930).

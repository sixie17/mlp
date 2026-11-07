# 06 — Dense layer and network

Code: `src/lib/layers.py`, `src/lib/network.py`

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
| `activation_fn`, `activation_derivative` | the functions, looked up in `ACTIVATIONS` |
| `init_fn` | the initializer, looked up in `INITIALIZERS` |
| `W`, `b` | parameters, shapes `(n_in, units)` and `(1, units)` |
| `x`, `z` | **cache** filled by `forward`, read by `backward` |

### Methods

**`build(input_dim, rng)`** — creates `W` with the initializer and `b` with zeros. Called by
`Network`, which knows the size of the previous layer.

**`forward(x)`** — the feedforward of one layer:

```python
self.x = x                        # remember the input
self.z = x @ self.W + self.b      # weighted sums
return self.activation_fn(self.z) # output
```

The input `x` and `z` are stored because backpropagation needs them:
`dW` depends on `x`, and the activation derivative depends on `z`.

**`backward(dz)`** — given `dz = ∂Loss/∂Z` for this layer (already divided by the batch size):

```python
dW = self.x.T @ dz                       # (n_in, batch) @ (batch, units) = (n_in, units)
db = np.sum(dz, axis=0, keepdims=True)   # (1, units)
dx = dz @ self.W.T                       # (batch, units) @ (units, n_in) = (batch, n_in)
return dW, db, dx
```

`dx` is the gradient for the previous layer's output. Derivation in [08](08-backpropagation.md).

**`activation_backward(da)`** — chain rule through the activation: `dz = da * f'(z)`. Raises an
error for softmax, which has no element-wise derivative.

**`get_config()`** — `{"units", "activation", "weights_initializer"}`: everything needed to rebuild
the layer when loading a model.

## `Network` (network.py)

A list of `DenseLayer`s.

```python
Network(layers, input_dim, seed=None)
```

### Validation at creation

- at least 2 layers (one hidden + the output);
- the last layer must be softmax;
- softmax is forbidden on hidden layers.

Then it builds every layer, chaining the sizes:

```python
in_dim = input_dim                 # 30
for layer in layers:
    layer.build(in_dim, rng)       # 30→24, then 24→24, then 24→2
    in_dim = layer.units
```

### Feedforward: `forward(x)`

```python
for layer in self.layers:
    x = layer.forward(x)           # each output becomes the next input
return x                           # (batch, 2) probabilities
```

`predict_proba(x)` is the same thing, named for readability where we only want predictions.
The predicted class is `np.argmax(probs, axis=1)`: the class with the higher probability
(equivalent to "P(M) > 0.5").

### Backpropagation: `backward(probs, y)`

```python
m = probs.shape[0]
dz = (probs - one_hot(y, probs.shape[1])) / m        # gradient at the output
grads = []
for i in reversed(range(len(self.layers))):          # last layer first
    dW, db, dx = self.layers[i].backward(dz)
    grads.append(db)
    grads.append(dW)
    if i > 0:
        dz = self.layers[i - 1].activation_backward(dx)
grads.reverse()                                       # [dW0, db0, dW1, db1, dW2, db2]
return grads
```

Explained in detail in [08](08-backpropagation.md). Note that `backward` only **computes**
gradients; it does not change any weight. The optimizer does that afterwards.

### Parameters and weights

- `parameters()` → `[W0, b0, W1, b1, W2, b2]`: the **actual arrays** inside the layers (not
  copies). The optimizer modifies them in place, which updates the network.
- `get_weights()` → copies of those arrays (used by early stopping to remember the best epoch).
- `set_weights(weights)` → puts copies back, checking that every shape matches.

### Description

- `get_config()` → `{"input_dim": 30, "layers": [layer configs]}`.
- `summary()` → the text printed at the start of training:

```
input: 30 features
hidden 1: 24 units, sigmoid, heUniform
hidden 2: 24 units, sigmoid, heUniform
output: 2 units, softmax, heUniform
```

## `create_network(...)`

```python
create_network(input_dim, hidden_layers, output_dim=2, activation="sigmoid",
               weights_initializer="heUniform", seed=None)
```

Shortcut used by training: one `DenseLayer` per number in `hidden_layers` (that is the `--layer`
option), then a 2-unit softmax output layer.

## Saving and loading the model

The subject requires saving the **topology and the weights**. We also save the **scaler** (training
mean and std), otherwise prediction could not standardize new data the same way.

### `save_model(path, network, scaler)`

Writes one NumPy `.npz` archive (a zip of arrays):

| Key | Content |
|-----|---------|
| `config` | a JSON string: `{"network": {input_dim, layers}, "scaler": {mean, std}}` |
| `param_0 … param_5` | `W0, b0, W1, b1, W2, b2` |

### `load_model(path)`

1. Opens the file with `allow_pickle=False` — a model file can only contain plain arrays and text,
   never executable Python objects, so loading a file is safe.
2. Reads the JSON config and the `param_*` arrays in order.
3. Rebuilds the layers with `DenseLayer(**layer_config)` and a `Network`.
4. `set_weights` replaces the random initial weights with the saved ones.
5. Returns `(network, scaler_dict)`.

We verified the round trip: `predict` on the validation set gives exactly the last epoch's
`val_loss` (0.0930).

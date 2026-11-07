# 04 — Activation functions

Code: `src/lib/activations.py`

## Why activations exist

A layer computes `Z = X @ W + b`, which is **linear**. Stacking linear layers gives another linear
function: `(X @ W1) @ W2 = X @ (W1 @ W2)`. A 10-layer linear network is no more powerful than a
single layer.

An activation `f` applied after each layer breaks this linearity, so stacked layers can represent
curved decision boundaries.

## The functions

Each activation comes with its **derivative**, which backpropagation needs. Derivatives are
computed from `Z` (the value before the activation), stored by the layer during the forward pass.

### sigmoid (default for hidden layers)

```
sigmoid(z)  = 1 / (1 + e^(−z))                   output in (0, 1)
sigmoid'(z) = sigmoid(z) · (1 − sigmoid(z))      maximum 0.25 at z = 0
```

- Smooth, "S"-shaped, squashes any number into (0, 1).
- Weakness: for large |z| the curve is flat, the derivative is close to 0, and the layers before it
  learn very slowly (*vanishing gradient*). Fine for 2–3 layers, bad for deep networks.
- Implementation detail: `np.clip(z, -500, 500)` before `np.exp` so `e^(500+)` never overflows.

### relu

```
relu(z)  = max(0, z)
relu'(z) = 1 if z > 0 else 0
```

- Very cheap, and its derivative is exactly 1 for positive inputs, so gradients don't shrink.
  The standard choice for deep networks.
- Weakness: a neuron whose `z` is negative for every sample outputs 0 and gets gradient 0 —
  it stops learning (*dead ReLU*).
- The derivative at exactly `z = 0` is undefined; we use 0.

### tanh

```
tanh(z)  = (e^z − e^(−z)) / (e^z + e^(−z))       output in (−1, 1)
tanh'(z) = 1 − tanh²(z)
```

- Like sigmoid but centered on 0, which usually trains a bit faster. Same vanishing-gradient
  weakness.

### identity

```
identity(z) = z,  identity'(z) = 1
```

No non-linearity. Available for experiments; a network with only identity hidden layers behaves
like logistic regression.

### softmax (output layer only)

```
softmax(z)_k = e^(z_k) / Σ_j e^(z_j)
```

Turns the output scores of each sample into **probabilities that sum to 1**:

```
z = [1.0, 3.0]  →  e^z = [2.72, 20.09]  →  softmax = [0.119, 0.881]     → predicts M
```

- Required by the subject on the output layer.
- **Numerical stability:** we subtract the row maximum first:
  `softmax(z) = softmax(z − max(z))` (the extra factor cancels out), so `e^z` never overflows.
- It has no element-wise derivative in the table (`None`): each output depends on **all** inputs.
  We never need it alone, because combined with cross-entropy the gradient simplifies to
  `probs − one_hot(y)` (see [08](08-backpropagation.md)). That is why `Network` refuses softmax on
  a hidden layer and requires it on the last one.

## The registry

```python
ACTIVATIONS = {
    "identity": (identity, identity_derivative),
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "softmax": (softmax, None),
}

def get_activation(name):          # returns (function, derivative) or raises ValueError
```

`DenseLayer` looks up its activation by name, which is what makes `--activation relu` work and
what lets the saved model store the activation as a simple string.

## Which one to choose

| Situation | Choice |
|-----------|--------|
| 2–3 hidden layers (our case) | sigmoid or tanh work fine; relu too |
| deeper networks | relu (+ He initialization) |
| output layer for classification | softmax (mandatory here) |

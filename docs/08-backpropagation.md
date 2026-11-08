# 08 — Backpropagation

Code: `Network.backward` in `src/lib/network.py`, `DenseLayer.backward` and
`DenseLayer.activation_backward` in `src/lib/layers.py`

## The question backpropagation answers

For **every** weight and bias: *if I increase it a tiny bit, does the loss go up or down, and by
how much?* That number is the **partial derivative** `∂L/∂w`. All of them together form the
**gradient**. Gradient descent then moves each parameter against its derivative
(see [09](09-gradient-descent-and-optimizers.md)).

There are 1 394 parameters. Trying each one separately would need 1 394 extra forward passes per
step. Backpropagation computes **all** of them in one backward pass, by reusing intermediate
results with the **chain rule**.

## The chain rule

If `L` depends on `z`, and `z` depends on `w`, then

```
∂L/∂w = ∂L/∂z · ∂z/∂w
```

A network is a chain of simple functions:

```
X → [Z0 = X W0 + b0] → [A0 = f(Z0)] → [Z1 = A0 W1 + b1] → [A1 = f(Z1)] → [Z2 = A1 W2 + b2] → [P = softmax(Z2)] → L
```

So we start from the loss and go **backwards**, multiplying local derivatives one step at a time.
Each layer only needs the gradient coming from the layer after it.

## Notation

- `m`: batch size (8)
- `dZ` means `∂L/∂Z`, `dW` means `∂L/∂W`, etc. Each has the same shape as the thing it is the
  derivative of.
- `*` is element-wise multiplication, `@` is matrix multiplication.

## Step 1 — the output layer: `dZ = (P − Y) / m`

The loss is the mean cross-entropy over the batch, with softmax output `P` and one-hot targets `Y`.
For one sample and output score `z_k`:

```
L = −Σⱼ yⱼ · log(pⱼ)              pⱼ = e^(zⱼ) / Σᵢ e^(zᵢ)

∂pⱼ/∂z_k = pⱼ (1 − p_k)   if j = k
         = −pⱼ p_k         if j ≠ k

∂L/∂z_k = −Σⱼ yⱼ · (1/pⱼ) · ∂pⱼ/∂z_k
        = −y_k (1 − p_k) + Σ_{j≠k} yⱼ p_k
        = −y_k + p_k · Σⱼ yⱼ
        = p_k − y_k                (because Σⱼ yⱼ = 1)
```

Softmax and cross-entropy cancel out into the very simple **`p − y`**: predicted probability minus
target. Because the loss is a **mean** over `m` samples, each sample's gradient is divided by `m`.

```python
dZ = (probabilities - one_hot(y, probabilities.shape[1])) / m
```

Example, one malignant tumor (target `[0, 1]`), prediction `[0.3, 0.7]`:
`dZ = [0.3, −0.3]` → "lower the B score, raise the M score".

## Step 2 — gradients of a dense layer

Given `dZ` for a layer with input `X` and `Z = X @ W + b`:

```
dW = Xᵀ @ dZ                (input_size, m) @ (m, output_size) → (input_size, output_size)
db = Σ over the batch of dZ  (1, output_size)
dX = dZ @ Wᵀ                (m, output_size) @ (output_size, input_size) → (m, input_size)
```

Why:

- `z_j = Σᵢ xᵢ wᵢⱼ + b_j`, so `∂z_j/∂wᵢⱼ = xᵢ`. Weight `wᵢⱼ` matters in proportion to the input
  `xᵢ` that flows through it. Summed over the batch: `Xᵀ @ dZ`.
- `∂z_j/∂b_j = 1`, so `db` is just `dZ` summed over the samples (the bias is added to every sample).
- `∂z_j/∂xᵢ = wᵢⱼ`; input `xᵢ` feeds every neuron `j`, so its gradient sums the contributions:
  `dZ @ Wᵀ`. That is the error "sent back" to the previous layer.

A good sanity check: every gradient has exactly the shape of its parameter.

```python
def backward(self, dZ):
    dW = self.X.T @ dZ
    db = np.sum(dZ, axis=0, keepdims=True)
    dX = dZ @ self.W.T
    return dW, db, dX
```

## Step 3 — through the activation of the previous layer

`dX` from layer `l` is `∂L/∂A` for layer `l−1` (its output). Since `A = f(Z)` element by element:

```
dZ_prev = dX * f'(Z_prev)
```

```python
def activation_backward(self, d_output):
    return d_output * self.activation_derivative(self.Z)     # z stored during forward
```

With sigmoid, `f'(z) = s(1 − s)` is at most 0.25: every sigmoid layer shrinks the gradient by at
least 4×. That is the *vanishing gradient* problem and why deep networks prefer ReLU.

## Putting it together: `Network.backward`

```python
dZ = (probabilities - one_hot(y)) / m                       # step 1
for i in reversed(range(len(self.layers))):         # 2, 1, 0
    dW, db, dX = self.layers[i].backward(dZ)        # step 2
    gradients.append(db); gradients.append(dW)
    if i > 0:
        dZ = self.layers[i - 1].activation_backward(dX)   # step 3
gradients.reverse()                                     # [dW0, db0, dW1, db1, dW2, db2]
```

For the default network:

```
dZ2 = (P − Y)/m                         (8, 2)
dW2 = A1ᵀ @ dZ2                         (24, 2)      db2 (1, 2)
dA1 = dZ2 @ W2ᵀ                         (8, 24)
dZ1 = dA1 * sigmoid'(Z1)                (8, 24)
dW1 = A0ᵀ @ dZ1                         (24, 24)     db1 (1, 24)
dA0 = dZ1 @ W1ᵀ                         (8, 24)
dZ0 = dA0 * sigmoid'(Z0)                (8, 24)
dW0 = Xᵀ @ dZ0                          (30, 24)     db0 (1, 24)
```

## Important details

- **The forward pass must run first**: `backward` reads the cached `x` and `z` of each layer.
- **All gradients are computed before any weight changes.** `dX = dZ @ Wᵀ` uses the same `W` that
  produced the prediction. The optimizer updates the weights only after `backward` returns.
- **Order matters**: `gradients` is in the same order as `network.parameters()`, so the optimizer
  can pair them with `zip`.

## How we know it is correct: gradient checking

A derivative can be approximated numerically:

```
∂L/∂w ≈ ( L(w + ε) − L(w − ε) ) / (2ε)      with ε = 1e-6
```

We compared this with backpropagation for every parameter of a small network: the difference was
about **1e-10** for sigmoid and tanh — the analytic gradients are correct. (With ReLU one value
differed because a neuron had `z` exactly 0, where ReLU has a kink and no derivative; that is
expected, not a bug.)

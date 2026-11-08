# 09 — Gradient descent and optimizers

Code: `src/lib/optimizers.py`

## Gradient descent

The gradient `∂L/∂w` points in the direction where the loss **increases** fastest. To decrease the
loss, move each parameter a small step in the **opposite** direction:

```
w ← w − η · ∂L/∂w           η (eta) = learning rate
```

Picture standing on a hilly landscape (the loss) in fog: you feel the slope under your feet and
take a step downhill. Repeat until you reach a valley.

### The learning rate

| Learning rate | Effect |
|---------------|--------|
| too small | very slow progress, may stop before reaching a good valley |
| good | the loss decreases steadily |
| too large | steps jump over the valley, the loss bounces or explodes |

Defaults: **0.05** for SGD (mandatory), **0.001** for Adam (bonus — Adam adapts its steps, so it
needs a smaller base rate).

### Batch, mini-batch, stochastic

The gradient can be computed on:

| Variant | Gradient computed on | Updates per epoch (456 rows) |
|---------|----------------------|-----------------------------:|
| batch gradient descent | all rows | 1 |
| **mini-batch** (what we use, `--batch_size 8`) | 8 rows | 57 |
| stochastic (SGD in the strict sense) | 1 row | 456 |

Mini-batches are the usual compromise: many updates per epoch (fast learning), matrix operations
on several rows at once (efficient), and a little noise in the gradient that helps avoid getting
stuck. In code, the class is called `SGD` as in most libraries, even though it is used with
mini-batches.

## The common interface

```python
class Optimizer:
    def __init__(self, learning_rate): ...
    def apply(self, parameters, gradients): ...     # updates parameters IN PLACE
```

- `parameters` = `network.parameters()` = `[W0, b0, W1, b1, W2, b2]` (the real arrays)
- `gradients`  = `network.backward(...)` = `[dW0, db0, …]` in the same order

**In place** (`p -= …`, not `p = p - …`) matters: `p -= …` modifies the array that lives inside the
layer; `p = p - …` would create a new array and leave the network unchanged.

Optimizers with memory (all except SGD) keep one state array per parameter, in a dictionary keyed
by the parameter's position `i`.

## SGD — mandatory part

```python
for p, g in zip(parameters, gradients):
    p -= self.learning_rate * g
```

Exactly the formula above. This is what the mandatory program always uses.

## Bonus optimizers

### Momentum

```
v ← μ·v − η·g          (μ = 0.9)
w ← w + v
```

`v` is a **velocity**: an exponential average of past steps. Like a ball rolling downhill, it keeps
going in consistent directions (faster progress) and the zigzags cancel out.

### Nesterov momentum

```
v ← μ·v − η·g
w ← w + μ·v − η·g
```

"Look ahead": the step is corrected using where momentum is about to take us. Usually a bit more
stable than plain momentum. (This is the common rewritten form used by Keras and PyTorch.)

### RMSprop

```
s ← ρ·s + (1 − ρ)·g²          (ρ = 0.9)
w ← w − η · g / (√s + ε)
```

`s` tracks the recent size of each parameter's gradient. Parameters with large gradients get
smaller steps, parameters with small gradients get larger steps: a **per-parameter learning rate**.
`ε = 1e-7` avoids division by zero.

### Adam (Adaptive Moment Estimation)

Momentum + RMSprop:

```
t ← t + 1
m ← β₁·m + (1 − β₁)·g          first moment: average gradient  (β₁ = 0.9)
v ← β₂·v + (1 − β₂)·g²         second moment: average squared gradient  (β₂ = 0.999)
m̂ = m / (1 − β₁ᵗ)              bias correction
v̂ = v / (1 − β₂ᵗ)
w ← w − η · m̂ / (√v̂ + ε)        (ε = 1e-8)
```

**Bias correction:** `m` and `v` start at 0, so during the first steps they are too small (biased
toward 0). Dividing by `1 − βᵗ` fixes that; as `t` grows, the correction tends to 1.

`t` increases once per `apply` call (once per mini-batch), not once per parameter.

## The registry

```python
OPTIMIZERS = {"sgd": ..., "momentum": ..., "nesterov": ..., "rmsprop": ..., "adam": ...}
def get_optimizer(name, learning_rate)
```

Used by the bonus `--optimizer` option.

## Comparison on this dataset (24-24 sigmoid, 70 epochs, seed-42 split)

| Optimizer | learning rate | final train loss | final validation loss | final validation accuracy |
|-----------|---:|-----------------:|---------------:|--------------:|
| sgd | 0.05 | 0.041 | 0.093 | 95.6% |
| nesterov | 0.01 | 0.030 | 0.107 | 94.7% |
| rmsprop | 0.001 | 0.032 | 0.137 | 94.7% |
| adam + early stopping | 0.001 | — | 0.078 | 97.3% |

The adaptive optimizers reach a lower **training** loss faster, and then start to overfit
(validation loss goes back up). With early stopping, Adam gives the best validation result.

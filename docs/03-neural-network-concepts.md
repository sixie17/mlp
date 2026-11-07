# 03 — Neural network concepts

No code in this file: these are the ideas behind everything else.

## The perceptron (one neuron)

A neuron takes inputs `x₀ … xₙ₋₁`, multiplies each by a **weight**, adds a **bias**, and passes
the result through an **activation function**:

```
z = x₀·w₀ + x₁·w₁ + … + xₙ₋₁·wₙ₋₁ + b        (weighted sum)
a = f(z)                                     (activation = output of the neuron)
```

- The **weights** say how important each input is (positive, negative, large, small).
- The **bias** shifts the threshold at which the neuron "activates".
- The **activation function** adds non-linearity (see [04](04-activations.md)).

Weights and biases are the **parameters**: the numbers the network learns.

## The multilayer perceptron (MLP)

Neurons are organized in **layers**. In a *dense* (fully connected) layer, every neuron receives
**all** the outputs of the previous layer.

```
input layer      hidden layer 1    hidden layer 2    output layer
(30 features)    (24 neurons)      (24 neurons)      (2 neurons, softmax)

  x₀ ──┐          ┌─ a ─┐           ┌─ a ─┐
  x₁ ──┼──────────┼─ a ─┼───────────┼─ a ─┼──────────  P(B)
  …  ──┤  all-to- ├─ …  ┤  all-to-  ├─ …  ┤  all-to-
  x₂₉──┘    all   └─ a ─┘    all    └─ a ─┘    all     P(M)
```

- **Input layer**: just the 30 standardized features. It has no weights.
- **Hidden layers**: where the network builds its own intermediate features
  (combinations like "large *and* irregular").
- **Output layer**: 2 neurons with softmax → a probability for each class.

It is called **feedforward** because data only flows forward, from input to output — no loops.

## From neurons to matrices

Computing each neuron one by one would be slow. A whole layer, for a whole batch of samples, is
one matrix multiplication:

```
Z = X @ W + b
A = f(Z)
```

| Symbol | Shape | Meaning |
|--------|-------|---------|
| `X` | (batch, n_in) | one row per sample, one column per input |
| `W` | (n_in, n_out) | column `j` = the weights of neuron `j` |
| `b` | (1, n_out) | one bias per neuron, added to every row |
| `Z` | (batch, n_out) | weighted sums |
| `A` | (batch, n_out) | layer output = next layer's `X` |

For the default network (`--layer 24 24`) and a batch of 8 tumors:

```
X  (8, 30)  @ W0 (30, 24) + b0 (1, 24) → sigmoid → (8, 24)
   (8, 24)  @ W1 (24, 24) + b1 (1, 24) → sigmoid → (8, 24)
   (8, 24)  @ W2 (24, 2)  + b2 (1, 2)  → softmax → (8, 2)    [P(B), P(M)] per tumor
```

**Number of parameters** of a layer = `n_in × n_out + n_out`:

| Layer | Weights | Biases | Total |
|-------|--------:|-------:|------:|
| hidden 1 | 30 × 24 = 720 | 24 | 744 |
| hidden 2 | 24 × 24 = 576 | 24 | 600 |
| output | 24 × 2 = 48 | 2 | 50 |
| **network** | | | **1 394** |

The subject's diagram writes a weight matrix as `W_{l0 l1}` of size (3, 4) — that is
(neurons in l₁, neurons in l₀). We store the transpose, (n_in, n_out), so that a batch of
row-vectors can be multiplied as `X @ W`. Same numbers, different orientation.

## What "learning" means

At first the weights are random, so the predictions are random. Learning repeats three steps:

1. **Feedforward** — compute the predictions for a few samples.
2. **Loss** — measure how wrong they are with one number (see [07](07-loss-functions.md)).
3. **Backpropagation + gradient descent** — compute, for every weight, in which direction it
   should move to reduce the loss, and move it a little (see [08](08-backpropagation.md) and
   [09](09-gradient-descent-and-optimizers.md)).

After a few thousand small steps the weights encode a rule that separates M from B.

## Vocabulary

| Term | Meaning |
|------|---------|
| sample / example | one tumor = one row |
| feature | one input column |
| label / target | the true answer (M or B) |
| parameters | the weights and biases |
| hyperparameters | settings you choose: layers, learning rate, batch size, epochs… |
| batch / mini-batch | the few samples used for one update (default 8) |
| epoch | one full pass over the training set |
| logits / scores | `Z` of the output layer, before softmax |
| gradient | the vector of partial derivatives of the loss w.r.t. the parameters |
| generalization | doing well on data never seen during training |
| overfitting | doing well on training data but worse on new data (memorizing) |

## Why it's called "deep" learning and why we need hidden layers

Without hidden layers (input → softmax directly) the model is **logistic regression**: it can only
draw a straight boundary (a hyperplane) between M and B. Each hidden layer with a non-linear
activation lets the network bend that boundary. Two hidden layers can approximate very complex
boundaries. For this dataset, which is almost linearly separable, two small hidden layers are more
than enough.

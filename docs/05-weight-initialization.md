# 05 — Weight initialization

Code: `src/utils/initializers.py`

## Why weights start random

If every weight started at the same value (say 0), all neurons of a layer would compute the same
output, receive the same gradient, and stay identical forever: 24 neurons would act like 1.
Random starting values **break the symmetry** so each neuron can learn something different.

Biases can start at 0 (`DenseLayer.build` uses `np.zeros`): the random weights are enough to make
neurons differ.

## Why the scale matters

- Weights **too large** → `Z` values are large → sigmoid/tanh saturate (flat region, derivative ≈ 0)
  or ReLU outputs explode from layer to layer.
- Weights **too small** → the signal shrinks a bit more at every layer and the gradients vanish.

The goal is to keep the **variance** of the activations roughly the same from layer to layer.
The right scale depends on how many inputs a neuron has (`input_size`, also called *fan-in*).

## The initializers

All take `(input_size, output_size, random_generator)` and return a matrix of shape `(input_size, output_size)`.

### Xavier / Glorot (for sigmoid, tanh)

```
xavierUniform:  W ~ Uniform(−L, +L),   L = √(6 / (input_size + output_size))
xavierNormal:   W ~ Normal(0, σ),      σ = √(2 / (input_size + output_size))
```

Balances the variance for the forward pass (depends on `input_size`) and the backward pass
(depends on `output_size`). Designed for activations that are symmetric around 0.

### He / Kaiming (for ReLU)

```
heUniform:  W ~ Uniform(−L, +L),   L = √(6 / input_size)
heNormal:   W ~ Normal(0, σ),      σ = √(2 / input_size)
```

ReLU sets about half of its inputs to 0, which halves the variance; He compensates with a factor
2 larger than Xavier.

Example for the first layer (30 → 24):

| Initializer | Range |
|-------------|-------|
| xavierUniform | ±√(6/54) = ±0.333 |
| heUniform | ±√(6/30) = ±0.447 |

`glorotUniform` / `glorotNormal` are aliases of the Xavier versions (same person, two names).

## Reproducibility

The `Network` creates `random_generator = np.random.default_rng(seed)` and passes it to every layer's `build`,
so with the same `--seed` you get exactly the same starting weights — and therefore the same
training run.

## Defaults in this project

`heUniform` is the default because the subject's example uses it. With sigmoid hidden layers
`xavierUniform` is the textbook match; on this small network both train well.

| Activation | Recommended |
|------------|-------------|
| relu | heUniform / heNormal |
| sigmoid, tanh | xavierUniform / xavierNormal |

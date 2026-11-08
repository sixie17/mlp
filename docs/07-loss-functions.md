# 07 — Loss functions

Code: `src/lib/losses.py`

## What a loss is

The **loss** turns "how wrong are the predictions" into one number. Training = making that number
as small as possible. It must be:

- small when the predictions are good, large when they are bad;
- differentiable, so we can compute which way to move each weight.

All our losses take `probabilities` (softmax output, shape `(n, 2)`) and integer labels `y` (shape `(n,)`).

## Cross-entropy, intuitively

For one sample, cross-entropy is `−log(probability given to the correct class)`:

| P(correct class) | loss |
|-----------------:|-----:|
| 0.99 | 0.010 |
| 0.90 | 0.105 |
| 0.50 | 0.693 |
| 0.10 | 2.303 |
| 0.01 | 4.605 |

Being confident and right costs almost nothing; being confident and **wrong** costs a lot. That
pushes the network to be both correct and well-calibrated.

A model that always answers 50/50 has loss `log 2 ≈ 0.693`. That is why the first epochs start
around 0.5–0.7.

## Binary cross-entropy — `binary_cross_entropy(probabilities, y)`

The formula in the subject (used by `predict`):

```
E = −(1/N) · Σₙ [ yₙ · log(pₙ) + (1 − yₙ) · log(1 − pₙ) ]
```

- `pₙ` = predicted probability of malignant = `probabilities[:, 1]`;
- if the tumor is M (`y = 1`) only `log(p)` counts; if B (`y = 0`) only `log(1 − p)` counts.

## Categorical cross-entropy — `categorical_cross_entropy(probabilities, y)`

The general form for K classes, with one-hot targets:

```
L = −(1/N) · Σₙ Σₖ yₙₖ · log(pₙₖ)
```

Only the term of the correct class is non-zero, so it is again `−log(P(correct class))`.

## Why both give the same number here

With a 2-unit softmax, `P(B) = 1 − P(M)`. For one sample:

```
categorical:  −[ y_B·log P(B) + y_M·log P(M) ]
binary:       −[ y·log P(M) + (1 − y)·log(1 − P(M)) ]
```

With `y_M = y`, `y_B = 1 − y` and `P(B) = 1 − P(M)`, the two lines are identical. So:

- `--loss categoricalCrossentropy` (default) and `--loss binaryCrossentropy` print the same values;
- the gradient used in backpropagation is the same (`probabilities − one_hot(y)`).

## Numerical safety

`log(0) = −∞`. If the network predicts exactly 0 for the correct class, the loss would be infinite.
We clip probabilities to `[1e-15, 1 − 1e-15]` (`EPSILON = 1e-15`) before taking the log.

## The registry

```python
LOSSES = {
    "binaryCrossentropy": binary_cross_entropy,
    "categoricalCrossentropy": categorical_cross_entropy,
}
def get_loss(name):     # function or ValueError
```

## Where losses are used

| Place | Loss |
|-------|------|
| `training.evaluate` (each epoch, train and validation) | the one chosen with `--loss` |
| `pipeline.run_predict` | always binary cross-entropy (subject requirement) |
| backpropagation | not called directly: its gradient `probabilities − one_hot(y)` is written in `Network.backward` |
